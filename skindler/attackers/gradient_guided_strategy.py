from typing import Dict, Any, List

import numpy as np
import torch
from copy import copy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from skindler.attackers import AttackerInput, AttackerOutput, Attacker
from skindler import MAX_LENGTH

extracted_grads = []


def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out)


def second_letter_is_uppercase(word: str) -> bool:
    if len(word) < 2:
        return False
    if word[0] == '▁' and word[1].isupper():
        return True
    else:
        return False


@Attacker.register("gradient_attack")
class GradientGuidedAttack(Attacker):
    def __init__(
            self,
            model_name,
            tokenizer_name,
            device: int = -1,
            constraint_of_lm: bool = False,
            threshold: float = 0.75,
            lm_threshold: float = 0.9,
            max_iteration: int = 100):
        super().__init__(device)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.tokenizer_vocab = self.tokenizer.get_vocab()
        self.model.model.shared.register_backward_hook(extract_grad_hook)
        self.indexes_starting_with_underscore = np.array(
            [i.startswith('▁') for i in list(self.tokenizer.get_vocab().keys())])
        self.indexes_first_uppercase = np.array(
            [i[0].isupper() for i in list(self.tokenizer.get_vocab().keys())])
        self.indexes_second_uppercase = np.array(
            [second_letter_is_uppercase(i) for i in list(self.tokenizer.get_vocab().keys())])
        self.constraint_of_lm = constraint_of_lm
        self.lm_threshold = lm_threshold
        self.threshold = threshold
        self.max_iteration = max_iteration

    def attack_dataset(
            self,
            dataloader,
            attack_first_n: int = -1,
            verbose=False) -> List[Any]:
        result = list()
        for batch_id, batch in enumerate(tqdm(dataloader)):
            if batch_id + 1 == attack_first_n:
                break
            result.append(self.attack_batch(batch, verbose=verbose))
            if verbose:
                print()
        return result

    def attack(self,
               data_to_attack: AttackerInput,
               verbose=False
               ) -> AttackerOutput:

        attack_input = self.prepare_attack_input(data_to_attack)
        attack_input = self.move_to_device(attack_input)
        data_attacked = self.gradient_attack(attack_input)
        attack_output = self.prepare_attack_output(
            data_to_attack, data_attacked)
        return attack_output

    def prepare_attack_input(
            self, data_to_attack: AttackerInput) -> Dict[str, Any]:
        attack_input = self.tokenizer(
            data_to_attack.x,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        with self.tokenizer.as_target_tokenizer():
            attack_input["labels"] = self.tokenizer(
                data_to_attack.y,
                max_length=MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )['input_ids']

        return attack_input

    def prepare_attack_output(self, data_to_attack: AttackerInput,
                              data_attacked: str) -> AttackerOutput:

        with torch.no_grad():
            translated = self.model.generate(torch.tensor(
                self.tokenizer.encode(
                    data_to_attack.x
                )).unsqueeze(0).to(self.device))
            y_trans = self.tokenizer.decode(
                translated[0], skip_special_tokens=True)

            att_translated = self.model.generate(torch.tensor(
                self.tokenizer.encode(
                    data_attacked.replace("[[", "").replace("]]", "")
                )).unsqueeze(0).to(self.device))
            y_trans_attacked = self.tokenizer.decode(
                att_translated[0], skip_special_tokens=True)

        output = AttackerOutput(
            x=data_to_attack.x,
            y=data_to_attack.y,
            x_attacked=data_attacked,
            y_trans=y_trans,
            y_trans_attacked=y_trans_attacked
        )
        return output

    def gradient_attack(
            self, attack_input: Dict[str, Any], verbose=False) -> str:

        losses_history = []
        input_history = [
            self.get_input_text(
                attack_input['input_ids'],
                self.tokenizer)]
        if verbose:
            print(input_history[0])
        already_flipped = [0, -1]  # not replacing first and last tokens
        stop_tokens = [self.tokenizer.convert_tokens_to_ids(
            i[1]) for i in self.tokenizer.special_tokens_map.items()]
        

        iteration = 0
        while iteration < self.max_iteration:
            iteration += 1
            global extracted_grads
            extracted_grads = []
            self.model.zero_grad()

            output = self.model(**attack_input)
            output['loss'].backward()
            input_gradient = extracted_grads[1][0]
            embedding_weight = self.model.model.shared.weight
            
            if self.constraint_of_lm:
                s = {'input_ids': attack_input['input_ids'], 'attention_mask': attack_input['attention_mask']}
                old_output_of_encoder = self.model.model.encoder(**s).last_hidden_state[0]
            '''
            choose position
            '''
            norm_of_input_gradients = torch.norm(input_gradient[0], 2, dim=1)
            norm_of_input_gradients[attack_input['input_ids'].shape[1]:] = -np.inf
            for i in already_flipped:
                norm_of_input_gradients[i] = -np.inf
            if np.isinf(norm_of_input_gradients.cpu().max().numpy()):
                break
            position = norm_of_input_gradients.argmax().item()
            already_flipped.append(position)

            '''
            choose token
            '''
            input_gradient = input_gradient[0][position].view(1, 1, -1)
            old_token_id = attack_input['input_ids'][0][position]
            embedding_of_old_token = self.model.model.shared(
                torch.tensor(old_token_id.view(1, -1)))
            cosine_distances = self.get_cosine_dist(
                embedding_weight, embedding_of_old_token)

            new_embed_dot_grad = torch.einsum(
                "bij,kj->bik", (input_gradient, embedding_weight))
            prev_embed_dot_grad = torch.einsum(
                "bij,bij->bi", (input_gradient, embedding_of_old_token)).unsqueeze(-1)
            difference = new_embed_dot_grad - prev_embed_dot_grad
            difference = difference.detach().cpu().reshape(-1)

            '''
            make replacement
            '''
            difference[cosine_distances < self.threshold] = - \
                np.inf  # out only tokens which have similar embedding due to cosine embedding
            # don't replace with the same token
            difference[attack_input['input_ids'][0][position].item()] = -np.inf
            old_token = self.tokenizer.convert_ids_to_tokens(
                [old_token_id.item()])[0]
            # don't replace with the same token(upper\lower case)
            similar_token = self.get_similar_token(old_token)
            if similar_token in self.tokenizer_vocab:
                difference[self.tokenizer_vocab[similar_token]] = -np.inf
            for i in stop_tokens:
                difference[i] = -np.inf

            # replace _... with _... / ... with ...
            if old_token.startswith('▁'):
                difference[~self.indexes_starting_with_underscore] = -np.inf
                if len(old_token) > 1:
                    if old_token[1].isupper():
                        difference[~self.indexes_second_uppercase] = -np.inf
                    else:
                        difference[self.indexes_second_uppercase] = -np.inf
            else:
                difference[self.indexes_starting_with_underscore] = -np.inf
                if old_token[0].isupper():
                    difference[~self.indexes_first_uppercase] = -np.inf
                else:
                    difference[self.indexes_first_uppercase] = -np.inf

            if np.isinf(difference.numpy().max()):
                break
                
            if self.constraint_of_lm:
                def get_cosine_between_old_and_new(difference):
                    new_token = difference.argmax()
                    check_input_ids = copy(attack_input['input_ids'][0].tolist())
                    check_input_ids[position] = new_token
                    check_input_ids = torch.tensor(check_input_ids).unsqueeze(0).to(self.device)
                    check_input_attention_mask = copy(attack_input['attention_mask'])
                    check_input = {'input_ids': check_input_ids, 'attention_mask': check_input_attention_mask}
                    new_output_of_encoder = self.model.model.encoder(**check_input).last_hidden_state[0]
                    new_vector = new_output_of_encoder[position]
                    old_vector = old_output_of_encoder[position]
                    return ((new_vector @ old_vector ) / (torch.norm(new_vector) * torch.norm(old_vector))).item()  
                
                cosine_distance = -1
                
                lm_iterations = 0
                new_token = None
                while cosine_distance < self.lm_threshold and not np.isinf(difference.numpy().max()) and lm_iterations < 15:
                    lm_iterations += 1
                    cosine_distance = get_cosine_between_old_and_new(difference)
#                     print(cosine_distance)
                    if cosine_distance >= self.lm_threshold:
                        new_token = difference.argmax()
                    else:
                        difference[difference.argmax()] = -np.inf
                        
                if np.isinf(difference.numpy().max()):
                    break
                if new_token is None:
                    break
                
            else:
                new_token = difference.argmax()
            
            if verbose:
                print(
                    iteration,
                    position,
                    self.tokenizer.decode(
                        [attack_input['input_ids'][0][position].item()]),
                    self.tokenizer.decode([new_token])
                )

            changed_input = copy(attack_input['input_ids'][0].tolist())
            if changed_input[position] == new_token:
                already_flipped = already_flipped[:-1]
                break
            changed_input[position] = new_token
            attack_input['input_ids'] = torch.tensor(
                changed_input).unsqueeze(0).to(self.device)

            losses_history.append(output['loss'].item())
            input_history.append(
                self.get_input_text(
                    attack_input['input_ids'],
                    self.tokenizer))

        input_history.append(
            self.get_input_text_flipped(
                attack_input['input_ids'],
                self.tokenizer,
                already_flipped))
        if verbose:
            print(input_history[-1])
        return input_history[-1]

    @staticmethod
    def replace_token(input_ids, position, token) -> List[int]:
        '''
        replace token after 1 attack step
        '''
        input_ids = copy(input_ids)
        input_ids[position] = token
        return input_ids

    @staticmethod
    def get_input_text(input_ids, tokenizer) -> str:
        '''
        decode sentence
        '''
        return tokenizer.decode(input_ids.tolist()[0]).replace("▁", " ")

    @staticmethod
    def get_input_text_flipped(input_ids, tokenizer, already_flipped) -> str:
        tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist()[0])
        decoded_string = tokenizer.decode(input_ids.tolist()[0])
        for flipped in already_flipped[2:]:
            tokens[flipped] = f"[[{tokens[flipped]}]]"
        result_string = ""
        for token_id, token in enumerate(tokens[:-1]):
            if token.startswith("▁"):
                result_string += " "
                result_string += token[1:]
            elif token.startswith("[[▁"):
                result_string += " "
                result_string += "[["
                result_string += token[3:]
            else:
                result_string += token

        result_string = result_string.replace("]][[", "").replace("▁", " ")
        return result_string

    @staticmethod
    def get_labels_text(labels, tokenizer) -> str:
        '''
        decode original labels
        '''
        return tokenizer.decode(labels.tolist()[0]).replace("▁", " ")

    @staticmethod
    def get_cosine_dist(
            all_embeddings: torch.tensor,
            embedding: torch.tensor) -> torch.tensor:
        '''
        get cosine distance between embedding and all other embeddings
        '''
        embedding = embedding.view(-1)
        return torch.einsum('kh,h->k', all_embeddings, embedding) / \
            (embedding.norm(2) * all_embeddings.norm(2, dim=1))

    @staticmethod
    def get_similar_token(old_token: str) -> str:
        if len(old_token) == 1:
            return old_token
        if old_token.startswith('▁'):
            if old_token[1].isupper():
                similar_token = old_token[0] + \
                    old_token[1].lower() + old_token[2:]
            else:
                similar_token = old_token[0] + \
                    old_token[1].upper() + old_token[2:]
        else:
            if old_token[0].isupper():
                similar_token = old_token[0].lower() + old_token[1:]
            else:
                similar_token = old_token[0].upper() + old_token[1:]
        return similar_token
