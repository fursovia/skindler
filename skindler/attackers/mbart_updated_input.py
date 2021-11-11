from typing import Dict, Any, List

import numpy as np
import torch
from copy import copy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MBartForConditionalGeneration, MBart50TokenizerFast

from skindler.attackers import AttackerInput, AttackerOutput, Attacker, GradientGuidedAttack
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


@Attacker.register("mbart_update_input")
class MbartGradientGuidedUpdateInput(GradientGuidedAttack, Attacker):
    def __init__(
            self,
            model_name: str,
            tokenizer_name: str,
            device: int = -1,
            checkpoint_path: str = 'pytorch_model.bin',
            threshold: float = 0.75,
            max_iteration: int = 100):
        Attacker.__init__(self, device)

        self.model = MBartForConditionalGeneration.from_pretrained(
            model_name).to(self.device)
        self.model.eval()
        self.tokenizer = MBart50TokenizerFast.from_pretrained(tokenizer_name)
        self.tokenizer.src_lang = 'en_XX'
        self.tokenizer.tgt_lang = 'ru_RU'
        self.load_layers_for_predicting_bertscore(checkpoint_path)

        self.tokenizer_vocab = self.tokenizer.get_vocab()
        self.model.model.shared.register_backward_hook(extract_grad_hook)
        self.indexes_starting_with_underscore = np.array(
            [i.startswith('▁') for i in list(self.tokenizer.get_vocab().keys())])
        self.indexes_first_uppercase = np.array(
            [i[0].isupper() for i in list(self.tokenizer.get_vocab().keys())])
        self.indexes_second_uppercase = np.array(
            [second_letter_is_uppercase(i) for i in list(self.tokenizer.get_vocab().keys())])
        self.threshold = threshold
        self.max_iteration = max_iteration

    def load_layers_for_predicting_bertscore(
            self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.linear1 = torch.nn.Linear(1024 * 2, 256)
        self.linear2 = torch.nn.Linear(256, 1)

        self.linear1.weight.data = checkpoint['linear1.weight']
        self.linear1.bias.data = checkpoint['linear1.bias']
        self.linear2.weight.data = checkpoint['linear2.weight']
        self.linear2.bias.data = checkpoint['linear2.bias']
        self.linear1.to(self.device)
        self.linear2.to(self.device)
        del checkpoint

    def get_logits(self, outputs: torch.Tensor) -> torch.Tensor:
        embeddings = torch.cat(
            (
                torch.sum(outputs, dim=1),
                torch.max(outputs, dim=1).values,
            ),
            dim=1
        )
        embeddings = self.linear1(embeddings)
        embeddings = torch.relu(embeddings)
        logits = self.linear2(embeddings)
        return logits

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
                )).unsqueeze(0).to(self.device),
                forced_bos_token_id=self.tokenizer.lang_code_to_id["ru_RU"]
            )
            y_trans = self.tokenizer.decode(
                translated[0], skip_special_tokens=True)

            att_translated = self.model.generate(torch.tensor(
                self.tokenizer.encode(
                    data_attacked.replace("[[", "").replace("]]", "")
                )).unsqueeze(0).to(self.device),
                forced_bos_token_id=self.tokenizer.lang_code_to_id["ru_RU"]
            )
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

        stop_tokens = [
            self.tokenizer.convert_tokens_to_ids(
                i[1]) for i in self.tokenizer.special_tokens_map.items()] + list(
            self.tokenizer.lang_code_to_id.values())

        already_flipped = [0, -1]  # not replacing first and last tokens
        already_flipped.extend([i_id for i_id, i in enumerate(
            attack_input['input_ids'].tolist()[0]) if i in stop_tokens])
        already_flipped = list(set(already_flipped))

        iteration = 0
        while iteration < self.max_iteration:
            iteration += 1
            global extracted_grads
            extracted_grads = []
            self.model.zero_grad()

            output = self.model.model.encoder(
                input_ids=attack_input['input_ids'],
                attention_mask=attack_input['attention_mask'],
            ).last_hidden_state
            predicted_bertscore = self.get_logits(output)
            # always want to increase loss (smaller loss == better translation)
            loss = torch.nn.functional.l1_loss(
                predicted_bertscore, torch.tensor(1.0).to(self.device))
            loss.backward()

            input_gradient = extracted_grads[0][0]
            embedding_weight = self.model.model.shared.weight
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

            losses_history.append(loss.item())
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
