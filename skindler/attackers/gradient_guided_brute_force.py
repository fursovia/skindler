from typing import Dict, Any, List

import numpy as np
import torch
from copy import copy, deepcopy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datasets

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


@Attacker.register("gradient_attack_brute_force")
class GradientGuidedBruteForceAttack(GradientGuidedAttack, Attacker):
    def __init__(
            self,
            model_name,
            tokenizer_name,
            device: int = -1,
            threshold: float = 0.75,
            max_iteration: int = 100,
            number_of_replacement: int = 50
    ):
        GradientGuidedAttack.__init__(
            self,
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            device=device,
            threshold=threshold,
            max_iteration=max_iteration
        )

        self.number_of_replacement = number_of_replacement
        self.model.model.shared.register_backward_hook(extract_grad_hook)
        self.coefficient_x_difference = 7
        self.metric = datasets.load_metric('bertscore')

    def gradient_attack(
            self, attack_input: Dict[str, Any]) -> str:

        losses_history = []
        input_history = [
            self.get_input_text(
                attack_input['input_ids'],
                self.tokenizer)]

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

            top_new_tokens = difference.topk(k=self.number_of_replacement)
            top_loss = -np.inf
            iteration = 0
            for replacement_index in top_new_tokens.indices:
                replacement_score = difference[replacement_index]
                if replacement_score != -np.inf:

                    def get_loss_modified(
                            replacement_index, position, attack_input):
                        attack_input_modified = deepcopy(attack_input)
                        attack_input_modified['input_ids'][0][position] = replacement_index
                        output = self.model(**attack_input_modified)
                        loss = output['loss'].item()

                        diff_x = self.metric.compute(
                            references=[[
                                input_history[0]]], predictions=[
                                self.tokenizer.decode(
                                    attack_input_modified['input_ids'][0], skip_special_tokens=True).replace(
                                    "▁", " ")], 
                            lang='en', 
                            rescale_with_baseline=True)['f1'][0]

                        diff_x_coef = diff_x * self.coefficient_x_difference
                        return loss, diff_x_coef

                    current_loss, diff_x_coef = get_loss_modified(
                        replacement_index, position, attack_input)
        
                    iteration += 1
                    if current_loss + diff_x_coef > top_loss:
                        top_loss = current_loss + diff_x_coef
                        new_token = replacement_index

            if top_loss == -np.inf:
                break

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
        return input_history[-1]
