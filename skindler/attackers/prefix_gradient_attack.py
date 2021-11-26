from copy import copy
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from skindler.attackers import Attacker, GradientGuidedAttack

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


@Attacker.register("prefix_gradient_attack")
class PrefixGradientGuidedAttack(GradientGuidedAttack, Attacker):
    def __init__(
            self,
            model_name,
            tokenizer_name,
            device: int = -1,
            max_iteration: int = 10):
        Attacker.__init__(self, device)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.tokenizer_vocab = self.tokenizer.get_vocab()
        self.model.model.shared.register_backward_hook(extract_grad_hook)
        self.max_iteration = max_iteration

    def gradient_attack(
            self, attack_input: Dict[str, Any], verbose=False) -> str:

        losses_history = []
        input_history = [
            self.get_input_text(
                attack_input['input_ids'],
                self.tokenizer)]
        if verbose:
            print(input_history[0])

        del attack_input['attention_mask']
        changed_input = copy(attack_input['input_ids'][0].tolist())
        changed_input.insert(0, self.tokenizer.pad_token_id)
        attack_input['input_ids'] = torch.tensor(
            changed_input).unsqueeze(0).to(self.device)

        iteration = 0
        while iteration < self.max_iteration:
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
            position = iteration

            '''
            choose token
            '''
            input_gradient = input_gradient[0][position].view(1, 1, -1)
            old_token_id = attack_input['input_ids'][0][position]
            embedding_of_old_token = self.model.model.shared(
                torch.tensor(old_token_id.view(1, -1)))

            new_embed_dot_grad = torch.einsum(
                "bij,kj->bik", (input_gradient, embedding_weight))
            prev_embed_dot_grad = torch.einsum(
                "bij,bij->bi", (input_gradient, embedding_of_old_token)).unsqueeze(-1)
            difference = new_embed_dot_grad - prev_embed_dot_grad
            difference = difference.detach().cpu().reshape(-1)

            '''
            make replacement
            '''
            new_token = difference.argmax()

            changed_input = copy(attack_input['input_ids'][0].tolist())
            changed_input[position] = new_token
            changed_input.insert(position + 1, self.tokenizer.pad_token_id)
            attack_input['input_ids'] = torch.tensor(
                changed_input).unsqueeze(0).to(self.device)

            losses_history.append(output['loss'].item())
            input_history.append(
                self.get_input_text(
                    attack_input['input_ids'],
                    self.tokenizer))
            iteration += 1

            if verbose:
                print(self.get_input_text_flipped(
                    attack_input['input_ids'],
                    self.tokenizer,
                    []))

        changed_input = copy(attack_input['input_ids'][0].tolist())
        del changed_input[iteration]
        attack_input['input_ids'] = torch.tensor(
            changed_input).unsqueeze(0).to(self.device)

        input_history.append(
            self.get_input_text_flipped(
                attack_input['input_ids'],
                self.tokenizer,
                []))
        return input_history[-1]
