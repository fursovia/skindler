from copy import copy
from typing import Dict, Any

import numpy as np
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers.modeling_outputs import BaseModelOutput

from skindler.attackers import AttackerInput, AttackerOutput, Attacker, MbartGradientGuidedUpdateInput


@Attacker.register("mbart_update_latent")
class MbartUpdateLatent(MbartGradientGuidedUpdateInput, Attacker):
    def __init__(
            self,
            model_name: str,
            tokenizer_name: str,
            device: int = -1,
            checkpoint_path: str = 'pytorch_model.bin',
            epsilon: float = 0.1,
            max_iteration: int = 10):
        Attacker.__init__(self, device)

        self.model = MBartForConditionalGeneration.from_pretrained(
            model_name).to(self.device)
        self.model.eval()
        self.tokenizer = MBart50TokenizerFast.from_pretrained(tokenizer_name)
        self.tokenizer.src_lang = 'en_XX'
        self.tokenizer.tgt_lang = 'ru_RU'
        self.load_layers_for_predicting_bertscore(checkpoint_path)

        self.epsilon = epsilon
        self.max_iteration = max_iteration

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

        embeddings = self.model.model.encoder(
            input_ids=attack_input['input_ids'],
            attention_mask=attack_input['attention_mask'],
        )

        iteration = 0
        emb = embeddings.last_hidden_state.detach()

        while iteration < self.max_iteration:
            iteration += 1

            emb = torch.from_numpy(emb.detach().cpu().numpy()).to(self.device)
            emb.requires_grad = True

            predicted_bertscore = self.get_logits(emb)
            loss = torch.nn.functional.l1_loss(
                predicted_bertscore, torch.tensor(
                    [1.0]).unsqueeze(0).to(
                    self.device))
            loss.backward()
#             print(emb.grad.data)
            emb = emb + self.epsilon * emb.grad.data + 0.5
            print(emb)
            inputs_for_generation = {'encoder_outputs': BaseModelOutput(emb)}

            decoded = self.tokenizer.decode(
                self.model.generate(
                    **inputs_for_generation,
                    forced_bos_token_id=attack_input['input_ids'][0][1].item(),
                    decoder_start_token_id=self.tokenizer.lang_code_to_id['en_XX'])[0].tolist(),
                skip_special_tokens=True)

            input_history.append(decoded)
            losses_history.append(loss.item())

        print(input_history)
        if verbose:
            print(input_history[-1])
        return input_history[-1]
