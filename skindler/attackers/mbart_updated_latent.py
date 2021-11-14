from typing import Dict, Any

import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers.modeling_outputs import BaseModelOutput

from skindler.attackers import Attacker, MbartGradientGuidedUpdateInput


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

    def gradient_attack(
            self, attack_input: Dict[str, Any], verbose=False) -> str:

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
        emb = embeddings.last_hidden_state
        for _ in range(self.max_iteration):

            emb = torch.from_numpy(emb.detach().cpu().numpy()).to(self.device)
            emb.requires_grad = True

            predicted_bertscore = self.get_logits(emb)
            loss = torch.nn.functional.l1_loss(
                predicted_bertscore, torch.tensor(
                    [1.0]).unsqueeze(0).to(
                    self.device))
            loss.backward()
            
            grad = emb.grad.detach()
            emb.requires_grad = False
#             print(emb.shape)
            
            emb[0][5:10] = emb[0][5:10] + self.epsilon * grad[0][5:10]
            inputs_for_generation = {'encoder_outputs': BaseModelOutput(emb)}

            decoded = self.tokenizer.decode(
                self.model.generate(
                    **inputs_for_generation,
                    max_length = attack_input['input_ids'][0].shape[0] + 10,
                    min_length = attack_input['input_ids'][0].shape[0] - 10,
                    forced_bos_token_id=attack_input['input_ids'][0][1].item(),
                    repetition_penalty = 1.5,
                    decoder_start_token_id=self.tokenizer.lang_code_to_id['en_XX'])[0].tolist(),
                skip_special_tokens=False)

            input_history.append(decoded)
#         for i in input_history:
#             print(i)
        if verbose:
            print(input_history[-1])
        return input_history[-1]