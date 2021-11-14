from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional
from nltk.translate.bleu_score import sentence_bleu
from transformers import MarianMTModel, MarianTokenizer
from transformers.trainer_utils import get_last_checkpoint

from skindler import MAX_LENGTH
from skindler.attackers import AttackerInput, AttackerOutput, Attacker
from skindler.models import MarianAutoEncoder, Bleuer


def calculate_metric(source: str, source_attacked: str, translation: str, translation_attacked: str) -> float:
    # should be large!
    source_bleu = sentence_bleu([source_attacked.lower()], source.lower())
    # should be small!
    target_bleu = sentence_bleu([translation_attacked.lower()], translation.lower())
    target_bleu_inversed = 1.0 - target_bleu

    if source_bleu or target_bleu_inversed:
        metric = 2 * (source_bleu * target_bleu_inversed) / (source_bleu + target_bleu_inversed)
    else:
        metric = 0.0
    return metric


@Attacker.register("bleuer")
class BleuerAttack(Attacker):
    def __init__(
            self,
            model_name,
            ae_dir,
            bl_dir,
            device: int = -1,
            epsilon: float = 0.25,
            max_iteration: int = 10,
            sign_mode: bool = False):

        super().__init__(device)

        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).eval().to(self.device)

        ae_dir = get_last_checkpoint(str(ae_dir)) or ae_dir
        ae_path = str(Path(ae_dir) / 'pytorch_model.bin')
        self.autoencoder = MarianAutoEncoder(model_name)
        self.autoencoder.load_state_dict(torch.load(ae_path)).to(self.device)

        bl_dir = get_last_checkpoint(str(bl_dir)) or bl_dir
        bl_path = str(Path(bl_dir) / 'pytorch_model.bin')
        self.bleuer = Bleuer(model_name)
        self.bleuer.load_state_dict(torch.load(bl_path)).to(self.device)
        for params in self.bleuer.parameters():
            params.grad = None

        self.epsilon = epsilon
        self.max_iteration = max_iteration
        self.sign_mode = sign_mode

    def attack(self,
               data_to_attack: AttackerInput
               ) -> AttackerOutput:

        attack_input = self.prepare_attack_input(data_to_attack)
        attack_input = self.move_to_device(attack_input)
        data_attacked = self.bleuer_attack(attack_input)
        data_attacked, y_trans_attacked = self.filter_attacked_sentences(data_to_attack, data_attacked)
        attack_output = self.prepare_attack_output(
            data_to_attack, data_attacked, y_trans_attacked)
        return attack_output

    def prepare_attack_input(
            self, data_to_attack: AttackerInput) -> Dict[str, Any]:

        attack_input = self.tokenizer(
            data_to_attack.x,
            max_length=MAX_LENGTH,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        return attack_input

    def prepare_attack_output(self, data_to_attack: AttackerInput,
                              data_attacked: str,
                              y_trans_attacked: str) -> AttackerOutput:
        with torch.no_grad():
            translated = self.model.generate(torch.tensor(
                self.tokenizer.encode(
                    data_to_attack.x
                )).unsqueeze(0).to(self.device))
            y_trans = self.tokenizer.decode(
                translated[0], skip_special_tokens=True)

        output = AttackerOutput(
            x=data_to_attack.x,
            y=data_to_attack.y,
            x_attacked=data_attacked,
            y_trans=y_trans,
            y_trans_attacked=y_trans_attacked
        )
        return output

    def bleuer_attack(
            self, attack_input: Dict[str, Any]) -> List[str]:

        # shape [1, num_tokens, 512]
        embeddings = self.autoencoder.get_embeddings(**attack_input)
        attacked_sentences = []
        for step in range(self.max_iteration):
            embeddings = torch.from_numpy(embeddings.detach().cpu().numpy()).to(self.device)
            embeddings.requires_grad = True

            # shape [1, 1] [0.2 L1 loss on validation set]
            bleu = torch.clamp(self.bleuer.get_logits(embeddings), 0.0, 1.0)
            loss = torch.nn.functional.l1_loss(bleu, torch.tensor(1.0, device=self.device))
            loss.backward()

            if self.sign_mode:
                embeddings = embeddings + self.epsilon * embeddings.grad.data.sign()
            else:
                embeddings = embeddings + self.epsilon * embeddings.grad.data
            # shape [1, num_tokens, vocab_size] [~0.02 cross entropy loss]
            logits = self.autoencoder.get_logits(embeddings)
            # shape [1, num_tokens]
            ids = logits.argmax(dim=-1)
            decoded = self.tokenizer.decode(ids[0].cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            decoded = decoded.replace('▁', ' ').strip()
            attacked_sentences.append(decoded)
        return attacked_sentences

    def filter_attacked_sentences(self,
                                  data_to_attack: AttackerInput,
                                  attacked_sentences: List[str]) -> Tuple[str, str]:
        best_metric = 0.0
        translation_history = []
        metric_history = []

        best_source_attacked = data_to_attack.x
        best_target_attacked = data_to_attack.y

        for i, x_attacked in enumerate(attacked_sentences):
            batch_text_inputs = self.tokenizer(
                x_attacked,
                max_length=MAX_LENGTH,
                return_tensors='pt',
                padding=True,
                truncation=True
            )
            batch_text_inputs.to(self.device)

            output = self.model.generate(**batch_text_inputs)
            y_trans_attacked = self.tokenizer.batch_decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
            translation_history.append(y_trans_attacked)
            if i == 0:
                best_source_attacked = x_attacked
                best_target_attacked = y_trans_attacked

            metric = calculate_metric(
                source=data_to_attack.x,
                source_attacked=x_attacked,
                translation=data_to_attack.y,
                translation_attacked=y_trans_attacked
            )
            metric_history.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_source_attacked = x_attacked
                best_target_attacked = y_trans_attacked

        return best_source_attacked, best_target_attacked
