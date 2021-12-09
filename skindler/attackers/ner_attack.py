import random
from typing import Dict, Any, List
import json

from skindler.attackers import AttackerInput, AttackerOutput, Attacker
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
# run '''python -m spacy download en_core_web_lg''' if spacy can't load
# en_core_web_lg
import en_core_web_lg


@Attacker.register("ner_attack")
class NerAttack(Attacker):
    def __init__(
            self,
            model_name,
            tokenizer_name,
            device: int = -1,
            ner_entity_file: str = "data/ner_examples.json",
            max_iteration: int = 2):
        super().__init__(device)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.nlp = en_core_web_lg.load()
        self.max_iteration = max_iteration
        with open(ner_entity_file, 'r') as f:
            self.ner_entity = json.load(f)

    def attack(self,
               data_to_attack: AttackerInput,
               verbose=False
               ) -> AttackerOutput:

        attack_input = data_to_attack.x
        data_attacked = self.ner_attack(attack_input)
        attack_output = self.prepare_attack_output(
            data_to_attack, data_attacked)
        return attack_output

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

    def ner_attack(
            self, attack_input: str, verbose=False) -> str:

        ner_output = self.nlp(attack_input)
        ents = ner_output.ents
        replace_words = list()
        if ents:
            for ent in ents:
                if ent.label_ in self.ner_entity:
                    ent_text_new = random.choice(self.ner_entity[ent.label_])
                    replace_words.append((ent.text, ent_text_new))
                    if len(replace_words) == self.max_iteration:
                        break

        for (ent_text, ent_text_new) in replace_words:
            attack_input = attack_input.replace(ent_text, ent_text_new)
        attack_output = attack_input
        return attack_output
