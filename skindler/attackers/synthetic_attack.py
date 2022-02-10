from typing import Dict, Any, List

import numpy as np
import random
import torch
from copy import copy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from skindler.attackers import AttackerInput, AttackerOutput, Attacker
from skindler import MAX_LENGTH


@Attacker.register("synthetic_attack")
class SyntheticAttacker(Attacker):
    def __init__(
            self,
            model_name,
            tokenizer_name,
            device: int = -1,
            number_of_replacements: int = 10,
            attack_type: str = 'random_word'
    ):
        super().__init__(device)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.number_of_replacements = number_of_replacements
        
    def attack(self,
               data_to_attack: AttackerInput,
               verbose=False
               ) -> AttackerOutput:

        data_attacked = self.synthetic_attack(data_to_attack)
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

    def synthetic_attack(
            self, attack_input: AttackerInput, verbose=False) -> str:
        s = attack_input.x
        s = s.split()
        for i in range(self.number_of_replacements):
            word_index = random.randint(0, len(s)-1)
            word = s[word_index]
            if len(word) > 1:
                first, second = random.sample(range(len(word)), 2)
                first, second = min(first, second), max(first, second)
                word = word[:first] + word[second] + word[first + 1: second] + word[first] + word[second+1:]
                s[word_index] = word

        s = ' '.join(s)
        
        return s

