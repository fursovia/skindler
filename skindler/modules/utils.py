from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import List
from skindler.modules.metrics import ALL_METRICS


@dataclass
class AttackOutput:
    x: str
    y: str
    x_attacked: str
    y_trans: str
    y_trans_attacked: str

    def __repr__(self) -> str:
        s = f"""
        Orig. sentence: {self.x}
        Att. sentence: {self.x_attacked}
        Orig. translation: {self.y}
        Translation: {self.y_trans}
        Att. translation: {self.y_trans_attacked}
        """
        return s

    def save_as_json(self, path: Path):
        with Path(path).open("a") as f:
            json.dump(asdict(self), f, ensure_ascii=False)
            f.write("\n")


def load_attacks(path: Path) -> List[AttackOutput]:
    attacks = []
    with Path(path).open("r") as f:
        for line in f:
            attacks.append(AttackOutput(**json.loads(line)))
    return attacks


def count_metrics(one_list, second_list):
    result = {}
    for metric in ['bleu', 'meteor', 'chrf', 'bertscore',
                   'calculate_wer_corpus', 'calculate_paraphrase_similarity']:
        result[metric] = ALL_METRICS[metric](one_list, second_list)
    result['wer'] = result.pop('calculate_wer_corpus')
    result['par.similarity'] = result.pop('calculate_paraphrase_similarity')
    return result
