from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import torch
from allenai_common import Registrable
from dataclasses_json import dataclass_json


@dataclass
class AttackerInput:
    x: str
    y: str = None

    def __repr__(self) -> str:
        s = f"""
        Orig. sentence: {self.x}
        Orig. translation: {self.y if self.y is not None else "translation is not available"}
        """
        return s


@dataclass_json
@dataclass
class AttackerOutput:
    x: str
    y: str
    x_attacked: str
    y_trans: str = None
    y_trans_attacked: str = None

    def __repr__(self) -> str:
        s = f"""
        Orig. sentence: {self.x}
        Attacked sentence: {self.x_attacked}
        Orig. translation: {self.y}
        Translation: {self.y_trans}
        Attacked translation: {self.y_trans_attacked}
        """
        return s


class Attacker(ABC, Registrable):

    def __init__(self, device: int = -1, ) -> None:
        self.device = torch.device(f'cuda:{device}') if device != -1 else -1

    @abstractmethod
    def attack(self, data_to_attack: AttackerInput) -> AttackerOutput:
        pass

    def move_to_device(
            self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.device != -1:
            return {key: val.to(device=self.device) if isinstance(
                val, torch.Tensor) else val for key, val in inputs.items()}
        else:
            return inputs
