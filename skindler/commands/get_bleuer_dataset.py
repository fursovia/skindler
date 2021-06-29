from typing import (
    Iterable,
    Iterator,
    List,
    TypeVar
)
from itertools import islice
from tqdm import tqdm
from pathlib import Path
import json
import math

import torch
from nltk.translate.bleu_score import sentence_bleu
from transformers import MarianMTModel, MarianTokenizer
from datasets import concatenate_datasets
from datasets import load_dataset
from typer import Typer

from skindler import DATSET_NAME, MODEL_NAME, MAX_LENGTH

app = Typer()

A = TypeVar("A")


def lazy_groups_of(iterable: Iterable[A], group_size: int) -> Iterator[List[A]]:
    """
    Takes an iterable and batches the individual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    """
    iterator = iter(iterable)
    while True:
        s = list(islice(iterator, group_size))
        if len(s) > 0:
            yield s
        else:
            break


@app.command()
def get_dataset(save_to: Path, batch_size: int = 128, sample: bool = False):
    dataset = load_dataset(*DATSET_NAME)
    splits_to_use = ["validation", "test"]
    if not sample:
        splits_to_use.append("train")

    dataset = concatenate_datasets([dataset[split] for split in splits_to_use])

    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    device = torch.device('cuda')
    model = model.to(device)
    model = model.eval()

    with save_to.open("w") as f:
        for batch in tqdm(lazy_groups_of(dataset, batch_size), total=math.ceil(dataset / batch_size)):
            en = [example['translation']['en'] for example in batch]
            ru = [example['translation']['ru'] for example in batch]

            batch_text_inputs = tokenizer(
                en,
                max_length=MAX_LENGTH,
                return_tensors='pt',
                padding=True,
                truncation=True
            )
            inputs = {}
            for k, v in batch_text_inputs.items():
                inputs[k] = v.to(device)

            output = model.generate(**inputs)
            translations = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            for en_sentence, ru_sentence, ru_translated_sentence in zip(en, ru, translations):
                bleu = sentence_bleu([ru_translated_sentence.lower()], ru_sentence.lower())
                f.write(
                    json.dumps(
                        {
                            "en": en_sentence,
                            "ru": ru_sentence,
                            "ru_trans": ru_translated_sentence,
                            "bleu": bleu
                        },
                        ensure_ascii=False
                    ) + "\n"
                )


if __name__ == '__main__':
    app()
