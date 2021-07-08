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

from skindler import DATASET_NAME, MODEL_NAME, MAX_LENGTH
from skindler.utils import lazy_groups_of

app = Typer()


@app.command()
def get_dataset(save_to: Path, batch_size: int = 128, sample: bool = False):
    dataset = load_dataset(*DATASET_NAME)
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
        for batch in tqdm(lazy_groups_of(dataset, batch_size), total=math.ceil(len(dataset) / batch_size)):
            en = [example['translation']['en'] for example in batch]
            ru = [example['translation']['ru'] for example in batch]

            batch_text_inputs = tokenizer(
                en,
                max_length=MAX_LENGTH,
                return_tensors='pt',
                padding=True,
                truncation=True
            )
            batch_text_inputs.to(device)

            output = model.generate(**batch_text_inputs)
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
