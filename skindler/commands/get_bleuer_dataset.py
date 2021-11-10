import json
import math
from pathlib import Path

import torch
from datasets import concatenate_datasets
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from typer import Typer

from skindler import DATASET_NAME, MODEL_NAME, MAX_LENGTH, MBART_NAME, SRC_LNG, TGT_LNG
from skindler.modules.metrics import ALL_METRICS
from skindler.utils import lazy_groups_of

app = Typer()

COLLECT_DATASET_CONFIG = {
    'bleuer': {'model': MODEL_NAME, 'target_metric_name': 'bleu', 'target_metric_function': sentence_bleu},
    'mbart_bertscore': {'model': MBART_NAME, 'target_metric_name': 'bertscore',
                        'target_metric_function': ALL_METRICS['bertscore']}
}


@app.command()
def get_dataset(task: str, save_to: Path, batch_size: int = 128, device: int = -1, sample: bool = False):
    assert task in COLLECT_DATASET_CONFIG.keys()
    config = COLLECT_DATASET_CONFIG[task]

    dataset = load_dataset(*DATASET_NAME)
    splits_to_use = ["validation", "test"]
    if not sample:
        splits_to_use.append("train")

    dataset = concatenate_datasets([dataset[split] for split in splits_to_use])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    device = torch.device(f'cuda:{device}') if device != -1 else -1
    model = model.to(device)
    model = model.eval()

    with save_to.open("w") as f:
        for batch in tqdm(lazy_groups_of(dataset, batch_size), total=math.ceil(len(dataset) / batch_size)):
            x = [example['translation'][SRC_LNG] for example in batch]
            y = [example['translation'][TGT_LNG] for example in batch]

            batch_text_inputs = tokenizer(
                x,
                max_length=MAX_LENGTH,
                return_tensors='pt',
                padding=True,
                truncation=True
            )
            batch_text_inputs.to(device)

            output = model.generate(**batch_text_inputs)
            # TODO: remove special tokens (__)
            translations = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            for x_sentence, y_sentence, y_translated_sentence in zip(x, y, translations):
                target_metric = config['target_metric_function']([y_translated_sentence.lower()],
                                                                 y_sentence.lower())
                f.write(
                    json.dumps(
                        {
                            "x": x_sentence,
                            "y": y_sentence,
                            "y_trans": y_translated_sentence,
                            config['target_metric_name']: target_metric
                        },
                        ensure_ascii=False
                    ) + "\n"
                )


if __name__ == '__main__':
    app()
