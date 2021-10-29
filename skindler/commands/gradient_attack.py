import json
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from skindler import SENTENCES_TO_ATTACK
from skindler.models import GradientGuidedSearchStrategy
from skindler.modules.gradient_guided_utils import prepare_dataloader, prepare_model_and_tokenizer, count_metrics



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--threshold", default=0.75, type=float)
    parser.add_argument("--max_iteration", default=100, type=int)
    parser.add_argument("--experiment_folder", default=Path('experiment/threshold_0.75/'), type=Path)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model, tokenizer = prepare_model_and_tokenizer(device)
    _, test_dataloader = prepare_dataloader(tokenizer)

    attacker = GradientGuidedSearchStrategy(model, tokenizer, threshold=args.threshold,
                                            max_iteration=args.max_iteration)

    result = attacker.attack_dataset(test_dataloader, SENTENCES_TO_ATTACK, verbose=False)
    x_perturbed_with_brackets = [i[-1] for i in result]
    x_perturbed = [i.replace("[[", "").replace("]]", "") for i in x_perturbed_with_brackets]

    # translate perturbed text
    y_attacked = []
    for perturbed_id, perturbed_text in enumerate(tqdm(x_perturbed)):
        with torch.no_grad():
            translated = model.generate(torch.tensor(tokenizer.encode(perturbed_text)).unsqueeze(0).to(device))
            translated = tokenizer.decode(translated[0], skip_special_tokens=True)
        y_attacked.append(translated)

    # translate origin text
    x = []
    y = []
    y_without_attack = []
    iter = 0
    for batch in tqdm(test_dataloader):
        iter += 1
        if iter == SENTENCES_TO_ATTACK:
            break
        batch = {i: j.to(device) for i, j in batch.items()}
        with torch.no_grad():
            translated = model.generate(batch['input_ids'])
            translated = tokenizer.decode(translated[0], skip_special_tokens=True)
        x.append(tokenizer.decode(batch['input_ids'][0].tolist()))
        y.append(tokenizer.decode(batch['labels'][0].tolist()))
        y_without_attack.append(translated)
    x = [i.replace("▁", " ") for i in x]

    # save everything

    with open(f"{str(args.experiment_folder)}/x.json", 'w') as f:
        json.dump(x, f)

    with open(f"{str(args.experiment_folder)}/x_perturbed_with_brackets.json", 'w') as f:
        json.dump(x_perturbed_with_brackets, f)

    with open(f"{str(args.experiment_folder)}/x_perturbed.json", 'w') as f:
        json.dump(x_perturbed, f)

    with open(f"{str(args.experiment_folder)}/y.json", 'w') as f:
        json.dump(y, f)

    with open(f"{str(args.experiment_folder)}/y_without_attack.json", 'w') as f:
        json.dump(y_without_attack, f)

    with open(f"{str(args.experiment_folder)}/y_attacked.json", 'w') as f:
        json.dump(y_attacked, f)

    # count metrics

    orig_translate_metrics = count_metrics(y, y_without_attack)
    attack_translate_metrics = count_metrics(y, y_attacked)
    x_metrics = count_metrics(x, x_perturbed)

    table = {'orig.input': x, 'pert.input': x_perturbed_with_brackets,
             'labels': y, 'orig.translation': y_without_attack, 'pert.translation': y_attacked}

    for metric_name, metric_value in orig_translate_metrics.items():
        table[f"orig.translation_{metric_name}"] = metric_value

    for metric_name, metric_value in attack_translate_metrics.items():
        table[f"attacked.translation_{metric_name}"] = metric_value

    for metric_name, metric_value in x_metrics.items():
        table[f"x_{metric_name}"] = metric_value

    table = pd.DataFrame(table)

    table.to_csv(f"{str(args.experiment_folder)}/table.csv")
