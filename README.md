# Skindler

## Prepare data for our attacks
Main idea: translate sentence, get metric, predict it, using differentiable model
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python skindler/commands/get_dataset.py DATASET_CONFIG_NAME PATH_TO_SAVE_DATA.jsonl --split train --device 0
```


## Train your models!

Model that predicts BLEU score
```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python skindler/models/bleuer.py
```

Model that predicts sentences from encoder outputs
```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python skindler/models/autoencoder.py
```

## Attack!

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python skindler/commands/attack_bleuer.py \
  data/valid.json --ae-dir experiments/ae --bl-dir experiments/bleuer/ \
  --save-to results/data_1.0.json --epsilon 1.0
```

Gradient attack
```bash
python skindler/commands/attack.py skindler/configs/attacks/gradient_attack.jsonnet  --out-dir gradient_attack_folder
```

## Get metrics of attack

```bash
python skindler/commands/validate.py path_to_attack_output
```