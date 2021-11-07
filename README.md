# Skindler


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
python skindler/commands/get_metrics.py --attack-path path/to/attack_output
```