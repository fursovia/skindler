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
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python skindler/commands/attack.py \
  data/valid.json --ae-dir experiments/ae --bl-dir experiments/bleuer/ \
  --save-to results/data_1.0.json --epsilon 1.0
```

Gradient attack
```bash
bash skindler/bin/gradient_attack.sh
```

## Get metrics of attack

```bash
python skindler/commands/get_metrics.py --attack-path path/to/attack_output
```