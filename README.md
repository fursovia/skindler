# Skindler


## Train your models!

Model that predicts BLEU score
```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python skindler/models/bleuer.py
```

Model that predicts sentences from encoder outputs
```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python skindler/models/autoencoder.py
```

## Attack!

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python skindler/commands/attack.py \
  data/valid.json --ae-dir experiments/ae --bl-dir experiments/bleuer/ \
  --save-to results/data_1.0.json --epsilon 1.0
```