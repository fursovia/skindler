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

Gradient attack
```bash
CUDA_VISIBLE_DEVICES=7 python skindler/commands/attack.py skindler/configs/attacks/bleuer.jsonnet  --out-dir bleuer_attack_folder
```

Gradient attack
```bash
CUDA_VISIBLE_DEVICES=7 python skindler/commands/attack.py skindler/configs/attacks/gradient_attack.jsonnet  --out-dir gradient_attack_folder
```

## Get metrics of attack

```bash
python skindler/commands/validate.py path_to_attack_output
```