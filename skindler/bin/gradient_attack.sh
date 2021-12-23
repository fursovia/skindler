#!/bin/bash

python skindler/commands/attack.py \
    skindler/configs/attacks/gradient_attack_brute_force.jsonnet \
    --out-dir ../skindler_data/attack_output/gradient_attack_brute_force/ \
    --samples 1000 
CUDA_VISIBLE_DEVICES=1 python skindler/commands/validate.py ../skindler_data/attack_output/gradient_attack_brute_force/data.json
    
done