#!/bin/bash

for i in {75..95}; do 
    echo $i; 
    python skindler/commands/attack.py \
        skindler/configs/attacks/gradient_attack.jsonnet \
        --out-dir ../skindler_data/attack_output/gradient_attack_lm_constraint/gradient_attack_${i}_lm_threshlod \
        --samples 100 \
        --lm-threshold ${i}
    CUDA_VISIBLE_DEVICES=1 python skindler/commands/validate.py ../skindler_data/attack_output/gradient_attack_lm_constraint/gradient_attack_${i}_lm_threshlod/data.json
    
done