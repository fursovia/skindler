#!/bin/bash
EXPERIMENTFOLDER='../grad_experiment'

for i in {60..90}; do 
    echo $i; 
    python skindler/commands/attack.py \
        skindler/configs/attacks/mbart_gradient_update_input_attack.jsonnet \
        --out-dir ../skindler_data/attack_output/mbart_attack/mbart_attack_${i}_threshlod \
        --samples 200 \
        --threshold ${i}
    
done