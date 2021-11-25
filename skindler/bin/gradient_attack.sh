#!/bin/bash

for i in {1..50}; do 
    echo $i; 
    python skindler/commands/attack.py \
        skindler/configs/attacks/prefix_gradient_attack.jsonnet \
        --out-dir ../skindler_data/attack_output/prefix_attack/prefix_attack_${i}_iteration \
        --samples 50 \
        --max_iteration ${i}
    python skindler/commands/validate.py ../skindler_data/attack_output/prefix_attack/prefix_attack_${i}_iteration/data.json
    
done