#!/bin/bash

for i in {1..35}; do 
    echo $i; 
#     python skindler/commands/attack.py \
#         skindler/configs/attacks/synthetic_attack.jsonnet \
#         --out-dir ../skindler_data/attack_output/synthetic_attack/synthetic_attack_${i}_iterations \
#         --samples 100 \
#         --number-of-replacements ${i}
    python skindler/commands/validate.py ../skindler_data/attack_output/synthetic_attack/synthetic_attack_${i}_iterations/data.json
    
done