#!/bin/bash

for i in {1..10}; do 
    echo $i; 
    python skindler/commands/attack.py \
        skindler/configs/attacks/ner_attack.jsonnet \
        --out-dir ../skindler_data/attack_output/ner_attack/ner_attack_${i}_iteration \
        --samples 50 \
        --max-iteration ${i}
    CUDA_VISIBLE_DEVICES=3 python skindler/commands/validate.py ../skindler_data/attack_output/ner_attack/ner_attack_${i}_iteration/data.json
    
done