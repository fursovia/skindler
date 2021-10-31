#!/bin/bash
EXPERIMENTFOLDER='../grad_experiment'

for i in {75..76}; do 
    echo $i; 
    mkdir -p ${EXPERIMENTFOLDER}/threshold_0.${i}
    python skindler/commands/gradient_attack.py --experiment-folder ${EXPERIMENTFOLDER}/threshold_0.${i} --threshold 0.${i}
    python skindler/commands/get_metrics.py --attack-path ${EXPERIMENTFOLDER}/threshold_0.${i}/attack_output
    
done