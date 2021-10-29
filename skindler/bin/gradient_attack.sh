#!/bin/bash
EXPERIMENTFOLDER='../experiment'

for i in {61..90}; do 
    echo $i; 
    mkdir -p ${EXPERIMENTFOLDER}/threshold_0.${i}
    python main_attack.py --threshold 0.${i} --experiment_folder ${EXPERIMENTFOLDER}/threshold_0.${i}
done