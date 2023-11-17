#!/bin/bash

d=(5)
array_len=$(( ${#d[@]} ))
echo $array_len

proj='linear_proj'
exp_edges=1
sem_type='nonlinear_scm'

output_file="out/${proj}-${sem_type}-er${exp_edges}-datagen-%A_%a.out"
command="sbatch --array=1-${array_len}%512 --job-name ${proj}-${sem_type}-er${exp_edges}-datagen --output ${output_file} scaling_ablation_datagen.sh ${proj} ${exp_edges} ${sem_type}}"   
echo ""
echo ${command}
echo ""

RES=$(${command})
job_id=${RES##* }
echo "Job ID"" ""${job_id}"
echo ""