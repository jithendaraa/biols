#!/bin/bash

biols_data_folder=$1
time=$2
lr=$3
learn_P='False'
config="BatchedBIOLS-learnP(${learn_P})-${biols_data_folder}"

seeds=(0 1 2 4)
array_len=$(( ${#seeds[@]} ))
echo $array_len

output_file="out/BIOLS_vector/batched-biols-vector-%A_%a.out"
echo "Train batched BIOLS vector: ${config}"

command="sbatch --array=1-${array_len}%512 --job-name ${config} --output ${output_file} --time ${time} scripts/batched_biols_job.sh ${biols_data_folder} ${learn_P} ${lr}"   
echo ""
echo ${command}
echo ""

RES=$(${command})
job_id=${RES##* }
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" >> out/job_logs.txt
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" 
echo ""


