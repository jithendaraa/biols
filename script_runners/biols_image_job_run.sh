#!/bin/bash

biols_data_folder=$1
time=$2
learn_P='False'
config="ImageBIOLS-learnP(${learn_P})-${biols_data_folder}"

seeds=(0 1 2 3 4)
array_len=$(( ${#seeds[@]} ))
echo $array_len

output_file="out/BIOLS_image/biols-image-%A_%a.out"
echo "Train BIOLS Image: ${config}"

command="sbatch --array=1-${array_len}%512 --job-name ${config} --output ${output_file} --time ${time} scripts/biols_image_job.sh ${biols_data_folder} ${learn_P}"   
echo ""
echo ${command}
echo ""

RES=$(${command})
job_id=${RES##* }
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" >> out/job_logs.txt
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" 
echo ""


