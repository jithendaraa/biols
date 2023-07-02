#!/bin/bash

dataset=$1
model=$2
time=$3
config=$4

seeds=(0 1 2 3 4)
array_len=$(( ${#seeds[@]} ))
echo $array_len

if [ ${dataset} == 'er' ]
then
    output_file="out/BIOLS_batched_vector/biols-batched-vector-%A_%a.out"
    echo "Train Batched BIOLS vector: ${config}"
else
    echo "Not implemented dataset ${dataset}" 
fi

command="sbatch --array=1-${array_len}%512 --job-name ${config} --output ${output_file} --time ${time} scripts/batched_biols_job.sh ${config}"   
echo ""
echo ${command}
echo ""

RES=$(${command})
job_id=${RES##* }
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" >> out/job_logs.txt
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" 
echo ""


