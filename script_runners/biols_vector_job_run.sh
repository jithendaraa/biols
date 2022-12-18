#!/bin/bash

dataset=$1
train=$2
model=$3
time=$4
config=$5

seeds=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
array_len=$(( ${#seeds[@]} ))
echo $array_len

if [ ${train} == 'train' ]
then
    if [ ${dataset} == 'er' ]
    then
        output_file="out/BIOLS_vector/biols-vector-%A_%a.out"
        echo "Train BIOLS vector: ${config}"
    else
        echo "Not implemented dataset ${dataset}" 
    fi
else
    echo "Not implemented dataset ${train}" 
fi

command="sbatch --array=1-${array_len}%512 --job-name ${config} --output ${output_file} --time ${time} scripts/biols_job.sh ${config}"   
echo ""
echo ${command}
echo ""

RES=$(${command})
job_id=${RES##* }
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" >> out/job_logs.txt
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" 
echo ""


