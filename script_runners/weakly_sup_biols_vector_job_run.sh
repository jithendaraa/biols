#!/bin/bash

dataset=$1
model=$2
time=$3
config=$4

seeds=(2 3 6 7 8 9 10 11)
array_len=$(( ${#seeds[@]} ))
echo $array_len

if [ ${dataset} == 'er' ]
then
    output_file="out/weakly_sup_biols_vector/ws-biols-vector-%A_%a.out"
    echo "Train WS-BIOLS vector: ${config}"
else
    echo "Not implemented dataset ${dataset}" 
fi


command="sbatch --array=1-${array_len}%512 --job-name ${config} --output ${output_file} --time ${time} scripts/weakly_sup_biols_vector_job.sh ${config}"   
echo ""
echo ${command}
echo ""

RES=$(${command})
job_id=${RES##* }
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" >> out/job_logs.txt
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" 
echo ""


