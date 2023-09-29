#!/bin/bash

config=$1
time=$2

seeds=(0 1 2 3 4)
array_len=$(( ${#seeds[@]} ))
echo $array_len

output_file="out/GraphVAE_chemdata/graphvae-chemdata-%A_%a.out"
command="sbatch --array=1-${array_len}%512 --job-name graph_vae_${config} --output ${output_file} --time ${time} scripts/graphvae_image_job.sh ${config} -x 'cn-g[005-012,017-026]'"   
echo ""
echo ${command}
echo ""

RES=$(${command})
job_id=${RES##* }
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" >> out/job_logs.txt
echo "Job ID"" ""${job_id}"" -> ""${config} ${args}" 
echo ""