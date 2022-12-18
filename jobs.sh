#!/bin/bash
exp_id=$1   
dataset=${2:-'er'} 
train=${3:-'train'}
def_time='1:00:00'
time=${4:-$def_time}
config=${5:-'linear_vector_multi_interv_learn_L'}


if [ ${exp_id} == '1' ]
then
    bash script_runners/biols_vector_job_run.sh ${dataset} ${train} 'BIOLS' ${time} ${config}

elif [ ${exp_id} == '2' ]
then
    bash script_runners/batched_biols_vector_job_run.sh ${dataset} ${train} 'BIOLS' ${time} ${config}

elif [ ${exp_id} == '3' ]
then
    bash script_runners/biols_image_job_run.sh ${dataset} ${train} 'BIOLS_Image' ${time} ${config}

elif [ ${exp_id} == '4' ] # ! vae baseline
then
    bash script_runners/vae_vector_baseline_job_run.sh ${dataset} ${train} 'VAE Baseline' ${time} ${config}

elif [ ${exp_id} == '5' ] # ! graphvae baseline
then
    bash script_runners/graphvae_vector_baseline_job_run.sh ${dataset} ${train} 'GraphVAE Baseline' ${time} ${config}

elif [ ${exp_id} == '6' ] # ! gin baseline
then
    bash script_runners/gin_baseline_job_run.sh ${dataset} ${train} 'GIN Baseline' ${time} ${config}


fi