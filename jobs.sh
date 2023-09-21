#!/bin/bash
exp_id=$1
data_config=${2:-'er1-ws_datagen_fix_noise_interv_noise-3_layer_mlpproj-d005-D0100-single-n_pairs2000-sets20-zerosinterv'}

def_time='9:00:00'
time=${3:-$def_time}


if [ ${exp_id} == '1' ]
then
    bash script_runners/biols_vector_job_run.sh ${data_config} ${time} 

elif [ ${exp_id} == '2' ]
then
    bash script_runners/batched_biols_vector_job_run.sh ${dataset} 'BIOLS' ${time} ${config}

elif [ ${exp_id} == '3' ]
then
    bash script_runners/biols_image_job_run.sh ${dataset} ${train} 'BIOLS_Image' ${time} ${config}

elif [ ${exp_id} == '4' ] # ! vae baseline
then
    bash script_runners/vae_vector_baseline_job_run.sh ${data_config} ${time} 

elif [ ${exp_id} == '5' ] # ! graphvae baseline
then
    bash script_runners/graphvae_vector_baseline_job_run.sh ${data_config} ${time} 

elif [ ${exp_id} == '6' ] # ! gin baseline
then
    bash script_runners/gin_baseline_job_run.sh ${dataset} ${train} 'GIN Baseline' ${time} ${config}
elif [ ${exp_id} == '7' ]
then
    bash script_runners/interv_biols_vector_job_run.sh ${dataset} 'BIOLS' ${time} ${config}

elif [ ${exp_id} == '8' ]
then
    bash script_runners/weakly_sup_biols_vector_job_run.sh ${dataset} 'BIOLS' ${time} ${config}
fi