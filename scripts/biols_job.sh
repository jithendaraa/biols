#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6

seeds=(1)
lr=0.00001
num_steps=20000
off_wandb='False'

biols_data_folder=$1
learn_P=$2
seed=${seeds[ $((  (${SLURM_ARRAY_TASK_ID})  % ${#seeds[@]} )) ]}

start=`date +%s`
echo "Script"

module load anaconda/3
conda activate biols
echo `date` "Python starting"

cd exps

if [ "$learn_P" = "False" ] 
then
    echo "python biols_vector_data.py --config defaults biols_learn_L --biols_data_folder ${biols_data_folder} --data_seed ${seed} --lr ${lr} --num_steps ${num_steps} --off_wandb ${off_wandb}"
    python biols_vector_data.py --config defaults biols_learn_L --biols_data_folder ${biols_data_folder} --data_seed ${seed} --lr ${lr} --num_steps ${num_steps} --off_wandb ${off_wandb}

elif [ "$learn_P" = "True" ]  
then
    echo "python biols_vector_data.py --config defaults biols_learn_P --biols_data_folder ${biols_data_folder} --data_seed ${seed} --lr ${lr} --num_steps ${num_steps} --off_wandb ${off_wandb}"
    python biols_vector_data.py --config defaults biols_learn_P --biols_data_folder ${biols_data_folder} --data_seed ${seed} --lr ${lr} --num_steps ${num_steps} --off_wandb ${off_wandb}

fi

# python biols_vector_data.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edges} --lr ${lr} --num_steps ${num_steps} --num_nodes ${num_nodes} --proj_dims ${proj_dims} --obs_data ${obs_data} --off_wandb ${off_wandb} --n_interv_sets ${n_interv_sets} --pts_per_interv ${pts_per_interv} --eq_noise_var ${eq_noise_var}
cd ..

echo $end
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"