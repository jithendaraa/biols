#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4

seeds=(0 1 2 3 4)
lr=0.0001
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
    echo "python biols_vector_data.py --config defaults biols_learn_L --biols_data_folder ${biols_data_folder} --data_seed ${seed} --lr ${lr} --off_wandb ${off_wandb}"
    python biols_vector_data.py --config defaults biols_learn_L --biols_data_folder ${biols_data_folder} --data_seed ${seed} --lr ${lr} --off_wandb ${off_wandb}

elif [ "$learn_P" = "True" ]  
then
    echo "python biols_vector_data.py --config defaults biols_learn_P --biols_data_folder ${biols_data_folder} --data_seed ${seed} --lr ${lr} --off_wandb ${off_wandb}"
    python biols_vector_data.py --config defaults biols_learn_P --biols_data_folder ${biols_data_folder} --data_seed ${seed} --lr ${lr} --off_wandb ${off_wandb}

fi

cd ..

echo $end
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"