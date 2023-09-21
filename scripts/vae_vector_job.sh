#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --exclude cn-g[005-012,017-026]

seeds=(0 1 2 3 4)
lr=0.0001
num_steps=10000
off_wandb='False'

biols_data_folder=$1
seed=${seeds[ $((  (${SLURM_ARRAY_TASK_ID})  % ${#seeds[@]} )) ]}

start=`date +%s`
echo "Script"

module load anaconda/3
conda activate biols
echo `date` "Python starting"
echo "python vae_vector_baseline.py --config defaults vae_linear_baseline --biols_data_folder ${biols_data_folder} --data_seed ${seed} --lr ${lr} --num_steps ${num_steps} --off_wandb ${off_wandb}"

cd exps
python vae_vector_baseline.py --config defaults vae_linear_baseline --biols_data_folder ${biols_data_folder} --data_seed ${seed} --lr ${lr} --num_steps ${num_steps} --off_wandb ${off_wandb}
cd ..

end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"