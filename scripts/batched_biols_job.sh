#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6

seeds=(0 1 2 3 4)
lr=0.0006
num_steps=10000

num_nodes=10
proj_dims=100
exp_edges=1.0
n_interv_sets=500
eq_noise_var='False'
min_interv_value=-5.0
max_interv_value=5.0

pts_per_interv=100
off_wandb='False'
obs_data=500

id=$1
seed=${seeds[ $(( ${SLURM_ARRAY_TASK_ID}  % ${#seeds[@]} )) ]}

start=`date +%s`
echo "Script"

module load anaconda/3
module load cuda/11.1
conda activate biols
echo `date` "Python starting"
echo "python batched_biols_vector_data.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edges} --lr ${lr} --num_steps ${num_steps} --num_nodes ${num_nodes} --proj_dims ${proj_dims} --obs_data ${obs_data} --off_wandb ${off_wandb} --n_interv_sets ${n_interv_sets} --pts_per_interv ${pts_per_interv} --eq_noise_var ${eq_noise_var} --min_interv_value ${min_interv_value} --max_interv_value ${max_interv_value}"

cd exps
python batched_biols_vector_data.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edges} --lr ${lr} --num_steps ${num_steps} --num_nodes ${num_nodes} --proj_dims ${proj_dims} --obs_data ${obs_data} --off_wandb ${off_wandb} --n_interv_sets ${n_interv_sets} --pts_per_interv ${pts_per_interv} --eq_noise_var ${eq_noise_var} --min_interv_value ${min_interv_value} --max_interv_value ${max_interv_value}
cd ..

echo $end
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"