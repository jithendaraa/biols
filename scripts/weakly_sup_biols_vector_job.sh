#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6

seeds=(2 3 6 7 8 9 10 11)
lr=0.0001
num_steps=10000

num_nodes=6
proj_dims=6
exp_edges=1.0
eq_noise_var='False'

n_interv_sets=30
pts_per_interv=200
off_wandb='False'
obs_data=6000

id=$1
seed=${seeds[ $((  (${SLURM_ARRAY_TASK_ID})  % ${#seeds[@]} )) ]}

start=`date +%s`
echo "Script"

module load anaconda/3
conda activate biols
echo `date` "Python starting"
echo "python weakly_supervised_biols_vector.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edges} --lr ${lr} --num_steps ${num_steps} --num_nodes ${num_nodes} --proj_dims ${proj_dims} --obs_data ${obs_data} --off_wandb ${off_wandb} --n_interv_sets ${n_interv_sets} --pts_per_interv ${pts_per_interv} --eq_noise_var ${eq_noise_var}"

cd exps
python weakly_supervised_biols_vector.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edges} --lr ${lr} --num_steps ${num_steps} --num_nodes ${num_nodes} --proj_dims ${proj_dims} --obs_data ${obs_data} --off_wandb ${off_wandb} --n_interv_sets ${n_interv_sets} --pts_per_interv ${pts_per_interv} --eq_noise_var ${eq_noise_var}
cd ..

echo $end
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"