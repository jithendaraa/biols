#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6

seeds=(1 2 3 4 5 6 7 8 9 10)
lrs=(0.0008)
num_steps=(2000)

num_nodes=5
exp_edges=(1.0)
n_interv_sets=20
eq_noise_var='False'

pts_per_interv=100
off_wandb='False'
obs_data=500
num_samples=(2500)

defg=$(( ${#exp_edges[@]} * ${#lrs[@]} * ${#num_steps[@]} * ${#num_samples[@]} ))
efg=$(( ${#lrs[@]} * ${#num_steps[@]} * ${#num_samples[@]} ))
fg=$(( ${#num_steps[@]} * ${#num_samples[@]} ))
g=$(( ${#num_samples[@]} ))

id=$1
seed=${seeds[ $((  (${SLURM_ARRAY_TASK_ID} / ${defg})  % ${#seeds[@]} )) ]}
exp_edge=${exp_edges[ $((  (${SLURM_ARRAY_TASK_ID} / ${efg}) % ${#exp_edges[@]} )) ]}
lr=${lrs[ $((  (${SLURM_ARRAY_TASK_ID} / ${fg}) % ${#lrs[@]} )) ]}
step=${num_steps[ $((  (${SLURM_ARRAY_TASK_ID} / ${g}) % ${#num_steps[@]} )) ]}
num_sample=${num_samples[ $((  ${SLURM_ARRAY_TASK_ID} % ${#num_samples[@]} )) ]}

start=`date +%s`
echo "Script"

module load anaconda/3
module unload cuda/11.2 && module load cuda/11.0
conda activate lbcd
echo `date` "Python starting"
echo "python biols_image_data.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --num_steps ${step} --num_samples ${num_sample} --num_nodes ${num_nodes} --obs_data ${obs_data} --off_wandb ${off_wandb} --n_interv_sets ${n_interv_sets} --pts_per_interv ${pts_per_interv} --eq_noise_var ${eq_noise_var}"

cd exps
python biols_image_data.py --config defaults ${id} --data_seed ${seed} --exp_edges ${exp_edge} --lr ${lr} --num_steps ${step} --num_samples ${num_sample} --num_nodes ${num_nodes} --obs_data ${obs_data} --off_wandb ${off_wandb} --n_interv_sets ${n_interv_sets} --pts_per_interv ${pts_per_interv} --eq_noise_var ${eq_noise_var}
cd ..

echo $end
end=`date +%s`
runtime=$((end-start))
echo "Program time: $runtime"