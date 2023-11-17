import sys
sys.path.append('..')
import wandb

api = wandb.Api(timeout=40)
runs = api.runs("structurelearning/BIOLS")
max_steps = 20000
reqd_keys = ['Evaluations/SHD', 'Evaluations/AUROC', 'L_MSE']

from helpers.num_interventions_ablation_helper import fetch_and_plot_num_intervs_ablation

basepath = '/home/mila/j/jithendaraa.subramanian/scratch/biols_datasets/graph_density_ablations'

fetch_and_plot_num_intervs_ablation(
    num_nodes=30, 
    num_intervs=[4000, 6000, 8000, 10000, 12000, 14000], 
    basepath=basepath, 
    runs=runs, 
    reqd_keys=reqd_keys, 
    proj='3_layer_mlp',
    interv_type='multi',
    box_widths=0.6
)

print()

fetch_and_plot_num_intervs_ablation(
    num_nodes=50, 
    num_intervs=[4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000], 
    basepath=basepath, 
    runs=runs, 
    reqd_keys=reqd_keys, 
    proj='3_layer_mlp',
    interv_type='multi',
    box_widths=0.75
)