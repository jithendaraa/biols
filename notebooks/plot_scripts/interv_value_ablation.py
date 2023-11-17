import sys
sys.path.append('..')
import wandb
from helpers.interv_value_ablation_helper import fetch_and_plot_interv_value_ablation

api = wandb.Api(timeout=40)
runs = api.runs("structurelearning/BIOLS")
max_steps = 20000
reqd_keys = ['Evaluations/SHD', 'Evaluations/AUROC', 'L_MSE']
basepath = '/home/mila/j/jithendaraa.subramanian/scratch/biols_datasets/interv_type_ablations'

# fetch_and_plot_interv_value_ablation(
#     num_nodes=20, 
#     num_intervs=[2000, 4000, 6000, 8000, 10000],
#     basepath='/home/mila/j/jithendaraa.subramanian/scratch/biols_datasets/interv_value_ablations', 
#     runs=runs, 
#     reqd_keys=reqd_keys, 
#     proj='linear',
#     interv_types=['multi'],
#     interv_values=['zeros', 'gaussian'],
#     box_widths=0.6
# )

# fetch_and_plot_interv_value_ablation(
#     num_nodes=30, 
#     num_intervs=[2000, 4000, 6000, 8000, 10000],
#     basepath='/home/mila/j/jithendaraa.subramanian/scratch/biols_datasets/interv_value_ablations', 
#     runs=runs, 
#     reqd_keys=reqd_keys, 
#     proj='linear',
#     interv_types=['multi'],
#     interv_values=['zeros', 'gaussian'],
#     box_widths=0.6
# )

# fetch_and_plot_interv_value_ablation(
#     num_nodes=50, 
#     num_intervs=[4000, 6000, 8000, 10000, 12000],
#     basepath='/home/mila/j/jithendaraa.subramanian/scratch/biols_datasets/interv_value_ablations', 
#     runs=runs, 
#     reqd_keys=reqd_keys, 
#     proj='linear',
#     interv_types=['multi'],
#     interv_values=['zeros', 'gaussian'],
#     box_widths=0.6
# )

fetch_and_plot_interv_value_ablation(
    num_nodes=20, 
    num_intervs=[2000, 4000, 6000, 8000, 10000],
    basepath='/home/mila/j/jithendaraa.subramanian/scratch/biols_datasets/interv_value_ablations', 
    runs=runs, 
    reqd_keys=reqd_keys, 
    proj='3_layer_mlp',
    interv_types=['multi'],
    interv_values=['zeros', 'gaussian'],
    box_widths=0.6
)

fetch_and_plot_interv_value_ablation(
    num_nodes=30, 
    num_intervs=[2000, 4000, 6000, 8000, 10000],
    basepath='/home/mila/j/jithendaraa.subramanian/scratch/biols_datasets/interv_value_ablations', 
    runs=runs, 
    reqd_keys=reqd_keys, 
    proj='3_layer_mlp',
    interv_types=['multi'],
    interv_values=['zeros', 'gaussian'],
    box_widths=0.6
)

fetch_and_plot_interv_value_ablation(
    num_nodes=50, 
    num_intervs=[4000, 6000, 8000, 10000, 12000],
    basepath='/home/mila/j/jithendaraa.subramanian/scratch/biols_datasets/interv_value_ablations', 
    runs=runs, 
    reqd_keys=reqd_keys, 
    proj='3_layer_mlp',
    interv_types=['multi'],
    interv_values=['zeros', 'gaussian'],
    box_widths=0.6
)