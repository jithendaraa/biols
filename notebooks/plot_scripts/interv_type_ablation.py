import sys
sys.path.append('..')
import wandb
from helpers.interv_type_ablation_helper import fetch_and_plot_interv_type_ablation

api = wandb.Api(timeout=40)
runs = api.runs("structurelearning/BIOLS")
max_steps = 20000
reqd_keys = ['Evaluations/SHD', 'Evaluations/AUROC', 'L_MSE']
basepath = '/home/mila/j/jithendaraa.subramanian/scratch/biols_datasets/interv_type_ablations'

# fetch_and_plot_interv_type_ablation(
#     num_nodes=20, 
#     num_intervs=[2000, 4000, 6000, 8000, 10000], 
#     basepath=basepath, 
#     runs=runs, 
#     reqd_keys=reqd_keys, 
#     proj='linear',
#     interv_types=['multi', 'single'],
#     box_widths=0.9
# )

# fetch_and_plot_interv_type_ablation(
#     num_nodes=30, 
#     num_intervs=[4000, 6000, 8000, 10000, 12000, 14000], 
#     basepath=basepath, 
#     runs=runs, 
#     reqd_keys=reqd_keys, 
#     proj='linear',
#     interv_types=['multi', 'single'],
#     box_widths=0.9
# )

# fetch_and_plot_interv_type_ablation(
#     num_nodes=40, 
#     num_intervs=[4000, 6000, 8000, 10000, 12000, 14000], 
#     basepath=basepath, 
#     runs=runs, 
#     reqd_keys=reqd_keys, 
#     proj='linear',
#     interv_types=['multi', 'single'],
#     box_widths=0.9
# )

# fetch_and_plot_interv_type_ablation(
#     num_nodes=50, 
#     num_intervs=[4000, 6000, 8000, 10000, 12000, 14000], 
#     basepath=basepath, 
#     runs=runs, 
#     reqd_keys=reqd_keys, 
#     proj='linear',
#     interv_types=['multi', 'single'],
#     box_widths=0.9
# )

fetch_and_plot_interv_type_ablation(
    num_nodes=20, 
    num_intervs=[2000, 4000, 6000, 8000, 10000], 
    basepath=basepath, 
    runs=runs, 
    reqd_keys=reqd_keys, 
    proj='3_layer_mlp',
    interv_types=['multi', 'single'],
    box_widths=0.9
)

fetch_and_plot_interv_type_ablation(
    num_nodes=30, 
    num_intervs=[4000, 6000, 8000, 10000, 12000, 14000], 
    basepath=basepath, 
    runs=runs, 
    reqd_keys=reqd_keys, 
    proj='3_layer_mlp',
    interv_types=['multi', 'single'],
    box_widths=0.9
)

# fetch_and_plot_interv_type_ablation(
#     num_nodes=40, 
#     num_intervs=[4000, 6000, 8000, 10000, 12000, 14000], 
#     basepath=basepath, 
#     runs=runs, 
#     reqd_keys=reqd_keys, 
#     proj='3_layer_mlp',
#     interv_types=['multi', 'single'],
#     box_widths=0.9
# )

fetch_and_plot_interv_type_ablation(
    num_nodes=50, 
    num_intervs=[4000, 6000, 8000, 10000, 12000, 14000], 
    basepath=basepath, 
    runs=runs, 
    reqd_keys=reqd_keys, 
    proj='3_layer_mlp',
    interv_types=['multi', 'single'],
    box_widths=0.9
)