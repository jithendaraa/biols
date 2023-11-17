import seaborn as sns
import pandas as pd
sns.set_style('dark')
from collections import defaultdict
import helpers.utils as utils


def get_plot_dataframe(data_folders, runs, reqd_keys):
    plot_data_dict = defaultdict(lambda: [])
    
    for data_folder in data_folders:
        exp_config = {'biols_data_folder': data_folder}
        exp_run = utils.get_reqd_runs(exp_config, runs, num_seeds=5)
        plotting_data = utils.get_plotting_data(exp_run, reqd_keys)
        splits = data_folder.split('-')
        exp_edges = int(splits[0][-1])
        proj = splits[2][:-4]
        if proj == '3_layer_mlp':   proj = 'nonlinear'
        d = int(splits[3][1:])
        D = int(splits[4][1:])
        interv_sets = int(splits[7][4:])
        interv_type = splits[5]
        interv_value_sampling = splits[-1][:-6]

        for key in reqd_keys:
            num_seeds = len(plotting_data[key][:, -1])
            lhs_key = key
            if 'SHD' in key:        lhs_key = 'SHD'
            elif 'AUROC' in key:    lhs_key = 'AUROC'
            plot_data_dict[lhs_key] += (plotting_data[key][:, -1]).tolist()
        
        rstring = r"$ER-{}, d={}, D={}\ $".format(exp_edges, d, D)
        plot_data_dict['Graph density'] += [rstring] * num_seeds
        plot_data_dict['Interventional Sets'] += [interv_sets] * num_seeds
        plot_data_dict['Intervention Target'] += [interv_type] * num_seeds
        plot_data_dict['Model'] += ['BIOLS'] * num_seeds
        plot_data_dict['Intervention Value'] += [interv_value_sampling] * num_seeds
        plot_data_dict['biols_data_folder'] += [exp_config['biols_data_folder']] * num_seeds

    plot_df = pd.DataFrame(plot_data_dict)
    name = f'er{exp_edges}_d{d}_D{D}_proj{proj}'
    return plot_df, name


def fetch_and_plot_interv_value_ablation(num_nodes, num_intervs, basepath, runs, reqd_keys, proj='linear', interv_types=['multi'], interv_values=['zeros', 'gaussian'], box_widths=0.8, fontsize=18):
    zfilled_nodes = str(num_nodes).zfill(3)
    data_folders = []
    for num_interv in num_intervs:
        for interv_type in interv_types:
            for interv_value in interv_values:
                datafolder = f'er1-ws_datagen_fix_noise_interv_noise-{proj}proj-d{zfilled_nodes}-D0100-{interv_type}-n_pairs{num_interv}-sets{int(num_interv/100)}-{interv_value}interv'
                data_folders.append(datafolder)

    plot_df, name = get_plot_dataframe(data_folders, runs, reqd_keys)
    utils.plot_interv_value_ablation(plot_df, basepath, name, reqd_keys, fontsize, interv_values=interv_values, box_widths=box_widths, title=None)
