import numpy as np
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_reqd_runs(exp_config, runs, num_seeds=5):
    reqd_runs = []
    for run in runs:
        reqd_run = True
        for k,v in exp_config.items():
            if k not in run.config.keys() or run.config[k] != v: 
                reqd_run = False
                break
        if reqd_run is False: continue
        else: reqd_runs.append(run)   # This is a required run
    try:
        assert len(reqd_runs) == num_seeds
    except:
        print(len(reqd_runs), exp_config)
    return reqd_runs


def get_plotting_data(reqd_runs, reqd_keys, max_steps=20000):
    seed_data = {}
    for key in reqd_keys: seed_data[key] = []
    for run in reqd_runs:
        plotting_data = run.scan_history(reqd_keys, max_steps)

        for key in reqd_keys:
            seed_data[key].append([data[key] for data in plotting_data])
    
    for key in reqd_keys:
        seed_data[key] = np.array([x for x in seed_data[key] if x])
    return seed_data


def adjust_box_widths(g, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:
        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


def interpolate_color(start_color, end_color, n):
    # Convert hex color codes to RGB values
    start_r, start_g, start_b = int(start_color[1:3], 16), int(start_color[3:5], 16), int(start_color[5:7], 16)
    end_r, end_g, end_b = int(end_color[1:3], 16), int(end_color[3:5], 16), int(end_color[5:7], 16)
    
    # Calculate the step size for each color channel
    step_r = (end_r - start_r) / (n - 1)
    step_g = (end_g - start_g) / (n - 1)
    step_b = (end_b - start_b) / (n - 1)
    
    colors = []
    
    for i in range(n):
        # Calculate the RGB values for the current step
        r = int(start_r + i * step_r)
        g = int(start_g + i * step_g)
        b = int(start_b + i * step_b)
        
        # Convert RGB values back to hex
        hex_color = "#{:02X}{:02X}{:02X}".format(r, g, b)
        colors.append(hex_color)
    
    return colors


def get_legend_plot(ax, handles, basepath, filename, ncol):
    figlegend = plt.figure(figsize=(2*ncol, 2))
    plt.legend(*handles, loc ='upper left', fontsize=32, ncol=ncol)
    plt.axis("off")
    filename=f"{basepath}/{filename}.pdf"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Legends saved at: {filename}")
    plt.show()



def plot_num_interventions_ablation(plot_df, basepath, name, reqd_keys, fontsize, num_intervs, 
    title=None, box_widths=0.8, key='Interventional Sets', x='Graph density'):

    num_rows = 1
    num_cols = 3

    # Create the subplots
    plt.rcParams.update({'font.size': fontsize})
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
    lines_labels = []
    
    start_color = '#81c361'
    end_color = '#8a59b7'
    colors = interpolate_color(start_color, end_color, n=len(num_intervs))

    for i, metric in enumerate(reqd_keys):
        # Extract the metric values for each num intervention sets and seed
        metric = metric.split('/')[-1]
        data = []
        for num_interv in num_intervs:
            model_data = plot_df[plot_df[key] == num_interv][metric]
            data.append(model_data)

        # Plot the data in the subplot
        ax = axs[i]
        PROPS = {
            'boxprops':{'alpha':0.5},
            'medianprops':{'alpha':0.7},
            'whiskerprops':{'alpha':0.7},
            'capprops':{'alpha':0.7}
        }
        
        sns.boxplot(ax=ax, x=x, y=metric, data=plot_df, hue=key, showfliers = False, palette=colors, dodge=0.4, **PROPS)
        sns.stripplot(
            ax=ax, 
            x=x, 
            y=metric, 
            data=plot_df, 
            hue=key, 
            dodge=True, 
            alpha=.7, 
            palette=colors, 
            ec='k',
            jitter=True,
            legend=False
        )

        ax.set_ylabel('')
        if metric == 'L_MSE':
            ax.title.set_text(r"$MSE(L, \hat{L})$")
        else:
            ax.title.set_text(metric)
        ax.grid(axis='y')

        if i == 0:
            lines_labels.append(ax.get_legend_handles_labels())
        
        ax.get_legend().remove()
        ax.set_xlabel('')

    adjust_box_widths(fig, box_widths)
    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()
    os.makedirs(basepath, exist_ok=True)
    filename = f'{basepath}/ablation_num_interventions_{name}.pdf'
    plt.savefig(filename)
    print(f'Saved to {filename}')
    plt.show()

    get_legend_plot(ax, lines_labels[0], basepath, f'{name}_num_intervention_ablation_legend', ncol=len(colors))



def plot_interv_value_ablation(plot_df, basepath, name, reqd_keys, fontsize, interv_values, 
    title=None, box_widths=0.8, key='Intervention Value', x='Interventional Sets'):
    num_rows = 1
    num_cols = 3
    
    # Create the subplots
    plt.rcParams.update({'font.size': fontsize})
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
    lines_labels = []
    
    start_color = '#81c361'
    end_color = '#8a59b7'
    colors = interpolate_color(start_color, end_color, n=len(interv_values))

    for i, metric in enumerate(reqd_keys):
        # Extract the metric values for each num intervention sets and seed
        metric = metric.split('/')[-1]
        data = []

        for interv_value in interv_values:
            model_data = plot_df[plot_df[key] == interv_value][metric]
            data.append(model_data)

        # Plot the data in the subplot
        ax = axs[i]
        PROPS = {
            'boxprops':{'alpha':0.5},
            'medianprops':{'alpha':0.7},
            'whiskerprops':{'alpha':0.7},
            'capprops':{'alpha':0.7}
        }
        
        sns.boxplot(ax=ax, x=x, y=metric, data=plot_df, hue=key, showfliers = False, palette=colors, dodge=0.4, **PROPS)
        sns.stripplot(
            ax=ax, 
            x=x, 
            y=metric, 
            data=plot_df, 
            hue=key, 
            dodge=True, 
            alpha=.7, 
            palette=colors, 
            ec='k',
            jitter=True,
            legend=False
        )

        ax.set_ylabel('')
        if metric == 'L_MSE':
            ax.title.set_text(r"$MSE(L, \hat{L})$")
        else:
            ax.title.set_text(metric)
        ax.grid(axis='y')

        if i == 0:
            lines_labels.append(ax.get_legend_handles_labels())
        
        ax.get_legend().remove()
        ax.set_xlabel('')

    adjust_box_widths(fig, box_widths)
    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()
    os.makedirs(basepath, exist_ok=True)
    filename = f'{basepath}/ablation_interv_value_{name}.pdf'
    plt.savefig(filename)
    print(f'Saved to {filename}')
    plt.show()

    get_legend_plot(ax, lines_labels[0], basepath, f'{name}_interv_value_ablation_legend', ncol=len(colors))



def plot_interv_type_ablation(plot_df, basepath, name, reqd_keys, fontsize, interv_types=['multi', 'single'], 
    title=None, box_widths=0.8, key='Intervention Target', x='Interventional Sets'):

    num_rows = 1
    num_cols = 3

    # Create the subplots
    plt.rcParams.update({'font.size': fontsize})
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 6*num_rows))
    lines_labels = []
    
    start_color = '#81c361'
    end_color = '#8a59b7'
    colors = interpolate_color(start_color, end_color, n=len(interv_types))

    for i, metric in enumerate(reqd_keys):
        # Extract the metric values for each num intervention sets and seed
        metric = metric.split('/')[-1]   
        
        # Plot the data in the subplot
        ax = axs[i]
        PROPS = {
            'boxprops':{'alpha':0.5},
            'medianprops':{'alpha':0.7},
            'whiskerprops':{'alpha':0.7},
            'capprops':{'alpha':0.7}
        }
        
        sns.boxplot(ax=ax, x=x, y=metric, data=plot_df, hue=key, showfliers = False, palette=colors, dodge=0.4, **PROPS)
        sns.stripplot(
            ax=ax, 
            x=x, 
            y=metric, 
            data=plot_df, 
            hue=key, 
            dodge=True, 
            alpha=.7, 
            palette=colors, 
            ec='k',
            jitter=True,
            legend=False
        )

        ax.set_ylabel('')
        if metric == 'L_MSE':
            ax.title.set_text(r"$MSE(L, \hat{L})$")
        else:
            ax.title.set_text(metric)
        ax.grid(axis='y')

        if i == 0:
            lines_labels.append(ax.get_legend_handles_labels())
        
        ax.get_legend().remove()
        ax.set_xlabel('')

    adjust_box_widths(fig, box_widths)
    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()
    os.makedirs(basepath, exist_ok=True)
    filename = f'{basepath}/ablation_interv_type_{name}.pdf'
    plt.savefig(filename)
    print(f'Saved to {filename}')
    plt.show()

    get_legend_plot(ax, lines_labels[0], basepath, f'{name}_interv_type_ablation_legend', ncol=len(colors))


def plot_scaling_curves(plot_df, basepath, name, reqd_keys, fontsize, title=None, 
    box_widths=0.8, key='Intervention Target', x='Nodes'):

    num_rows = 1
    num_cols = 3

    # Create the subplots
    plt.rcParams.update({'font.size': fontsize})
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
    lines_labels = []
    color = ['#8a59b7']

    for i, metric in enumerate(reqd_keys):
        # Extract the metric values for each num intervention sets and seed
        metric = metric.split('/')[-1]

        # Plot the data in the subplot
        ax = axs[i]
        sns.lineplot(ax=ax, x=x, y=metric, data=plot_df, hue=key, markers=True, dashes=False, style=key, palette=color)

        ax.set_ylabel('')
        if metric == 'L_MSE':
            ax.title.set_text(r"$MSE(L, \hat{L})$")
        else:
            ax.title.set_text(metric)
        ax.grid(axis='y')

        if i == 0:
            lines_labels.append(ax.get_legend_handles_labels())
        
        ax.get_legend().remove()
        ax.set_xlabel('')

    adjust_box_widths(fig, box_widths)
    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()
    os.makedirs(basepath, exist_ok=True)
    filename = f'{basepath}/scaling_{name}.pdf'
    plt.savefig(filename)
    print(f'Saved to {filename}')
    plt.show()


def plot_graph_density_ablation(plot_df, basepath, name, reqd_keys, fontsize, graph_densities, title=None, box_widths=0.8, 
    key='Graph density', x='x-caption'):

    num_rows, num_cols = 1, 3

    # Create the subplots
    plt.rcParams.update({'font.size': fontsize})
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
    lines_labels = []
    
    start_color = '#81c361'
    end_color = '#8a59b7'
    colors = interpolate_color(start_color, end_color, n=len(graph_densities))

    for i, metric in enumerate(reqd_keys):
        # Extract the metric values for each num intervention sets and seed
        metric = metric.split('/')[-1]
        data = []
        for graph_density in graph_densities:
            model_data = plot_df[plot_df[key] == graph_density][metric]
            data.append(model_data)

        # Plot the data in the subplot
        ax = axs[i]
        PROPS = {
            'boxprops':{'alpha':0.5},
            'medianprops':{'alpha':0.7},
            'whiskerprops':{'alpha':0.7},
            'capprops':{'alpha':0.7}
        }
        
        sns.boxplot(ax=ax, x=x, y=metric, data=plot_df, hue=key, showfliers = False, palette=colors, dodge=0.4, **PROPS)
        sns.stripplot(
            ax=ax, 
            x=x, 
            y=metric, 
            data=plot_df, 
            hue=key, 
            dodge=True, 
            alpha=.7, 
            palette=colors, 
            ec='k',
            jitter=True,
            legend=False
        )

        ax.set_ylabel('')
        if metric == 'L_MSE':
            ax.title.set_text(r"$MSE(L, \hat{L})$")
        else:
            ax.title.set_text(metric)
        ax.grid(axis='y')

        if i == 0:
            lines_labels.append(ax.get_legend_handles_labels())
        
        ax.get_legend().remove()
        ax.set_xlabel('')

    adjust_box_widths(fig, box_widths)
    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()
    os.makedirs(basepath, exist_ok=True)
    filename = f'{basepath}/ablation_graph_density_{name}.pdf'
    plt.savefig(filename)
    print(f'Saved to {filename}')
    plt.show()

    get_legend_plot(ax, lines_labels[0], basepath, f'{name}_graph_density_ablation_legend', ncol=len(colors))
