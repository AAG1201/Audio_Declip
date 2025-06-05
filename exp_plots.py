import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import scipy.io as sio
import json


# Set the output directory for saving plots
output_dir = "/data2/AAG/Audio_Declip/saved_plots"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Common plot settings
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'axes.linewidth': 1.2,
    'axes.labelpad': 10
})

# Color scheme for consistency across plots
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ---- Experiment 1: Duration Analysis ----
def plot_exp1():
    print("Plotting Experiment 1: Duration Analysis")
    
    # Dataset file paths - make sure these paths are correct
    file_paths = {
        'Bird': '/data2/AAG/Audio_Declip/exp1_new/bird_sound/evaluation_results_baseline_model_SDR_.xlsx',
        'Heart': '/data2/AAG/Audio_Declip/exp1_new/heart_sound/evaluation_results_baseline_model_SDR_.xlsx',
        'Lung': '/data2/AAG/Audio_Declip/exp1_new/lung_sound/evaluation_results_baseline_model_SDR_.xlsx',
        'Speech': '/data2/AAG/Audio_Declip/exp1_new/speech_sound/evaluation_results_baseline_model_SDR_.xlsx'
    }
    
    # Load and prepare datasets
    datasets = {}
    for name, path in file_paths.items():
        df = pd.read_excel(path)
        df['duration'] = df['duration'].astype(float)
        df['sdr_orig'] = df['sdr_orig'].astype(float)
        df['sdr_orig_rounded'] = df['sdr_orig'].round(1)
        datasets[name] = df
    
    durations = [1, 2, 4, 8]
    
    # Metrics to plot (removed cycles)
    metrics = {
        'delta_sdr': 'ΔSDR (dB)',
        'processing_time': 'Processing Time (s)'
    }
    
    # Generate plots for each metric
    for metric, ylabel in metrics.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        fig.subplots_adjust(top=0.88)
        axes = axes.flatten()
        letters = ['(a)', '(b)', '(c)', '(d)']
        
        for ax, (label, df), letter in zip(axes, datasets.items(), letters):
            unique_sdrs = sorted(df['sdr_orig_rounded'].unique())
            grouped = df.groupby(['sdr_orig_rounded', 'duration'])[metric].agg(['mean', 'std']).unstack()
            
            x = np.arange(len(unique_sdrs))
            width = 0.2
            
            for i, dur in enumerate(durations):
                means = grouped['mean'][dur].reindex(unique_sdrs).values
                stds = grouped['std'][dur].reindex(unique_sdrs).values
                ax.bar(
                    x + i * width,
                    means,
                    width,
                    yerr=stds,
                    label=f'{dur} sec',
                    color=colors[i],
                    capsize=4,
                    edgecolor='black',
                    linewidth=0.7
                )
            
            ax.set_xticks(x + 1.5 * width)
            ax.set_xticklabels(unique_sdrs, rotation=45)
            ax.set_title(f"{letter} {label}", loc='left')
            ax.grid(True, linestyle='--', alpha=0.4)
        
        # Set axis labels
        axes[0].set_ylabel(ylabel)
        axes[2].set_ylabel(ylabel)
        axes[2].set_xlabel("Input SDR (dB)")
        axes[3].set_xlabel("Input SDR (dB)")
        
        # Add common legend
        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4,
                   fontsize=14, title="Durations", title_fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        save_path = os.path.join(output_dir, f"exp1_{metric}.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close(fig)

        # Export table of values
        all_tables = []
        for label, df in datasets.items():
            df['sdr_orig_rounded'] = df['sdr_orig'].round(1)
            summary = df.groupby(['sdr_orig_rounded', 'duration'])[metric].agg(['mean', 'std']).reset_index()
            summary['Dataset'] = label
            summary['Metric'] = metric
            all_tables.append(summary)

        result_df = pd.concat(all_tables, ignore_index=True)
        csv_path = os.path.join(output_dir, f"exp1_{metric}_summary.csv")
        result_df.to_csv(csv_path, index=False)
        print(f"Saved table of values to {csv_path}")
        


# ---- Experiment 2: Sampling Rate Analysis ----
def plot_exp2():
    print("Plotting Experiment 2: Sampling Rate Analysis")
    
    # Dataset file paths - make sure these paths are correct
    file_paths = {
        'Bird': '/data2/AAG/Audio_Declip/exp2_new/bird_sound/evaluation_results_baseline_model_SDR_.xlsx',
        'Heart': '/data2/AAG/Audio_Declip/exp2_new/heart_sound/evaluation_results_baseline_model_SDR_.xlsx',
        'Lung': '/data2/AAG/Audio_Declip/exp2_new/lung_sound/evaluation_results_baseline_model_SDR_.xlsx',
        'Speech': '/data2/AAG/Audio_Declip/exp2_new/speech_sound/evaluation_results_baseline_model_SDR_.xlsx'
    }
    
    # Load and prepare datasets
    datasets = {}
    for label, path in file_paths.items():
        df = pd.read_excel(path)
        df['sdr_orig'] = df['sdr_orig'].astype(float)
        df['sdr_orig_rounded'] = df['sdr_orig'].round(1)
        df['target_fs'] = df['target_fs'].astype(int)
        datasets[label] = df
    
    # Metrics to plot (removed cycles)
    metrics = {
        'delta_sdr': 'ΔSDR (dB)',
        'processing_time': 'Processing Time (s)'
    }
    
        # Generate plots for each metric
    for metric, ylabel in metrics.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        letters = ['(a)', '(b)', '(c)', '(d)']

        all_tables = []  # <-- Add this to collect summary data

        for ax, (label, df), letter in zip(axes, datasets.items(), letters):
            unique_sdrs = sorted(df['sdr_orig_rounded'].unique())
            fs_values = sorted(df['target_fs'].unique())
            max_fs = max(fs_values)
            relative_fs_labels = {
                max_fs // 8: 'fs/8',
                max_fs // 4: 'fs/4',
                max_fs // 2: 'fs/2',
                max_fs: 'fs'
            }
            num_fs = len(fs_values)
            grouped = df.groupby(['sdr_orig_rounded', 'target_fs'])[metric].agg(['mean', 'std']).unstack()

            x = np.arange(len(unique_sdrs))
            width = 0.8 / num_fs

            for i, fs in enumerate(fs_values):
                means = grouped['mean'][fs].reindex(unique_sdrs).values
                stds = grouped['std'][fs].reindex(unique_sdrs).values
                ax.bar(
                    x + i * width,
                    means,
                    width,
                    yerr=stds,
                    label=relative_fs_labels.get(fs, f'{fs} Hz'),
                    color=colors[i % len(colors)],
                    capsize=4,
                    edgecolor='black',
                    linewidth=0.7
                )

                # Collect data for the table
                table_df = pd.DataFrame({
                    'Dataset': label,
                    'Input SDR (dB)': unique_sdrs,
                    'Sampling Rate (Hz)': fs,
                    f'Mean {ylabel}': means,
                    f'Std {ylabel}': stds
                })
                all_tables.append(table_df)

            ax.set_xticks(x + (width * num_fs) / 2)
            ax.set_xticklabels(unique_sdrs, rotation=45)
            ax.set_title(f"{letter} {label}", loc='left')
            ax.grid(True, linestyle='--', alpha=0.4)

        # Set axis labels
        axes[0].set_ylabel(ylabel)
        axes[2].set_ylabel(ylabel)
        axes[2].set_xlabel("Input SDR (dB)")
        axes[3].set_xlabel("Input SDR (dB)")

        # Add common legend
        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=num_fs,
                   fontsize=14, title="Target Sampling Rate (Hz)", title_fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(output_dir, f"exp2_{metric}.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close(fig)

        # Save table
        result_df = pd.concat(all_tables, ignore_index=True)
        csv_path = os.path.join(output_dir, f"exp2_{metric}_summary.csv")
        result_df.to_csv(csv_path, index=False)
        print(f"Saved table of values to {csv_path}")



# ---- Experiment 3: Window Length Analysis ----
def plot_exp3():
    print("Plotting Experiment 3: Window Length Analysis")
    
    # Dataset file paths - make sure these paths are correct
    file_paths = {
        'Bird': '/data2/AAG/Audio_Declip/exp3_new/bird_sound/evaluation_results_baseline_model_SDR_timewise_len.xlsx',
        'Heart': '/data2/AAG/Audio_Declip/exp3_new/heart_sound/evaluation_results_baseline_model_SDR_timewise_len.xlsx',
        'Lung': '/data2/AAG/Audio_Declip/exp3_new/lung_sound/evaluation_results_baseline_model_SDR_timewise_len.xlsx',
        'Speech': '/data2/AAG/Audio_Declip/exp3_new/speech_sound/evaluation_results_baseline_model_SDR_timewise_len.xlsx'
    }
    
    # Load and prepare datasets
    datasets = {}
    for label, path in file_paths.items():
        df = pd.read_excel(path)
        
        for col in ['delta_sdr', 'processing_time', 'sdr_orig', 'winlen']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        df['sdr_orig_rounded'] = df['sdr_orig'].round(1)
        datasets[label] = df
    
    # Metrics to plot (removed cycles)
    metrics = {
        'delta_sdr': 'ΔSDR (dB)',
        'processing_time': 'Processing Time (s)'
    }
    
    # Generate line plots for each metric
    for metric, ylabel in metrics.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=False)
        axes = axes.flatten()
        letters = ['(a)', '(b)', '(c)', '(d)']
        
        legend_handles_labels = {}
        
        for ax, (label, df), letter in zip(axes, datasets.items(), letters):
            sdr_levels = sorted(df['sdr_orig_rounded'].unique())
            
            for i, sdr in enumerate(sdr_levels):
                subset = df[df['sdr_orig_rounded'] == sdr]
                grouped = subset.groupby('winlen')[metric].mean()
                stds = subset.groupby('winlen')[metric].std()
                
                line, = ax.plot(
                    grouped.index,
                    grouped.values,
                    label=f'{sdr} dB',
                    marker='o',
                    linewidth=2,
                    color=colors[i % len(colors)]
                )
                
                # Add error shading (optional)
                ax.fill_between(grouped.index, grouped.values - stds, grouped.values + stds, 
                                alpha=0.2, color=colors[i % len(colors)])
                
                if f'{sdr} dB' not in legend_handles_labels:
                    legend_handles_labels[f'{sdr} dB'] = line
            
            ax.set_title(f"{letter} {label}", loc='left')
            ax.grid(True, linestyle='--', alpha=0.4)
        
        # Set axis labels
        axes[0].set_ylabel(ylabel)
        axes[2].set_ylabel(ylabel)
        axes[2].set_xlabel("Window Length")
        axes[3].set_xlabel("Window Length")
        
        # Add common legend
        handles = list(legend_handles_labels.values())
        labels_ = list(legend_handles_labels.keys())
        fig.legend(handles, labels_, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4,
                   fontsize=14, title="Input SDR (dB)", title_fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(output_dir, f"exp3_{metric}.png")
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close(fig)

                # Export table of values
        all_tables = []
        for label, df in datasets.items():
            df['sdr_orig_rounded'] = df['sdr_orig'].round(1)
            grouped = df.groupby(['sdr_orig_rounded', 'winlen'])[metric].agg(['mean', 'std']).reset_index()
            grouped['Dataset'] = label
            grouped['Metric'] = metric
            all_tables.append(grouped)

        result_df = pd.concat(all_tables, ignore_index=True)
        csv_path = os.path.join(output_dir, f"exp3_{metric}_summary.csv")
        result_df.to_csv(csv_path, index=False)
        print(f"Saved table of values to {csv_path}")



# def block_analysis(i):
#         # Load baseline model results
#     results_mat_path = "/data2/AAG/Audio_Declip/exp/baseline/evaluation_results_baseline_model_SDR_.mat"
#     data_baseline = sio.loadmat(results_mat_path)
#     a = data_baseline['results']['objVal'][0][0][0][i][0][data_baseline['results']['Block'][0][0][0][i][0] == 1]

#     # Load dynamic model results
#     results_mat_path1 = "/data2/AAG/Audio_Declip/exp/dynamic/evaluation_results_dynamic_model_SDR_.mat"
#     data_dynamic = sio.loadmat(results_mat_path1)
#     b = data_dynamic['results']['objVal'][0][0][0][i][0][data_dynamic['results']['Block'][0][0][0][i][0] == 1]

#     # Plotting
#     plt.figure(figsize=(10, 5))
#     plt.plot(a, label='Baseline Model')
#     plt.plot(b, label='Dynamic Model')
#     plt.xlabel("Iteration")
#     plt.ylabel("Objective Value (SDR)")
#     plt.title("Comparison of Objective Value (SDR) for Block 0")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     a = data_baseline['results']['k'][0][0][0][i][0][data_baseline['results']['Block'][0][0][0][i][0] == 1]
#     b = data_dynamic['results']['k'][0][0][0][i][0][data_dynamic['results']['Block'][0][0][0][i][0] == 1]

#     # Plotting
#     plt.figure(figsize=(10, 5))
#     plt.plot(a, label='Baseline Model')
#     plt.plot(b, label='Dynamic Model')
#     plt.xlabel("Iteration")
#     plt.ylabel("k value")
#     plt.title("Comparison of Objective Value (SDR) for Block 0")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()



def block_analysis(u, idx):
    # Set professional plot style with adjusted font sizes
    plt.rcParams.update({
    'font.size': 14,            # Slightly larger font
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 20,
    'figure.dpi': 300,
    'axes.linewidth': 1.2,      # Slightly bolder axes
    'axes.labelpad': 10,        # More spacing between label and axis
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'lines.markersize': 5,      # Slightly larger markers
    'lines.linewidth': 1.8,     # Slightly thicker lines for better visibility
    'legend.frameon': False,    # Remove legend box
    'grid.alpha': 0.4           # Slightly more visible grid
    })
    
    # Define paths for all sound types
    sound_types = ['bird_sound', 'heart_sound', 'lung_sound', 'speech_sound']
    sound_labels = ['Bird', 'Heart', 'Lung', 'Speech']
    
    # Initialize data storage for summary tables
    summary_data = []
    detailed_data = {'Sound_Type': [], 'Model': [], 'Iteration': [], 'SDR': [], 'K_Value': []}
    
    # Create figures with professional styling
    fig_obj, axes_obj = plt.subplots(2, 2, figsize=(12, 8))
    fig_obj.subplots_adjust(top=0.90, hspace=0.3, wspace=0.3)
    axes_obj = axes_obj.flatten()
    
    fig_k, axes_k = plt.subplots(2, 2, figsize=(12, 8))
    fig_k.subplots_adjust(top=0.90, hspace=0.3, wspace=0.3)
    axes_k = axes_k.flatten()
    
    # Professional color scheme with reduced line thickness
    colors = ['#1f77b4', '#ff7f0e']  # Blue for baseline, Orange for dynamic
    
    for i, (sound_type, sound_label) in enumerate(zip(sound_types, sound_labels)):
        try:
            # Load baseline model results
            results_mat_path = f"/data2/AAG/Audio_Declip/exp/comparison/{sound_type}/Block_evaluation_results_baseline_model_SDR_.mat"
            data_baseline = sio.loadmat(results_mat_path)
            a_obj = data_baseline['results']['objVal'][0][0][0][u][0][data_baseline['results']['Block'][0][0][0][u][0] == idx]
            a_k = data_baseline['results']['k'][0][0][0][u][0][data_baseline['results']['Block'][0][0][0][u][0] == idx]

            # Load dynamic model results
            results_mat_path1 = f"/data2/AAG/Audio_Declip/exp/comparison/{sound_type}/Block_evaluation_results_dynamic_model_SDR_.mat"
            data_dynamic = sio.loadmat(results_mat_path1)
            b_obj = data_dynamic['results']['objVal'][0][0][0][u][0][data_dynamic['results']['Block'][0][0][0][u][0] == idx]
            b_k = data_dynamic['results']['k'][0][0][0][u][0][data_dynamic['results']['Block'][0][0][0][u][0] == idx]

            # Store detailed data for tables
            for iter_num, (sdr_base, k_base) in enumerate(zip(a_obj, a_k)):
                detailed_data['Sound_Type'].append(sound_label)
                detailed_data['Model'].append('Baseline')
                detailed_data['Iteration'].append(iter_num + 1)
                detailed_data['SDR'].append(float(sdr_base))
                detailed_data['K_Value'].append(float(k_base))
            
            for iter_num, (sdr_dyn, k_dyn) in enumerate(zip(b_obj, b_k)):
                detailed_data['Sound_Type'].append(sound_label)
                detailed_data['Model'].append('Dynamic')
                detailed_data['Iteration'].append(iter_num + 1)
                detailed_data['SDR'].append(float(sdr_dyn))
                detailed_data['K_Value'].append(float(k_dyn))

            # Calculate summary statistics
            summary_stats = {
                'Sound_Type': sound_label,
                'Baseline_SDR_Mean': np.mean(a_obj),
                'Baseline_SDR_Std': np.std(a_obj),
                'Baseline_SDR_Final': a_obj[-1] if len(a_obj) > 0 else np.nan,
                'Dynamic_SDR_Mean': np.mean(b_obj),
                'Dynamic_SDR_Std': np.std(b_obj),
                'Dynamic_SDR_Final': b_obj[-1] if len(b_obj) > 0 else np.nan,
                'SDR_Improvement': (np.mean(b_obj) - np.mean(a_obj)),
                'Baseline_K_Mean': np.mean(a_k),
                'Dynamic_K_Mean': np.mean(b_k),
                'K_Difference': (np.mean(b_k) - np.mean(a_k)),
                'Convergence_Iterations': min(len(a_obj), len(b_obj))
            }
            summary_data.append(summary_stats)

            # Plot objective values with reduced line thickness
            ax_obj = axes_obj[i]
            iterations_base = range(1, len(a_obj) + 1)
            iterations_dyn = range(1, len(b_obj) + 1)
            
            # Plot lines with reduced thickness and smaller markers
            ax_obj.plot(iterations_base, a_obj, 
                       color=colors[0], linewidth=1.5, markersize=4,
                       label='Baseline Model', markerfacecolor='white', 
                       markeredgecolor=colors[0], markeredgewidth=1.5)
            ax_obj.plot(iterations_dyn, b_obj, 
                       color=colors[1], linewidth=1.5, markersize=4,
                       label='Dynamic Model', markerfacecolor='white',
                       markeredgecolor=colors[1], markeredgewidth=1.5)
            
            # Customize axis labels based on subplot position
            row, col = divmod(i, 2)
            if row == 0 and col == 0:  # (0,0) - only y axis label
                ax_obj.set_ylabel("Objective Value (SDR)")
            elif row == 0 and col == 1:  # (0,1) - no axis labels
                pass
            elif row == 1 and col == 0:  # (1,0) - both x and y axis labels
                ax_obj.set_xlabel("Iteration")
                ax_obj.set_ylabel("Objective Value (SDR)")
            elif row == 1 and col == 1:  # (1,1) - only x axis label
                ax_obj.set_xlabel("Iteration")
            
            # Professional axes styling without titles
            ax_obj.grid(True, linestyle='--', alpha=0.3)
            ax_obj.spines['top'].set_visible(False)
            ax_obj.spines['right'].set_visible(False)
            
            # Plot k values with reduced line thickness
            ax_k = axes_k[i]
            ax_k.plot(iterations_base, a_k, 
                     color=colors[0], linewidth=1.5, markersize=4,
                     label='Baseline Model', markerfacecolor='white',
                     markeredgecolor=colors[0], markeredgewidth=1.5)
            ax_k.plot(iterations_dyn, b_k, 
                     color=colors[1], linewidth=1.5, markersize=4,
                     label='Dynamic Model', markerfacecolor='white',
                     markeredgecolor=colors[1], markeredgewidth=1.5)
            
            # Customize axis labels based on subplot position
            if row == 0 and col == 0:  # (0,0) - only y axis label
                ax_k.set_ylabel("k Value")
            elif row == 0 and col == 1:  # (0,1) - no axis labels
                pass
            elif row == 1 and col == 0:  # (1,0) - both x and y axis labels
                ax_k.set_xlabel("Iteration")
                ax_k.set_ylabel("k Value")
            elif row == 1 and col == 1:  # (1,1) - only x axis label
                ax_k.set_xlabel("Iteration")
            
            ax_k.grid(True, linestyle='--', alpha=0.3)
            ax_k.spines['top'].set_visible(False)
            ax_k.spines['right'].set_visible(False)
            
        except Exception as e:
            print(f"Error processing {sound_type}: {str(e)}")
            continue
    
    # Add single common legend at the top center
    handles, labels = axes_obj[0].get_legend_handles_labels()
    fig_obj.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
                   ncol=2, fontsize=18, frameon=False)
    
    handles_k, labels_k = axes_k[0].get_legend_handles_labels()
    fig_k.legend(handles_k, labels_k, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
                 ncol=2, fontsize=18, frameon=False)
    
    # Adjust layout with professional spacing
    fig_obj.tight_layout(rect=[0, 0, 1, 0.92])
    fig_k.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Create output directory
    output_dir = "/data2/AAG/Audio_Declip/exp/comparison/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save professional figures
    obj_path = os.path.join(output_dir, f"objective_value_comparison_u{u}_idx{idx}.png")
    fig_obj.savefig(obj_path, bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)
    print(f"Saved objective value comparison plot to {obj_path}")
    
    k_path = os.path.join(output_dir, f"k_value_comparison_u{u}_idx{idx}.png")
    fig_k.savefig(k_path, bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)
    print(f"Saved k value comparison plot to {k_path}")
    
    # Create and save summary table
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.round(4)
        
        # Save summary table
        summary_csv_path = os.path.join(output_dir, f"summary_statistics_u{u}_idx{idx}.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Saved summary statistics to {summary_csv_path}")
        
    plt.close('all')  # Close all figures to free memory
    
    return summary_df if summary_data else None



def plot_comparison():
    print("Plotting Experiment: Baseline vs Dynamic Model Comparison")

    # File paths
    base_dir = '/data2/AAG/Audio_Declip/exp/comparison'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    models = ['baseline', 'dynamic']
    datasets = ['Bird', 'Heart', 'Lung', 'Speech']

    # Define paths
    def construct_path(model, dataset):
        return os.path.join(base_dir, dataset.lower() + '_sound', f'evaluation_results_{model}_model_SDR_.xlsx')

    paths = {model: {ds: construct_path(model, ds) for ds in datasets} for model in models}

    # Load and filter data
    def load_filtered_data(filepath):
        df = pd.read_excel(filepath)
        df['duration'] = df['duration'].astype(float)
        df['sdr_orig'] = df['sdr_orig'].astype(float)
        df['sdr_orig_rounded'] = df['sdr_orig'].round(1)
        return df[df['duration'] == 1.0]

    data = {model: {ds: load_filtered_data(paths[model][ds]) for ds in datasets} for model in models}

    # Visualization parameters
    colors = {'baseline': '#1f77b4', 'dynamic': '#ff7f0e'}
    # metrics = {
    #     'delta_sdr': 'ΔSDR (dB)',
    #     'processing_time': 'Processing Time (s)',
    #     'cycles': 'Cycles'
    # }
    metrics = {
        'delta_sdr': 'ΔSDR (dB)',
        'cycles': 'Cycles'
    }
    letters = ['(a)', '(b)', '(c)', '(d)']

    for metric, ylabel in metrics.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        fig.subplots_adjust(top=0.88)

        for ax, ds, letter in zip(axes, datasets, letters):
            # SDR is switched for delta_sdr
            # Always keep the same model colors and labels
            baseline_df = data['baseline'][ds]
            dynamic_df = data['dynamic'][ds]

            # # For delta_sdr, reverse the data (dynamic as ref)
            # if metric == 'delta_sdr':
            #     ref_df, alt_df = dynamic_df, baseline_df
            # else:
            #     ref_df, alt_df = baseline_df, dynamic_df

            ref_df, alt_df = baseline_df, dynamic_df


            unique_sdrs = sorted(ref_df['sdr_orig_rounded'].unique())

            def grouped_stats(df):
                return df.groupby('sdr_orig_rounded')[metric].agg(['mean', 'std']).reindex(unique_sdrs)

            ref_stats = grouped_stats(ref_df)
            alt_stats = grouped_stats(alt_df)
            x = np.arange(len(unique_sdrs))
            width = 0.35

            ax.bar(
                x - width/2, ref_stats['mean'], width,
                yerr=ref_stats['std'], color=colors['baseline'], edgecolor='black',
                linewidth=0.7, alpha=0.8, capsize=4, label='Baseline' if ds == 'Bird' else ""
            )
            ax.bar(
                x + width/2, alt_stats['mean'], width,
                yerr=alt_stats['std'], color=colors['dynamic'], edgecolor='black',
                linewidth=0.7, alpha=0.8, capsize=4, label='Dynamic' if ds == 'Bird' else ""
            )

            ax.set_xticks(x)
            ax.set_xticklabels(unique_sdrs, rotation=45)
            ax.set_title(f"{letter} {ds}", loc='left', fontsize=14, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.4)

        for ax in [axes[0], axes[2]]:
            ax.set_ylabel(ylabel, fontsize=18)
        for ax in [axes[2], axes[3]]:
            ax.set_xlabel("Input SDR (dB)", fontsize=18)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2,
                   fontsize=18)

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        output_path = os.path.join(base_dir,'plots', f'baseline_vs_dynamic_{metric}_1sec.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved comparison plot to {output_path}")
        plt.close(fig)

        # Save statistics as CSV
        summary_data = []
        for ds in datasets:
            for model in models:
                df = data[model][ds]
                summary = df.groupby('sdr_orig_rounded')[metric].agg(['mean', 'std']).reset_index()
                summary['Dataset'] = ds
                summary['Model'] = model.capitalize()
                summary['Duration'] = 1.0
                summary['Metric'] = metric
                summary_data.append(summary)

        comparison_df = pd.concat(summary_data, ignore_index=True)
        csv_path = os.path.join(base_dir,'plots', f'baseline_vs_dynamic_{metric}_1sec.csv')
        comparison_df.to_csv(csv_path, index=False)
        print(f"Saved comparison table to {csv_path}")

        # Summary statistics
        print(f"\n=== {metric.upper()} SUMMARY (1 sec duration) ===")
        for ds in datasets:
            base_avg = data['baseline'][ds][metric].mean()
            dyn_avg = data['dynamic'][ds][metric].mean()
            improvement = ((dyn_avg - base_avg) / base_avg) * 100 if base_avg != 0 else 0
            print(f"{ds}: Baseline={base_avg:.3f}, Dynamic={dyn_avg:.3f}, Improvement={improvement:+.1f}%")


def plot_losses_from_json(json_path, plot_type='all', save_path='.', 
                          train_colors=['tab:blue', 'tab:green', 'tab:orange'],
                          val_colors=['tab:red', 'tab:purple', 'tab:brown'],
                          line_width=2, marker_size=4, 
                          title_size=16, font_size=14, tick_size=12, legend_size=12):
    """
    Plot training/validation loss from a saved JSON file.

    Parameters:
    - json_path: str, path to the loss_history.json
    - plot_type: str, one of ['train', 'val', 'compare', 'all']
    - save_path: str, directory to save plots
    - *_colors, *_size: optional style configurations
    """
    # Load JSON
    with open(json_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['total_loss']) + 1)

    def save_fig(fig, name):
        fig.tight_layout(pad=3.0)
        fig.savefig(os.path.join(save_path, f"{name}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    if plot_type == 'train':
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        titles = ['Total Training Loss', 'DFT Training Loss', 'Sparsity Training Loss']
        keys = ['total_loss', 'dft_loss', 'sparsity_loss']

        for i, ax in enumerate(axes):
            ax.plot(epochs, history[keys[i]], color=train_colors[i], linewidth=line_width, marker='o', markersize=marker_size)
            ax.set_title(titles[i], fontsize=title_size, fontweight='bold')
            ax.set_xlabel("Epoch", fontsize=font_size)
            ax.set_ylabel("Loss", fontsize=font_size)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=tick_size)

        save_fig(fig, "training_losses")

    elif plot_type == 'val':
        if 'val_total_loss' not in history or len(history['val_total_loss']) == 0:
            print("Validation losses not found in JSON.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        titles = ['Total Validation Loss', 'DFT Validation Loss', 'Sparsity Validation Loss']
        keys = ['val_total_loss', 'val_dft_loss', 'val_sparsity_loss']

        for i, ax in enumerate(axes):
            ax.plot(epochs, history[keys[i]], color=val_colors[i], linewidth=line_width, marker='s', markersize=marker_size)
            ax.set_title(titles[i], fontsize=title_size, fontweight='bold')
            ax.set_xlabel("Epoch", fontsize=font_size)
            ax.set_ylabel("Loss", fontsize=font_size)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=tick_size)

        save_fig(fig, "validation_losses")

    elif plot_type == 'compare':
        if 'val_total_loss' not in history or len(history['val_total_loss']) == 0:
            print("Validation losses not found in JSON.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        titles = ['Total Loss Comparison', 'DFT Loss Comparison', 'Sparsity Loss Comparison']
        train_keys = ['total_loss', 'dft_loss', 'sparsity_loss']
        val_keys = ['val_total_loss', 'val_dft_loss', 'val_sparsity_loss']

        for i, ax in enumerate(axes):
            ax.plot(epochs, history[train_keys[i]], label='Train', color=train_colors[i], linewidth=line_width, marker='o', markersize=marker_size)
            ax.plot(epochs, history[val_keys[i]], label='Val', color=val_colors[i], linewidth=line_width, linestyle='--', marker='s', markersize=marker_size)
            ax.set_title(titles[i], fontsize=title_size, fontweight='bold')
            ax.set_xlabel("Epoch", fontsize=font_size)
            ax.set_ylabel("Loss", fontsize=font_size)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=legend_size)
            ax.tick_params(axis='both', labelsize=tick_size)

        save_fig(fig, "comparison_losses")

    elif plot_type == 'all':
        fig = plt.figure(figsize=(12, 8))

        plt.plot(epochs, history['total_loss'], label='Total (Train)', color=train_colors[0], linewidth=line_width, marker='o', markersize=marker_size)
        plt.plot(epochs, history['dft_loss'], label='DFT (Train)', color=train_colors[1], linewidth=line_width, marker='o', markersize=marker_size)
        plt.plot(epochs, history['sparsity_loss'], label='Sparsity (Train)', color=train_colors[2], linewidth=line_width, marker='o', markersize=marker_size)

        if 'val_total_loss' in history and len(history['val_total_loss']) > 0:
            plt.plot(epochs, history['val_total_loss'], label='Total (Val)', color=val_colors[0], linewidth=line_width, linestyle='--', marker='s', markersize=marker_size)
            plt.plot(epochs, history['val_dft_loss'], label='DFT (Val)', color=val_colors[1], linewidth=line_width, linestyle='--', marker='s', markersize=marker_size)
            plt.plot(epochs, history['val_sparsity_loss'], label='Sparsity (Val)', color=val_colors[2], linewidth=line_width, linestyle='--', marker='s', markersize=marker_size)

        plt.title("All Loss Metrics", fontsize=title_size, fontweight='bold')
        plt.xlabel("Epoch", fontsize=font_size)
        plt.ylabel("Loss", fontsize=font_size)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=legend_size)
        plt.tick_params(axis='both', labelsize=tick_size)

        save_fig(fig, "all_losses")

    else:
        print(f"Invalid plot_type: '{plot_type}'. Choose from ['train', 'val', 'compare', 'all'].")




def plot_comparison_all_models(base_dir, batch_size):
    print(f"Plotting Experiment: All Models Comparison for {base_dir}")

    # File paths
    # base_dir = '/data2/AAG/Audio_Declip/exp_ml/heart_sound'
    os.makedirs(base_dir, exist_ok=True)
    output_dir = os.path.join(base_dir, f'all_model_plots_{batch_size}')
    os.makedirs(output_dir, exist_ok=True)

    # Input Excel file paths
    model_files = {
        'Baseline': f'{base_dir}/evaluation_results_baseline_model_SDR_.xlsx',
        'Dynamic': f'{base_dir}/evaluation_results_dynamic_model_SDR_.xlsx',
        'ML1': f'{base_dir}/evaluation_results_ml_model_SDR_without_refinement.xlsx',
        'ML2': f'{base_dir}/evaluation_results_ml_model_SDR_with_refinement.xlsx'
    }

    # Load and filter data
    def load_filtered_data(filepath):
        df = pd.read_excel(filepath)
        df['duration'] = df['duration'].astype(float)
        df['sdr_orig'] = df['sdr_orig'].astype(float)
        df['sdr_orig_rounded'] = df['sdr_orig'].round(1)
        return df[df['duration'] == 1.0]

    data = {model: load_filtered_data(path) for model, path in model_files.items()}

    # Visualization parameters
    colors = {
        'Baseline': '#1f77b4',
        'Dynamic': '#ff7f0e',
        'ML1': '#2ca02c',
        'ML2': '#d62728'
    }

    metrics = {
        'delta_sdr': 'ΔSDR (dB)',
        'cycles': 'Cycles',
        'processing_time': 'Processing Time (s)'
    }

    for metric, ylabel in metrics.items():
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.subplots_adjust(top=0.85)

        # Get union of SDR bins
        all_sdrs = sorted(set().union(*[df['sdr_orig_rounded'].unique() for df in data.values()]))

        x = np.arange(len(all_sdrs))
        width = 0.18

        for i, (model, df) in enumerate(data.items()):
            stats = df.groupby('sdr_orig_rounded')[metric].agg(['mean', 'std']).reindex(all_sdrs)
            ax.bar(
                x + (i - 1.5) * width, stats['mean'], width,
                yerr=stats['std'], capsize=4,
                color=colors[model], edgecolor='black', linewidth=0.7, alpha=0.85,
                label=model
            )

        ax.set_xticks(x)
        ax.set_xticklabels(all_sdrs, rotation=45)
        ax.set_title(f"Heart Dataset: {ylabel} Comparison (1 sec)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Input SDR (dB)", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plot_path = os.path.join(output_dir, f'all_models_{metric}_1sec.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot to {plot_path}")
        plt.close(fig)

        # Save statistics
        summary_data = []
        for model, df in data.items():
            summary = df.groupby('sdr_orig_rounded')[metric].agg(['mean', 'std']).reset_index()
            summary['Model'] = model
            summary['Duration'] = 1.0
            summary['Metric'] = metric
            summary_data.append(summary)

        comparison_df = pd.concat(summary_data, ignore_index=True)
        csv_path = os.path.join(output_dir, f'all_models_{metric}_1sec.csv')
        comparison_df.to_csv(csv_path, index=False)
        print(f"Saved comparison table to {csv_path}")

        # Summary stats print
        print(f"\n=== {metric.upper()} SUMMARY (SDR-wise Across Models) ===")

        # Collect SDR bins
        all_sdrs = sorted(set().union(*[df['sdr_orig_rounded'].unique() for df in data.values()]))

        # Prepare a dictionary: {sdr_value: {model: mean_metric}}
        sdr_summary = {sdr: {} for sdr in all_sdrs}
        for model, df in data.items():
            grouped = df.groupby('sdr_orig_rounded')[metric].mean()
            for sdr_val in all_sdrs:
                sdr_summary[sdr_val][model] = grouped.get(sdr_val, np.nan)

        # Print header
        header = "SDR (dB)".ljust(10) + "".join(model.ljust(12) for model in data.keys())
        print(header)
        print("-" * len(header))

        # Print row-wise
        for sdr_val in all_sdrs:
            row = f"{sdr_val:<10.1f}"
            for model in data.keys():
                val = sdr_summary[sdr_val].get(model, np.nan)
                row += f"{val:<12.3f}" if not np.isnan(val) else f"{'NaN':<12}"
            print(row)




def plot_batch_size_comparison(base_dir, batch_size_1, batch_size_2):
    """
    Compare performance between two batch sizes for all models - Optimized for 1x1 plot
    
    Args:
        base_dir: Base directory containing the experiment results
        batch_size_1: First batch size to compare
        batch_size_2: Second batch size to compare
    """
    print(f"Plotting Batch Size Comparison: {batch_size_1} vs {batch_size_2} for {base_dir}")

    # Create output directory
    os.makedirs(base_dir, exist_ok=True)
    output_dir = os.path.join(base_dir, f'batch_comparison_{batch_size_1}_vs_{batch_size_2}')
    os.makedirs(output_dir, exist_ok=True)

    # Model names and their corresponding file patterns
    models = ['ML1']
    model_file_patterns = {
        'ML1': 'evaluation_results_ml_model_SDR_without_refinement.xlsx'
    }

    def load_batch_data(batch_size):
        """Load data for a specific batch size from CSV files in all_model_plots_{batch_size} directory"""
        batch_data = {}
        batch_dir = os.path.join(base_dir, f'all_model_plots_{batch_size}')  # Match your directory structure
        
        # CSV file patterns (generated from your previous function)
        csv_file_patterns = {
            'ML1': f'all_models_delta_sdr_1sec.csv'
        }
        
        # Load CSV files for each metric
        for metric in ['delta_sdr', 'cycles', 'processing_time']:
            csv_file = os.path.join(batch_dir, f'all_models_{metric}_1sec.csv')
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                # The CSV has columns: sdr_orig_rounded, mean, std, Model, Duration, Metric
                for model in models:
                    model_data = df[df['Model'] == model].copy()
                    if not model_data.empty:
                        if model not in batch_data:
                            batch_data[model] = {}
                        batch_data[model][metric] = model_data
            else:
                print(f"Warning: CSV file not found - {csv_file}")
        
        return batch_data

    # Load data for both batch sizes
    data_batch_1 = load_batch_data(batch_size_1)
    data_batch_2 = load_batch_data(batch_size_2)

    # Color schemes for batch sizes - more distinct colors for better visibility
    colors_batch_1 = {
        'ML1': '#1f77b4'  # Strong blue
    }
    
    colors_batch_2 = {
        'ML1': '#ff7f0e'  # Strong orange for contrast
    }

    # Metrics to compare
    metrics = {
        'delta_sdr': 'ΔSDR (dB)',
        'cycles': 'Cycles',
        'processing_time': 'Processing Time (s)'
    }

    # Create comparison plots for each metric - optimized for single plot
    for metric, ylabel in metrics.items():
        # Larger figure size for single plot with better visibility
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Model (only ML1 in this case)
        model = 'ML1'
        
        # Get data for both batch sizes
        model_data_1 = data_batch_1.get(model, {})
        model_data_2 = data_batch_2.get(model, {})
        
        metric_data_1 = model_data_1.get(metric, pd.DataFrame())
        metric_data_2 = model_data_2.get(metric, pd.DataFrame())
        
        if metric_data_1.empty and metric_data_2.empty:
            ax.text(0.5, 0.5, f'No data available for {model}', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=16)
            ax.set_title(f'Batch Size Comparison: {batch_size_1} vs {batch_size_2} - {ylabel}', 
                        fontsize=20, fontweight='bold', pad=20)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'batch_comparison_{metric}_{batch_size_1}_vs_{batch_size_2}.png')
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            print(f"Saved batch comparison plot to {plot_path}")
            plt.close(fig)
            continue
        
        # Get union of SDR bins from both datasets
        sdr_bins_1 = set(metric_data_1['sdr_orig_rounded'].unique()) if not metric_data_1.empty else set()
        sdr_bins_2 = set(metric_data_2['sdr_orig_rounded'].unique()) if not metric_data_2.empty else set()
        all_sdrs = sorted(sdr_bins_1.union(sdr_bins_2))
        
        if not all_sdrs:
            ax.text(0.5, 0.5, f'No SDR data available for {model}', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=16)
            ax.set_title(f'Batch Size Comparison: {batch_size_1} vs {batch_size_2} - {ylabel}', 
                        fontsize=20, fontweight='bold', pad=20)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'batch_comparison_{metric}_{batch_size_1}_vs_{batch_size_2}.png')
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            print(f"Saved batch comparison plot to {plot_path}")
            plt.close(fig)
            continue
        
        x = np.arange(len(all_sdrs))
        width = 0.35
        
        # Prepare data for plotting
        means_1 = []
        stds_1 = []
        means_2 = []
        stds_2 = []
        
        for sdr_val in all_sdrs:
            # Batch 1 data
            if not metric_data_1.empty:
                row_1 = metric_data_1[metric_data_1['sdr_orig_rounded'] == sdr_val]
                if not row_1.empty:
                    means_1.append(row_1['mean'].iloc[0])
                    stds_1.append(row_1['std'].iloc[0])
                else:
                    means_1.append(np.nan)
                    stds_1.append(np.nan)
            else:
                means_1.append(np.nan)
                stds_1.append(np.nan)
            
            # Batch 2 data
            if not metric_data_2.empty:
                row_2 = metric_data_2[metric_data_2['sdr_orig_rounded'] == sdr_val]
                if not row_2.empty:
                    means_2.append(row_2['mean'].iloc[0])
                    stds_2.append(row_2['std'].iloc[0])
                else:
                    means_2.append(np.nan)
                    stds_2.append(np.nan)
            else:
                means_2.append(np.nan)
                stds_2.append(np.nan)
        
        # Plot bars with error bars
        valid_1 = ~np.isnan(means_1)
        valid_2 = ~np.isnan(means_2)
        
        if np.any(valid_1):
            ax.bar(x[valid_1] - width/2, np.array(means_1)[valid_1], width,
                  yerr=np.array(stds_1)[valid_1], capsize=5,
                  color=colors_batch_1[model], edgecolor='black', linewidth=1.0, alpha=0.8,
                  label=f'Batch Size {batch_size_1}')
        
        if np.any(valid_2):
            ax.bar(x[valid_2] + width/2, np.array(means_2)[valid_2], width,
                  yerr=np.array(stds_2)[valid_2], capsize=5,
                  color=colors_batch_2[model], edgecolor='black', linewidth=1.0, alpha=0.8,
                  label=f'Batch Size {batch_size_2}')
        
        # Enhanced formatting for single plot
        ax.set_xticks(x)
        ax.set_xticklabels([f'{sdr:.1f}' for sdr in all_sdrs], rotation=45, fontsize=14)
        
        # Main title with larger font
        ax.set_title(f'Batch Size Comparison: {batch_size_1} vs {batch_size_2} - {ylabel}', 
                    fontsize=20, fontweight='bold', pad=20)
        
        # Axis labels with larger fonts
        ax.set_xlabel('Input SDR (dB)', fontsize=16, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
        
        # Enhanced grid and legend
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=1)
        legend = ax.legend(fontsize=14, loc='best', frameon=True, 
                          fancybox=True, shadow=True, framealpha=0.9)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')
        
        # Tick parameters for better visibility
        ax.tick_params(axis='both', which='major', labelsize=12, width=1, length=6)
        ax.tick_params(axis='both', which='minor', width=0.5, length=3)
        
        # Adjust layout with more padding
        plt.tight_layout(pad=2.0)
        
        # Save plot with higher quality
        plot_path = os.path.join(output_dir, f'batch_comparison_{metric}_{batch_size_1}_vs_{batch_size_2}.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300, facecolor='white', edgecolor='none')
        print(f"Saved batch comparison plot to {plot_path}")
        plt.close(fig)

    # Create comprehensive comparison CSV
    comparison_data = []
    
    for model in models:
        model_data_1 = data_batch_1.get(model, {})
        model_data_2 = data_batch_2.get(model, {})
        
        for metric in metrics.keys():
            metric_data_1 = model_data_1.get(metric, pd.DataFrame())
            metric_data_2 = model_data_2.get(metric, pd.DataFrame())
            
            # Get all SDR bins for this model and metric
            sdr_bins_1 = set(metric_data_1['sdr_orig_rounded'].unique()) if not metric_data_1.empty else set()
            sdr_bins_2 = set(metric_data_2['sdr_orig_rounded'].unique()) if not metric_data_2.empty else set()
            all_sdrs = sorted(sdr_bins_1.union(sdr_bins_2))
            
            for sdr_val in all_sdrs:
                row_data = {
                    'Model': model,
                    'SDR_Input': sdr_val,
                    'Metric': metric
                }
                
                # Batch 1 statistics
                if not metric_data_1.empty:
                    row_1 = metric_data_1[metric_data_1['sdr_orig_rounded'] == sdr_val]
                    if not row_1.empty:
                        row_data[f'Batch_{batch_size_1}_Mean'] = row_1['mean'].iloc[0]
                        row_data[f'Batch_{batch_size_1}_Std'] = row_1['std'].iloc[0]
                        row_data[f'Batch_{batch_size_1}_Count'] = 1  # From aggregated data
                    else:
                        row_data[f'Batch_{batch_size_1}_Mean'] = np.nan
                        row_data[f'Batch_{batch_size_1}_Std'] = np.nan
                        row_data[f'Batch_{batch_size_1}_Count'] = 0
                else:
                    row_data[f'Batch_{batch_size_1}_Mean'] = np.nan
                    row_data[f'Batch_{batch_size_1}_Std'] = np.nan
                    row_data[f'Batch_{batch_size_1}_Count'] = 0
                
                # Batch 2 statistics
                if not metric_data_2.empty:
                    row_2 = metric_data_2[metric_data_2['sdr_orig_rounded'] == sdr_val]
                    if not row_2.empty:
                        row_data[f'Batch_{batch_size_2}_Mean'] = row_2['mean'].iloc[0]
                        row_data[f'Batch_{batch_size_2}_Std'] = row_2['std'].iloc[0]
                        row_data[f'Batch_{batch_size_2}_Count'] = 1  # From aggregated data
                    else:
                        row_data[f'Batch_{batch_size_2}_Mean'] = np.nan
                        row_data[f'Batch_{batch_size_2}_Std'] = np.nan
                        row_data[f'Batch_{batch_size_2}_Count'] = 0
                else:
                    row_data[f'Batch_{batch_size_2}_Mean'] = np.nan
                    row_data[f'Batch_{batch_size_2}_Std'] = np.nan
                    row_data[f'Batch_{batch_size_2}_Count'] = 0
                
                # Calculate difference and percentage change
                mean1 = row_data[f'Batch_{batch_size_1}_Mean']
                mean2 = row_data[f'Batch_{batch_size_2}_Mean']
                
                if not (np.isnan(mean1) or np.isnan(mean2)):
                    row_data['Difference'] = mean2 - mean1
                    if mean1 != 0:
                        row_data['Percentage_Change'] = ((mean2 - mean1) / abs(mean1)) * 100
                    else:
                        row_data['Percentage_Change'] = np.nan
                else:
                    row_data['Difference'] = np.nan
                    row_data['Percentage_Change'] = np.nan
                
                comparison_data.append(row_data)
    
    # Save comprehensive comparison CSV
    comparison_df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(output_dir, f'batch_comparison_detailed_{batch_size_1}_vs_{batch_size_2}.csv')
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved detailed comparison CSV to {csv_path}")
    
    # Create summary statistics
    print(f"\n=== BATCH SIZE COMPARISON SUMMARY: {batch_size_1} vs {batch_size_2} ===")
    
    for metric in metrics.keys():
        print(f"\n--- {metric.upper()} ---")
        metric_data = comparison_df[comparison_df['Metric'] == metric]
        
        for model in models:
            model_data = metric_data[metric_data['Model'] == model]
            if not model_data.empty:
                avg_diff = model_data['Difference'].mean()
                avg_pct_change = model_data['Percentage_Change'].mean()
                print(f"{model:>10}: Avg Diff = {avg_diff:>8.3f}, Avg % Change = {avg_pct_change:>8.1f}%")
    
    # Create a summary table for easier interpretation
    summary_data = []
    for metric in metrics.keys():
        for model in models:
            model_metric_data = comparison_df[
                (comparison_df['Model'] == model) & 
                (comparison_df['Metric'] == metric)
            ]
            
            if not model_metric_data.empty:
                summary_row = {
                    'Metric': metric,
                    'Model': model,
                    'Avg_Difference': model_metric_data['Difference'].mean(),
                    'Avg_Percentage_Change': model_metric_data['Percentage_Change'].mean(),
                    'Max_Difference': model_metric_data['Difference'].max(),
                    'Min_Difference': model_metric_data['Difference'].min(),
                    'Std_Difference': model_metric_data['Difference'].std()
                }
                summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(output_dir, f'batch_comparison_summary_{batch_size_1}_vs_{batch_size_2}.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary comparison CSV to {summary_csv_path}")
    
    return output_dir


def plot_comparison_all_models_speech(base_dir, batch_size, factor):
    print(f"Plotting Experiment: All Models Comparison for {base_dir}")

    # File paths
    # base_dir = '/data2/AAG/Audio_Declip/exp_ml/heart_sound'
    os.makedirs(base_dir, exist_ok=True)
    output_dir = os.path.join(base_dir, f'all_model_plots_{batch_size}_{factor}')
    os.makedirs(output_dir, exist_ok=True)

    # Input Excel file paths
    model_files = {
        'Baseline': f'{base_dir}/evaluation_results_baseline_model_SDR_.xlsx',
        'Dynamic': f'{base_dir}/evaluation_results_dynamic_model_SDR_.xlsx',
        'ML1': f'{base_dir}/evaluation_results_ml_model_SDR_without_refinement.xlsx',
        'ML2': f'{base_dir}/evaluation_results_ml_model_SDR_with_refinement.xlsx'
    }

    # Load and filter data
    def load_filtered_data(filepath):
        df = pd.read_excel(filepath)
        df['duration'] = df['duration'].astype(float)
        df['sdr_orig'] = df['sdr_orig'].astype(float)
        df['sdr_orig_rounded'] = df['sdr_orig'].round(1)
        return df[df['duration'] == 1.0]

    data = {model: load_filtered_data(path) for model, path in model_files.items()}

    # Visualization parameters
    colors = {
        'Baseline': '#1f77b4',
        'Dynamic': '#ff7f0e',
        'ML1': '#2ca02c',
        'ML2': '#d62728'
    }

    metrics = {
        'delta_sdr': 'ΔSDR (dB)',
        'cycles': 'Cycles',
        'processing_time': 'Processing Time (s)',
        'delta_pesq': 'ΔPESQ'
    }

    for metric, ylabel in metrics.items():
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.subplots_adjust(top=0.85)

        # Get union of SDR bins
        all_sdrs = sorted(set().union(*[df['sdr_orig_rounded'].unique() for df in data.values()]))

        x = np.arange(len(all_sdrs))
        width = 0.18

        for i, (model, df) in enumerate(data.items()):
            stats = df.groupby('sdr_orig_rounded')[metric].agg(['mean', 'std']).reindex(all_sdrs)
            ax.bar(
                x + (i - 1.5) * width, stats['mean'], width,
                yerr=stats['std'], capsize=4,
                color=colors[model], edgecolor='black', linewidth=0.7, alpha=0.85,
                label=model
            )

        ax.set_xticks(x)
        ax.set_xticklabels(all_sdrs, rotation=45)
        ax.set_title(f"Heart Dataset: {ylabel} Comparison (1 sec)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Input SDR (dB)", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plot_path = os.path.join(output_dir, f'all_models_{metric}_1sec.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot to {plot_path}")
        plt.close(fig)

        # Save statistics
        summary_data = []
        for model, df in data.items():
            summary = df.groupby('sdr_orig_rounded')[metric].agg(['mean', 'std']).reset_index()
            summary['Model'] = model
            summary['Duration'] = 1.0
            summary['Metric'] = metric
            summary_data.append(summary)

        comparison_df = pd.concat(summary_data, ignore_index=True)
        csv_path = os.path.join(output_dir, f'all_models_{metric}_1sec.csv')
        comparison_df.to_csv(csv_path, index=False)
        print(f"Saved comparison table to {csv_path}")

        # Summary stats print
        print(f"\n=== {metric.upper()} SUMMARY (SDR-wise Across Models) ===")

        # Collect SDR bins
        all_sdrs = sorted(set().union(*[df['sdr_orig_rounded'].unique() for df in data.values()]))

        # Prepare a dictionary: {sdr_value: {model: mean_metric}}
        sdr_summary = {sdr: {} for sdr in all_sdrs}
        for model, df in data.items():
            grouped = df.groupby('sdr_orig_rounded')[metric].mean()
            for sdr_val in all_sdrs:
                sdr_summary[sdr_val][model] = grouped.get(sdr_val, np.nan)

        # Print header
        header = "SDR (dB)".ljust(10) + "".join(model.ljust(12) for model in data.keys())
        print(header)
        print("-" * len(header))

        # Print row-wise
        for sdr_val in all_sdrs:
            row = f"{sdr_val:<10.1f}"
            for model in data.keys():
                val = sdr_summary[sdr_val].get(model, np.nan)
                row += f"{val:<12.3f}" if not np.isnan(val) else f"{'NaN':<12}"
            print(row)