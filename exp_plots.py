import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import scipy.io as sio
import json
import glob

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
            ax_obj.set_title(f"{sound_label} Sound", fontsize=15, pad=12)
            
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
            ax_k.set_title(f"{sound_label} Sound", fontsize=15, pad=12)
            
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
        # ax.set_title(f"Heart Dataset: {ylabel} Comparison (1 sec)", fontsize=14, fontweight='bold')
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


def plot_comparison_all_models_speech(base_dir, batch_size, factor, duration):
    print(f"Plotting Experiment: All Models Comparison for {base_dir}")

    # File paths
    # base_dir = '/data2/AAG/Audio_Declip/exp_ml/heart_sound'
    os.makedirs(base_dir, exist_ok=True)
    output_dir = os.path.join(base_dir, f'all_model_plots_{batch_size}_{factor}_{duration}')
    os.makedirs(output_dir, exist_ok=True)

    # Input Excel file paths
    model_files = {
        'Baseline': f'{base_dir}/evaluation_results_baseline_model_SDR_.xlsx',
        'Dynamic': f'{base_dir}/evaluation_results_dynamic_model_SDR_.xlsx',
        'U-Net(FAA)': f'{base_dir}/evaluation_results_ml_model_SDR_without_refinement.xlsx',
        'NeuraDyne': f'{base_dir}/evaluation_results_ml_model_SDR_with_refinement.xlsx'
    }

    # Load and filter data
    def load_filtered_data(filepath):
        df = pd.read_excel(filepath)
        df['duration'] = df['duration'].astype(float)
        df['sdr_orig'] = df['sdr_orig'].astype(float)
        df['sdr_orig_rounded'] = df['sdr_orig'].round(1)
        return df[df['duration'] == duration]

    data = {model: load_filtered_data(path) for model, path in model_files.items()}

    # Visualization parameters
    colors = {
        'Baseline': '#1f77b4',
        'Dynamic': '#ff7f0e',
        'U-Net(FAA)': '#2ca02c',
        'NeuraDyne': '#d62728'
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
        # ax.set_title(f"Speech Dataset: {ylabel} Comparison (1 sec)", fontsize=14, fontweight='bold')
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
            summary['Duration'] = duration
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



def plot_comparison_all_models_cross(base_dir, batch_size):
    print(f"Plotting Experiment: All Models Comparison for {base_dir}")

    os.makedirs(base_dir, exist_ok=True)
    output_dir = os.path.join(base_dir,'cross_model', f'all_model_plots_{batch_size}')
    os.makedirs(output_dir, exist_ok=True)

    # model_files = {
    #     'Baseline': os.path.join(base_dir, 'evaluation_results_baseline_model_SDR_.xlsx'),
    #     'Dynamic': os.path.join(base_dir, 'evaluation_results_dynamic_model_SDR_.xlsx'),
    #     'ML1': os.path.join(base_dir, 'evaluation_results_ml_model_SDR_without_refinement.xlsx'),
    #     'ML2': os.path.join(base_dir, 'evaluation_results_ml_model_SDR_with_refinement.xlsx'),
    #     'ML1_Cross': os.path.join(base_dir,'cross_model', 'evaluation_results_ml_model_SDR_without_refinement.xlsx'),
    #     'ML2_Cross': os.path.join(base_dir, 'cross_model', 'evaluation_results_ml_model_SDR_with_refinement.xlsx')
    # }

    model_files = {
        'Baseline': os.path.join(base_dir, 'evaluation_results_baseline_model_SDR_.xlsx'),
        'Dynamic': os.path.join(base_dir, 'evaluation_results_dynamic_model_SDR_.xlsx'),
        'ML2': os.path.join(base_dir, 'evaluation_results_ml_model_SDR_with_refinement.xlsx'),
        'ML2_Cross': os.path.join(base_dir, 'cross_model', 'evaluation_results_ml_model_SDR_with_refinement.xlsx')
    }

    # Load and filter data
    def load_filtered_data(filepath):
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return None
        df = pd.read_excel(filepath)
        if 'duration' not in df or 'sdr_orig' not in df:
            print(f"Required columns missing in {filepath}")
            return None
        df['duration'] = df['duration'].astype(float)
        df['sdr_orig'] = df['sdr_orig'].astype(float)
        df['sdr_orig_rounded'] = df['sdr_orig'].round(1)
        return df[df['duration'] == 1.0]

    data = {}
    for model, path in model_files.items():
        df = load_filtered_data(path)
        if df is not None:
            data[model] = df

    if not data:
        print("No valid data loaded.")
        return

    # Visualization parameters
    colors = {
        'Baseline': '#1f77b4',
        'Dynamic': '#ff7f0e',
        'ML2': '#d62728',
        'ML2_Cross': '#8c564b'
    }

    # colors = {
    #     'Baseline': '#1f77b4',
    #     'Dynamic': '#ff7f0e',
    #     'ML1': '#2ca02c',
    #     'ML2': '#d62728',
    #     'ML1_Cross': '#9467bd',
    #     'ML2_Cross': '#8c564b'
    # }

    metrics = {
        'delta_sdr': 'ΔSDR (dB)',
        'cycles': 'Cycles',
        'processing_time': 'Processing Time (s)'
    }

    for metric, ylabel in metrics.items():
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.subplots_adjust(top=0.85)

        all_sdrs = sorted(set().union(*[df['sdr_orig_rounded'].unique() for df in data.values()]))
        x = np.arange(len(all_sdrs))
        n_models = len(data)
        width = 0.8 / n_models  # dynamically adjust bar width

        for i, (model, df) in enumerate(data.items()):
            if metric not in df.columns:
                print(f"Metric '{metric}' missing in model '{model}'. Skipping.")
                continue
            stats = df.groupby('sdr_orig_rounded')[metric].agg(['mean', 'std']).reindex(all_sdrs)
            offset = (i - n_models / 2) * width + width / 2
            ax.bar(
                x + offset, stats['mean'], width,
                yerr=stats['std'], capsize=4,
                color=colors.get(model, '#999999'), edgecolor='black', linewidth=0.7, alpha=0.85,
                label=model
            )

        ax.set_xticks(x)
        ax.set_xticklabels(all_sdrs, rotation=45)
        # ax.set_title(f"Heart Dataset: {ylabel} Comparison (1 sec)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Input SDR (dB)", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plot_path = os.path.join(output_dir, f'all_models_{metric}_1sec.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot to {plot_path}")
        plt.close(fig)

        # Save summary CSV
        summary_data = []
        for model, df in data.items():
            if metric not in df.columns:
                continue
            summary = df.groupby('sdr_orig_rounded')[metric].agg(['mean', 'std']).reset_index()
            summary['Model'] = model
            summary['Duration'] = 1.0
            summary['Metric'] = metric
            summary_data.append(summary)

        comparison_df = pd.concat(summary_data, ignore_index=True)
        csv_path = os.path.join(output_dir, f'all_models_{metric}_1sec.csv')
        comparison_df.to_csv(csv_path, index=False)
        print(f"Saved comparison table to {csv_path}")

        # Print summary table
        print(f"\n=== {metric.upper()} SUMMARY (SDR-wise Across Models) ===")
        sdr_summary = {sdr: {} for sdr in all_sdrs}
        for model, df in data.items():
            if metric not in df.columns:
                continue
            grouped = df.groupby('sdr_orig_rounded')[metric].mean()
            for sdr_val in all_sdrs:
                sdr_summary[sdr_val][model] = grouped.get(sdr_val, np.nan)

        header = "SDR (dB)".ljust(10) + "".join(model.ljust(12) for model in data.keys())
        print(header)
        print("-" * len(header))
        for sdr_val in all_sdrs:
            row = f"{sdr_val:<10.1f}"
            for model in data.keys():
                val = sdr_summary[sdr_val].get(model, np.nan)
                row += f"{val:<12.3f}" if not np.isnan(val) else f"{'NaN':<12}"
            print(row)






######################################## Fold Plottings ########################################




def load_and_process_data(MODELS, FOLDS, BASE_DIR, DURATION, METRICS):
    """Load and process data from all folds and models."""
    results = {metric: {model: [] for model in MODELS} for metric in METRICS}
    
    for fold in FOLDS:
        fold_dir = os.path.join(BASE_DIR, fold, "all_model_plots_16_0.3")
        
        for metric in METRICS:
            file_path = os.path.join(fold_dir, f"all_models_{metric}_1sec.csv")
            try:
                if not os.path.exists(file_path):
                    print(f"Warning: File not found: {file_path}")
                    continue
                    
                df = pd.read_csv(file_path)
                
                # Check if Duration column exists and filter
                if 'Duration' in df.columns:
                    df = df[df['Duration'] == float(DURATION)]
                else:
                    print(f"Warning: Duration column not found in {file_path}")
                
                df['fold'] = fold
                
                for model in MODELS:
                    if 'Model' not in df.columns:
                        print(f"Warning: Model column not found in {file_path}")
                        continue
                        
                    model_df = df[df['Model'] == model].copy()
                    
                    if model_df.empty:
                        print(f"Warning: No data found for model {model} in {file_path}")
                        continue
                    
                    # Rename column if exists
                    if 'sdr_orig_rounded' in model_df.columns:
                        model_df.rename(columns={'sdr_orig_rounded': 'sdr_orig'}, inplace=True)
                    
                    # Check if required columns exist
                    required_cols = ['sdr_orig', 'mean']
                    missing_cols = [col for col in required_cols if col not in model_df.columns]
                    if missing_cols:
                        print(f"Warning: Missing columns {missing_cols} in {file_path} for model {model}")
                        continue
                    
                    results[metric][model].append(model_df)
                    
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue

    # Combine folds for each metric/model
    combined = {}
    for metric in METRICS:
        combined[metric] = {}
        for model in MODELS:
            if results[metric][model]:
                combined[metric][model] = pd.concat(results[metric][model], ignore_index=True)
            else:
                print(f"Warning: No data available for metric {metric}, model {model}")
    
    return combined

def compute_stats(df):
    """Compute mean and std for each sdr_orig level."""
    if df.empty or 'sdr_orig' not in df.columns or 'mean' not in df.columns:
        return pd.DataFrame(columns=['sdr_orig', 'mean', 'std'])
    
    return df.groupby('sdr_orig')['mean'].agg(['mean', 'std']).reset_index()

def compute_relative_improvement(combined, METRICS, MODELS):
    """Compute relative improvement over baseline for each metric."""
    relative_improvement = {metric: {} for metric in METRICS if metric != 'cycles'}
    
    for metric in relative_improvement.keys():
        if metric not in combined or 'Baseline' not in combined[metric]:
            print(f"Warning: Baseline data not available for metric {metric}")
            continue
            
        baseline_stats = compute_stats(combined[metric]['Baseline'])
        if baseline_stats.empty:
            print(f"Warning: Empty baseline stats for metric {metric}")
            continue
            
        baseline_stats = baseline_stats.set_index('sdr_orig')
        
        for model in MODELS:
            if model == 'Baseline' or model not in combined[metric]:
                continue
            
            model_stats = compute_stats(combined[metric][model])
            if model_stats.empty:
                print(f"Warning: Empty model stats for metric {metric}, model {model}")
                continue
                
            model_stats = model_stats.set_index('sdr_orig')
            
            # Align indices
            joined = model_stats.join(baseline_stats, lsuffix='_model', rsuffix='_baseline', how='inner')
            
            if joined.empty:
                print(f"Warning: No matching SDR levels between {model} and Baseline for metric {metric}")
                continue
            
            # Avoid division by zero
            valid_baseline = joined['mean_baseline'] != 0
            if not valid_baseline.any():
                print(f"Warning: All baseline values are zero for metric {metric}")
                continue
            
            # Percentage improvement = (model - baseline)/baseline * 100
            rel_impr = pd.Series(index=joined.index, dtype=float)
            rel_impr[valid_baseline] = ((joined.loc[valid_baseline, 'mean_model'] - 
                                       joined.loc[valid_baseline, 'mean_baseline']) / 
                                      joined.loc[valid_baseline, 'mean_baseline']) * 100
            
            relative_improvement[metric][model] = rel_impr.reset_index()
    
    return relative_improvement

def compute_speedup_ratio(combined, MODELS):
    """Compute speedup ratio (baseline_cycles / model_cycles)."""
    speedup = {}
    if 'cycles' not in combined or 'Baseline' not in combined['cycles']:
        print("Warning: Cycles data not available for speedup computation")
        return speedup
    
    baseline_stats = compute_stats(combined['cycles']['Baseline'])
    if baseline_stats.empty:
        print("Warning: Empty baseline stats for cycles")
        return speedup
        
    baseline_stats = baseline_stats.set_index('sdr_orig')
    
    for model in MODELS:
        if model == 'Baseline' or model not in combined['cycles']:
            continue
            
        model_stats = compute_stats(combined['cycles'][model])
        if model_stats.empty:
            continue
            
        model_stats = model_stats.set_index('sdr_orig')
        joined = model_stats.join(baseline_stats, lsuffix='_model', rsuffix='_baseline', how='inner')
        
        if joined.empty:
            continue
        
        # Avoid division by zero
        valid_model = joined['mean_model'] != 0
        if not valid_model.any():
            continue
        
        speedup_ratio = pd.Series(index=joined.index, dtype=float)
        speedup_ratio[valid_model] = (joined.loc[valid_model, 'mean_baseline'] / 
                                     joined.loc[valid_model, 'mean_model'])
        
        speedup[model] = speedup_ratio.reset_index()
    
    return speedup

def compute_composite_score(combined, MODELS, W_SDR, W_PESQ, W_CYCLES):
    """
    Composite = W_SDR * normalized_delta_sdr + W_PESQ * normalized_delta_pesq + W_CYCLES * normalized_cycles_inv
    Normalization is min-max over all models and sdr_orig levels for each metric.
    """
    # Check if all required metrics are available
    required_metrics = ['delta_sdr', 'delta_pesq', 'cycles']
    missing_metrics = [m for m in required_metrics if m not in combined or not combined[m]]
    if missing_metrics:
        print(f"Warning: Missing metrics for composite score: {missing_metrics}")
        return pd.DataFrame()
    
    # Get common sdr_orig levels across all metrics and models
    all_sdr_levels = set()
    for metric in required_metrics:
        for model in MODELS:
            if model in combined[metric] and not combined[metric][model].empty:
                all_sdr_levels.update(combined[metric][model]['sdr_orig'].unique())
    
    if not all_sdr_levels:
        print("Warning: No common SDR levels found across metrics")
        return pd.DataFrame()
    
    sdr_levels = sorted(all_sdr_levels)
    
    # Compute mean per model per metric
    metric_means = {}
    for metric in required_metrics:
        metric_means[metric] = {}
        for model in MODELS:
            if model in combined[metric] and not combined[metric][model].empty:
                stats = compute_stats(combined[metric][model])
                if not stats.empty:
                    metric_means[metric][model] = stats.set_index('sdr_orig')['mean']
                else:
                    metric_means[metric][model] = pd.Series(index=sdr_levels, data=np.nan)
            else:
                metric_means[metric][model] = pd.Series(index=sdr_levels, data=np.nan)
    
    # Concatenate all values to find global min/max per metric
    all_sdr_index = pd.Index(sdr_levels)
    
    try:
        all_deltasdr = pd.concat([metric_means['delta_sdr'][m].reindex(all_sdr_index) for m in MODELS])
        all_deltapesq = pd.concat([metric_means['delta_pesq'][m].reindex(all_sdr_index) for m in MODELS])
        all_cycles = pd.concat([metric_means['cycles'][m].reindex(all_sdr_index) for m in MODELS])
        
        # Remove NaN values for min/max computation
        all_deltasdr = all_deltasdr.dropna()
        all_deltapesq = all_deltapesq.dropna()
        all_cycles = all_cycles.dropna()
        
        if all_deltasdr.empty or all_deltapesq.empty or all_cycles.empty:
            print("Warning: Insufficient data for composite score computation")
            return pd.DataFrame()
        
        min_sdr, max_sdr = all_deltasdr.min(), all_deltasdr.max()
        min_pesq, max_pesq = all_deltapesq.min(), all_deltapesq.max()
        min_cycles, max_cycles = all_cycles.min(), all_cycles.max()
        
        # Avoid division by zero in normalization
        if max_sdr == min_sdr or max_pesq == min_pesq or max_cycles == min_cycles:
            print("Warning: Cannot normalize - all values are identical for one or more metrics")
            return pd.DataFrame()
        
        composite_scores = {}
        
        for model in MODELS:
            sdr_vals = metric_means['delta_sdr'][model].reindex(all_sdr_index)
            pesq_vals = metric_means['delta_pesq'][model].reindex(all_sdr_index)
            cycles_vals = metric_means['cycles'][model].reindex(all_sdr_index)
            
            # Normalize (handle NaN values)
            sdr_norm = (sdr_vals - min_sdr) / (max_sdr - min_sdr)
            pesq_norm = (pesq_vals - min_pesq) / (max_pesq - min_pesq)
            cycles_norm = (cycles_vals - min_cycles) / (max_cycles - min_cycles)
            
            # Invert cycles_norm so lower cycles = higher score
            cycles_norm_inv = 1 - cycles_norm
            
            comp = (W_SDR * sdr_norm) + (W_PESQ * pesq_norm) + (W_CYCLES * cycles_norm_inv)
            composite_scores[model] = comp
        
        composite_df = pd.DataFrame({'sdr_orig': sdr_levels})
        for model in MODELS:
            composite_df[model] = composite_scores[model].values
        
        return composite_df
        
    except Exception as e:
        print(f"Error computing composite score: {str(e)}")
        return pd.DataFrame()

def plot_metrics_per_fold(data, fold, MODELS, METRICS, BASE_DIR, BAR_WIDTH, FIG_SIZE):
    """Plot delta_sdr, delta_pesq, cycles for one fold with bar plots for each model."""
    plot_dir = os.path.join(BASE_DIR, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    for metric in METRICS:
        plt.figure(figsize=FIG_SIZE)
        
        # Collect data for plotting
        plot_data = []
        for i, model in enumerate(MODELS):
            if model not in data[metric]:
                continue
            
            df = data[metric][model]
            df_fold = df[df['fold'] == fold]
            if df_fold.empty:
                continue
                
            stats = df_fold.groupby('sdr_orig')['mean'].agg(['mean', 'std']).reset_index()
            if stats.empty:
                continue
                
            plot_data.append((i, model, stats))
        
        if not plot_data:
            print(f"Warning: No data available for plotting {metric} in {fold}")
            plt.close()
            continue
        
        # Plot bars
        x = np.arange(len(plot_data[0][2]))  # Use first model's data for x-axis
        for i, model, stats in plot_data:
            plt.bar(x + i * BAR_WIDTH, stats['mean'], BAR_WIDTH, 
                   label=model, yerr=stats['std'], capsize=3)
        
        plt.xticks(x + (len(plot_data) - 1) * BAR_WIDTH / 2, plot_data[0][2]['sdr_orig'])
        plt.xlabel('Original SDR (dB)')
        plt.ylabel(metric.upper())
        plt.title(f'{metric.upper()} Comparison - {fold}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(plot_dir, f"{fold}_{metric}_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()

def plot_grouped_metric(data, metric, ylabel, filename, MODELS, BASE_DIR, BAR_WIDTH, FIG_SIZE):
    """Plot grouped metric across all models."""
    if metric not in data or not data[metric]:
        print(f"Warning: No data available for plotting {metric}")
        return
    
    # Find common SDR levels
    all_sdr_levels = set()
    valid_models = []
    
    for model in MODELS:
        if model in data[metric] and not data[metric][model].empty:
            all_sdr_levels.update(data[metric][model]['sdr_orig'].unique())
            valid_models.append(model)
    
    if not valid_models:
        print(f"Warning: No valid models found for metric {metric}")
        return
    
    sdr_levels = sorted(all_sdr_levels)
    x = np.arange(len(sdr_levels))
    
    plt.figure(figsize=FIG_SIZE)
    
    plotted_models = []
    for i, model in enumerate(valid_models):
        stats = data[metric][model].groupby('sdr_orig')['mean'].agg(['mean', 'std']).reset_index()
        if stats.empty:
            continue
        
        # Align with common SDR levels
        stats_aligned = stats.set_index('sdr_orig').reindex(sdr_levels).reset_index()
        
        # Only plot if we have data
        valid_data = ~stats_aligned['mean'].isna()
        if not valid_data.any():
            continue
        
        plt.bar(x + i * BAR_WIDTH, stats_aligned['mean'], BAR_WIDTH, 
               label=model, yerr=stats_aligned['std'], capsize=3)
        plotted_models.append(model)
    
    if not plotted_models:
        print(f"Warning: No data could be plotted for metric {metric}")
        plt.close()
        return
    
    plt.xticks(x + (len(plotted_models) - 1) * BAR_WIDTH / 2, sdr_levels)
    plt.xlabel('Original SDR (dB)')
    plt.ylabel(ylabel)
    plt.title(f'{metric.upper()} Comparison Across Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_dir = os.path.join(BASE_DIR, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    full_path = os.path.join(plot_dir, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {full_path}")

def save_summary_tables(combined, rel_impr, speedup, composite_df, output_dir, BASE_DIR, FOLDS, MODELS, METRICS, DURATION, W_SDR, W_PESQ, W_CYCLES):
    """Save comprehensive summary tables as CSV files."""
    print(f"\nSaving statistical results to: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save mean and std per metric & model
    print("Saving mean and std statistics...")
    for metric in METRICS:
        all_model_stats = []
        
        for model in MODELS:
            if metric not in combined or model not in combined[metric]:
                continue
                
            stats = compute_stats(combined[metric][model])
            if stats.empty:
                continue
            
            # Add model column for identification
            stats['model'] = model
            stats['metric'] = metric
            all_model_stats.append(stats)
        
        if all_model_stats:
            # Combine all models for this metric
            metric_df = pd.concat(all_model_stats, ignore_index=True)
            # Reorder columns for better readability
            metric_df = metric_df[['metric', 'model', 'sdr_orig', 'mean', 'std']]
            
            filename = f"stats_{metric}_mean_std.csv"
            filepath = os.path.join(output_dir, filename)
            metric_df.to_csv(filepath, index=False, float_format='%.6f')
            print(f"  Saved: {filename}")
    
    # 2. Save all metrics combined in one file
    print("Saving combined statistics...")
    all_combined_stats = []
    for metric in METRICS:
        for model in MODELS:
            if metric not in combined or model not in combined[metric]:
                continue
                
            stats = compute_stats(combined[metric][model])
            if stats.empty:
                continue
            
            stats['model'] = model
            stats['metric'] = metric
            all_combined_stats.append(stats)
    
    if all_combined_stats:
        combined_df = pd.concat(all_combined_stats, ignore_index=True)
        combined_df = combined_df[['metric', 'model', 'sdr_orig', 'mean', 'std']]
        
        filepath = os.path.join(output_dir, "stats_all_metrics_combined.csv")
        combined_df.to_csv(filepath, index=False, float_format='%.6f')
        print(f"  Saved: stats_all_metrics_combined.csv")
    
    # 3. Save relative improvement data
    if rel_impr:
        print("Saving relative improvement statistics...")
        all_rel_impr = []
        
        for metric, models_dict in rel_impr.items():
            for model, df in models_dict.items():
                if df.empty:
                    continue
                
                df_copy = df.copy()
                df_copy['model'] = model
                df_copy['metric'] = metric
                
                # The df from compute_relative_improvement should have columns: ['sdr_orig', improvement_values]
                # We need to rename the second column appropriately
                cols = list(df_copy.columns)
                if len(cols) >= 2:
                    # Find the column that's not 'sdr_orig', 'model', or 'metric'
                    value_col = None
                    for col in cols:
                        if col not in ['sdr_orig', 'model', 'metric']:
                            value_col = col
                            break
                    
                    if value_col is not None:
                        df_copy = df_copy.rename(columns={value_col: 'relative_improvement_pct'})
                
                all_rel_impr.append(df_copy)
        
        if all_rel_impr:
            rel_impr_df = pd.concat(all_rel_impr, ignore_index=True)
            
            # Ensure we have the expected columns
            expected_cols = ['metric', 'model', 'sdr_orig', 'relative_improvement_pct']
            available_cols = [col for col in expected_cols if col in rel_impr_df.columns]
            
            if len(available_cols) >= 3:  # At minimum we need metric, model, sdr_orig
                rel_impr_df = rel_impr_df[available_cols]
                
                filepath = os.path.join(output_dir, "stats_relative_improvement.csv")
                rel_impr_df.to_csv(filepath, index=False, float_format='%.4f')
                print(f"  Saved: stats_relative_improvement.csv")
            else:
                print("  Warning: Could not save relative improvement - missing required columns")
    
    # 4. Save speedup data
    if speedup:
        print("Saving speedup statistics...")
        all_speedup = []
        
        for model, df in speedup.items():
            if df.empty:
                continue
            
            df_copy = df.copy()
            df_copy['model'] = model
            df_copy['metric'] = 'cycles_speedup'
            
            # The df from compute_speedup_ratio should have columns: ['sdr_orig', speedup_values]
            # We need to rename the second column appropriately
            cols = list(df_copy.columns)
            if len(cols) >= 2:
                # Find the column that's not 'sdr_orig', 'model', or 'metric'
                value_col = None
                for col in cols:
                    if col not in ['sdr_orig', 'model', 'metric']:
                        value_col = col
                        break
                
                if value_col is not None:
                    df_copy = df_copy.rename(columns={value_col: 'speedup_ratio'})
            
            all_speedup.append(df_copy)
        
        if all_speedup:
            speedup_df = pd.concat(all_speedup, ignore_index=True)
            
            # Ensure we have the expected columns
            expected_cols = ['metric', 'model', 'sdr_orig', 'speedup_ratio']
            available_cols = [col for col in expected_cols if col in speedup_df.columns]
            
            if len(available_cols) >= 3:  # At minimum we need metric, model, sdr_orig
                speedup_df = speedup_df[available_cols]
                
                filepath = os.path.join(output_dir, "stats_speedup_ratios.csv")
                speedup_df.to_csv(filepath, index=False, float_format='%.4f')
                print(f"  Saved: stats_speedup_ratios.csv")
            else:
                print("  Warning: Could not save speedup ratios - missing required columns")
    
    # 5. Save composite scores
    if not composite_df.empty:
        print("Saving composite scores...")
        
        # Add metadata about weights
        weights_info = pd.DataFrame({
            'metric': ['delta_sdr', 'delta_pesq', 'cycles'],
            'weight': [W_SDR, W_PESQ, W_CYCLES],
            'description': [
                'Higher is better - normalized 0-1',
                'Higher is better - normalized 0-1', 
                'Lower is better - inverted and normalized 0-1'
            ]
        })
        
        # Save weights info
        weights_filepath = os.path.join(output_dir, "composite_score_weights.csv")
        weights_info.to_csv(weights_filepath, index=False)
        print(f"  Saved: composite_score_weights.csv")
        
        # Save composite scores
        composite_filepath = os.path.join(output_dir, "stats_composite_scores.csv")
        composite_df.to_csv(composite_filepath, index=False, float_format='%.6f')
        print(f"  Saved: stats_composite_scores.csv")
    
    # 6. Create a summary report
    print("Creating summary report...")
    summary_lines = []
    summary_lines.append("AUDIO DECLIPPING ANALYSIS SUMMARY")
    summary_lines.append("=" * 50)
    summary_lines.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Base Directory: {BASE_DIR}")
    summary_lines.append(f"Processed Folds: {', '.join(FOLDS)}")
    summary_lines.append(f"Models Analyzed: {', '.join(MODELS)}")
    summary_lines.append(f"Metrics Evaluated: {', '.join(METRICS)}")
    summary_lines.append(f"Duration Filter: {DURATION}s")
    summary_lines.append("")
    
    # Data loading summary
    summary_lines.append("DATA LOADING SUMMARY:")
    summary_lines.append("-" * 30)
    for metric in METRICS:
        summary_lines.append(f"\n{metric.upper()}:")
        for model in MODELS:
            if metric in combined and model in combined[metric]:
                count = len(combined[metric][model])
                summary_lines.append(f"  {model}: {count} records")
            else:
                summary_lines.append(f"  {model}: 0 records")
    
    summary_lines.append("\nFILES GENERATED:")
    summary_lines.append("-" * 20)
    
    # List generated files
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    for csv_file in sorted(csv_files):
        summary_lines.append(f"  {csv_file}")
    
    # Composite score info
    if not composite_df.empty:
        summary_lines.append(f"\nCOMPOSITE SCORE WEIGHTS:")
        summary_lines.append(f"  SDR Weight: {W_SDR}")
        summary_lines.append(f"  PESQ Weight: {W_PESQ}")
        summary_lines.append(f"  Cycles Weight: {W_CYCLES}")
    
    # Save summary
    summary_filepath = os.path.join(output_dir, "analysis_summary.txt")
    with open(summary_filepath, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"  Saved: analysis_summary.txt")
    print(f"\nAll statistical results saved to: {output_dir}")
    print(f"Total CSV files generated: {len([f for f in os.listdir(output_dir) if f.endswith('.csv')])}")



def plot_comparison_all_folds_speech(BASE_DIR):
    # Constants
    # BASE_DIR = "/data2/AAG/Audio_Declip/exp_ml/speech_sound/kfold_data"
    FOLDS = [f"fold_{i}" for i in range(1, 6)]  # fold_1 to fold_4
    MODELS = ["Baseline", "Dynamic", "ML2"]
    METRICS = ["delta_sdr", "delta_pesq", "cycles", "processing_time"]
    DURATION = "4.0"
    BAR_WIDTH = 0.15
    FIG_SIZE = (12, 6)

    # Composite weights (adjust as needed)
    W_SDR = 0.3
    W_PESQ = 0.5
    W_CYCLES = 0.2  # Since cycles is "better" when smaller, we'll subtract normalized cycles
    """Main execution function."""
    print("Starting audio declipping analysis...")
    print(f"Base directory: {BASE_DIR}")
    print(f"Processing folds: {FOLDS}")
    print(f"Models: {MODELS}")
    print(f"Metrics: {METRICS}")
    
    # Load and process data
    print("\nLoading and processing data...")
    combined = load_and_process_data(MODELS, FOLDS, BASE_DIR, DURATION, METRICS)
    
    # Verify data was loaded
    data_summary = {}
    for metric in METRICS:
        data_summary[metric] = {}
        for model in MODELS:
            if metric in combined and model in combined[metric]:
                data_summary[metric][model] = len(combined[metric][model])
            else:
                data_summary[metric][model] = 0
    
    # print("\nData loading summary:")
    # for metric in METRICS:
    #     print(f"\n{metric}:")
    #     for model in MODELS:
    #         print(f"  {model}: {data_summary[metric][model]} records")
    
    # Create plots directory
    plot_dir = os.path.join(BASE_DIR, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nPlots will be saved to: {plot_dir}")
    
    # Plot per fold
    print("\nGenerating per-fold plots...")
    for fold in FOLDS:
        plot_metrics_per_fold(combined, fold, MODELS, METRICS, BASE_DIR, BAR_WIDTH, FIG_SIZE)
    
    # Plot grouped metrics across all folds combined
    print("\nGenerating grouped plots...")
    plot_grouped_metric(combined, 'delta_sdr', 'ΔSDR (dB)', 'delta_sdr_grouped.png', MODELS, BASE_DIR, BAR_WIDTH, FIG_SIZE)
    plot_grouped_metric(combined, 'delta_pesq', 'ΔPESQ', 'delta_pesq_grouped.png', MODELS, BASE_DIR, BAR_WIDTH, FIG_SIZE)
    plot_grouped_metric(combined, 'cycles', 'Cycles', 'cycles_grouped.png', MODELS, BASE_DIR, BAR_WIDTH, FIG_SIZE) 
    
    # Compute analysis metrics
    print("\nComputing analysis metrics...")
    rel_impr = compute_relative_improvement(combined, METRICS, MODELS)
    speedup = compute_speedup_ratio(combined, MODELS)
    composite_df = compute_composite_score(combined, MODELS, W_SDR, W_PESQ, W_CYCLES)
    
    # Save all statistics to CSV files
    stats_dir = os.path.join(BASE_DIR, 'statistics')
    save_summary_tables(combined, rel_impr, speedup, composite_df, stats_dir, BASE_DIR, FOLDS, MODELS, METRICS, DURATION, W_SDR, W_PESQ, W_CYCLES)
    
    print(f"\nAnalysis complete!")
    print(f"  - Plots saved to: {plot_dir}")
    print(f"  - Statistics saved to: {stats_dir}")



def combine_fold_metric_csvs(base_dir,
                             folds=(1, 2, 3, 4, 5),
                             plots_subdir="all_model_plots_16_0.3_4.0",
                             outfile="all_folds_all_metrics.csv",
                             rename_sdr_col=True):
    """
    Scan `base_dir/fold_x/<plots_subdir>/all_models_*_1sec.csv` for every fold,
    add 'fold' and 'metric' columns, concatenate, and save one grand CSV.

    Parameters
    ----------
    base_dir : str
        Path that contains the `fold_1`, `fold_2`, … directories.
    folds : iterable
        Which fold numbers to include. Default (1-5).
    plots_subdir : str
        The sub-folder under each fold that holds the CSV files.
    outfile : str
        Name (or absolute path) for the combined file that will be written
        inside `base_dir/statistics/`.
    rename_sdr_col : bool
        When True, renames the column `sdr_orig_rounded` → `sdr_orig`
        so downstream code can treat all files the same.

    Returns
    -------
    pandas.DataFrame
        The concatenated, annotated table.
    """

    all_frames = []

    for fold in folds:
        fold_dir = os.path.join(base_dir, f"fold_{fold}", plots_subdir)
        pattern  = os.path.join(fold_dir, "all_models_*_1sec.csv")

        for csv_path in glob.glob(pattern):
            # metric name is the bit between 'all_models_' and '_1sec.csv'
            metric = os.path.basename(csv_path)[
                len("all_models_") : -len("_1sec.csv")
            ]

            df = pd.read_csv(csv_path)

            if rename_sdr_col and "sdr_orig_rounded" in df.columns:
                df = df.rename(columns={"sdr_orig_rounded": "sdr_orig"})

            df["fold"]   = f"fold_{fold}"
            df["metric"] = metric           # cycles / delta_pesq / delta_sdr …

            all_frames.append(df)

    if not all_frames:
        raise FileNotFoundError("No matching CSV files were found.")

    combined = pd.concat(all_frames, ignore_index=True)

    # Consistent column ordering (optional — adjust to taste)
    preferred_order = (
        ["metric", "fold", "Model", "sdr_orig", "mean", "std", "Duration"]
        if "sdr_orig" in combined.columns
        else None
    )
    if preferred_order:
        keep = [c for c in preferred_order if c in combined.columns]
        rest = [c for c in combined.columns if c not in keep]
        combined = combined[keep + rest]

    # Make an output folder alongside your other statistics
    out_dir = os.path.join(base_dir, "statistics")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, outfile)
    combined.to_csv(out_path, index=False, float_format="%.6f")

    print(f"✔ Combined table written to {out_path}")
    print(f"   Rows: {len(combined):,} · Columns: {combined.shape[1]}")

    return combined