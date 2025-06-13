# List 

# To copy files of duration atleats 2s
# Copy files from one dir to other 
# Data Generation
# To check number of files
# Normal  audio declipping
# GIT PUSH
# Training


###############################################################################################################################

# To copy files of duration atleats 2s

import os
import shutil
import wave

# Source and destination directories
src_dir = '/data2/AAG/MTech_Project_Data/speech_data'
dst_dir = '/data2/AAG/MTech_Project_Data/speech_data_filter'

# Create destination directory if it doesn't exist
os.makedirs(dst_dir, exist_ok=True)

# Minimum duration in seconds
min_duration = 2.0

# Go through each file in the source directory
for file_name in os.listdir(src_dir):
    if file_name.lower().endswith('.wav'):
        src_path = os.path.join(src_dir, file_name)
        try:
            with wave.open(src_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
                
                if duration >= min_duration:
                    dst_path = os.path.join(dst_dir, file_name)
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied: {file_name} ({duration:.2f}s)")
        except wave.Error as e:
            print(f"Skipping {file_name} due to error: {e}")

###############################################################################################################################

# Copy files from one dir to other 

import os
import shutil

# Source and destination directories
src_dir = '/data2/AAG/amartya_sounds'
dst_dir = '/data2/AAG/MTech_Project_Data/speech_data'

# Create destination directory if it doesn't exist
os.makedirs(dst_dir, exist_ok=True)

# Walk through all directories and subdirectories
for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.lower().endswith('.wav'):
            src_path = os.path.join(root, file)
            dst_path = os.path.join(dst_dir, file)
            
            # To avoid overwriting if duplicate filenames exist
            base, ext = os.path.splitext(file)
            count = 1
            while os.path.exists(dst_path):
                dst_path = os.path.join(dst_dir, f"{base}_{count}{ext}")
                count += 1

            shutil.copy2(src_path, dst_path)
            print(f"Copied: {src_path} -> {dst_path}")


###############################################################################################################################


# Data Generation

# nohup python train_data_gen.py --audio_dir "/data/AAG/MTech_Project_Data/speech_data_filter" --cnt 1000 --train_dir "train_data" --test_dir "test_data" --output_path "pkl_data" --target_fs_values 16000 --clipping_thresholds 0.1 0.2 --time_clip 1 --win_len 500 --win_shift 125 --delta 300 --s_ratio 0.9 > output_log.txt 2>&1 &


# # Path to your .pkl file
# file_path = "/data/AAG/MTech_Project_Data/pkl_data/training_data.pkl"

# # Load the pickle file
# with open(file_path, "rb") as f:
#     data = pickle.load(f)

###############################################################################################################################


# To check number of files

# find /data2/AAG/MTech_Project_Speech/speech_data -type f -name "*.wav" | wc -l

###############################################################################################################################

# Normal declipping audio


# win_len = 250
# win_shift = int(win_len / 4)

# !python process.py --audio_dir custom_sound \
#     --output_path output_sound \
#     --time_clip 1 \
#     --target_fs_values 16000 \
#     --clipping_thresholds 0.2 \
#     --dynamic 1 \
#     --saving 0 \
#     --plotting 0 \
#     --delta 0 \
#     --win_len {win_len} \
#     --win_shift {win_shift}

###############################################################################################################################


# GIT PUSH

# cd /data2/AAG/MTech_Project_Second_Part
# git init
# echo -e "*.wav\n*.pkl\n*.pth\n*.log\naagproj/\n__pycache__/\n*.pyc\n.env\n.DS_Store" > .gitignore
# cat .gitignore
# git reset
# git add .
# git commit -m "Initial backup (excluding .wav files)"
# git remote add origin https://github.com/AAG1201/Audio_Declip.git
# git branch -M main
# git push -u origin main

###############################################################################################################################


# Training

# nohup python training.py > training_log.txt 2>&1 &

# nohup python training.py --pkl_path pkl_data/training_data.pkl --epochs 500 --batch_size 128 --save_path saved_models --plot_path loss_plots --checkpoint_freq 50 --resume > training_log.txt 2>&1 &

###############################################################################################################################

# Attention maps


from MTech_Project_Second_Part_mod.pipeline import load_model, visualize_frequency_attention, ComplexDFTUNet, ComplexDFTDataset, prepare_training_data_with_masks
from torch.utils.data import Dataset, DataLoader

import pickle

# Path to your .pkl file
file_path = "pkl_data/training_data.pkl"

# Load the pickle file
with open(file_path, "rb") as f:
    data = pickle.load(f)

inputs, masks, targets_dft, targets_sparsity = prepare_training_data_with_masks(data)

train_dataset = ComplexDFTDataset(inputs, targets_dft, masks, targets_sparsity, max_sparsity=250)

# Later, loading the model
loaded_model = ComplexDFTUNet(dft_size=500, mask_channels=3, max_sparsity=250)
loaded_model, checkpoint = load_model(loaded_model, "saved_models/complex_dft_unet_final_epoch100.pt")


train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
visualize_frequency_attention(loaded_model, train_loader)

###############################################################################################################################



# To upload data to matlab

# gdown --folder https://drive.google.com/drive/folders/1Zuc2mpSK9NSEpuEcMvK1hK-wCa7G0hyn



#  github

# ssh-keygen -t ed25519 -C "adityaag@iisc.ac.in" -f ~/.ssh/github_AAG
# cat ~/.ssh/github_AAG.pub

# rsync -avP wtc3@10.64.26.95:/data2/AAG/Audio_Declip/pkl_data/ /mnt/HDD_8TB/AAG/Audio_Declip/pkl_data

# To upload data to matlab

# nohup rclone copy /data/AAG/MTech_Project_Final/saved_models gdrive: --drive-root-folder-id 1qMC818ggFpiL7YZ8FpsZGSW3iVOSZmOu --progress

# gdown --folder https://drive.google.com/drive/folders/1Zuc2mpSK9NSEpuEcMvK1hK-wCa7G0hyn


# CUDA_VISIBLE_DEVICES=3 nohup python training.py --pkl_path pkl_data/training_data.pkl --epochs 500 --batch_size 2048 --save_path saved_models --plot_path loss_plots --checkpoint_freq 50 --val_split 0.1 --resume --val > training_log.txt 2>&1 &


#  nohup python training.py --pkl_path pkl_data/training_data.pkl --epochs 500 --batch_size 128 --save_path saved_models --plot_path loss_plots --checkpoint_freq 50 --val_split 0.1 --resume --val > training_log.txt 2>&1 &




###############################################################################################################################


# Small dataset generation



# import pickle

# # Define the file path on the server
# file_path = '/data2/AAG/Audio_Declip/pkl_data/training_data.pkl'

# # Load the .pkl file
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

# # Extract a small portion of the data (e.g., first 1000 rows if it's a DataFrame or a subset if it's a different structure)
# small_data = data[:10000] if isinstance(data, list) else data

# # Save the small data as a new .pkl file
# small_data_path = '/data2/AAG/Audio_Declip/pkl_data/small_training_data.pkl'
# with open(small_data_path, 'wb') as file:
#     pickle.dump(small_data, file)

# print(f"Small data saved to {small_data_path}")


###############################################################################################################################


# Small training


# !python training.py --pkl_path pkl_data/small_training_data.pkl \
#     --epochs 50 \
#     --batch_size 512 \
#     --save_path saved_models \
#     --plot_path loss_plots \
#     --checkpoint_freq 5 \
#     --val \
#     --val_split 0.2 \
#     --resume 




###############################################################################################################################

# LOSS PLOTTING

# !python plot_loss_history.py --history_file saved_models/loss_history.json --output_path final_plots


###############################################################################################################################



# FILE RENAMING

# import os

# # Path to the folder
# folder_path = "/data2/AAG/Audio_Declip/midterm_sounds/Speech_sounds"

# # Get a list of all .wav files
# wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

# # Sort the files to ensure consistent ordering
# wav_files.sort()

# # Rename each file
# for i, filename in enumerate(wav_files, start=1):
#     new_name = f"sound{i}.wav"
#     old_path = os.path.join(folder_path, filename)
#     new_path = os.path.join(folder_path, new_name)
#     os.rename(old_path, new_path)
#     print(f"Renamed '{filename}' to '{new_name}'")




###############################################################################################################################


# TO CALCULATE CONSISTENT WINDOWS BASED ONF Fs


import numpy as np
from toolbox.gabwin import gabwin

# Define your sampling rates
fs_values = [11025, 4000, 16000, 2000]

# Create non-uniform window lengths with more sampling at smaller values
# Method 1: Logarithmic spacing
win_len_ms = np.round(np.logspace(np.log10(10), np.log10(400), 10))
print(f"Original window lengths (ms): {win_len_ms}")

# Function to check and adjust window lengths for consistency
def make_consistent(n):
    """
    Check if window length n is consistent with gabwin(n) length.
    If not consistent, adjust n by ±1 until consistent.
    """
    original_n = n
    g = gabwin(n)
    
    # If already consistent, return original n
    if n == len(g):
        return n
    
    # Try increasing and decreasing by 1 until consistent
    offset = 1
    while True:
        # Try increasing by offset
        n_plus = original_n + offset
        g_plus = gabwin(n_plus)
        if n_plus == len(g_plus):
            return n_plus
            
        # Try decreasing by offset
        n_minus = original_n - offset
        if n_minus > 0:  # Ensure window length is positive
            g_minus = gabwin(n_minus)
            if n_minus == len(g_minus):
                return n_minus
                
        # Increase offset for next iteration
        offset += 1
        
        # Safety check to prevent infinite loops
        if offset > 10:
            print(f"Warning: Could not find consistent window length near {original_n}")
            return original_n

# Process each sampling rate
print("\n=== FINAL WINDOW LENGTHS ===")
for fs in fs_values:
    print(f"\nSampling rate: {fs} Hz")
    
    # Convert window lengths from milliseconds to samples
    original_win_len_samples = np.round(win_len_ms * fs / 1000).astype(int)
    
    # Check and adjust for consistency
    consistent_win_len_samples = np.array([make_consistent(n) for n in original_win_len_samples])
    
    # Convert back to milliseconds for comparison
    consistent_win_len_ms = consistent_win_len_samples * 1000 / fs

    print (consistent_win_len_samples)


from toolbox.gabwin import gabwin
def consistency_check(n):
    g = gabwin(n)
    if n == len(g):
        print("consistent")
    else:
        print("not consistent")



###############################################################################################################################




# RANDOM CODES

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File paths
file_paths = {
    'Bird': '/data2/AAG/Audio_Declip/exp3_new/bird_sound/evaluation_results_baseline_model_SDR_timewise_len.xlsx',
    'Heart': '/data2/AAG/Audio_Declip/exp3_new/heart_sound/evaluation_results_baseline_model_SDR_timewise_len.xlsx',
    'Lung': '/data2/AAG/Audio_Declip/exp3_new/lung_sound/evaluation_results_baseline_model_SDR_timewise_len.xlsx',
    'Speech': '/data2/AAG/Audio_Declip/exp3_new/speech_sound/evaluation_results_baseline_model_SDR_timewise_len.xlsx'
}

# Load datasets
datasets = {}
for label, path in file_paths.items():
    df = pd.read_excel(path)

    for col in ['delta_sdr', 'processing_time', 'cycles', 'sdr_orig', 'winlen']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['sdr_orig'] = df['sdr_orig'].round(1)
    df['sdr_orig_rounded'] = df['sdr_orig']

    datasets[label] = df

# Function to analyze delta SDR for each window length across all SDR levels
def find_best_common_winlen():
    results = {}
    available_winlens = set()
    
    # Find all available window lengths across datasets
    for df in datasets.values():
        available_winlens.update(df['winlen'].unique())
    available_winlens = sorted(available_winlens)
    
    for audio_type, df in datasets.items():
        # Group by window length and original SDR
        grouped = df.groupby(['winlen', 'sdr_orig_rounded'])['delta_sdr'].mean().reset_index()
        
        # For each window length, calculate average delta SDR across all SDR levels
        winlen_performance = {}
        sdr_levels = sorted(df['sdr_orig_rounded'].unique())
        
        for winlen in available_winlens:
            winlen_data = grouped[grouped['winlen'] == winlen]
            
            if len(winlen_data) == len(sdr_levels):  # Only consider window lengths that have data for all SDR levels
                avg_delta_sdr = winlen_data['delta_sdr'].mean()
                min_delta_sdr = winlen_data['delta_sdr'].min()
                
                # Store performance metrics for this window length
                winlen_performance[winlen] = {
                    'avg_delta_sdr': avg_delta_sdr,
                    'min_delta_sdr': min_delta_sdr,
                    'delta_sdr_by_sdr': {row['sdr_orig_rounded']: row['delta_sdr'] for _, row in winlen_data.iterrows()}
                }
        
        # Find the window length with the highest average delta SDR
        if winlen_performance:
            best_winlen = max(winlen_performance.items(), key=lambda x: x[1]['avg_delta_sdr'])[0]
            results[audio_type] = {
                'best_common_winlen': best_winlen,
                'avg_delta_sdr': winlen_performance[best_winlen]['avg_delta_sdr'],
                'min_delta_sdr': winlen_performance[best_winlen]['min_delta_sdr'],
                'delta_sdr_by_sdr': winlen_performance[best_winlen]['delta_sdr_by_sdr'],
                'all_winlen_performance': winlen_performance
            }
    
    return results

# Function to find the best window length for each specific SDR level
def find_best_winlen_per_sdr():
    results = {}
    
    for audio_type, df in datasets.items():
        results[audio_type] = {}
        sdr_levels = sorted(df['sdr_orig_rounded'].unique())
        
        for sdr in sdr_levels:
            subset = df[df['sdr_orig_rounded'] == sdr]
            # Group by window length and find the one with max delta_sdr
            best_winlen = subset.groupby('winlen')['delta_sdr'].mean().idxmax()
            max_delta_sdr = subset.groupby('winlen')['delta_sdr'].mean().max()
            
            results[audio_type][sdr] = {
                'best_winlen': best_winlen,
                'max_delta_sdr': max_delta_sdr
            }
    
    return results

# Function to create a bar chart showing delta SDR for each window length
def plot_delta_sdr_by_winlen():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    common_winlen_results = find_best_common_winlen()
    best_winlen_per_sdr = find_best_winlen_per_sdr()
    
    for i, (audio_type, df) in enumerate(datasets.items()):
        ax = axes[i]
        
        # Get all unique window lengths
        winlens = sorted(df['winlen'].unique())
        sdr_levels = sorted(df['sdr_orig_rounded'].unique())
        
        # Group by window length and calculate average delta SDR
        grouped = df.groupby(['winlen'])['delta_sdr'].mean().reset_index()
        
        # Plot bar chart
        bars = ax.bar(range(len(winlens)), [grouped[grouped['winlen'] == w]['delta_sdr'].values[0] for w in winlens])
        
        # Highlight the best common window length
        if audio_type in common_winlen_results:
            best_idx = winlens.index(common_winlen_results[audio_type]['best_common_winlen'])
            bars[best_idx].set_color('red')
            ax.text(best_idx, grouped[grouped['winlen'] == winlens[best_idx]]['delta_sdr'].values[0] + 0.1, 
                   f"Best\n{int(winlens[best_idx])}", ha='center', fontweight='bold')
        
        ax.set_title(f'{audio_type} Audio', fontsize=16)
        ax.set_xlabel('Window Length', fontsize=14)
        ax.set_ylabel('Average ΔSDR (dB)', fontsize=14)
        ax.set_xticks(range(len(winlens)))
        ax.set_xticklabels([int(w) for w in winlens], rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add text at the top showing the best window length for each SDR level
        if audio_type in best_winlen_per_sdr:
            sdr_text = "Best winlen for each SDR level:\n"
            for sdr in sorted(sdr_levels):
                if sdr in best_winlen_per_sdr[audio_type]:
                    best_w = int(best_winlen_per_sdr[audio_type][sdr]['best_winlen'])
                    delta = round(best_winlen_per_sdr[audio_type][sdr]['max_delta_sdr'], 2)
                    sdr_text += f"SDR {sdr} dB: {best_w} (Δ: {delta} dB)\n"
            
            ax.text(0.98, 0.98, sdr_text, 
                   transform=ax.transAxes, 
                   verticalalignment='top', 
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('Average ΔSDR by Window Length for Each Audio Type', y=1.02, fontsize=18)
    plt.show()

# Function to create a heatmap showing delta SDR for each window length and SDR level
def create_delta_sdr_heatmap():
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    common_winlen_results = find_best_common_winlen()
    
    for i, (audio_type, df) in enumerate(datasets.items()):
        ax = axes[i]
        
        # Get unique window lengths and SDR levels
        winlens = sorted(df['winlen'].unique())
        sdr_levels = sorted(df['sdr_orig_rounded'].unique())
        
        # Create a matrix for the heatmap
        heatmap_data = np.zeros((len(winlens), len(sdr_levels)))
        
        # Fill the matrix with delta SDR values
        for wi, winlen in enumerate(winlens):
            for si, sdr in enumerate(sdr_levels):
                subset = df[(df['winlen'] == winlen) & (df['sdr_orig_rounded'] == sdr)]
                if not subset.empty:
                    heatmap_data[wi, si] = subset['delta_sdr'].mean()
        
        # Create heatmap
        im = ax.imshow(heatmap_data, cmap='viridis')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label('Average ΔSDR (dB)')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(sdr_levels)))
        ax.set_yticks(np.arange(len(winlens)))
        ax.set_xticklabels([f"{sdr} dB" for sdr in sdr_levels])
        ax.set_yticklabels([int(w) for w in winlens])
        
        # Rotate tick labels and set alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Highlight the best common window length
        if audio_type in common_winlen_results:
            best_winlen = common_winlen_results[audio_type]['best_common_winlen']
            best_idx = winlens.index(best_winlen)
            ax.axhline(y=best_idx, color='red', linestyle='-', linewidth=2, alpha=0.5)
            ax.text(-0.5, best_idx, f"Best: {int(best_winlen)}", 
                   verticalalignment='center', color='red', fontweight='bold')
        
        # Add annotations with delta SDR values
        for wi in range(len(winlens)):
            for si in range(len(sdr_levels)):
                if heatmap_data[wi, si] > 0:
                    text = ax.text(si, wi, f"{heatmap_data[wi, si]:.2f}",
                                 ha="center", va="center", 
                                 color="white" if heatmap_data[wi, si] > np.max(heatmap_data)/2 else "black",
                                 fontsize=9)
        
        ax.set_title(f'{audio_type} Audio', fontsize=16)
        ax.set_xlabel('Original SDR (dB)', fontsize=14)
        ax.set_ylabel('Window Length', fontsize=14)
    
    plt.tight_layout()
    plt.suptitle('ΔSDR (dB) by Window Length and Original SDR', y=1.02, fontsize=18)
    plt.show()

# Function to generate summary table of best common window lengths
def generate_best_common_winlen_table():
    common_winlen_results = find_best_common_winlen()
    best_winlen_per_sdr = find_best_winlen_per_sdr()
    
    # Create a dataframe for the summary table
    summary_data = []
    
    for audio_type in sorted(datasets.keys()):
        if audio_type in common_winlen_results:
            result = common_winlen_results[audio_type]
            
            # Get performance difference between best common winlen and optimal winlen for each SDR
            performance_loss = {}
            for sdr, data in best_winlen_per_sdr[audio_type].items():
                best_specific_delta = data['max_delta_sdr']
                common_delta = result['delta_sdr_by_sdr'].get(sdr, 0)
                performance_loss[sdr] = best_specific_delta - common_delta
            
            # Calculate average and maximum performance loss
            avg_loss = sum(performance_loss.values()) / len(performance_loss) if performance_loss else 0
            max_loss = max(performance_loss.values()) if performance_loss else 0
            
            summary_data.append([
                audio_type,
                int(result['best_common_winlen']),
                round(result['avg_delta_sdr'], 2),
                round(result['min_delta_sdr'], 2),
                round(avg_loss, 2),
                round(max_loss, 2)
            ])
    
    summary_df = pd.DataFrame(summary_data, columns=[
        'Audio Type', 
        'Best Common Window Length', 
        'Average ΔSDR (dB)', 
        'Minimum ΔSDR (dB)',
        'Avg Performance Loss (dB)',
        'Max Performance Loss (dB)'
    ])
    
    return summary_df

# Run the analyses
best_common_winlen_table = generate_best_common_winlen_table()
print("Best Common Window Length for Each Audio Type to Maximize ΔSDR:")
print(best_common_winlen_table)

# Create visualizations
plot_delta_sdr_by_winlen()
create_delta_sdr_heatmap()

# Generate detailed recommendation for each audio type
def generate_detailed_recommendations():
    common_winlen_results = find_best_common_winlen()
    best_winlen_per_sdr = find_best_winlen_per_sdr()
    
    print("\n=== DETAILED WINDOW LENGTH RECOMMENDATIONS FOR MAXIMUM ΔSDR ===\n")
    
    for audio_type in sorted(datasets.keys()):
        print(f"For {audio_type} Audio:")
        
        if audio_type in common_winlen_results:
            result = common_winlen_results[audio_type]
            best_common_winlen = int(result['best_common_winlen'])
            avg_delta_sdr = round(result['avg_delta_sdr'], 2)
            
            print(f"  • Best common window length: {best_common_winlen}")
            print(f"  • Average ΔSDR across all SDR levels: {avg_delta_sdr} dB")
            
            # Show ΔSDR for each SDR level using the common window length
            print(f"  • ΔSDR performance by original SDR level (using winlen={best_common_winlen}):")
            for sdr in sorted(result['delta_sdr_by_sdr'].keys()):
                delta = round(result['delta_sdr_by_sdr'][sdr], 2)
                print(f"    - SDR {sdr} dB: {delta} dB")
            
            # Compare with optimal window length for each SDR level
            print(f"\n  • Comparison with optimal window length for each SDR level:")
            for sdr in sorted(best_winlen_per_sdr[audio_type].keys()):
                best_specific = best_winlen_per_sdr[audio_type][sdr]
                best_specific_winlen = int(best_specific['best_winlen'])
                best_specific_delta = round(best_specific['max_delta_sdr'], 2)
                common_delta = round(result['delta_sdr_by_sdr'].get(sdr, 0), 2)
                loss = round(best_specific_delta - common_delta, 2)
                
                if best_specific_winlen == best_common_winlen:
                    print(f"    - SDR {sdr} dB: Common winlen {best_common_winlen} is optimal (ΔSDR: {common_delta} dB)")
                else:
                    print(f"    - SDR {sdr} dB: Optimal winlen {best_specific_winlen} (ΔSDR: {best_specific_delta} dB) vs Common winlen {best_common_winlen} (ΔSDR: {common_delta} dB) - Loss: {loss} dB")
        
        print()

# Generate detailed recommendations
generate_detailed_recommendations()
















import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File paths
file_paths = {
    'Bird': '/data2/AAG/Audio_Declip/exp3/bird_sound/evaluation_results_baseline_model_SDR.xlsx',
    'Heart': '/data2/AAG/Audio_Declip/exp3/heart_sound/evaluation_results_baseline_model_SDR.xlsx',
    'Lung': '/data2/AAG/Audio_Declip/exp3/lung_sound/evaluation_results_baseline_model_SDR.xlsx',
    'Speech': '/data2/AAG/Audio_Declip/exp3/speech_sound/evaluation_results_baseline_model_SDR.xlsx'
}

# Plot settings
metrics = {
    'delta_sdr': 'ΔSDR (dB)',
    'processing_time': 'Processing Time (s)',
    'cycles': 'Cycles'
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
sdr_order = [1, 3, 5, 7]  # fixed SDR order

plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 150,
    'axes.linewidth': 1.2,
    'axes.labelpad': 10
})

# Load datasets
datasets = {}
for label, path in file_paths.items():
    df = pd.read_excel(path)

    for col in ['delta_sdr', 'processing_time', 'cycles', 'sdr_orig', 'blocks', 'winlen']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['sdr_orig'] = df['sdr_orig'].round(1)
    datasets[label] = df

# Plotting
for metric, ylabel in metrics.items():
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    axs = axs.flatten()

    for idx, (label, df) in enumerate(datasets.items()):
        ax = axs[idx]

        blocks_order = sorted(df['blocks'].unique())
        width = 0.15
        x = np.arange(len(blocks_order))

        grouped = df.groupby(['sdr_orig', 'blocks'])[metric].agg(['mean', 'std']).unstack()
        grouped_mean = grouped['mean']
        grouped_std = grouped['std']

        actual_sdrs = [s for s in sdr_order if s in grouped_mean.index]

        for i, sdr in enumerate(actual_sdrs):
            means = grouped_mean.loc[sdr].reindex(blocks_order).values
            stds = grouped_std.loc[sdr].reindex(blocks_order).values

            ax.bar(
                x + i * width,
                means,
                width,
                yerr=stds,
                label=f'SDR {sdr} dB',
                color=colors[i % len(colors)],
                capsize=4,
                edgecolor='black',
                linewidth=0.7
            )

        ax.set_xticks(x + width * len(actual_sdrs) / 2)
        ax.set_xticklabels(blocks_order, rotation=0)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Blocks')
        ax.set_title(f'{label}')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(title='Input SDR')

    plt.suptitle(f'{ylabel} vs Blocks for Different Sound Types', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


##############################################################################################


