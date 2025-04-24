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

# nohup python training.py --pkl_path pkl_data/training_data.pkl --epochs 100 --batch_size 1024 --save_path saved_models --plot_path loss_plots --checkpoint_freq 10 --resume > training_log.txt 2>&1 &

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
