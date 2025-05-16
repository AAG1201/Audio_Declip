import numpy as np
import os
import soundfile as sf
from scipy.signal import resample
from toolbox.clip_sdr_modified import clip_sdr_modified
from spade_segmentation import spade_segmentation
from typing import List
import pickle
import shutil
import multiprocessing
import argparse

def generate_training_data(audio_file, audio_dir, target_fs, clipping_threshold, time_clip, 
                          win_len, win_shift, delta, F_red=2):
    """
    Generate training data for a single audio file
    
    Parameters:
    -----------
    audio_file : str
        Name of the audio file
    audio_dir : str
        Directory containing the audio file
    target_fs : int
        Target sampling frequency
    clipping_threshold : float
        Clipping threshold
    time_clip : int
        Time clip value in seconds
    win_len : int
        Window length
    win_shift : int
        Window shift
    delta : int
        Delta value
    F_red : int, default=1
        Frequency reduction factor
    
    Returns:
    --------
    list
        Training data for the audio file
    """
    print(f"Processing: {audio_file}", flush=True)
    data, fs = sf.read(os.path.join(audio_dir, audio_file))
    
    # Preprocessing steps
    if len(data.shape) > 1:
        data = data[:, 0]  # Take only first channel if stereo
    
    data = data[delta : ((fs * time_clip) + delta)]
    data = data / max(np.abs(data))  # Normalize
    
    # Resample to target frequency
    resampled_data = resample(data, int(target_fs * time_clip))
    
    # Setup parameters
    Ls = len(resampled_data)
    
    # ASPADE parameters
    ps_s = 1
    ps_r = 2
    ps_epsilon = 0.1
    ps_maxit = np.ceil(np.floor(win_len * F_red / 2 + 1) * ps_r / ps_s)

    # Generate clipped signal
    clipped_signal, masks, _, _, _ = clip_sdr_modified(resampled_data, clipping_threshold)
    
    # Process with SPADE segmentation
    _, _, training_data, _ = spade_segmentation(
        clipped_signal, resampled_data, Ls, win_len, win_shift,
        ps_maxit, ps_epsilon, ps_r, ps_s, F_red, masks,
        0, None, 1, 0, 0, 0
    )

    return training_data

def process_batch(batch_params):
    """Process a single batch of files with specific parameters"""
    audio_files, audio_dir, target_fs, clipping_threshold, time_clip, win_len, win_shift, delta, batch_id = batch_params
    
    print(f"Processing batch {batch_id}: fs={target_fs}, clip={clipping_threshold}, {len(audio_files)} files", flush=True)
    
    batch_training_data = []
    for audio_file in audio_files:
        audio_file_name = os.path.basename(audio_file)
        try:
            file_data = generate_training_data(
                audio_file_name, audio_dir, target_fs, clipping_threshold, 
                time_clip, win_len, win_shift, delta
            )
            batch_training_data.extend(file_data)
        except Exception as e:
            print(f"Error processing {audio_file_name}: {str(e)}", flush=True)
    
    print(f"Completed batch {batch_id}: fs={target_fs}, clip={clipping_threshold}", flush=True)
    return batch_training_data

def main():
    parser = argparse.ArgumentParser(description="Generate training dataset with parallel processing")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to audio directory (training files)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output pickle file")
    parser.add_argument("--target_fs_values", type=int, nargs='+', required=True, help="List of target sampling frequencies")
    parser.add_argument("--clipping_thresholds", type=float, nargs='+', required=True, help="List of clipping thresholds")
    parser.add_argument("--time_clip", type=int, required=True, help="Time clip value in seconds")
    parser.add_argument("--win_len", type=int, required=True, help="Window length")
    parser.add_argument("--win_shift", type=int, required=True, help="Window Shift")
    parser.add_argument("--delta", type=int, required=True, help="Delta value")
    parser.add_argument("--num_processes", type=int, default=6, help="Number of parallel processes to use")
    parser.add_argument("--num_batches", type=int, default=4, help="Number of batches to split the data into")
    parser.add_argument("--n_files", type=int, default=None, help="Limit the number of audio files to process")
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # Get list of all .wav files in the training directory
    wav_files = [f for f in os.listdir(args.audio_dir) if f.endswith(".wav")]

    if args.n_files is not None:
        wav_files = wav_files[:args.n_files]

    print(f"Found {len(wav_files)} training files", flush=True)
    
    # Split files into batches
    total_files = len(wav_files)
    batch_size = max(1, total_files // args.num_batches)
    file_batches = []
    
    for i in range(args.num_batches - 1):
        file_batches.append(wav_files[i*batch_size:(i+1)*batch_size])
    # Last batch includes any remaining files
    file_batches.append(wav_files[(args.num_batches-1)*batch_size:])
    
    # Print batch sizes for verification
    for i, batch in enumerate(file_batches):
        print(f"Batch {i+1} size: {len(batch)} files", flush=True)
    
    # Prepare batch parameters
    batch_params = []
    batch_id = 1
    
    for target_fs in args.target_fs_values:
        for clipping_threshold in args.clipping_thresholds:
            for batch in file_batches:
                batch_params.append((
                    batch, 
                    args.audio_dir, 
                    target_fs, 
                    clipping_threshold, 
                    args.time_clip,
                    args.win_len,
                    args.win_shift,
                    args.delta,
                    batch_id
                ))
                batch_id += 1
    
    # Process batches in parallel and collect all results
    print("Processing data in parallel...", flush=True)
    with multiprocessing.Pool(processes=min(args.num_processes, len(batch_params))) as pool:
        all_results = pool.map(process_batch, batch_params)
    
    # Combine all results into one list
    print("Combining all results...", flush=True)
    combined_training_data = []
    for result in all_results:
        combined_training_data.extend(result)
    
    # Save the combined data to a single pickle file
    combined_output_file = os.path.join(args.output_path, 'training_data.pkl')
    print(f"Saving combined data to {combined_output_file}", flush=True)
    
    with open(combined_output_file, 'wb') as f:
        pickle.dump(combined_training_data, f)
    
    print(f"Done! All data saved to {combined_output_file}", flush=True)
    print(f"Total training examples: {len(combined_training_data)}", flush=True)

if __name__ == "__main__":
    main()