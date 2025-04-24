import argparse
import numpy as np
import os
from tqdm import tqdm
import soundfile as sf
from scipy.signal import resample
from spade_segmentation import spade_segmentation
import pandas as pd
from toolbox.clip_sdr_modified import clip_sdr_modified
from typing import List, Dict
from toolbox.sdr import sdr
from pesq import pesq
import sys


def evaluate_model(test_audio_dir: str,
                   output_dir: str,
                   target_fs_values: List[int],
                   clipping_thresholds: List[float],
                   time_clip: List[int],
                   factor: float,
                   model_path: str,
                   train_gen_mode: int,
                   eval_mode: int,
                   dynamic: int,
                   save: int) -> Dict:

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize results storage
    results = {
        'file': [],
        'fs': [],
        'duration': [],
        'threshold': [],
        'clipped_percentage': [],
        'sdr_orig': [],
        'delta_sdr': [],
        'pesq_orig': [],
        'delta_pesq': [],
        'cycles': [],
        'processing_time': []
    }


    # Get all WAV files
    wav_files = [f for f in os.listdir(test_audio_dir) if f.endswith(".wav")]
    total_configs = len(target_fs_values) * len(clipping_thresholds) * len(time_clip) * len(wav_files)
    pbar = tqdm(total=total_configs, desc="Processing files",position=0, leave=True)

    for target_fs in target_fs_values:
        for clipping_threshold in clipping_thresholds:
            for tc in time_clip:
                for audio_file in wav_files:
                    print(f"\nProcessing: {audio_file} (fs={target_fs}, threshold={clipping_threshold}, duration={tc})")

                    audio_path = os.path.join(test_audio_dir, audio_file)

                    try:
                        # Load and preprocess audio
                        data, fs = sf.read(audio_path)
                    except Exception as e:
                        print(f"Error reading {audio_file}: {e}")
                        pbar.update(1)
                        continue
                    
                    if len(data.shape) > 1:
                        data = data[:, 0]  # Convert stereo to mono

                    # Clip to desired duration and normalize
                    max_samples = min(len(data), fs * tc)
                    data = data[:max_samples]
                    if np.max(np.abs(data)) > 0:  # Prevent division by zero
                        data = data / np.max(np.abs(data))

                    # Resample to target frequency
                    resampled_data = resample(data, int(target_fs * tc))

                    # Setup parameters
                    Ls = len(resampled_data)
                    win_len = np.floor(Ls / 32)
                    win_shift = np.floor(win_len / 4)
                    F_red = 1

                    # ASPADE parameters
                    ps_s = 1
                    ps_r = 2
                    ps_epsilon = 0.1
                    ps_maxit = np.ceil(np.floor(win_len * F_red / 2 + 1) * ps_r / ps_s)

                    # Generate clipped signal
                    print("Generating clipped signal...")
                    clipped_signal, masks, theta, sdr_original, clipped_percentage = \
                        clip_sdr_modified(resampled_data, clipping_threshold)

                    # Perform reconstruction
                   
                    reconstructed_signal, cycles, processing_time = spade_segmentation(
                        clipped_signal, resampled_data, Ls, win_len, win_shift,
                        ps_maxit, ps_epsilon, ps_r, ps_s, F_red, masks, dynamic, model_path,
                        train_gen_mode,  eval_mode, factor)

                    # Resample back to target_fs (not original fs)
                    reconstructed_signal = resample(reconstructed_signal, int(fs * tc))

                    clipped_signal,_,_,_,_ = clip_sdr_modified(data, clipping_threshold)

                    clipped_signal = clipped_signal / max(np.abs(clipped_signal))         # normalization
                    reconstructed_signal = reconstructed_signal / max(np.abs(reconstructed_signal))         # normalization



                    pesq_i = pesq(16000, data, clipped_signal, 'wb')
                    pesq_f = pesq(16000, data, reconstructed_signal, 'wb')
                    pesq_imp = pesq_f - pesq_i

                    sdr_clip = sdr(data, clipped_signal)
                    sdr_rec = sdr(data, reconstructed_signal)
                    sdr_imp = sdr_rec - sdr_clip

                    if save:
                        # Save reconstructed audio
                        dir_name = f"fs_{fs}_threshold_{clipping_threshold:.2f}"
                        full_dir_path = os.path.join(output_dir, dir_name)
                        os.makedirs(full_dir_path, exist_ok=True)
                        output_path = os.path.join(full_dir_path, f"reconstructed_{audio_file}")
                        sf.write(output_path, reconstructed_signal, fs)

                    # Store results
                    results['file'].append(audio_file)
                    results['fs'].append(fs)
                    results['duration'].append(tc)
                    results['threshold'].append(clipping_threshold)
                    results['clipped_percentage'].append(clipped_percentage)
                    results['sdr_orig'].append(sdr_clip)
                    results['delta_sdr'].append(sdr_imp)
                    results['pesq_orig'].append(pesq_i)
                    results['delta_pesq'].append(pesq_imp)
                    results['cycles'].append(cycles)
                    results['processing_time'].append(processing_time)
                
                    pbar.update(1)  # Update progress bar after each file

    pbar.close()

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)

    # Save Excel file for verification
    results_excel_path = os.path.join(output_dir, 'evaluation_results.xlsx')
    results_df.to_excel(results_excel_path, index=False)




def main(args):

    # Run evaluation
    loss_history = evaluate_model(
        test_audio_dir=args.test_audio_dir,
        output_dir=args.output_dir,
        target_fs_values=args.target_fs_values,
        clipping_thresholds=args.clipping_thresholds,
        time_clip=args.time_clip,
        factor=args.factor,
        model_path=args.model_path,
        train_gen_mode=args.train_gen_mode,
        eval_mode=args.eval_mode,
        dynamic=args.dynamic,
        save=args.save
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ComplexDFTUNet on test audio data.")

    parser.add_argument("--model_path", type=str, default="dft_model_checkpoints/complex_dft_unet_final_model.pth",
                        help="Path to the saved model file.")
    parser.add_argument("--test_audio_dir", type=str, default="custom_sound",
                        help="Directory containing test audio files.")
    parser.add_argument("--output_dir", type=str, default="test_data_output",
                        help="Directory to save output results.")
    parser.add_argument("--target_fs_values", type=int, nargs="+", default=[16000],
                        help="List of target sampling frequencies.")
    parser.add_argument("--clipping_thresholds", type=float, nargs="+", default=[0.2],
                        help="List of clipping thresholds.")
    parser.add_argument("--time_clip", type=int, nargs="+", default=[1],
                        help="List of time clipping values.")
    parser.add_argument("--factor", type=float, default=1,
                        help="Decrease the initialised k")
    parser.add_argument("--train_gen_mode", type=float, default=0,
                        help="In data generation mode")
    parser.add_argument("--eval_mode", type=float, default=0,
                        help="Evaluation using ML ASPADE")
    parser.add_argument("--dynamic", type=float, default=0,
                        help="Evaluation using Baseline/Dynamic ASPADE")
    parser.add_argument("--save", type=int, default=0,
                        help="Save reconstructed sounds")

    args = parser.parse_args()
    main(args)



