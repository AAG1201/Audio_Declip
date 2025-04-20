import os
import numpy as np
import soundfile as sf
from scipy.signal import resample
from scipy.io.wavfile import write
from time import time

from toolbox.clip_sdr_modified import clip_sdr_modified
from spade_segmentation import spade_segmentation
from toolbox.sdr import sdr

import matplotlib.pyplot as plt 
import argparse
import pandas as pd
from pesq import pesq

F_red = 1  # redundancy
s = 1    #  stepsize 
r = 2    #  steprate
epsilon = 0.1  # epsilon

# Helper function to ensure directories exist
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def recon(audio_dir, output_path, time_clip, target_fs_values, clipping_thresholds, dynamic, plotting, saving, delta, win_len, win_shift):

    sdr_clip_mean_array = []    # Clipped signal sdr storage
    sdr_clip_std_array = []
    time_mean_array = []    
    time_std_array = []
    theta_c_mean_array = []
    samples_clipped_mean_array = []
    samples_clipped_std_array = []
    sdr_imp_mean_array = []             # Tiem storage
    sdr_imp_std_array = []
    pesq_mean_array = []
    pesq_std_array = []
    pesq_imp_mean_array = []
    pesq_imp_std_array = []
    cycles_mean_array = []
    cycles_std_array = []

    for target_fs in target_fs_values:
        for clipping_threshold in clipping_thresholds:

            # Directory for the current target_fs and threshold combination
            dir_name = f"fs_{target_fs}_threshold_{clipping_threshold:.2f}"
            full_dir_path = os.path.join(output_path, dir_name)
            ensure_dir(full_dir_path)

            for tc in time_clip:
                wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
                n_files = len(wav_files) 
                # Limit the files to process to 'n_files'
                wav_files = wav_files[:n_files]             

                sdr_clip_array = []
                time_array = []
                theta_c_array = []
                samples_clipped_array = []
                sdr_imp_array = []
                pesq_array = []
                pesq_imp_array = []
                cycles_array = []

                for i, audio_file in enumerate(wav_files):
                    print(f"Loading audio: {audio_file}")
                    data, fs = sf.read(os.path.join(audio_dir, audio_file))

                    # Ensure mono signal
                    if len(data.shape) > 1:
                        data = data[:, 0]

                    # data = data[: fs * tc]      # fs is original data sampling rate
                    data = data[delta : delta + (fs * tc)] 

                    data = data / max(np.abs(data))         # normalization

                    # Resample to target sampling frequency
                    resampled_data = resample(data, int(target_fs * tc))

                    Ls = len(resampled_data) # length of resampled (not original)
                    inputTheta = clipping_threshold # threshold
                    # win_len = np.floor(Ls/K)  # window length
                    # win_len = 4096
                    # win_shift = int(np.floor(win_len/4))   # window shift
                    #win_type = 'hann'
                    
                    maxit = np.ceil(np.floor(win_len * F_red / 2 + 1) * r / s)          # max iterations

                    print("Generating clipped signal \n")
                    # Clipping
                    clipped_signal, masks, theta, true_sdr, percentage = clip_sdr_modified(resampled_data, clipping_threshold)
                    print(f"Clipping threshold {theta:.3f}, true SDR: {true_sdr:.2f} dB, Clipped samples: {percentage:.2f}%, Time: {tc}sec")
                    
                    

                    # Reconstruction
                    start_time = time()
                    reconstructed_signal, cycles = spade_segmentation(clipped_signal, resampled_data, Ls, win_len, win_shift, maxit, epsilon, r, s, F_red, masks,dynamic)
                    elapsed_time = time() - start_time

                    reconstructed_signal = resample(reconstructed_signal, int(fs * tc))

                    clipped_signal,_,_,_,_ = clip_sdr_modified(data, clipping_threshold)

                    clipped_signal = clipped_signal / max(np.abs(clipped_signal))         # normalization
                    reconstructed_signal = reconstructed_signal / max(np.abs(reconstructed_signal))         # normalization


                    sdr_clip = sdr(data, clipped_signal)
                    sdr_rec = sdr(data, reconstructed_signal)
                    sdr_imp = sdr_rec - sdr_clip
                    pesq_val  = pesq(fs, data, clipped_signal, 'wb')
                    pesq_imp = pesq(fs, data, reconstructed_signal, 'wb') - pesq(fs, data, clipped_signal, 'wb') 

                    # Store metrics
                    sdr_clip_array.append(sdr_clip)
                    theta_c_array.append(theta)
                    samples_clipped_array.append(percentage)
                    sdr_imp_array.append(sdr_imp)
                    time_array.append(elapsed_time)
                    pesq_array.append(pesq_val)
                    pesq_imp_array.append(pesq_imp)
                    cycles_array.append(cycles)

                    if (plotting):

                        # Plot the original signal (resampled_data)
                        plt.figure(figsize=(8, 4))
                        plt.plot(resampled_data, color='blue', linewidth=1.5)
                        plt.title("Original Signal")
                        plt.xlabel("Samples")
                        plt.ylabel("Amplitude")
                        plt.ylim(-1, 1)  # Set y-axis limits with a margin
                        plt.grid(True, linestyle='--', alpha=0.6)
                        plt.show()

                        # Plot the clipped signal with xlim and ylim
                        plt.figure(figsize=(8, 4))
                        plt.plot(clipped_signal, color='red', linewidth=1.5)
                        plt.title("Clipped Signal")
                        plt.xlabel("Samples")
                        plt.ylabel("Amplitude")
                        plt.ylim(-1, 1)  # Set y-axis limits with a margin
                        plt.grid(True, linestyle='--', alpha=0.6)
                        plt.show()


                        # Plot the reconstructed signal
                        plt.figure(figsize=(8, 4))
                        plt.plot(reconstructed_signal, color='green', linewidth=1.5)
                        plt.title("Reconstructed Signal (Dynamic ASPADE)")
                        plt.xlabel("Samples")
                        plt.ylabel("Amplitude")
                        plt.ylim(-1, 1)  # Set y-axis limits with a margin
                        plt.grid(True, linestyle='--', alpha=0.6)
                        plt.show()

                    if (saving):

                        # Save audio files
                        audios_dir = os.path.join(full_dir_path, "audios")
                        ensure_dir(audios_dir)

                        sf.write(os.path.join(audios_dir, f"{audio_file}_clipped_{tc}s.wav"), clipped_signal, target_fs)
                        sf.write(os.path.join(audios_dir, f"{audio_file}_reconstructed_{tc}s.wav"), reconstructed_signal, fs)
                        sf.write(os.path.join(audios_dir, f"{audio_file}_original_{tc}s.wav"), data, fs)

                # Calculate mean and std
                sdr_clip_mean_array.append(np.mean(sdr_clip_array))
                sdr_clip_std_array.append(np.std(sdr_clip_array))
                theta_c_mean_array.append(np.mean(theta_c_array))
                samples_clipped_mean_array.append(np.mean(samples_clipped_array))
                samples_clipped_std_array.append(np.std(samples_clipped_array))
                sdr_imp_mean_array.append(np.mean(sdr_imp_array))
                sdr_imp_std_array.append(np.std(sdr_imp_array))
                time_mean_array.append(np.mean(time_array))
                time_std_array.append(np.std(time_array))
                pesq_mean_array.append(np.mean(pesq_array))
                pesq_std_array.append(np.std(pesq_array))
                pesq_imp_mean_array.append(np.mean(pesq_imp_array))
                pesq_imp_std_array.append(np.std(pesq_imp_array))
                cycles_mean_array.append(np.mean(cycles_array))
                cycles_std_array.append(np.std(cycles_array))


            # Save results to csv file

            results = {
            "sdr_clip_mean_array": sdr_clip_mean_array,
            "sdr_clip_std_array": sdr_clip_std_array,
            "theta_c_array": theta_c_array,
            "samples_clipped_mean_array": samples_clipped_mean_array,
            "samples_clipped_std_array": samples_clipped_std_array,
            "sdr_imp_mean_array": sdr_imp_mean_array,
            "sdr_imp_std_array": sdr_imp_std_array,
            "time_mean_array": time_mean_array,
            "time_std_array": time_std_array,
            "pesq_mean_array": pesq_mean_array,
            "pesq_std_array": pesq_std_array,
            "pesq_imp_mean_array": pesq_imp_mean_array,
            "pesq_imp_std_array" : pesq_imp_std_array,
            "cycles_mean_array" : cycles_mean_array,
            "cycles_std_array" : cycles_std_array
            }


            # Flatten and format to 2 decimal places
            results_cleaned = [float(np.round(x.item(), 2)) if isinstance(x, np.ndarray) and x.size == 1 else float(np.round(x[0], 2)) for x in results.values()]
            df = pd.DataFrame([results_cleaned], columns=results.keys())

            # Save as Excel file
            output_file = os.path.join(full_dir_path, f"results_{tc}s.xlsx")  # ← Change to .xlsx
            df.to_excel(output_file, index=False)
            print(f"Results saved to {output_file}")
    

def main():
    parser = argparse.ArgumentParser(description="Audio Reconstruction")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to audio directory")
    parser.add_argument("--output_path", type=str, help="Path to the output directory.")
    parser.add_argument("--dynamic", type=int, default=0, required=True, help="Enable Dynamic ASPADE")
    parser.add_argument("--plotting", type=int, default=0, required=False, help="Enable plotting")
    parser.add_argument("--saving", type=int, default=0, required=False, help="Enable saving")
    parser.add_argument("--delta", type=int, default=0, required=False, help="Starting point in audio")
    parser.add_argument("--target_fs_values", type=int, nargs='+', help="List of target sampling frequencies.")
    parser.add_argument("--clipping_thresholds", type=float, nargs='+', help="List of clipping thresholds.")
    parser.add_argument("--time_clip", type=int, nargs='+', help="List of time clipping values.")
    parser.add_argument("--win_len", type=int, required=True, help="Window length")
    parser.add_argument("--win_shift", type=int, required=True, help="Window Shift")
    
    args = parser.parse_args()

    recon(
        audio_dir=args.audio_dir,
        output_path=args.output_path,
        time_clip=args.time_clip,
        target_fs_values=args.target_fs_values,
        clipping_thresholds=args.clipping_thresholds,
        dynamic=args.dynamic,
        plotting=args.plotting,
        saving=args.saving,
        delta=args.delta,
        win_len=args.win_len,
        win_shift=args.win_shift
    )


if __name__ == "__main__":
    main()