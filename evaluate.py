import argparse
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import soundfile as sf
from scipy.signal import resample
from spade_segmentation import spade_segmentation
import pandas as pd
from toolbox.clip_sdr_modified import clip_sdr_modified
from toolbox.clip_sdr import clip_sdr
from typing import List, Dict
from toolbox.sdr import sdr
from pesq import pesq
import scipy.io as sio



def evaluate_model(test_audio_dir: str,
                   output_dir: str,
                   target_fs_values: List[int],
                   clipping_thresholds: List[float],
                   input_sdrs: List[float],
                   time_clip: List[int],
                   factor: float,
                   model_path: str,
                   train_gen_mode: int,
                   eval_mode: int,
                   dynamic: int,
                   delta: int,
                   save: int,
                   custom_wins: List[int],
                   restrict_mode: int,
                   sdr_mode: int,
                   pesq_mode: int,
                   n_files: int,
                   exp_name: str,
                   verbose: int,
                   stepsize: int,
                   steprate: int,
                   block_metrics: int,
                   mask_size: int,
                   max_sparsity: int) -> Dict:

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize results storage
    results = {
        'file': [],
        'fs': [],
        'target_fs': [],
        'duration': [],
        'threshold': [],
        'clipped_percentage': [],
        'sdr_orig': [],
        'delta_sdr': [],
        'pesq_orig': [],
        'delta_pesq': [],
        'cycles': [],
        'winlen': [],
        'blocks': [],
        'processing_time': []
    }

    if block_metrics:

        results1 = {
            'Block_summary': [],
            'Iterations': [],
            'ProcessingTime(s)': [],
            'Block': [],
            'Iteration': [],
            'k': [],
            'objVal': []
        }



    # Get all WAV files
    wav_files = [f for f in os.listdir(test_audio_dir) if f.endswith(".wav")]

    if n_files:
        wav_files = wav_files[:n_files]  # Use only first n files

    if sdr_mode:
        total_configs = len(target_fs_values) * len(input_sdrs) * len(time_clip) * len(wav_files) * len(custom_wins)
    else:
        total_configs = len(target_fs_values) * len(clipping_thresholds) * len(time_clip) * len(wav_files) * len(custom_wins)

    
    pbar = tqdm(total=total_configs, desc="Processing files",position=0, leave=True)

    if sdr_mode:
        for target_fs in target_fs_values:
            for input_sdr in input_sdrs:
                for tc in time_clip:
                    for custom_win in custom_wins:
                        for audio_file in wav_files:
                            print(f"\nProcessing: {audio_file} (fs={target_fs}, input_sdr={input_sdr}, duration={tc}, winlen={custom_win})")

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

                            data = data[delta : delta + (fs * tc)] 

                            # Clip to desired duration and normalize
                            max_samples = min(len(data), fs * tc)
                            data = data[:max_samples]
                            if np.max(np.abs(data)) > 0:  # Prevent division by zero
                                data = data / np.max(np.abs(data))

                            # Resample to target frequency
                            resampled_data = resample(data, int(target_fs * tc))

                            # Setup parameters
                            Ls = len(resampled_data)
                            # win_len = np.floor(Ls / 32)
                            # win_shift = np.floor(win_len / 4)
                            win_len = custom_win
                            win_shift = np.floor (custom_win / 4)
                            F_red = 2

                            # ASPADE parameters
                            ps_s = stepsize
                            ps_r = steprate
                            ps_epsilon = 0.1
                            ps_maxit = np.ceil(np.floor(win_len * F_red / 2 + 1) * ps_r / ps_s)

                            # Generate clipped signal
                            print("Generating clipped signal...")

                            clipped_signal, masks, clipping_threshold, sdr_original, clipped_percentage = \
                                clip_sdr(resampled_data, input_sdr)
                            
                        
                            # Perform reconstruction
                            reconstructed_signal, cycles, blocks, processing_time, all_k_values, all_objVal_values, all_iterations, all_processing_times = spade_segmentation(
                                clipped_signal, resampled_data, Ls, win_len, win_shift,
                                ps_maxit, ps_epsilon, ps_r, ps_s, F_red, masks, dynamic, model_path,
                                train_gen_mode,  eval_mode, restrict_mode, factor, verbose, mask_size, max_sparsity)


                            # Resample back to target_fs (not original fs)
                            reconstructed_signal = resample(reconstructed_signal, int(fs * tc))
                            
                            clipped_signal,_,_,_,_ = clip_sdr(data, input_sdr)

                            if pesq_mode == 0:
                                pesq_i = 0
                                pesq_f = 0
                            else:
                                pesq_i = pesq(fs, data, clipped_signal, 'wb')
                                pesq_f = pesq(fs, data, reconstructed_signal, 'wb')
                                
                            pesq_imp = pesq_f - pesq_i

                            sdr_clip = sdr(data, clipped_signal)
                            sdr_rec = sdr(data, reconstructed_signal)
                            sdr_imp = sdr_rec - sdr_clip
                            

                            if save:
                                clipped_signal = clipped_signal / max(np.abs(clipped_signal))         # normalization
                                reconstructed_signal = reconstructed_signal / max(np.abs(reconstructed_signal))         # normalization
                                # Save reconstructed audio
                                dir_name = f"fs_{fs}_input_sdr_{input_sdr:.2f}"
                                full_dir_path = os.path.join(output_dir, dir_name)
                                os.makedirs(full_dir_path, exist_ok=True)
                                output_path = os.path.join(full_dir_path, f"reconstructed_{audio_file}")
                                sf.write(output_path, reconstructed_signal, fs)
                                # Save clipped signal
                                output_path_clipped = os.path.join(full_dir_path, f"clipped_{audio_file}")
                                sf.write(output_path_clipped, clipped_signal, fs)

                                # Save original signal (data)
                                output_path_original = os.path.join(full_dir_path, f"original_{audio_file}")
                                sf.write(output_path_original, data, fs)

                        

                            # Store results
                            results['file'].append(audio_file)
                            results['fs'].append(fs)
                            results['target_fs'].append(target_fs)
                            results['duration'].append(tc)
                            results['threshold'].append(clipping_threshold)
                            results['clipped_percentage'].append(clipped_percentage)
                            results['sdr_orig'].append(sdr_clip)
                            results['delta_sdr'].append(sdr_imp)
                            results['pesq_orig'].append(pesq_i)
                            results['delta_pesq'].append(pesq_imp)
                            results['cycles'].append(cycles)
                            results['winlen'].append(custom_win)
                            results['blocks'].append(blocks)
                            results['processing_time'].append(processing_time)

                            if block_metrics:

                                # Lists for the detailed evolution file
                                block_list = []
                                iteration_list = []
                                k_list = []
                                objVal_list = []

                                for block_idx, (k_vals, obj_vals) in enumerate(zip(all_k_values, all_objVal_values)):
                                    k_vals = list(k_vals)
                                    obj_vals = list(obj_vals)
                                    for iter_idx, (k, obj) in enumerate(zip(k_vals, obj_vals)):
                                        block_list.append(block_idx)
                                        iteration_list.append(iter_idx)
                                        k_list.append(float(k))
                                        objVal_list.append(float(obj))

                                # Lists for the summary file
                                block_summary_list = list(range(len(all_iterations)))
                                iterations_summary_list = [int(i) for i in all_iterations]
                                processing_time_summary_list = [float(t) for t in all_processing_times]


                                
                                results1['Block_summary'].append(block_summary_list)
                                results1['Iterations'].append(iterations_summary_list)
                                results1['ProcessingTime(s)'].append(processing_time_summary_list)
                                results1['Block'].append(block_list)
                                results1['Iteration'].append(iteration_list)
                                results1['k'].append(k_list)
                                results1['objVal'].append(objVal_list)

                                                    
                        
                            pbar.update(1)  # Update progress bar after each file
    
    else:

        for target_fs in target_fs_values:
            for clipping_threshold in clipping_thresholds:
                for tc in time_clip:
                    for custom_win in custom_wins:
                        for audio_file in wav_files:
                            print(f"\nProcessing: {audio_file} (fs={target_fs}, threshold={clipping_threshold}, duration={tc}, winlen={custom_win})")

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

                            data = data[delta : delta + (fs * tc)] 

                            # Clip to desired duration and normalize
                            max_samples = min(len(data), fs * tc)
                            data = data[:max_samples]
                            if np.max(np.abs(data)) > 0:  # Prevent division by zero
                                data = data / np.max(np.abs(data))

                            # Resample to target frequency
                            resampled_data = resample(data, int(target_fs * tc))

                            # Setup parameters
                            Ls = len(resampled_data)
                            # win_len = np.floor(Ls / 32)
                            # win_shift = np.floor(win_len / 4)
                            win_len = np.floor(custom_win)
                            win_shift = np.floor(win_len / 4)
                            F_red = 2

                            # ASPADE parameters
                            ps_s = stepsize
                            ps_r = steprate
                            ps_epsilon = 0.1
                            ps_maxit = np.ceil(np.floor(win_len * F_red / 2 + 1) * ps_r / ps_s)

                            # Generate clipped signal
                            print("Generating clipped signal...")

                            clipped_signal, masks, theta, sdr_original, clipped_percentage = \
                                clip_sdr_modified(resampled_data, clipping_threshold)

                            # Perform reconstruction
                            reconstructed_signal, cycles, blocks, processing_time, all_k_values, all_objVal_values, all_iterations, all_processing_times = spade_segmentation(
                                clipped_signal, resampled_data, Ls, win_len, win_shift,
                                ps_maxit, ps_epsilon, ps_r, ps_s, F_red, masks, dynamic, model_path,
                                train_gen_mode,  eval_mode, restrict_mode, factor, verbose, mask_size, max_sparsity)


                            # Resample back to target_fs (not original fs)
                            reconstructed_signal = resample(reconstructed_signal, int(fs * tc))
                            
                            clipped_signal,_,_,_,_ = clip_sdr_modified(data, clipping_threshold)

                            if pesq_mode == 0:
                                pesq_i = 0
                                pesq_f = 0
                            else:
                                pesq_i = pesq(fs, data, clipped_signal, 'wb')
                                pesq_f = pesq(fs, data, reconstructed_signal, 'wb')
                                
                            pesq_imp = pesq_f - pesq_i

                            sdr_clip = sdr(data, clipped_signal)
                            sdr_rec = sdr(data, reconstructed_signal)
                            sdr_imp = sdr_rec - sdr_clip

                            
                            if save:
                                clipped_signal = clipped_signal / max(np.abs(clipped_signal))         # normalization
                                reconstructed_signal = reconstructed_signal / max(np.abs(reconstructed_signal))         # normalization

                                # Save reconstructed audio
                                dir_name = f"fs_{fs}_threshold_{clipping_threshold:.2f}"
                                full_dir_path = os.path.join(output_dir, dir_name)
                                os.makedirs(full_dir_path, exist_ok=True)
                                output_path = os.path.join(full_dir_path, f"reconstructed_{audio_file}")
                                sf.write(output_path, reconstructed_signal, fs)
                                # Save clipped signal
                                output_path_clipped = os.path.join(full_dir_path, f"clipped_{audio_file}")
                                sf.write(output_path_clipped, clipped_signal, fs)

                                # Save original signal (data)
                                output_path_original = os.path.join(full_dir_path, f"original_{audio_file}")
                                sf.write(output_path_original, data, fs)

                             

                            # Store results
                            results['file'].append(audio_file)
                            results['fs'].append(fs)
                            results['target_fs'].append(target_fs)
                            results['duration'].append(tc)
                            results['threshold'].append(clipping_threshold)
                            results['clipped_percentage'].append(clipped_percentage)
                            results['sdr_orig'].append(sdr_clip)
                            results['delta_sdr'].append(sdr_imp)
                            results['pesq_orig'].append(pesq_i)
                            results['delta_pesq'].append(pesq_imp)
                            results['cycles'].append(cycles)
                            results['winlen'].append(custom_win)
                            results['blocks'].append(blocks)
                            results['processing_time'].append(processing_time)


                            if block_metrics:

                                # Lists for the detailed evolution file
                                block_list = []
                                iteration_list = []
                                k_list = []
                                objVal_list = []

                                for block_idx, (k_vals, obj_vals) in enumerate(zip(all_k_values, all_objVal_values)):
                                    k_vals = list(k_vals)
                                    obj_vals = list(obj_vals)
                                    for iter_idx, (k, obj) in enumerate(zip(k_vals, obj_vals)):
                                        block_list.append(block_idx)
                                        iteration_list.append(iter_idx)
                                        k_list.append(float(k))
                                        objVal_list.append(float(obj))

                                # Lists for the summary file
                                block_summary_list = list(range(len(all_iterations)))
                                iterations_summary_list = [int(i) for i in all_iterations]
                                processing_time_summary_list = [float(t) for t in all_processing_times]


                                
                                results1['Block_summary'].append(block_summary_list)
                                results1['Iterations'].append(iterations_summary_list)
                                results1['ProcessingTime(s)'].append(processing_time_summary_list)
                                results1['Block'].append(block_list)
                                results1['Iteration'].append(iteration_list)
                                results1['k'].append(k_list)
                                results1['objVal'].append(objVal_list)


                        
                            pbar.update(1)  # Update progress bar after each file

    pbar.close()

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)

    # Determine configuration type
    if eval_mode == 1 and dynamic == 0:
        config_type = 'ml_model'
    elif eval_mode == 0 and dynamic == 1:
        config_type = 'dynamic_model'
    elif eval_mode == 0 and dynamic == 0:
        config_type = 'baseline_model'
    else:
        config_type = 'unknown_config'

    # # Format clipping thresholds into a string
    # clip_str = '_'.join([str(th) for th in clipping_thresholds])

    if sdr_mode:
        sdr_config = 'SDR'
    else:
        sdr_config = 'Threshold'

    # Construct filename
    filename = f"evaluation_results_{config_type}_{sdr_config}_{exp_name}.xlsx"
    results_excel_path = os.path.join(output_dir, filename)

    # Save Excel file
    results_df.to_excel(results_excel_path, index=False)

    if block_metrics:

        # Create a filename for the .mat file
        mat_filename = f"Block_evaluation_results_{config_type}_{sdr_config}_{exp_name}.mat"
        results_mat_path = os.path.join(output_dir, mat_filename)

        # Save as .mat
        sio.savemat(results_mat_path, {'results': results1})



def main(args):

    # Run evaluation
    loss_history = evaluate_model(
        test_audio_dir=args.test_audio_dir,
        output_dir=args.output_dir,
        target_fs_values=args.target_fs_values,
        clipping_thresholds=args.clipping_thresholds,
        input_sdrs=args.input_sdrs,
        time_clip=args.time_clip,
        factor=args.factor,
        model_path=args.model_path,
        train_gen_mode=args.train_gen_mode,
        eval_mode=args.eval_mode,
        dynamic=args.dynamic,
        delta=args.delta,
        save=args.save,
        custom_wins=args.c_win,
        restrict_mode=args.r_mode,
        sdr_mode=args.sdr_mode,
        pesq_mode=args.pesq_mode,
        n_files=args.n_files,
        exp_name=args.exp_name,
        verbose=args.verbose,
        stepsize=args.stepsize,
        steprate=args.steprate,
        block_metrics=args.block_metrics,
        mask_size=args.mask_size,
        max_sparsity=args.max_sparsity
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
    parser.add_argument("--input_sdrs", type=float, nargs="+", default=[1],
                        help="List of input sdrs.")
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
    parser.add_argument("--delta", type=int, default=0, 
                        required=False, help="Starting point in audio")
    parser.add_argument("--c_win", type=int, nargs="+", default=[500], 
                        required=False, help="custom window length")
    parser.add_argument("--r_mode", type=int, default=0,
                        help="Evaluation using ML ASPADE without refinement")
    parser.add_argument("--sdr_mode", type=int, default=0,
                        help="input sdr instead of threshold")
    parser.add_argument("--pesq_mode", type=int, default=1,
                        help="enable pesq calculation")
    parser.add_argument("--n_files", type=int, default=0,
                        help="limits the number of files used")
    parser.add_argument("--exp_name", type=str, default="",
                        help="custom experiment name")
    parser.add_argument("--verbose", type=int, default=0,
                        help="enable processing updates in output")
    parser.add_argument("--stepsize", type=int, default=1,
                        help="Stepsize for the algorithm")
    parser.add_argument("--steprate", type=int, default=2,
                        help="Steprate for the algorithm")
    parser.add_argument("--block_metrics", type=int, default=0,
                            help="Enable block wise analysis")
    parser.add_argument("--mask_size", type=int,
                            help="Mask length of each channel")
    parser.add_argument("--max_sparsity", type=int,
                            help="Maximum expected sparsity of input")
    args = parser.parse_args()
    main(args)



