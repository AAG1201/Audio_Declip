# import os
# import argparse
# import subprocess

# def run_evaluations(model_path, test_audio_dir, output_dir, target_fs_values, 
#                    input_sdrs, time_clip, factor, save, delta, c_win, 
#                    mask_size, max_sparsity, n_files, verbose=0, sdr_mode=1, pesq_mode=0,
#                    config_to_run="all"):
#     """
#     Run evaluations with different configurations
    
#     Args:
#         model_path: Path to the trained model
#         test_audio_dir: Directory containing test audio files
#         output_dir: Output directory for results
#         target_fs_values: Target sampling frequency
#         input_sdrs: Input SDR values
#         time_clip: Time clipping parameter
#         factor: Factor parameter
#         save: Save parameter (0 or 1)
#         delta: Delta parameter
#         c_win: Window size parameter
#         mask_size: Mask size parameter
#         max_sparsity: Maximum sparsity parameter
#         n_files: Number of files to process
#         verbose: Verbose mode (default: 0)
#         sdr_mode: SDR mode (default: 1)
#         pesq_mode: PESQ mode (default: 0)
#         config_to_run: Which config to run ("all", "with_refinement", "without_refinement", "dynamic", "baseline")
#     """
    
#     # Base command with all common arguments
#     base_command = (
#         f"python evaluate.py --model_path \"{model_path}\" "
#         f"--test_audio_dir \"{test_audio_dir}\" "
#         f"--output_dir \"{output_dir}\" "
#         f"--target_fs_values {target_fs_values} "
#         f"--input_sdrs {' '.join(str(x) for x in input_sdrs)} "
#         f"--time_clip {time_clip} "
#         f"--factor {factor} "
#         f"--save {save} "
#         f"--delta {delta} "
#         f"--c_win {c_win} "
#         f"--verbose {verbose} "
#         f"--sdr_mode {sdr_mode} "
#         f"--pesq_mode {pesq_mode} "
#         f"--mask_size {mask_size} "
#         f"--max_sparsity {max_sparsity} "
#         f"--n_files {n_files} "
#     )
    
#     # Dictionary of all available configurations
#     all_configs = {
#         "with_refinement": (1, 0, 0, "with_refinement"),     # ML model with refinement
#         "without_refinement": (1, 0, 1, "without_refinement"), # ML model without refinement
#         "dynamic": (0, 1, 0, ""),                            # Dynamic model
#         "baseline": (0, 0, 0, ""),                           # Baseline model
#     }
    
#     # Determine which configs to run
#     if config_to_run == "all":
#         configs_to_run = all_configs
#         print("Running all configurations...")
#     elif config_to_run in all_configs:
#         configs_to_run = {config_to_run: all_configs[config_to_run]}
#         print(f"Running single configuration: {config_to_run}")
#     else:
#         print(f"Error: Unknown configuration '{config_to_run}'")
#         print(f"Available configurations: {list(all_configs.keys())} or 'all'")
#         return
    
#     # Run selected configurations
#     for config_name, (eval_mode, dynamic, r_mode, exp_name) in configs_to_run.items():
#         command = f"{base_command} --eval_mode {eval_mode} --dynamic {dynamic} --r_mode {r_mode}"
        
#         # Add exp_name only if it's not empty
#         if exp_name:
#             command += f" --exp_name \"{exp_name}\""
        
#         print(f"\n{'='*60}")
#         print(f"Running configuration: {config_name}")
#         print(f"Parameters: eval_mode={eval_mode}, dynamic={dynamic}, r_mode={r_mode}, exp_name='{exp_name}'")
#         print(f"{'='*60}")
        
#         subprocess.run(command, shell=True)
#         print(f"Completed configuration: {config_name}")


# def main():
#     parser = argparse.ArgumentParser(description='Run evaluations with different configurations')
    
#     # Required arguments
#     parser.add_argument('--model_path', required=True, type=str,
#                        help='Path to the trained model')
#     parser.add_argument('--test_audio_dir', required=True, type=str,
#                        help='Directory containing test audio files')
#     parser.add_argument('--output_dir', required=True, type=str,
#                        help='Output directory for results')
#     parser.add_argument('--target_fs_values', required=True, type=int,
#                        help='Target sampling frequency')
#     parser.add_argument("--input_sdrs", type=int, nargs="+", default=[1],
#                         help="List of input sdrs.")
#     parser.add_argument('--time_clip', required=True, type=int,
#                        help='Time clipping parameter')
#     parser.add_argument('--factor', required=True, type=float,
#                        help='Factor parameter')
#     parser.add_argument('--save', required=True, type=int, choices=[0, 1],
#                        help='Save parameter (0 or 1)')
#     parser.add_argument('--delta', required=True, type=int,
#                        help='Delta parameter')
#     parser.add_argument('--c_win', required=True, type=int,
#                        help='Window size parameter')
#     parser.add_argument('--mask_size', required=True, type=int,
#                        help='Mask size parameter')
#     parser.add_argument('--max_sparsity', required=True, type=int,
#                        help='Maximum sparsity parameter')
#     parser.add_argument('--n_files', required=True, type=int,
#                        help='Total files to process')

#     # Optional arguments with defaults
#     parser.add_argument('--verbose', type=int, default=0,
#                        help='Verbose mode (default: 0)')
#     parser.add_argument('--sdr_mode', type=int, default=1,
#                        help='SDR mode (default: 1)')
#     parser.add_argument('--pesq_mode', type=int, default=0,
#                        help='PESQ mode (default: 0)')
    
#     # New argument for config selection
#     parser.add_argument('--config', type=str, default="all",
#                        choices=["all", "with_refinement", "without_refinement", "dynamic", "baseline"],
#                        help='Configuration to run: "all" for all configs, or specific config name (default: all)')
    
#     args = parser.parse_args()
    
#     # Call the function with parsed arguments
#     run_evaluations(
#         model_path=args.model_path,
#         test_audio_dir=args.test_audio_dir,
#         output_dir=args.output_dir,
#         target_fs_values=args.target_fs_values,
#         input_sdrs=args.input_sdrs,
#         time_clip=args.time_clip,
#         factor=args.factor,
#         save=args.save,
#         delta=args.delta,
#         c_win=args.c_win,
#         mask_size=args.mask_size,
#         max_sparsity=args.max_sparsity,
#         n_files=args.n_files,
#         verbose=args.verbose,
#         sdr_mode=args.sdr_mode,
#         pesq_mode=args.pesq_mode,
#         config_to_run=args.config
#     )

# if __name__ == "__main__":
#     main()


import os
import sys
import argparse
import subprocess

def run_evaluations(model_path, test_audio_dir, output_dir, target_fs_values, 
                   input_sdrs, time_clip, factor, save, delta, c_win, 
                   mask_size, max_sparsity, n_files, verbose=0, sdr_mode=1, pesq_mode=0,
                   config_to_run="all"):
    """
    Run evaluations with different configurations
    
    Args:
        model_path: Path to the trained model
        test_audio_dir: Directory containing test audio files
        output_dir: Output directory for results
        target_fs_values: Target sampling frequency
        input_sdrs: Input SDR values
        time_clip: Time clipping parameter
        factor: Factor parameter
        save: Save parameter (0 or 1)
        delta: Delta parameter
        c_win: Window size parameter
        mask_size: Mask size parameter
        max_sparsity: Maximum sparsity parameter
        n_files: Number of files to process
        verbose: Verbose mode (default: 0)
        sdr_mode: SDR mode (default: 1)
        pesq_mode: PESQ mode (default: 0)
        config_to_run: Which config to run ("all", "with_refinement", "without_refinement", "dynamic", "baseline")
    """
    
    # Base command with all common arguments
    base_command = (
        f"python evaluate.py --model_path \"{model_path}\" "
        f"--test_audio_dir \"{test_audio_dir}\" "
        f"--output_dir \"{output_dir}\" "
        f"--target_fs_values {target_fs_values} "
        f"--input_sdrs {' '.join(str(x) for x in input_sdrs)} "
        f"--time_clip {time_clip} "
        f"--factor {factor} "
        f"--save {save} "
        f"--delta {delta} "
        f"--c_win {c_win} "
        f"--verbose {verbose} "
        f"--sdr_mode {sdr_mode} "
        f"--pesq_mode {pesq_mode} "
        f"--mask_size {mask_size} "
        f"--max_sparsity {max_sparsity} "
        f"--n_files {n_files} "
    )
    
    # Dictionary of all available configurations
    all_configs = {
        "with_refinement": (1, 0, 0, "with_refinement"),     # ML model with refinement
        "without_refinement": (1, 0, 1, "without_refinement"), # ML model without refinement
        "dynamic": (0, 1, 0, ""),                            # Dynamic model
        "baseline": (0, 0, 0, ""),                           # Baseline model
    }
    
    # Determine which configs to run
    if config_to_run == "all":
        configs_to_run = all_configs
        print("Running all configurations...", flush=True)
    elif config_to_run in all_configs:
        configs_to_run = {config_to_run: all_configs[config_to_run]}
        print(f"Running single configuration: {config_to_run}", flush=True)
    else:
        print(f"Error: Unknown configuration '{config_to_run}'", flush=True)
        print(f"Available configurations: {list(all_configs.keys())} or 'all'", flush=True)
        return
    
    # Run selected configurations
    for config_name, (eval_mode, dynamic, r_mode, exp_name) in configs_to_run.items():
        command = f"{base_command} --eval_mode {eval_mode} --dynamic {dynamic} --r_mode {r_mode}"
        
        # Add exp_name only if it's not empty
        if exp_name:
            command += f" --exp_name \"{exp_name}\""
        
        print(f"\n{'='*60}", flush=True)
        print(f"Running configuration: {config_name}", flush=True)
        print(f"Parameters: eval_mode={eval_mode}, dynamic={dynamic}, r_mode={r_mode}, exp_name='{exp_name}'", flush=True)
        print(f"{'='*60}", flush=True)
        
        # Run subprocess with unbuffered output
        result = subprocess.run(command, shell=True, stdout=sys.stdout, stderr=sys.stderr)
        print(f"Completed configuration: {config_name} (exit code: {result.returncode})", flush=True)


def main():
    # Force line buffering for better output visibility with nohup
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    parser = argparse.ArgumentParser(description='Run evaluations with different configurations')
    
    # Required arguments
    parser.add_argument('--model_path', required=True, type=str,
                       help='Path to the trained model')
    parser.add_argument('--test_audio_dir', required=True, type=str,
                       help='Directory containing test audio files')
    parser.add_argument('--output_dir', required=True, type=str,
                       help='Output directory for results')
    parser.add_argument('--target_fs_values', required=True, type=int,
                       help='Target sampling frequency')
    parser.add_argument("--input_sdrs", type=int, nargs="+", default=[1],
                        help="List of input sdrs.")
    parser.add_argument('--time_clip', required=True, type=int,
                       help='Time clipping parameter')
    parser.add_argument('--factor', required=True, type=float,
                       help='Factor parameter')
    parser.add_argument('--save', required=True, type=int, choices=[0, 1],
                       help='Save parameter (0 or 1)')
    parser.add_argument('--delta', required=True, type=int,
                       help='Delta parameter')
    parser.add_argument('--c_win', required=True, type=int,
                       help='Window size parameter')
    parser.add_argument('--mask_size', required=True, type=int,
                       help='Mask size parameter')
    parser.add_argument('--max_sparsity', required=True, type=int,
                       help='Maximum sparsity parameter')
    parser.add_argument('--n_files', required=True, type=int,
                       help='Total files to process')

    # Optional arguments with defaults
    parser.add_argument('--verbose', type=int, default=0,
                       help='Verbose mode (default: 0)')
    parser.add_argument('--sdr_mode', type=int, default=1,
                       help='SDR mode (default: 1)')
    parser.add_argument('--pesq_mode', type=int, default=0,
                       help='PESQ mode (default: 0)')
    
    # New argument for config selection
    parser.add_argument('--config', type=str, default="all",
                       choices=["all", "with_refinement", "without_refinement", "dynamic", "baseline"],
                       help='Configuration to run: "all" for all configs, or specific config name (default: all)')
    
    args = parser.parse_args()
    
    print(f"Starting evaluation script at {os.getcwd()}", flush=True)
    print(f"Python version: {sys.version}", flush=True)
    
    # Call the function with parsed arguments
    run_evaluations(
        model_path=args.model_path,
        test_audio_dir=args.test_audio_dir,
        output_dir=args.output_dir,
        target_fs_values=args.target_fs_values,
        input_sdrs=args.input_sdrs,
        time_clip=args.time_clip,
        factor=args.factor,
        save=args.save,
        delta=args.delta,
        c_win=args.c_win,
        mask_size=args.mask_size,
        max_sparsity=args.max_sparsity,
        n_files=args.n_files,
        verbose=args.verbose,
        sdr_mode=args.sdr_mode,
        pesq_mode=args.pesq_mode,
        config_to_run=args.config
    )
    
    print("Script completed successfully!", flush=True)

if __name__ == "__main__":
    main()