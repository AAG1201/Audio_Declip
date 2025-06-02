import os
import argparse

def run_evaluations(model_path, test_audio_dir, output_dir, target_fs_values, 
                   input_sdrs, time_clip, factor, save, delta, c_win, 
                   mask_size, max_sparsity, n_files, verbose=0, sdr_mode=1, pesq_mode=0):
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
        verbose: Verbose mode (default: 0)
        sdr_mode: SDR mode (default: 1)
        pesq_mode: PESQ mode (default: 0)
    """
    
    # Base command with all common arguments
    base_command = (
        f"python evaluate.py --model_path \"{model_path}\" "
        f"--test_audio_dir \"{test_audio_dir}\" "
        f"--output_dir \"{output_dir}\" "
        f"--target_fs_values {target_fs_values} "
        f"--input_sdrs {input_sdrs} "
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
    
    # List of (eval_mode, dynamic, r_mode, exp_name) configurations
    configs = [
        (1, 0, 0, "with_refinement"),  # ML model with refinement
        (1, 0, 1, ""),                 # ML model
        (0, 1, 0, ""),                 # Dynamic model
        (0, 0, 0, ""),                 # Baseline model
    ]
    
    # Run all configurations
    for eval_mode, dynamic, r_mode, exp_name in configs:
        command = f"{base_command} --eval_mode {eval_mode} --dynamic {dynamic} --r_mode {r_mode}"
        
        # Add exp_name only if it's not empty
        if exp_name:
            command += f" --exp_name \"{exp_name}\""
        
        print(f"\nRunning: eval_mode {eval_mode} dynamic {dynamic} r_mode {r_mode} exp_name \"{exp_name}\" \n")
        os.system(command)



def main():
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
    parser.add_argument('--input_sdrs', required=True, type=int,
                       help='Input SDR values')
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
    parser.add_argument('--sdr_mode', type=int, default=0,
                       help='SDR mode (default: 1)')
    parser.add_argument('--pesq_mode', type=int, default=0,
                       help='PESQ mode (default: 0)')
    
    args = parser.parse_args()
    
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
        pesq_mode=args.pesq_mode
    )

if __name__ == "__main__":
    main()