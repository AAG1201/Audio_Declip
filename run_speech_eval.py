import subprocess
from exp_plots import plot_comparison_all_models_speech

factors = [0.8, 0.6, 0.4]

for factor in factors:
    print(f"\n--- Running evaluation for factor = {factor} ---\n")

    # Base command
    cmd = [
        "python", "run_all_configs.py",
        "--model_path", "exp_ml/speech_sound/saved_models_batch16/final/complex_dft_unet_final.pth",
        "--test_audio_dir", "exp_ml/speech_sound/test_data",
        "--output_dir", "exp_ml/speech_sound",
        "--target_fs_values", "16000",
        "--input_sdrs", "1", "3", "5", "7",
        "--time_clip", "1",
        "--factor", str(factor),
        "--save", "0",
        "--delta", "300",
        "--c_win", "1024",
        "--verbose", "0",
        "--sdr_mode", "1",
        "--pesq_mode", "1",
        "--mask_size", "1024",
        "--max_sparsity", "1024",
        "--n_files", "20"
    ]

    # Add config for 0.6 and 0.4
    if factor in [0.6, 0.4]:
        cmd += ["--config", "with_refinement"]

    # Run evaluation
    subprocess.run(cmd, check=True)

    # Run plotting
    print(f"\n--- Plotting for factor = {factor} ---\n")
    plot_comparison_all_models_speech("exp_ml/speech_sound", 16, factor)
