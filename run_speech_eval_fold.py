import subprocess
from exp_plots import plot_comparison_all_models_speech

fold_no = [1, 2, 3, 4, 5]
factors = [0.3]

for factor in factors:
    print(f"\n--- Running evaluation for factor = {factor} ---\n")

    for fold in fold_no:
        model_path = f"exp_ml/speech_sound/kfold_data/fold_{fold}/saved_models_batch16/final/complex_dft_unet_final.pth"
        test_audio_dir = f"exp_ml/speech_sound/kfold_data/fold_{fold}/test"
        output_dir = f"exp_ml/speech_sound/kfold_data/unseen/fold_{fold}"

        cmd = [
            "python", "run_all_configs.py",
            "--model_path", model_path,
            "--test_audio_dir", test_audio_dir,
            "--output_dir", output_dir,
            "--target_fs_values", "16000",
            "--input_sdrs", "2", "4", "6", "8",
            "--time_clip", "4",
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

        # Optional config (uncomment if needed)
        # if factor in [0.3]:
        #     cmd += ["--config", "with_refinement"]

        # Run evaluation
        subprocess.run(cmd, check=True)

        # Run plotting
        print(f"\n--- Plotting for factor = {factor}, fold = {fold} ---\n")
        plot_comparison_all_models_speech(output_dir, 16, factor, 4.0)

