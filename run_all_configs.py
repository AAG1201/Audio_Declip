import os

# Common arguments
base_command = (
    "python evaluate.py "
    "--model_path 'saved_models/final/complex_dft_unet_final.pth' "
    "--test_audio_dir 'test_data' "
    "--output_dir 'custom_output' "
    "--target_fs_values 16000 "
    "--clipping_thresholds 0.1 0.3 "
    "--time_clip 1 "
    "--factor 0.2 "
    "--save 0 "
    "--delta 300 "
)

# List of (eval_mode, dynamic) configurations
configs = [
    (1, 0),  # ML model
    (0, 1),  # Dynamic model
    (0, 0),  # Baseline model
]

# # List of (eval_mode, dynamic) configurations
# configs = [
#     (1, 0)
# ]

# Run all configurations
for eval_mode, dynamic in configs:
    command = f"{base_command} --eval_mode {eval_mode} --dynamic {dynamic}"
    print(f"\nRunning: {command}\n")
    os.system(command)
