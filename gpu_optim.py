import subprocess
import time

# Training commands for each fold
commands = [
    'python training.py --pkl_path "exp_ml/speech_sound/kfold_data/fold_1/training_data.pkl" --epochs 100 --batch_size 16 --save_path "exp_ml/speech_sound/kfold_data/fold_1/saved_models_batch16" --plot_path "exp_ml/speech_sound/kfold_data/fold_1/loss_plots" --checkpoint_freq 50 --val_split 0.2 --resume --val --max_sparsity 1024 --mask_size 1024',
    'python training.py --pkl_path "exp_ml/speech_sound/kfold_data/fold_2/training_data.pkl" --epochs 100 --batch_size 16 --save_path "exp_ml/speech_sound/kfold_data/fold_2/saved_models_batch16" --plot_path "exp_ml/speech_sound/kfold_data/fold_2/loss_plots" --checkpoint_freq 50 --val_split 0.2 --resume --val --max_sparsity 1024 --mask_size 1024',
    'python training.py --pkl_path "exp_ml/speech_sound/kfold_data/fold_3/training_data.pkl" --epochs 100 --batch_size 16 --save_path "exp_ml/speech_sound/kfold_data/fold_3/saved_models_batch16" --plot_path "exp_ml/speech_sound/kfold_data/fold_3/loss_plots" --checkpoint_freq 50 --val_split 0.2 --resume --val --max_sparsity 1024 --mask_size 1024',
    'python training.py --pkl_path "exp_ml/speech_sound/kfold_data/fold_4/training_data.pkl" --epochs 100 --batch_size 16 --save_path "exp_ml/speech_sound/kfold_data/fold_4/saved_models_batch16" --plot_path "exp_ml/speech_sound/kfold_data/fold_4/loss_plots" --checkpoint_freq 50 --val_split 0.2 --resume --val --max_sparsity 1024 --mask_size 1024',
    'python training.py --pkl_path "exp_ml/speech_sound/kfold_data/fold_5/training_data.pkl" --epochs 100 --batch_size 16 --save_path "exp_ml/speech_sound/kfold_data/fold_5/saved_models_batch16" --plot_path "exp_ml/speech_sound/kfold_data/fold_5/loss_plots" --checkpoint_freq 50 --val_split 0.2 --resume --val --max_sparsity 1024 --mask_size 1024',
]

# Number of parallel jobs to run (tune to fit your memory, 1 or 2 is safe)
max_parallel = 2

processes = []

for i, cmd in enumerate(commands):
    print(f"[INFO] Starting Fold {i+1} ...")
    log_file = f"logs/training_speech_16_fold{i+1}.log"
    with open(log_file, "w") as log:
        p = subprocess.Popen(cmd, shell=True, stdout=log, stderr=log)
        processes.append(p)

    # Wait if max_parallel reached or last command
    if len(processes) == max_parallel or i == len(commands) - 1:
        for p in processes:
            p.wait()
        processes = []

    # Optional: sleep a bit to let memory clear
    time.sleep(10)

print("[DONE] All training processes completed.")
