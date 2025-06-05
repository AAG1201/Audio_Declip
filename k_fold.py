import os
import shutil
import random
import argparse
from math import ceil

def k_fold_split(audio_dir, output_dir, k=5, cnt=None):
    """
    Perform K-Fold cross-validation split on audio files.

    Parameters:
    -----------
    audio_dir : str
        Path to the directory containing audio files.
    output_dir : str
        Base directory where k-fold train/test directories will be saved.
    k : int
        Number of folds.
    cnt : int or None
        Number of files to sample randomly (optional).
    """
    # Get all .wav files
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

    # Sample if needed
    if cnt is not None and cnt < len(wav_files):
        wav_files = random.sample(wav_files, cnt)
    
    # Shuffle files
    random.shuffle(wav_files)
    
    # Divide into k roughly equal chunks
    fold_size = ceil(len(wav_files) / k)
    folds = [wav_files[i*fold_size : (i+1)*fold_size] for i in range(k)]

    # Generate k train/test splits
    for i in range(k):
        test_files = folds[i]
        train_files = [f for j in range(k) if j != i for f in folds[j]]

        train_dir = os.path.join(output_dir, f"fold_{i+1}", "train")
        test_dir = os.path.join(output_dir, f"fold_{i+1}", "test")
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Copy files
        for f in train_files:
            shutil.copy2(os.path.join(audio_dir, f), os.path.join(train_dir, f))
        for f in test_files:
            shutil.copy2(os.path.join(audio_dir, f), os.path.join(test_dir, f))

        print(f"Fold {i+1}: {len(train_files)} train files, {len(test_files)} test files")

def main():
    parser = argparse.ArgumentParser(description="K-Fold split for audio data")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to audio directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save k-fold splits")
    parser.add_argument("--k", type=int, default=5, help="Number of folds")
    parser.add_argument("--cnt", type=int, default=None, help="Number of files to sample")

    args = parser.parse_args()

    # Clear output directory
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)

    k_fold_split(args.audio_dir, args.output_dir, args.k, args.cnt)

if __name__ == "__main__":
    main()
