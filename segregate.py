import os
import shutil
import random
import argparse

def segregate_data(audio_dir, train_dir, test_dir, cnt=None, s_ratio=0.9):
    """
    Segregate audio files into training and test sets
    
    Parameters:
    -----------
    audio_dir : str
        Path to directory containing audio files
    train_dir : str
        Path to output training directory
    test_dir : str
        Path to output test directory
    cnt : int, optional
        Number of files to use (sample randomly if less than total available)
    s_ratio : float, default=0.9
        Split ratio for train/test (e.g., 0.9 means 90% train, 10% test)
    """
    # Create output directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get list of all .wav files
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    
    # Sample random subset if cnt is specified
    if cnt is not None and cnt < len(wav_files):
        wav_files = random.sample(wav_files, cnt)
    
    # Shuffle files randomly
    random.shuffle(wav_files)
    
    # Split based on ratio
    split_idx = int(s_ratio * len(wav_files))
    train_files = wav_files[:split_idx]
    test_files = wav_files[split_idx:]
    
    # Copy files to respective directories
    for f in train_files:
        shutil.copy2(os.path.join(audio_dir, f), os.path.join(train_dir, f))
    
    for f in test_files:
        shutil.copy2(os.path.join(audio_dir, f), os.path.join(test_dir, f))
    
    print(f"Copied {len(train_files)} files to {train_dir}")
    print(f"Copied {len(test_files)} files to {test_dir}")
    
    return train_files, test_files

def main():
    parser = argparse.ArgumentParser(description="Segregate audio files into train and test sets")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to audio directory")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to train directory")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test directory")
    parser.add_argument("--cnt", type=int, default=None, help="Number of files to use (sample randomly if less than total available)")
    parser.add_argument("--s_ratio", type=float, default=0.9, help="Split ratio (e.g., 0.9 means 90% train, 10% test)")
    
    args = parser.parse_args()
    
    # Delete existing train and test directories if they exist
    for dir_path in [args.train_dir, args.test_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)
    
    # Perform the segregation
    segregate_data(args.audio_dir, args.train_dir, args.test_dir, args.cnt, args.s_ratio)

if __name__ == "__main__":
    main()