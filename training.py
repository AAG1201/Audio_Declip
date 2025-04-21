import argparse
import pickle
import torch
import os
import json
from torch.utils.data import DataLoader, random_split
from pipeline import (
    ComplexDFTDataset,
    ComplexDFTUNet,
    train_complex_dft_unet,
    prepare_training_data_with_masks
)
import sys
sys.stdout.reconfigure(line_buffering=True)

def main(args):
    # Load training data
    with open(args.pkl_path, "rb") as f:
        data = pickle.load(f)

    # Prepare inputs, masks, targets
    inputs, masks, targets_dft, targets_sparsity = prepare_training_data_with_masks(data)

    # Create dataset and loader
    train_dataset = ComplexDFTDataset(inputs, targets_dft, masks, targets_sparsity, max_sparsity=args.max_sparsity)
    
    if args.val:
        # Determine validation split
        dataset_size = len(train_dataset)
        val_size = int(dataset_size * args.val_split)
        train_size = dataset_size - val_size
        
        # Split the dataset
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
        
        print(f"Training samples: {len(train_subset)}, Validation samples: {len(val_subset)}")
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = None
        print(f"Training samples: {len(train_dataset)}, No validation")

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = ComplexDFTUNet(dft_size=args.dft_size, mask_channels=args.mask_channels, max_sparsity=args.max_sparsity)
    model = model.to(device)
    
    # Initialize training parameters
    start_epoch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    history = {'total_loss': [], 'dft_loss': [], 'sparsity_loss': []}
    best_loss = float('inf')
    
    # Check for checkpoint and load if exists
    checkpoint_path = os.path.join(args.save_path, "checkpoint.pt")
    history_path = os.path.join(args.save_path, "loss_history.json")
    
    # In the checkpoint loading code
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        # Load checkpoint to CPU first to avoid device mismatch
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Load model state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)  # Move model to device after loading state dict
        
        # Create a fresh optimizer and load its state
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        optimizer_state = checkpoint['optimizer_state_dict']
        
        # Manually move optimizer state to the correct device
        for state in optimizer_state['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    
        optimizer.load_state_dict(optimizer_state)
        
        # Create a fresh scheduler and load its state if available
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        
        # Load history, ensuring validation keys exist if needed
        history = checkpoint['history']
        if args.val:
            # Make sure validation keys exist in the history
            for key in ['val_total_loss', 'val_dft_loss', 'val_sparsity_loss']:
                if key not in history:
                    history[key] = []
        
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Resuming training from epoch {start_epoch}")
        
        # Check if loss history file exists and load it if available
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    saved_history = json.load(f)
                # Update the history with saved values (in case they're more comprehensive)
                for key in saved_history:
                    if key in history and len(saved_history[key]) >= len(history[key]):
                        history[key] = saved_history[key]
                print(f"Loaded loss history from {history_path}")
            except Exception as e:
                print(f"Error loading loss history file: {e}")

    # Create directories if they don't exist
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.plot_path, exist_ok=True)

    # Train
    loss_history = train_complex_dft_unet(
        model,
        train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        start_epoch=start_epoch,
        optimizer=optimizer,
        scheduler=scheduler,
        history=history,
        best_loss=best_loss,
        save_path=args.save_path,
        plot_path=args.plot_path,
        checkpoint_freq=args.checkpoint_freq,
        lr=args.learning_rate
    )
    
    # Save the loss history to a separate JSON file for easy plotting later
    history_path = os.path.join(args.save_path, "loss_history.json")
    try:
        with open(history_path, 'w') as f:
            json.dump(loss_history, f, indent=4)
        print(f"Saved loss history to {history_path}")
    except Exception as e:
        print(f"Error saving loss history file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ComplexDFTUNet on prepared data with checkpoint support")
    parser.add_argument("--pkl_path", type=str, default="pkl_data/training_data.pkl", help="Path to the training data pickle file")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--save_path", type=str, default="saved_models", help="Directory to save trained models")
    parser.add_argument("--dft_size", type=int, default=500, help="Size of the DFT (input signal length / 2)")
    parser.add_argument("--mask_channels", type=int, default=3, help="Number of mask input channels")
    parser.add_argument("--max_sparsity", type=int, default=250, help="Maximum expected sparsity of input")
    parser.add_argument("--plot_path", type=str, default="loss_plots", help="Path to save loss history plots")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--checkpoint_freq", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint if available")
    parser.add_argument("--val", action="store_true", help="Use a validation set for training")
    parser.add_argument("--val_split", type=float, default=0.1, help="Proportion of data to use for validation (default: 0.1)")

    args = parser.parse_args()
    main(args)