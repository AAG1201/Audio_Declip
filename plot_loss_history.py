import matplotlib.pyplot as plt
import json
import os
import argparse

def plot_loss_history(history_file, output_path):
    """
    Generate beautiful plots from saved loss history
    
    Args:
        history_file: Path to the saved loss history JSON file
        output_path: Directory to save the generated plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Load history data
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Set style parameters
    plt.style.use('seaborn-v0_8-whitegrid')
    line_width = 2.5
    font_size = 14
    title_size = 18
    legend_size = 12
    tick_size = 12
    
    # Create colors with better contrast
    train_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    val_colors = ['#d62728', '#9467bd', '#8c564b']    # Red, Purple, Brown
    
    # 1. Plot training losses
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Total Loss subplot
    axes[0].plot(range(1, len(history['total_loss']) + 1), history['total_loss'], color=train_colors[0], linewidth=line_width, marker='o', markersize=4)
    axes[0].set_title("Total Training Loss", fontsize=title_size, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=font_size)
    axes[0].set_ylabel("Loss", fontsize=font_size)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='both', labelsize=tick_size)
    
    # DFT Loss subplot
    axes[1].plot(range(1, len(history['total_loss']) + 1), history['dft_loss'], color=train_colors[1], linewidth=line_width, marker='o', markersize=4)
    axes[1].set_title("DFT Training Loss", fontsize=title_size, fontweight='bold')
    axes[1].set_xlabel("Epoch", fontsize=font_size)
    axes[1].set_ylabel("Loss", fontsize=font_size)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='both', labelsize=tick_size)
        
    # Sparsity Loss subplot
    axes[2].plot(range(1, len(history['total_loss']) + 1), history['sparsity_loss'], color=train_colors[2], linewidth=line_width, marker='o', markersize=4)
    axes[2].set_title("Sparsity Training Loss", fontsize=title_size, fontweight='bold')
    axes[2].set_xlabel("Epoch", fontsize=font_size)
    axes[2].set_ylabel("Loss", fontsize=font_size)
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(axis='both', labelsize=tick_size)
    
    # Adjust layout and save
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_path, "training_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot validation losses (if available)
    if 'val_total_loss' in history and len(history['val_total_loss']) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Total Loss subplot
        axes[0].plot(range(1, len(history['total_loss']) + 1), history['val_total_loss'], color=val_colors[0], linewidth=line_width, marker='o', markersize=4)
        axes[0].set_title("Total Validation Loss", fontsize=title_size, fontweight='bold')
        axes[0].set_xlabel("Epoch", fontsize=font_size)
        axes[0].set_ylabel("Loss", fontsize=font_size)
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='both', labelsize=tick_size)
        
        # DFT Loss subplot
        axes[1].plot(range(1, len(history['total_loss']) + 1), history['val_dft_loss'], color=val_colors[1], linewidth=line_width, marker='o', markersize=4)
        axes[1].set_title("DFT Validation Loss", fontsize=title_size, fontweight='bold')
        axes[1].set_xlabel("Epoch", fontsize=font_size)
        axes[1].set_ylabel("Loss", fontsize=font_size)
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='both', labelsize=tick_size)
        
        # Sparsity Loss subplot
        axes[2].plot(range(1, len(history['total_loss']) + 1), history['val_sparsity_loss'], color=val_colors[2], linewidth=line_width, marker='o', markersize=4)
        axes[2].set_title("Sparsity Validation Loss", fontsize=title_size, fontweight='bold')
        axes[2].set_xlabel("Epoch", fontsize=font_size)
        axes[2].set_ylabel("Loss", fontsize=font_size)
        axes[2].grid(True, alpha=0.3)
        axes[2].tick_params(axis='both', labelsize=tick_size)
        
        # Adjust layout and save
        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(output_path, "validation_loss.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Plot comparison (train vs validation)
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Total Loss subplot
        axes[0].plot(range(1, len(history['total_loss']) + 1), history['total_loss'], label='Training', color=train_colors[0], linewidth=line_width, marker='o', markersize=4)
        axes[0].plot(history['val_total_loss'], label='Validation', color=val_colors[0], linewidth=line_width, marker='s', markersize=4)
        axes[0].set_title("Total Loss Comparison", fontsize=title_size, fontweight='bold')
        axes[0].set_xlabel("Epoch", fontsize=font_size)
        axes[0].set_ylabel("Loss", fontsize=font_size)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=legend_size)
        axes[0].tick_params(axis='both', labelsize=tick_size)
        
        # DFT Loss subplot
        axes[1].plot(range(1, len(history['total_loss']) + 1), history['dft_loss'], label='Training', color=train_colors[1], linewidth=line_width, marker='o', markersize=4)
        axes[1].plot(history['val_dft_loss'], label='Validation', color=val_colors[1], linewidth=line_width, marker='s', markersize=4)
        axes[1].set_title("DFT Loss Comparison", fontsize=title_size, fontweight='bold')
        axes[1].set_xlabel("Epoch", fontsize=font_size)
        axes[1].set_ylabel("Loss", fontsize=font_size)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=legend_size)
        axes[1].tick_params(axis='both', labelsize=tick_size)
        
        # Sparsity Loss subplot
        axes[2].plot(range(1, len(history['total_loss']) + 1), history['sparsity_loss'], label='Training', color=train_colors[2], linewidth=line_width, marker='o', markersize=4)
        axes[2].plot(history['val_sparsity_loss'], label='Validation', color=val_colors[2], linewidth=line_width, marker='s', markersize=4)
        axes[2].set_title("Sparsity Loss Comparison", fontsize=title_size, fontweight='bold')
        axes[2].set_xlabel("Epoch", fontsize=font_size)
        axes[2].set_ylabel("Loss", fontsize=font_size)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=legend_size)
        axes[2].tick_params(axis='both', labelsize=tick_size)
        
        # Adjust layout and save
        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(output_path, "comparison_loss.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. One consolidated plot showing all losses
    plt.figure(figsize=(12, 8))
    
    # Training losses
    plt.plot(history['total_loss'], label='Total (Train)', color=train_colors[0], linewidth=line_width, marker='o', markersize=4)
    plt.plot(history['dft_loss'], label='DFT (Train)', color=train_colors[1], linewidth=line_width, marker='o', markersize=4)
    plt.plot(history['sparsity_loss'], label='Sparsity (Train)', color=train_colors[2], linewidth=line_width, marker='o', markersize=4)
    
    # Validation losses if available
    if 'val_total_loss' in history and len(history['val_total_loss']) > 0:
        plt.plot(history['val_total_loss'], label='Total (Val)', color=val_colors[0], linewidth=line_width, linestyle='--', marker='s', markersize=4)
        plt.plot(history['val_dft_loss'], label='DFT (Val)', color=val_colors[1], linewidth=line_width, linestyle='--', marker='s', markersize=4)
        plt.plot(history['val_sparsity_loss'], label='Sparsity (Val)', color=val_colors[2], linewidth=line_width, linestyle='--', marker='s', markersize=4)
    
    plt.title("All Loss Metrics", fontsize=title_size, fontweight='bold')
    plt.xlabel("Epoch", fontsize=font_size)
    plt.ylabel("Loss", fontsize=font_size)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=legend_size)
    plt.tick_params(axis='both', labelsize=tick_size)
    
    # Add smooth trend lines
    try:
        import numpy as np
        from scipy.signal import savgol_filter
        
        # Apply Savitzky-Golay filter to smooth the curves for trend visualization
        # Only do this if we have enough data points (at least 7)
        if len(history['total_loss']) >= 7:
            window_size = min(7, len(history['total_loss']) // 2 * 2 - 1)  # Must be odd
            if window_size >= 3:
                x = np.arange(len(history['total_loss']))
                
                # Add trend lines for each loss
                for i, key in enumerate(['total_loss', 'dft_loss', 'sparsity_loss']):
                    if len(history[key]) > 0:
                        y_smooth = savgol_filter(history[key], window_size, 3)
                        plt.plot(x, y_smooth, color=train_colors[i], alpha=0.5, linewidth=line_width*1.5)
                
                # Add trend lines for validation losses if available
                if 'val_total_loss' in history and len(history['val_total_loss']) >= window_size:
                    for i, key in enumerate(['val_total_loss', 'val_dft_loss', 'val_sparsity_loss']):
                        if len(history[key]) > 0:
                            y_smooth = savgol_filter(history[key], window_size, 3)
                            plt.plot(x[:len(y_smooth)], y_smooth, color=val_colors[i], alpha=0.5, linewidth=line_width*1.5)
    except ImportError:
        # Skip trend lines if scipy is not available
        pass
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "all_losses.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Log scale plot for better visualization of small changes
    plt.figure(figsize=(12, 8))
    
    # Training losses
    plt.semilogy(history['total_loss'], label='Total (Train)', color=train_colors[0], linewidth=line_width, marker='o', markersize=4)
    plt.semilogy(history['dft_loss'], label='DFT (Train)', color=train_colors[1], linewidth=line_width, marker='o', markersize=4)
    plt.semilogy(history['sparsity_loss'], label='Sparsity (Train)', color=train_colors[2], linewidth=line_width, marker='o', markersize=4)
    
    # Validation losses if available
    if 'val_total_loss' in history and len(history['val_total_loss']) > 0:
        plt.semilogy(history['val_total_loss'], label='Total (Val)', color=val_colors[0], linewidth=line_width, linestyle='--', marker='s', markersize=4)
        plt.semilogy(history['val_dft_loss'], label='DFT (Val)', color=val_colors[1], linewidth=line_width, linestyle='--', marker='s', markersize=4)
        plt.semilogy(history['val_sparsity_loss'], label='Sparsity (Val)', color=val_colors[2], linewidth=line_width, linestyle='--', marker='s', markersize=4)
    
    plt.title("All Loss Metrics (Log Scale)", fontsize=title_size, fontweight='bold')
    plt.xlabel("Epoch", fontsize=font_size)
    plt.ylabel("Loss (log scale)", fontsize=font_size)
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=legend_size)
    plt.tick_params(axis='both', labelsize=tick_size)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "all_losses_log_scale.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate beautiful plots from saved loss history")
    parser.add_argument("--history_file", type=str, required=True, help="Path to the loss history JSON file")
    parser.add_argument("--output_path", type=str, default="loss_plots", help="Directory to save the generated plots")
    
    args = parser.parse_args()
    plot_loss_history(args.history_file, args.output_path)