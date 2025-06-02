import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json


class FrequencyAwareAttention(nn.Module):
    """
    Frequency-Aware Attention module for complex DFT data.
    """
    def __init__(self, in_channels, reduction=8):
        super(FrequencyAwareAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, max(in_channels // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_channels // reduction, 8), in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, f = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class DoubleConv(nn.Module):
    """
    Double convolution block with frequency-aware attention.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.attention = FrequencyAwareAttention(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.attention(x)  # Apply frequency-aware attention
        x = self.relu2(x)
        
        return x

class Down(nn.Module):
    """
    Downscaling block with maxpool and double convolution.
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.double_conv(x)
        return x

class Up(nn.Module):
    """
    Upscaling block with transposed convolution and double convolution.
    """
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Adjust dimensions if needed
        diff = x2.size(2) - x1.size(2)
        if diff > 0:
            x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        
        x = torch.cat([x2, x1], dim=1)
        x = self.double_conv(x)
        return x

class ComplexDFTUNet(nn.Module):
    """
    U-Net architecture for complex DFT processing with frequency-aware attention.
    
    Parameters:
    -----------
    dft_size : int, options are 512, 1024, 2048
        Size of the DFT data (complex values)
    mask_size : int, options are 256, 512, 1024
        Size of the masks - can be different from dft_size
    mask_channels : int, default=3
        Number of mask channels
    max_sparsity : int, half of dft_size
        Maximum value for sparsity prediction
    """
    def __init__(self, dft_size, mask_size, mask_channels, max_sparsity):
        super(ComplexDFTUNet, self).__init__()
        self.dft_size = dft_size
        self.mask_size = mask_size
        self.max_sparsity = max_sparsity
        
        # Feature dimensions

        if dft_size <= 512:
            features = 32
        elif dft_size <= 1024:
            features = 48
        else:  # 2048
            features = 64

        
        # features = 32
        
        # For asymmetric case (dft_size=1024, mask_size=512):
        # We'll need to handle masks separately
        
        # Complex data processing branch
        self.complex_conv = nn.Conv1d(2, features // 2, kernel_size=3, padding=1)
        
        # Mask processing branch - handle size mismatch
        self.mask_conv = nn.Conv1d(mask_channels, features // 2, kernel_size=3, padding=1)
        
        # Encoder path
        self.inc = DoubleConv(features, features)  # features -> features
        self.down1 = Down(features, features * 2)  # features -> features*2
        self.down2 = Down(features * 2, features * 4)  # features*2 -> features*4
        self.down3 = Down(features * 4, features * 8)  # features*4 -> features*8
        
        # Middle frequency-aware attention
        self.mid_attention = FrequencyAwareAttention(features * 8)
        
        # Decoder path
        self.up1 = Up(features * 8, features * 4)  # features*8 -> features*4
        self.up2 = Up(features * 4, features * 2)  # features*4 -> features*2
        self.up3 = Up(features * 2, features)  # features*2 -> features
        
        # Output layers
        self.out_conv = nn.Conv1d(features, 2, kernel_size=1)  # features -> 2 (real, imag)
        
        # Sparsity prediction head
        self.sparsity_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, masks):
        # For dft_size=1024 and mask_size=512:
        # Input x is [batch, 2048] (1024 real + 1024 imag values flattened)
        # Reshape to [batch, 2, 1024]
        batch_size = x.size(0)
        x_real = x[:, :self.dft_size].view(batch_size, 1, self.dft_size)
        x_imag = x[:, self.dft_size:].view(batch_size, 1, self.dft_size)
        x_complex = torch.cat([x_real, x_imag], dim=1)  # [batch, 2, 1024]
        
        # Process complex data branch
        x_complex_features = self.complex_conv(x_complex)  # [batch, features//2, 1024]
        
        # Handle masks separately - masks are [batch, 3, 512]
        # First, process the masks through their own convolutional layer
        mask_features = self.mask_conv(masks)  # [batch, features//2, 512]
        
        # Now we need to upsample mask features to match dft_size
        if self.mask_size != self.dft_size:
            mask_features = F.interpolate(
                mask_features, 
                size=self.dft_size,
                mode='linear', 
                align_corners=False
            )  # [batch, features//2, 1024]
        
        # Concatenate the complex features and mask features along channel dimension
        x = torch.cat([x_complex_features, mask_features], dim=1)  # [batch, features, 1024]
        
        # Encoder path
        x1 = self.inc(x)          # [batch, features, 1024]
        x2 = self.down1(x1)       # [batch, features*2, 512]
        x3 = self.down2(x2)       # [batch, features*4, 256]
        x4 = self.down3(x3)       # [batch, features*8, 128]
        
        # Middle attention
        x4 = self.mid_attention(x4)
        
        # Decoder path with skip connections
        x = self.up1(x4, x3)      # [batch, features*4, 256]
        x = self.up2(x, x2)       # [batch, features*2, 512]
        x = self.up3(x, x1)       # [batch, features, 1024]
        
        # Output processing
        dft_out = self.out_conv(x)  # [batch, 2, 1024]
        # Reshape to [batch, 2048] - 1024 real followed by 1024 imaginary
        dft_out = dft_out.reshape(batch_size, -1)  # [batch, 2048]
        
        # Sparsity prediction
        sparsity = self.sparsity_head(x)  # [batch, 1]
        sparsity = torch.clamp(sparsity, 0, self.max_sparsity)
        
        return dft_out, sparsity.squeeze(1)

class ComplexDFTDataset(torch.utils.data.Dataset):
    """
    Dataset for Complex DFT training with masks and sparsity targets.
    """
    def __init__(self, inputs, targets_dft, masks, targets_sparsity, max_sparsity):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets_dft = torch.tensor(targets_dft, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)
        self.targets_sparsity = torch.tensor(targets_sparsity, dtype=torch.float32)
        self.max_sparsity = max_sparsity
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            'input': self.inputs[idx],
            'mask': self.masks[idx],
            'target_dft': self.targets_dft[idx],
            'target_sparsity': self.targets_sparsity[idx]
        }

def train_complex_dft_unet(model, train_loader, val_loader=None, device='cpu', epochs=100, start_epoch=0, 
                           optimizer=None, scheduler=None, history=None, best_loss=float('inf'),
                           save_path="saved_models", plot_path=None, 
                           checkpoint_freq=5, lr=0.001):
    """
    Train the complex DFT U-Net model with checkpoint support and validation.
    """

    os.makedirs(save_path, exist_ok=True)
    if plot_path:
        os.makedirs(plot_path, exist_ok=True)

    save_path_final = os.path.join(save_path, f"final")
    os.makedirs(save_path_final, exist_ok=True)

    model = model.to(device)  # Ensure model is on the correct device

    # Initialize optimizer and scheduler if not provided
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    if history is None:
        history = {
            'total_loss': [],
            'dft_loss': [],
            'sparsity_loss': []
        }
        
    # Add this code right after the history initialization:
    # Make sure validation keys exist if validation is enabled
    if val_loader is not None and history is not None:
        for key in ['val_total_loss', 'val_dft_loss', 'val_sparsity_loss']:
            if key not in history:
                history[key] = []

    dft_criterion = nn.MSELoss()
    sparsity_criterion = nn.MSELoss()

    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        epoch_dft_loss = 0
        epoch_sparsity_loss = 0

        for batch in train_loader:
            inputs = batch['input'].to(device)
            masks = batch['mask'].to(device)
            targets_dft = batch['target_dft'].to(device)
            targets_sparsity = batch['target_sparsity'].to(device)

            optimizer.zero_grad()
            pred_dft, pred_sparsity = model(inputs, masks)

            dft_loss = dft_criterion(pred_dft, targets_dft)
            sparsity_loss = sparsity_criterion(pred_sparsity, targets_sparsity / model.max_sparsity)

            total_loss = 1.0 * dft_loss + 0.5 * sparsity_loss
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_dft_loss += dft_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()

        avg_loss = epoch_loss / len(train_loader)
        avg_dft_loss = epoch_dft_loss / len(train_loader)
        avg_sparsity_loss = epoch_sparsity_loss / len(train_loader)

        # Validation phase
        val_loss = 0
        val_dft_loss = 0
        val_sparsity_loss = 0
        
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch['input'].to(device)
                    masks = batch['mask'].to(device)
                    targets_dft = batch['target_dft'].to(device)
                    targets_sparsity = batch['target_sparsity'].to(device)
                    
                    pred_dft, pred_sparsity = model(inputs, masks)
                    
                    dft_loss = dft_criterion(pred_dft, targets_dft)
                    sparsity_loss = sparsity_criterion(pred_sparsity, targets_sparsity / model.max_sparsity)
                    
                    total_loss = 1.0 * dft_loss + 0.5 * sparsity_loss
                    
                    val_loss += total_loss.item()
                    val_dft_loss += dft_loss.item()
                    val_sparsity_loss += sparsity_loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                avg_val_dft_loss = val_dft_loss / len(val_loader)
                avg_val_sparsity_loss = val_sparsity_loss / len(val_loader)
                
                # Use validation loss for scheduler if available
                scheduler.step(avg_val_loss if val_loader else avg_loss)
                
                # # Update best loss tracking based on validation loss
                # if avg_val_loss < best_loss:
                #     best_loss = avg_val_loss
                #     # Save best model
                #     best_model_path = os.path.join(save_path, "best_model.pth")
                #     torch.save({
                #         'epoch': epoch,
                #         'model_state_dict': model.state_dict(),
                #         'optimizer_state_dict': optimizer.state_dict(),
                #         'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                #         'loss': avg_val_loss,
                #         'best_loss': best_loss,
                #         'history': history
                #     }, best_model_path)
                #     print(f"New best model saved with validation loss: {avg_val_loss:.6f}")
                
                # Add validation metrics to history
                history['val_total_loss'].append(avg_val_loss)
                history['val_dft_loss'].append(avg_val_dft_loss)
                history['val_sparsity_loss'].append(avg_val_sparsity_loss)
        else:
            # Without validation, use training loss for scheduler
            scheduler.step(avg_loss)
            # Check if current model is the best
            # if avg_loss < best_loss:
            #     best_loss = avg_loss
            #     # Save best model
            #     best_model_path = os.path.join(save_path, "best_model.pth")
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            #         'loss': avg_loss,
            #         'best_loss': best_loss,
            #         'history': history
            #     }, best_model_path)
            #     print(f"New best model saved with training loss: {avg_loss:.6f}")

        # Add training metrics to history
        history['total_loss'].append(avg_loss)
        history['dft_loss'].append(avg_dft_loss)
        history['sparsity_loss'].append(avg_sparsity_loss)

        # Print progress
        if val_loader is not None:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}, '
                  f'Train DFT Loss: {avg_dft_loss:.6f}, Val DFT Loss: {avg_val_dft_loss:.6f}, '
                  f'Train Sparsity Loss: {avg_sparsity_loss:.6f}, Val Sparsity Loss: {avg_val_sparsity_loss:.6f}')
        else:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, '
                  f'DFT Loss: {avg_dft_loss:.6f}, Sparsity Loss: {avg_sparsity_loss:.6f}')

        # Save checkpoint every checkpoint_freq epochs
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(save_path, "checkpoint.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': avg_val_loss if val_loader else avg_loss,
                'best_loss': best_loss,
                'history': history
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
        # Plot and save loss history
        if plot_path and (epoch + 1) % checkpoint_freq == 0:
            os.makedirs(plot_path, exist_ok=True)
            
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
            
            # 1. Plot training and validation separately
            
            # Training Loss Plots
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            
            # Total Loss subplot
            axes[0].plot(history['total_loss'], color=train_colors[0], linewidth=line_width, marker='o', markersize=4)
            axes[0].set_title("Total Training Loss", fontsize=title_size, fontweight='bold')
            axes[0].set_xlabel("Epoch", fontsize=font_size)
            axes[0].set_ylabel("Loss", fontsize=font_size)
            axes[0].grid(True, alpha=0.3)
            axes[0].tick_params(axis='both', labelsize=tick_size)
            
            # DFT Loss subplot
            axes[1].plot(history['dft_loss'], color=train_colors[1], linewidth=line_width, marker='o', markersize=4)
            axes[1].set_title("DFT Training Loss", fontsize=title_size, fontweight='bold')
            axes[1].set_xlabel("Epoch", fontsize=font_size)
            axes[1].set_ylabel("Loss", fontsize=font_size)
            axes[1].grid(True, alpha=0.3)
            axes[1].tick_params(axis='both', labelsize=tick_size)
            
            # Sparsity Loss subplot
            axes[2].plot(history['sparsity_loss'], color=train_colors[2], linewidth=line_width, marker='o', markersize=4)
            axes[2].set_title("Sparsity Training Loss", fontsize=title_size, fontweight='bold')
            axes[2].set_xlabel("Epoch", fontsize=font_size)
            axes[2].set_ylabel("Loss", fontsize=font_size)
            axes[2].grid(True, alpha=0.3)
            axes[2].tick_params(axis='both', labelsize=tick_size)
            
            # Adjust layout and save
            plt.tight_layout(pad=3.0)
            plt.savefig(os.path.join(plot_path, f"training_loss_epoch_{epoch+1}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Validation Loss Plots (if available)
            if val_loader is not None and 'val_total_loss' in history and len(history['val_total_loss']) > 0:
                fig, axes = plt.subplots(1, 3, figsize=(20, 6))
                
                # Total Loss subplot
                axes[0].plot(history['val_total_loss'], color=val_colors[0], linewidth=line_width, marker='o', markersize=4)
                axes[0].set_title("Total Validation Loss", fontsize=title_size, fontweight='bold')
                axes[0].set_xlabel("Epoch", fontsize=font_size)
                axes[0].set_ylabel("Loss", fontsize=font_size)
                axes[0].grid(True, alpha=0.3)
                axes[0].tick_params(axis='both', labelsize=tick_size)
                
                # DFT Loss subplot
                axes[1].plot(history['val_dft_loss'], color=val_colors[1], linewidth=line_width, marker='o', markersize=4)
                axes[1].set_title("DFT Validation Loss", fontsize=title_size, fontweight='bold')
                axes[1].set_xlabel("Epoch", fontsize=font_size)
                axes[1].set_ylabel("Loss", fontsize=font_size)
                axes[1].grid(True, alpha=0.3)
                axes[1].tick_params(axis='both', labelsize=tick_size)
                
                # Sparsity Loss subplot
                axes[2].plot(history['val_sparsity_loss'], color=val_colors[2], linewidth=line_width, marker='o', markersize=4)
                axes[2].set_title("Sparsity Validation Loss", fontsize=title_size, fontweight='bold')
                axes[2].set_xlabel("Epoch", fontsize=font_size)
                axes[2].set_ylabel("Loss", fontsize=font_size)
                axes[2].grid(True, alpha=0.3)
                axes[2].tick_params(axis='both', labelsize=tick_size)
                
                # Adjust layout and save
                plt.tight_layout(pad=3.0)
                plt.savefig(os.path.join(plot_path, f"validation_loss_epoch_{epoch+1}.png"), dpi=300, bbox_inches='tight')
                plt.close()
            
            # 2. Combined comparison plots (train vs validation)
            if val_loader is not None and 'val_total_loss' in history and len(history['val_total_loss']) > 0:
                fig, axes = plt.subplots(1, 3, figsize=(20, 6))
                
                # Total Loss subplot
                axes[0].plot(history['total_loss'], label='Training', color=train_colors[0], linewidth=line_width, marker='o', markersize=4)
                axes[0].plot(history['val_total_loss'], label='Validation', color=val_colors[0], linewidth=line_width, marker='s', markersize=4)
                axes[0].set_title("Total Loss Comparison", fontsize=title_size, fontweight='bold')
                axes[0].set_xlabel("Epoch", fontsize=font_size)
                axes[0].set_ylabel("Loss", fontsize=font_size)
                axes[0].grid(True, alpha=0.3)
                axes[0].legend(fontsize=legend_size)
                axes[0].tick_params(axis='both', labelsize=tick_size)
                
                # DFT Loss subplot
                axes[1].plot(history['dft_loss'], label='Training', color=train_colors[1], linewidth=line_width, marker='o', markersize=4)
                axes[1].plot(history['val_dft_loss'], label='Validation', color=val_colors[1], linewidth=line_width, marker='s', markersize=4)
                axes[1].set_title("DFT Loss Comparison", fontsize=title_size, fontweight='bold')
                axes[1].set_xlabel("Epoch", fontsize=font_size)
                axes[1].set_ylabel("Loss", fontsize=font_size)
                axes[1].grid(True, alpha=0.3)
                axes[1].legend(fontsize=legend_size)
                axes[1].tick_params(axis='both', labelsize=tick_size)
                
                # Sparsity Loss subplot
                axes[2].plot(history['sparsity_loss'], label='Training', color=train_colors[2], linewidth=line_width, marker='o', markersize=4)
                axes[2].plot(history['val_sparsity_loss'], label='Validation', color=val_colors[2], linewidth=line_width, marker='s', markersize=4)
                axes[2].set_title("Sparsity Loss Comparison", fontsize=title_size, fontweight='bold')
                axes[2].set_xlabel("Epoch", fontsize=font_size)
                axes[2].set_ylabel("Loss", fontsize=font_size)
                axes[2].grid(True, alpha=0.3)
                axes[2].legend(fontsize=legend_size)
                axes[2].tick_params(axis='both', labelsize=tick_size)
                
                # Adjust layout and save
                plt.tight_layout(pad=3.0)
                plt.savefig(os.path.join(plot_path, f"comparison_loss_epoch_{epoch+1}.png"), dpi=300, bbox_inches='tight')
                plt.close()
            
            # 3. One additional consolidated plot showing all losses
            plt.figure(figsize=(12, 8))
            
            # Training losses
            plt.plot(history['total_loss'], label='Total (Train)', color=train_colors[0], linewidth=line_width, marker='o', markersize=4)
            plt.plot(history['dft_loss'], label='DFT (Train)', color=train_colors[1], linewidth=line_width, marker='o', markersize=4)
            plt.plot(history['sparsity_loss'], label='Sparsity (Train)', color=train_colors[2], linewidth=line_width, marker='o', markersize=4)
            
            # Validation losses if available
            if val_loader is not None and 'val_total_loss' in history and len(history['val_total_loss']) > 0:
                plt.plot(history['val_total_loss'], label='Total (Val)', color=val_colors[0], linewidth=line_width, linestyle='--', marker='s', markersize=4)
                plt.plot(history['val_dft_loss'], label='DFT (Val)', color=val_colors[1], linewidth=line_width, linestyle='--', marker='s', markersize=4)
                plt.plot(history['val_sparsity_loss'], label='Sparsity (Val)', color=val_colors[2], linewidth=line_width, linestyle='--', marker='s', markersize=4)
            
            plt.title("All Loss Metrics", fontsize=title_size, fontweight='bold')
            plt.xlabel("Epoch", fontsize=font_size)
            plt.ylabel("Loss", fontsize=font_size)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=legend_size)
            plt.tick_params(axis='both', labelsize=tick_size)
            
            # Add epoch marker
            plt.axvline(x=epoch, color='gray', linestyle='--', alpha=0.5)
            plt.text(epoch, plt.gca().get_ylim()[1]*0.9, f'Current: Epoch {epoch+1}', 
                    fontsize=12, ha='center', va='center', 
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_path, f"all_losses_epoch_{epoch+1}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save the raw data for later plotting
            history_file = os.path.join(plot_path, "loss_history.json")
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=4)

    # Save final model
    final_model_path = os.path.join(save_path_final, f"complex_dft_unet_final.pth")
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': avg_val_loss if val_loader else avg_loss,
        'history': history
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Remove checkpoint file after successful completion
    checkpoint_path = os.path.join(save_path, "checkpoint.pt")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Training completed successfully. Removed checkpoint file: {checkpoint_path}")

    return history



def prepare_training_data_with_masks(intermediate_data, dft_size, mask_size):
    """
    Prepare training data with masks for DFT size of 1000 and mask size of 500.
    
    Parameters:
    -----------
    intermediate_data : list
        List of intermediate data tuples
    dft_size : int, default=1000
        Size of the DFT data (complex values)
    mask_size : int, default=500
        Size of the masks
    
    Returns:
    --------
    inputs : numpy.ndarray
        Array of input data with shape (N, 2*dft_size)
    masks : numpy.ndarray
        Array of masks with shape (N, 3, mask_size)
    targets_dft : numpy.ndarray
        Array of target DFT data with shape (N, 2*dft_size)
    targets_sparsity : numpy.ndarray
        Array of target sparsity values with shape (N,)
    """
    N = len(intermediate_data)
    inputs = np.zeros((N, 2 * dft_size))     # dft_size real + dft_size imag
    masks = np.zeros((N, 3, mask_size))      # 3 masks, each with mask_size elements
    targets_dft = np.zeros((N, 2 * dft_size))  # dft_size real + dft_size imag
    targets_sparsity = np.zeros(N)

    for i, example in enumerate(intermediate_data):
        freq_domain = example[0]['frequency_domain']
        mask_data = example[0]['masks']
        target = example[1]
        sparsity = example[2]

        # Real and imaginary parts for input - full 1000 DFT values
        # Ensure we have enough data or pad with zeros
        real_data = np.real(freq_domain)
        imag_data = np.imag(freq_domain)
        
        # Handle cases where input data might be smaller than dft_size
        real_len = min(len(real_data), dft_size)
        imag_len = min(len(imag_data), dft_size)
        
        inputs[i, :real_len] = real_data[:real_len]
        inputs[i, dft_size:dft_size+imag_len] = imag_data[:imag_len]

        # Pack masks into the correct format - keep at 500 size
        # Assuming mask_data is a dictionary with keys 'Mh', 'Ml', 'Mr'
        if isinstance(mask_data, dict):
            for j, key in enumerate(['Mh', 'Ml', 'Mr']):
                if key in mask_data:
                    # Ensure each mask is mask_size elements long
                    mask_len = min(len(mask_data[key]), mask_size)
                    masks[i, j, :mask_len] = mask_data[key][:mask_len]
        else:
            # Assuming mask_data is already properly formatted as [3, mask_size]
            # or can be reshaped to that
            if len(mask_data.shape) == 1:
                # If it's a flattened array of 3*mask_size elements
                masks[i] = mask_data.reshape(3, mask_size)
            else:
                masks[i] = mask_data[:3, :mask_size]

        # Real and imaginary parts for target - full 1000 DFT values
        real_target = np.real(target)
        imag_target = np.imag(target)
        
        real_target_len = min(len(real_target), dft_size)
        imag_target_len = min(len(imag_target), dft_size)
        
        targets_dft[i, :real_target_len] = real_target[:real_target_len]
        targets_dft[i, dft_size:dft_size+imag_target_len] = imag_target[:imag_target_len]

        # Sparsity
        targets_sparsity[i] = sparsity

    return inputs, masks, targets_dft, targets_sparsity

def load_model(model, model_path):
    # Load checkpoint
    checkpoint = torch.load(model_path)

    # Load model state_dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # Return the model after loading the state dict
    return model

def predict_with_model(model, input_data, masks, dft_size=None, mask_size=None):
    """
    Make predictions with the model using input data and masks.
    
    Parameters:
    -----------
    model : ComplexDFTUNet
        The model to use for predictions
    input_data : numpy.ndarray or torch.Tensor
        Input data of shape [batch, 2*dft_size]
    masks : dict or numpy.ndarray or torch.Tensor
        Mask data of shape [batch, 3, mask_size]
    dft_size : int, optional
        Size of the DFT data, defaults to model.dft_size if None
    mask_size : int, optional
        Size of the masks, defaults to model.mask_size if None
        
    Returns:
    --------
    tuple
        Tuple of (pred_dft, pred_sparsity)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Use the model's parameters if not specified
    if dft_size is None:
        dft_size = model.dft_size
    if mask_size is None:
        mask_size = model.mask_size
    
    # Convert inputs to tensors if they're not already
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data, dtype=torch.float32)
    
    # Handle input_data shape mismatch
    batch_size = input_data.shape[0]
    input_size = input_data.shape[1] // 2  # Assuming [batch, 2*some_size]
    
    if input_size != dft_size:
        print(f"Warning: Input size {input_size} doesn't match model's dft_size {dft_size}. Reshaping...")
        # Resize input_data appropriately
        if input_size > dft_size:
            # Truncate if input is larger
            new_input = torch.zeros((batch_size, 2 * dft_size), dtype=input_data.dtype, device=input_data.device)
            new_input[:, :dft_size] = input_data[:, :dft_size]  # First half (real)
            new_input[:, dft_size:] = input_data[:, input_size:input_size+dft_size]  # Second half (imag)
            input_data = new_input
        else:
            # Pad with zeros if input is smaller
            new_input = torch.zeros((batch_size, 2 * dft_size), dtype=input_data.dtype, device=input_data.device)
            new_input[:, :input_size] = input_data[:, :input_size]  # First half (real)
            new_input[:, dft_size:dft_size+input_size] = input_data[:, input_size:]  # Second half (imag)
            input_data = new_input
    
    # Handle masks based on their type
    if not isinstance(masks, torch.Tensor):
        if isinstance(masks, dict):
            # Process masks using the utility function
            masks = prepare_mask_for_inference(masks, mask_size)
        else:
            masks = torch.tensor(masks, dtype=torch.float32)
            
            # Ensure masks have shape [batch, 3, mask_size]
            if masks.dim() == 2 and masks.shape[1] == 3:
                # [batch, 3] -> [batch, 3, 1]
                masks = masks.unsqueeze(-1)
            
            if masks.shape[-1] != mask_size:
                if masks.shape[-1] == 1:
                    # Broadcast to full size if just one value per channel
                    masks = masks.expand(-1, -1, mask_size)
                else:
                    # Reshape masks to match expected size
                    print(f"Warning: Mask shape {masks.shape} doesn't match mask_size={mask_size}. Reshaping...")
                    # Try to reshape or interpolate
                    if masks.shape[-1] > mask_size:
                        # Truncate if larger
                        masks = masks[..., :mask_size]
                    else:
                        # Pad with zeros if smaller
                        new_masks = torch.zeros((masks.shape[0], masks.shape[1], mask_size), 
                                              dtype=masks.dtype, device=masks.device)
                        new_masks[..., :masks.shape[-1]] = masks
                        masks = new_masks
                    
    # Make sure we have a batch dimension
    if input_data.dim() == 1:
        input_data = input_data.unsqueeze(0)
    if masks.dim() == 2:
        masks = masks.unsqueeze(0)
    
    # Additional check for mask shape
    if masks.shape[1] != 3:
        print(f"Warning: Mask should have 3 channels, got {masks.shape[1]}. Adjusting...")
        new_masks = torch.zeros((masks.shape[0], 3, mask_size), 
                              dtype=masks.dtype, device=masks.device)
        chan_count = min(masks.shape[1], 3)
        new_masks[:, :chan_count] = masks[:, :chan_count]
        masks = new_masks
    
    input_data = input_data.to(device)
    masks = masks.to(device)
    
    with torch.no_grad():
        pred_dft, pred_sparsity = model(input_data, masks)
        
    # Convert predictions back to numpy arrays for easier handling
    pred_dft = pred_dft.cpu().numpy()
    pred_sparsity = pred_sparsity.cpu().numpy()
    
    return pred_dft, pred_sparsity

def prepare_mask_for_inference(mask_data, mask_size):
    """
    Convert a dict or numpy mask to torch.Tensor of shape [1, 3, mask_size]
    
    Parameters:
    -----------
    mask_data : dict or numpy.ndarray or torch.Tensor
        The mask data to be prepared
    mask_size : int, default=500
        The size of the mask
        
    Returns:
    --------
    torch.Tensor
        The prepared mask as a tensor of shape [1, 3, mask_size]
    """
    if isinstance(mask_data, dict):
        # Create an empty mask
        processed_mask = torch.zeros(1, 3, mask_size, dtype=torch.float32)
        for j, key in enumerate(['Mh', 'Ml', 'Mr']):
            if key in mask_data:
                mask_array = np.array(mask_data[key], dtype=np.float32)
                length = min(len(mask_array), mask_size)
                processed_mask[0, j, :length] = torch.tensor(mask_array[:length])
        return processed_mask
    elif isinstance(mask_data, np.ndarray) or isinstance(mask_data, torch.Tensor):
        mask_tensor = torch.tensor(mask_data, dtype=torch.float32) if isinstance(mask_data, np.ndarray) else mask_data
        if mask_tensor.dim() == 1 and mask_tensor.numel() == 3 * mask_size:
            return mask_tensor.view(1, 3, mask_size)
        elif mask_tensor.shape == (3, mask_size):
            return mask_tensor.unsqueeze(0)
        elif mask_tensor.shape == (mask_size, 3):
            return mask_tensor.permute(1, 0).unsqueeze(0)
        elif mask_tensor.shape == (1, 3, mask_size):
            return mask_tensor
        else:
            raise ValueError(f"Unsupported mask shape: {mask_tensor.shape}. Expected dimensions compatible with mask_size={mask_size}")
    else:
        raise TypeError("Unsupported mask type")


# def visualize_frequency_attention(model, dataloader):
#     """
#     Visualize the frequency attention maps from the model.
#     """
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     model.eval()
    
#     # Get a batch of data
#     batch = next(iter(dataloader))
#     inputs = batch['input'].to(device)
#     masks = batch['mask'].to(device)
    
#     # Create hooks to capture attention maps
#     attention_maps = []
    
#     def hook_fn(module, input, output):
#         attention_maps.append(output.detach().cpu().numpy())
    
#     # Register hooks on attention modules
#     hooks = []
#     for name, module in model.named_modules():
#         if isinstance(module, FrequencyAwareAttention):
#             hooks.append(module.register_forward_hook(hook_fn))
    
#     # Forward pass
#     with torch.no_grad():
#         model(inputs, masks)
    
#     # Remove hooks
#     for hook in hooks:
#         hook.remove()
    
#     # Visualize attention maps
#     plt.figure(figsize=(15, 10))
#     for i, attention_map in enumerate(attention_maps):
#         if i < 4:  # Only show first few attention maps
#             plt.subplot(2, 2, i+1)
#             # Take one example and average over channels
#             avg_attention = np.mean(attention_map[0], axis=0)
#             plt.plot(avg_attention)
#             plt.title(f'Attention Map {i+1}')
#             plt.xlabel('Frequency')
#             plt.ylabel('Attention Weight')
    
#     plt.tight_layout()
#     plt.show()

# import numpy as np



# def evaluate_complex_dft_unet(model, dataloader):
#     """
#     Evaluate the complex DFT U-Net model.
#     """
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     model.eval()
    
#     dft_criterion = nn.MSELoss()
#     sparsity_criterion = nn.MSELoss()
    
#     total_dft_loss = 0
#     total_sparsity_loss = 0
    
#     with torch.no_grad():
#         for batch in dataloader:
#             inputs = batch['input'].to(device)
#             masks = batch['mask'].to(device)
#             targets_dft = batch['target_dft'].to(device)
#             targets_sparsity = batch['target_sparsity'].to(device)
            
#             # Forward pass
#             pred_dft, pred_sparsity = model(inputs, masks)
            
#             # Calculate losses
#             dft_loss = dft_criterion(pred_dft, targets_dft)
#             sparsity_loss = sparsity_criterion(pred_sparsity, targets_sparsity / model.max_sparsity)
            
#             # Accumulate losses
#             total_dft_loss += dft_loss.item()
#             total_sparsity_loss += sparsity_loss.item()
    
#     # Average losses
#     avg_dft_loss = total_dft_loss / len(dataloader)
#     avg_sparsity_loss = total_sparsity_loss / len(dataloader)
    
#     metrics = {
#         'dft_mse': avg_dft_loss,
#         'sparsity_mse': avg_sparsity_loss
#     }
    
#     return metrics
