"""
UNet trainer for building segmentation tasks.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import wandb
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from training.train import BaseTrainer
from utils.losses import BCEWithDiceLoss, DiceLoss
from evaluation.metrics import compute_metrics, compute_metrics_batch


class UNetTrainer(BaseTrainer):
    """
    Trainer class for UNet model with building segmentation tasks.
    Supports both mask and contour outputs.
    """
    def __init__(self, model, train_loader, val_loader, device='cuda', 
                learning_rate=1e-4, model_save_dir='./checkpoints',
                mask_weight=0.7, contour_weight=0.3, use_wandb=True, 
                wandb_project='building_seg', wandb_run_name=None):
        """
        Initialize the UNetTrainer.
        
        Args:
            model: The UNet model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to use for training ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            model_save_dir: Directory to save model checkpoints
            mask_weight: Weight for mask loss
            contour_weight: Weight for contour loss
            use_wandb: Whether to use wandb for logging
            wandb_project: Project name for wandb
            wandb_run_name: Run name for wandb (optional)
        """
        super().__init__(model, train_loader, val_loader, device, learning_rate, model_save_dir)
        self.mask_weight = mask_weight
        self.contour_weight = contour_weight
        self.use_wandb = use_wandb
        
        # Initialize wandb if enabled
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    'learning_rate': learning_rate,
                    'mask_weight': mask_weight,
                    'contour_weight': contour_weight,
                    'device': device,
                    'model_type': 'UNet'
                }
            )
            # Watch the model for gradient and parameter tracking
            wandb.watch(self.model, log='all', log_freq=100)
        
    def _get_optimizer(self):
        """
        Get Adam optimizer for UNet.
        
        Returns:
            Adam optimizer
        """
        return Adam(self.model.parameters(), lr=self.learning_rate)
    
    def _get_loss_fn(self):
        """
        Get combined loss function for UNet.
        
        Returns:
            Dictionary of loss functions
        """
        return {
            'mask': BCEWithDiceLoss(dice_weight=0.5, bce_weight=0.5),
            'contour': BCEWithDiceLoss(dice_weight=0.7, bce_weight=0.3)
        }
    
    def _create_validation_visualizations(self, images, masks, predictions, contours=None, contour_preds=None, max_samples=4):
        """
        Create visualizations for wandb logging during validation.
        
        Args:
            images: Input images tensor
            masks: Ground truth masks tensor
            predictions: Predicted masks tensor
            contours: Ground truth contours tensor (optional)
            contour_preds: Predicted contours tensor (optional)
            max_samples: Maximum number of samples to visualize
            
        Returns:
            List of wandb.Image objects for logging
        """
        wandb_images = []
        batch_size = min(images.shape[0], max_samples)
        
        for i in range(batch_size):
            # Convert tensors to numpy and move to CPU
            img = images[i].cpu().numpy()
            mask_gt = masks[i].cpu().numpy().squeeze()
            mask_pred = torch.sigmoid(predictions[i]).cpu().numpy().squeeze()
            
            # Normalize image for display (assuming it's in [0,1] or needs normalization)
            if img.shape[0] == 3:  # RGB image
                img = np.transpose(img, (1, 2, 0))
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
            elif img.shape[0] == 1:  # Grayscale
                img = img.squeeze()
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
            
            # Create figure
            if contours is not None and contour_preds is not None:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                contour_gt = contours[i].cpu().numpy().squeeze()
                contour_pred = torch.sigmoid(contour_preds[i]).cpu().numpy().squeeze()
                
                # Top row: masks
                axes[0, 0].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                axes[0, 0].set_title('Input Image')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(mask_gt, cmap='Blues', alpha=0.7)
                axes[0, 1].set_title('Ground Truth Mask')
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(mask_pred, cmap='Blues', alpha=0.7)
                axes[0, 2].set_title('Predicted Mask')
                axes[0, 2].axis('off')
                
                # Bottom row: contours
                axes[1, 0].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                axes[1, 0].set_title('Input Image')
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(contour_gt, cmap='Reds', alpha=0.7)
                axes[1, 1].set_title('Ground Truth Contour')
                axes[1, 1].axis('off')
                
                axes[1, 2].imshow(contour_pred, cmap='Reds', alpha=0.7)
                axes[1, 2].set_title('Predicted Contour')
                axes[1, 2].axis('off')
            else:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                axes[0].set_title('Input Image')
                axes[0].axis('off')
                
                axes[1].imshow(mask_gt, cmap='Blues', alpha=0.7)
                axes[1].set_title('Ground Truth Mask')
                axes[1].axis('off')
                
                axes[2].imshow(mask_pred, cmap='Blues', alpha=0.7)
                axes[2].set_title('Predicted Mask')
                axes[2].axis('off')
            
            plt.tight_layout()
            
            # Convert to wandb image
            wandb_images.append(wandb.Image(fig, caption=f"Sample {i+1}"))
            plt.close(fig)
        
        return wandb_images
    
    def _train_epoch(self, epoch):
        """
        Train UNet for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        running_loss = 0.0
        running_mask_loss = 0.0
        running_contour_loss = 0.0
        step = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            # Check if batch includes contour maps
            if len(batch) == 3:
                images, masks, contours = batch
                images = images.to(self.device)
                masks = masks.to(self.device)
                contours = contours.to(self.device)
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                mask_pred, contour_pred = self.model(images)
                
                # Calculate losses
                mask_loss = self.loss_fn['mask'](mask_pred, masks)
                contour_loss = self.loss_fn['contour'](contour_pred, contours)
                loss = self.mask_weight * mask_loss + self.contour_weight * contour_loss
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Update running loss
                running_loss += loss.item()
                running_mask_loss += mask_loss.item()
                running_contour_loss += contour_loss.item()
                
                # Log to wandb every 10 steps
                if self.use_wandb and step % 10 == 0:
                    wandb.log({
                        'train/step_loss': loss.item(),
                        'train/step_mask_loss': mask_loss.item(),
                        'train/step_contour_loss': contour_loss.item(),
                        'train/step': step,
                        'epoch': epoch
                    })
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'mask_loss': mask_loss.item(),
                    'contour_loss': contour_loss.item()
                })
                
                step += 1
            else:
                # Handle case without contour maps
                images, masks = batch
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Forward pass (might still return two outputs)
                outputs = self.model(images)
                
                if isinstance(outputs, tuple):
                    mask_pred = outputs[0]  # Just use the mask output
                else:
                    mask_pred = outputs
                
                # Calculate loss
                loss = self.loss_fn['mask'](mask_pred, masks)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Update running loss
                running_loss += loss.item()
                running_mask_loss += loss.item()
                
                # Log to wandb every 10 steps
                if self.use_wandb and step % 10 == 0:
                    wandb.log({
                        'train/step_loss': loss.item(),
                        'train/step_mask_loss': loss.item(),
                        'train/step': step,
                        'epoch': epoch
                    })
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
                
                step += 1
        
        # Calculate average loss
        avg_loss = running_loss / len(self.train_loader)
        avg_mask_loss = running_mask_loss / len(self.train_loader)
        
        # Log epoch-level training metrics to wandb
        if self.use_wandb:
            log_dict = {
                'train/epoch_loss': avg_loss,
                'train/epoch_mask_loss': avg_mask_loss,
                'epoch': epoch
            }
            if running_contour_loss > 0:  # Only log if we have contour losses
                avg_contour_loss = running_contour_loss / len(self.train_loader)
                log_dict['train/epoch_contour_loss'] = avg_contour_loss
            
            wandb.log(log_dict)
        
        return avg_loss
    
    def _validate_epoch(self, epoch):
        """
        Validate UNet for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average validation loss, metrics dictionary)
        """
        self.model.eval()
        running_loss = 0.0
        all_metrics = {
            'iou': 0.0,
            'dice': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
        
        contour_metrics = {
            'iou': 0.0,
            'dice': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
        
        has_contours = False
        validation_images = []  # Store samples for visualization
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                # Check if batch includes contour maps
                if len(batch) == 3:
                    images, masks, contours = batch
                    has_contours = True
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    contours = contours.to(self.device)
                    
                    # Forward pass
                    mask_pred, contour_pred = self.model(images)
                    
                    # Calculate losses
                    mask_loss = self.loss_fn['mask'](mask_pred, masks)
                    contour_loss = self.loss_fn['contour'](contour_pred, contours)
                    loss = self.mask_weight * mask_loss + self.contour_weight * contour_loss
                    
                    # Calculate metrics for mask
                    batch_metrics = compute_metrics_batch(mask_pred, masks)
                    for k, v in batch_metrics.items():
                        all_metrics[k] += v
                    
                    # Calculate metrics for contour
                    batch_contour_metrics = compute_metrics_batch(contour_pred, contours)
                    for k, v in batch_contour_metrics.items():
                        contour_metrics[k] += v
                    
                    # Store samples for visualization (first batch only)
                    if len(validation_images) == 0 and self.use_wandb:
                        validation_images.append({
                            'images': images[:4].clone(),  # Store first 4 samples
                            'masks': masks[:4].clone(),
                            'mask_preds': mask_pred[:4].clone(),
                            'contours': contours[:4].clone(),
                            'contour_preds': contour_pred[:4].clone()
                        })
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'mask_loss': mask_loss.item(),
                        'contour_loss': contour_loss.item(),
                        'iou': batch_metrics['iou']
                    })
                else:
                    # Handle case without contour maps
                    images, masks = batch
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    
                    if isinstance(outputs, tuple):
                        mask_pred = outputs[0]  # Just use the mask output
                    else:
                        mask_pred = outputs
                    
                    # Calculate loss
                    loss = self.loss_fn['mask'](mask_pred, masks)
                    
                    # Calculate metrics
                    batch_metrics = compute_metrics_batch(mask_pred, masks)
                    for k, v in batch_metrics.items():
                        all_metrics[k] += v
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'iou': batch_metrics['iou']
                    })
                
                # Update running loss
                running_loss += loss.item()
        
        # Calculate average loss and metrics
        avg_loss = running_loss / len(self.val_loader)
        
        for k in all_metrics:
            all_metrics[k] /= len(self.val_loader)
        
        if has_contours:
            for k in contour_metrics:
                contour_metrics[k] /= len(self.val_loader)
            
            # Add contour metrics to the metrics dictionary
            all_metrics.update({f'contour_{k}': v for k, v in contour_metrics.items()})
        
        return avg_loss, all_metrics
