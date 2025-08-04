"""
UNet trainer for building segmentation tasks.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

from .train import BaseTrainer
from ..utils.losses import BCEWithDiceLoss, DiceLoss
from ..evaluation.metrics import compute_metrics, compute_metrics_batch


class UNetTrainer(BaseTrainer):
    """
    Trainer class for UNet model with building segmentation tasks.
    Supports both mask and contour outputs.
    """
    def __init__(self, model, train_loader, val_loader, device='cuda', 
                learning_rate=1e-4, model_save_dir='./checkpoints',
                mask_weight=0.7, contour_weight=0.3):
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
        """
        super().__init__(model, train_loader, val_loader, device, learning_rate, model_save_dir)
        self.mask_weight = mask_weight
        self.contour_weight = contour_weight
        
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
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'mask_loss': mask_loss.item(),
                    'contour_loss': contour_loss.item()
                })
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
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average loss
        avg_loss = running_loss / len(self.train_loader)
        
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
