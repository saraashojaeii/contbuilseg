"""
DeepLabV3+ trainer for building segmentation tasks.
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
from utils.losses import BCEWithDiceLoss, DiceLoss, L1Loss
from evaluation.metrics import compute_metrics, compute_metrics_batch


class DeepLabV3PlusTrainer(BaseTrainer):
    """
    Trainer class for DeepLabV3+ model with building segmentation tasks.
    Supports both mask and contour outputs.
    """
    def __init__(self, model, train_loader, val_loader, device='cuda', 
                learning_rate=1e-4, model_save_dir='./checkpoints',
                mask_weight=0.7, contour_weight=0.3, use_wandb=True, 
                wandb_project='building_seg', wandb_run_name=None, dataset_name='dataset'):
        """
        Initialize the DeepLabV3PlusTrainer.
        
        Args:
            model: The DeepLabV3+ model to train
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
            dataset_name: Name of the dataset for prefixing saved files
        """
        super().__init__(model, train_loader, val_loader, device, learning_rate, model_save_dir)
        self.mask_weight = mask_weight
        self.contour_weight = contour_weight
        self.use_wandb = use_wandb
        self.dataset_name = dataset_name
        
        # Create directories for saving models and predictions
        import os
        self.model_save_dir = os.path.join(model_save_dir, f"{dataset_name}_models")
        self.predictions_save_dir = os.path.join(model_save_dir, f"{dataset_name}_predictions")
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.predictions_save_dir, exist_ok=True)
        
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
                    'model_type': 'DeepLabV3Plus'
                }
            )
            # Watch the model for gradient and parameter tracking
            wandb.watch(self.model, log='all', log_freq=100)
        
    def _get_optimizer(self):
        """
        Get Adam optimizer for DeepLabV3+.
        
        Returns:
            Adam optimizer
        """
        return Adam(self.model.parameters(), lr=self.learning_rate)
    
    def _get_loss_fn(self):
        """
        Get combined loss function for DeepLabV3+.
        Binary mask uses BCE+Dice loss, contour uses L1 loss for distance transform.
        
        Returns:
            Dictionary of loss functions
        """
        return {
            'mask': BCEWithDiceLoss(dice_weight=0.5, bce_weight=0.5),
            'contour': L1Loss(reduction='mean')
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
            mask_pred = predictions[i].cpu().numpy().squeeze()  # Already sigmoid-activated
            
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
                contour_pred = contour_preds[i].cpu().numpy().squeeze()  # Distance transform, no sigmoid
                
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
    
    def _save_model_checkpoint(self, epoch, val_loss, metrics):
        """
        Save model checkpoint at the end of an epoch.
        
        Args:
            epoch: Current epoch number
            val_loss: Validation loss
            metrics: Dictionary of validation metrics
        """
        import os
        import torch
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'mask_weight': self.mask_weight,
            'contour_weight': self.contour_weight,
            'learning_rate': self.learning_rate
        }
        
        # Save checkpoint with epoch number
        checkpoint_path = os.path.join(self.model_save_dir, f"{self.dataset_name}_epoch_{epoch:03d}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest checkpoint
        latest_path = os.path.join(self.model_save_dir, f"{self.dataset_name}_latest.pth")
        torch.save(checkpoint, latest_path)
        
        print(f"Model checkpoint saved: {checkpoint_path}")
    
    def _save_validation_predictions(self, epoch, validation_images):
        """
        Save validation predictions as images.
        
        Args:
            epoch: Current epoch number
            validation_images: List of validation image data dictionaries
        """
        import os
        import torch
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
        
        if not validation_images:
            return
        
        # Create epoch-specific directory
        epoch_dir = os.path.join(self.predictions_save_dir, f"epoch_{epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        sample_data = validation_images[0]
        images = sample_data['images']
        masks = sample_data['masks']
        mask_preds = sample_data['mask_preds']
        contours = sample_data.get('contours')
        contour_preds = sample_data.get('contour_preds')
        
        batch_size = min(images.shape[0], 8)  # Save up to 8 samples
        
        for i in range(batch_size):
            # Convert tensors to numpy and move to CPU
            img = images[i].cpu().numpy()
            mask_gt = masks[i].cpu().numpy().squeeze()
            mask_pred = mask_preds[i].cpu().numpy().squeeze()  # Already sigmoid-activated
            
            # Normalize image for display
            if img.shape[0] == 3:  # RGB image
                img = np.transpose(img, (1, 2, 0))
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
            elif img.shape[0] == 1:  # Grayscale
                img = img.squeeze()
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
            
            # Save individual components
            sample_dir = os.path.join(epoch_dir, f"sample_{i+1:02d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save input image
            if len(img.shape) == 3:
                Image.fromarray(img).save(os.path.join(sample_dir, "input.png"))
            else:
                Image.fromarray(img.astype(np.uint8)).save(os.path.join(sample_dir, "input.png"))
            
            # Save ground truth mask
            mask_gt_img = (mask_gt * 255).astype(np.uint8)
            Image.fromarray(mask_gt_img).save(os.path.join(sample_dir, "mask_gt.png"))
            
            # Save predicted mask
            mask_pred_img = (mask_pred * 255).astype(np.uint8)
            Image.fromarray(mask_pred_img).save(os.path.join(sample_dir, "mask_pred.png"))
            
            # Save contours if available
            if contours is not None and contour_preds is not None:
                contour_gt = contours[i].cpu().numpy().squeeze()
                contour_pred = contour_preds[i].cpu().numpy().squeeze()  # Distance transform, no sigmoid
                
                contour_gt_img = (contour_gt * 255).astype(np.uint8)
                contour_pred_img = (contour_pred * 255).astype(np.uint8)
                
                Image.fromarray(contour_gt_img).save(os.path.join(sample_dir, "contour_gt.png"))
                Image.fromarray(contour_pred_img).save(os.path.join(sample_dir, "contour_pred.png"))
            
            # Create and save combined visualization
            if contours is not None and contour_preds is not None:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                contour_gt = contours[i].cpu().numpy().squeeze()
                contour_pred = contour_preds[i].cpu().numpy().squeeze()  # Distance transform, no sigmoid
                
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
            plt.savefig(os.path.join(sample_dir, "combined_visualization.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        print(f"Validation predictions saved: {epoch_dir}")
    
    def _train_epoch(self, epoch):
        """
        Train DeepLabV3+ for one epoch.
        
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
        
        # Initialize metrics tracking for training
        train_metrics = {
            'iou': 0.0,
            'dice': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
        
        train_contour_metrics = {
            'iou': 0.0,
            'dice': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
        
        has_contours = False
        
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
                
                # Calculate metrics for mask
                with torch.no_grad():
                    batch_metrics = compute_metrics_batch(mask_pred, masks)
                    for k, v in batch_metrics.items():
                        train_metrics[k] += v
                    
                    # Calculate metrics for contour
                    batch_contour_metrics = compute_metrics_batch(contour_pred, contours)
                    for k, v in batch_contour_metrics.items():
                        train_contour_metrics[k] += v
                    
                    has_contours = True
                
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
                
                # Calculate metrics for mask
                with torch.no_grad():
                    batch_metrics = compute_metrics_batch(mask_pred, masks)
                    for k, v in batch_metrics.items():
                        train_metrics[k] += v
                
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
        
        # Calculate average loss and metrics
        avg_loss = running_loss / len(self.train_loader)
        avg_mask_loss = running_mask_loss / len(self.train_loader)
        
        # Average the metrics
        for k in train_metrics:
            train_metrics[k] /= len(self.train_loader)
        
        if has_contours:
            for k in train_contour_metrics:
                train_contour_metrics[k] /= len(self.train_loader)
        
        # Log epoch-level training metrics to wandb
        if self.use_wandb:
            log_dict = {
                'train/epoch_loss': avg_loss,
                'train/epoch_mask_loss': avg_mask_loss,
                'train/iou': train_metrics['iou'],
                'train/dice': train_metrics['dice'],
                'train/precision': train_metrics['precision'],
                'train/recall': train_metrics['recall'],
                'train/f1': train_metrics['f1'],
                'epoch': epoch
            }
            if running_contour_loss > 0:  # Only log if we have contour losses
                avg_contour_loss = running_contour_loss / len(self.train_loader)
                log_dict['train/epoch_contour_loss'] = avg_contour_loss
                log_dict['train/contour_iou'] = train_contour_metrics['iou']
                log_dict['train/contour_dice'] = train_contour_metrics['dice']
                log_dict['train/contour_precision'] = train_contour_metrics['precision']
                log_dict['train/contour_recall'] = train_contour_metrics['recall']
                log_dict['train/contour_f1'] = train_contour_metrics['f1']
            
            wandb.log(log_dict)
        
        return avg_loss
    
    def _validate_epoch(self, epoch):
        """
        Validate DeepLabV3+ for one epoch.
        
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
                    
                    # Store samples for visualization (first batch only)
                    if len(validation_images) == 0 and self.use_wandb:
                        validation_images.append({
                            'images': images[:4].clone(),  # Store first 4 samples
                            'masks': masks[:4].clone(),
                            'mask_preds': mask_pred[:4].clone(),
                            'contours': None,
                            'contour_preds': None
                        })
                    
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
        
        # Log validation metrics and visualizations to wandb
        if self.use_wandb:
            # Prepare validation metrics for logging
            val_log_dict = {
                'val/epoch_loss': avg_loss,
                'val/iou': all_metrics['iou'],
                'val/dice': all_metrics['dice'],
                'val/precision': all_metrics['precision'],
                'val/recall': all_metrics['recall'],
                'val/f1': all_metrics['f1'],
                'epoch': epoch
            }
            
            # Add contour metrics if available
            if has_contours:
                val_log_dict.update({
                    'val/contour_iou': all_metrics['contour_iou'],
                    'val/contour_dice': all_metrics['contour_dice'],
                    'val/contour_precision': all_metrics['contour_precision'],
                    'val/contour_recall': all_metrics['contour_recall'],
                    'val/contour_f1': all_metrics['contour_f1']
                })
            
            # Create and log visualizations
            if validation_images:
                sample_data = validation_images[0]
                if sample_data['contours'] is not None:
                    # Case with contours
                    wandb_images = self._create_validation_visualizations(
                        sample_data['images'],
                        sample_data['masks'],
                        sample_data['mask_preds'],
                        sample_data['contours'],
                        sample_data['contour_preds'],
                        max_samples=4
                    )
                else:
                    # Case without contours
                    wandb_images = self._create_validation_visualizations(
                        sample_data['images'],
                        sample_data['masks'],
                        sample_data['mask_preds'],
                        max_samples=4
                    )
                
                val_log_dict['val/predictions'] = wandb_images
            
            # Log everything to wandb
            wandb.log(val_log_dict)
        
        # Save model checkpoint and validation predictions
        self._save_model_checkpoint(epoch, avg_loss, all_metrics)
        if validation_images:
            self._save_validation_predictions(epoch, validation_images)
        
        return avg_loss, all_metrics
