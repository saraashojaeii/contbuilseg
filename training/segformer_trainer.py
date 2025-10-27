"""
SegFormer trainer for building segmentation tasks.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import time
import wandb
import matplotlib.pyplot as plt 

from .train import BaseTrainer
from utils.losses import BCEWithDiceLoss, BCEWithLogitsDiceLoss, L1Loss, MergeSeparationLoss
from evaluation.metrics import compute_metrics_batch


class SegFormerTrainer(BaseTrainer):
    """
    Trainer class for SegFormer model with semantic segmentation for buildings.
    """
    def __init__(self, model, train_loader, val_loader, device='cuda', 
                learning_rate=2e-5, model_save_dir='./checkpoints',
                use_wandb=True, wandb_project='building_seg', wandb_run_name=None,
                dataset_name='dataset', mask_weight=0.7, contour_weight=0.3,
                merge_weight=0.0, merge_boundary_width: int = 1):
        """
        Initialize the SegFormerTrainer.
        
        Args:
            model: The SegFormer model to train (from Hugging Face transformers)
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to use for training ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            model_save_dir: Directory to save model checkpoints
            use_wandb: Whether to enable Weights & Biases logging
            wandb_project: W&B project name
            wandb_run_name: Optional W&B run name
            dataset_name: Dataset name for logging context
        """
        # Set attributes needed by _get_loss_fn BEFORE calling BaseTrainer.__init__
        self.use_wandb = use_wandb
        self.dataset_name = dataset_name
        self.mask_weight = mask_weight
        self.contour_weight = contour_weight
        self.merge_weight = merge_weight
        self.merge_boundary_width = merge_boundary_width
        
        # Now call BaseTrainer initializer (which invokes _get_loss_fn)
        super().__init__(model, train_loader, val_loader, device, learning_rate, model_save_dir)

        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    'learning_rate': learning_rate,
                    'device': device,
                    'model_type': 'SegFormer',
                    'dataset': dataset_name,
                    'mask_weight': mask_weight,
                    'contour_weight': contour_weight,
                    'merge_weight': merge_weight,
                    'merge_boundary_width': merge_boundary_width
                }
            )
            wandb.watch(self.model, log='all', log_freq=100)
        
    def _get_optimizer(self):
        """
        Get AdamW optimizer for SegFormer.
        
        Returns:
            AdamW optimizer
        """
        # Two-tier learning rates: smaller for encoder, larger for heads
        lr_backbone = self.learning_rate
        lr_head = self.learning_rate * 10.0

        encoder_params = []
        head_params = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # Parameters inside the HF encoder live under 'backbone.segformer'
            if 'backbone.segformer' in name:
                encoder_params.append(p)
            else:
                # includes backbone.decode_head and our contour_head
                head_params.append(p)

        param_groups = [
            { 'params': encoder_params, 'lr': lr_backbone },
            { 'params': head_params, 'lr': lr_head },
        ]
        return AdamW(param_groups, lr=lr_backbone)
    
    def _get_loss_fn(self):
        """Return combined loss fns for mask and contour."""
        return {
            'mask': BCEWithLogitsDiceLoss(dice_weight=0.5, bce_weight=0.5),
            'contour': L1Loss(reduction='mean'),
            'merge': MergeSeparationLoss(boundary_width=self.merge_boundary_width)
        }
    
    def _train_epoch(self, epoch):
        """
        Train SegFormer for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        running_loss = 0.0
        running_mask_loss = 0.0
        running_contour_loss = 0.0
        running_merge_loss = 0.0
        step = 0
        
        # Initialize metrics tracking for training
        train_metrics = {
            'iou': 0.0,
            'dice': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0
        }
        
        train_contour_metrics = {
            'iou': 0.0,
            'dice': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0
        }
        
        has_contours = False
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch in pbar:
            pixel_values = batch['pixel_values'].to(self.device)
            masks = batch['mask'].to(self.device)
            contours = batch.get('contours')
            if contours is not None:
                contours = contours.to(self.device)

            self.optimizer.zero_grad()

            mask_logits, contour_map = self.model(pixel_values)
            # Align predictions to GT spatial size if needed
            if mask_logits.shape[-2:] != masks.shape[-2:]:
                mask_logits = torch.nn.functional.interpolate(
                    mask_logits, size=masks.shape[-2:], mode='bilinear', align_corners=False
                )
                if contour_map is not None and contour_map.shape[-2:] != masks.shape[-2:]:
                    contour_map = torch.nn.functional.interpolate(
                        contour_map, size=masks.shape[-2:], mode='bilinear', align_corners=False
                    )
            # loss on logits
            mask_loss = self.loss_fn['mask'](mask_logits, masks)
            merge_loss = self.loss_fn['merge'](mask_logits, masks)
            # probabilities for metrics only
            mask_pred = torch.sigmoid(mask_logits)
            if contours is not None:
                contour_loss = self.loss_fn['contour'](contour_map, contours)
                loss = (
                    self.mask_weight * mask_loss +
                    self.contour_weight * contour_loss +
                    self.merge_weight * merge_loss
                )
            else:
                contour_loss = torch.tensor(0.0, device=self.device)
                loss = self.mask_weight * mask_loss + self.merge_weight * merge_loss

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_mask_loss += mask_loss.item()
            running_contour_loss += contour_loss.item() if contours is not None else 0.0
            # Track merge loss separately
            running_merge_loss += merge_loss.item()

            # Metrics
            with torch.no_grad():
                batch_metrics = compute_metrics_batch(mask_pred, masks)
                for k, v in batch_metrics.items():
                    train_metrics[k] += v
                
                # Calculate metrics for contour if available
                if contours is not None:
                    batch_contour_metrics = compute_metrics_batch(contour_map, contours)
                    for k, v in batch_contour_metrics.items():
                        train_contour_metrics[k] += v
                    has_contours = True

            pbar.set_postfix({
                'loss': loss.item(),
                'mask_loss': mask_loss.item(),
                'contour_loss': contour_loss.item() if contours is not None else 0.0,
                'merge_loss': merge_loss.item(),
                'iou': batch_metrics['iou']
            })

            if self.use_wandb and step % 10 == 0:
                log = {
                    'train/step_loss': loss.item(),
                    'train/step_mask_loss': mask_loss.item(),
                    'train/step': step,
                    'epoch': epoch
                }
                if contours is not None:
                    log['train/step_contour_loss'] = contour_loss.item()
                log['train/step_merge_loss'] = merge_loss.item()
                wandb.log(log)
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
            if running_contour_loss > 0:
                avg_contour_loss = running_contour_loss / len(self.train_loader)
                log_dict['train/epoch_contour_loss'] = avg_contour_loss
                log_dict['train/contour_iou'] = train_contour_metrics['iou']
                log_dict['train/contour_dice'] = train_contour_metrics['dice']
                log_dict['train/contour_precision'] = train_contour_metrics['precision']
                log_dict['train/contour_recall'] = train_contour_metrics['recall']
                log_dict['train/contour_f1'] = train_contour_metrics['f1']
            if running_merge_loss > 0:
                avg_merge_loss = running_merge_loss / len(self.train_loader)
                log_dict['train/epoch_merge_loss'] = avg_merge_loss
            wandb.log(log_dict)
        
        return avg_loss
    
    def _validate_epoch(self, epoch):
        """
        Validate SegFormer for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average validation loss, metrics dictionary)
        """
        self.model.eval()
        running_loss = 0.0
        running_merge_loss = 0.0
        # Accumulators for metrics
        sum_metrics = { 'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0 }
        contour_metrics = { 'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0 }
        has_contours = False
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        # Store first batch for visualization
        first_batch_data = None
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                pixel_values = batch['pixel_values'].to(self.device)
                masks = batch['mask'].to(self.device)
                contours = batch.get('contours')
                if contours is not None:
                    contours = contours.to(self.device)

                mask_logits, contour_map = self.model(pixel_values)
                # Align predictions to GT spatial size if needed
                if mask_logits.shape[-2:] != masks.shape[-2:]:
                    mask_logits = torch.nn.functional.interpolate(
                        mask_logits, size=masks.shape[-2:], mode='bilinear', align_corners=False
                    )
                    if contour_map is not None and contour_map.shape[-2:] != masks.shape[-2:]:
                        contour_map = torch.nn.functional.interpolate(
                            contour_map, size=masks.shape[-2:], mode='bilinear', align_corners=False
                        )
                # loss on logits
                mask_loss = self.loss_fn['mask'](mask_logits, masks)
                merge_loss = self.loss_fn['merge'](mask_logits, masks)
                mask_pred = torch.sigmoid(mask_logits)
                if contours is not None:
                    contour_loss = self.loss_fn['contour'](contour_map, contours)
                    loss = (
                        self.mask_weight * mask_loss +
                        self.contour_weight * contour_loss +
                        self.merge_weight * merge_loss
                    )
                else:
                    loss = self.mask_weight * mask_loss + self.merge_weight * merge_loss

                metrics = compute_metrics_batch(mask_pred, masks)
                # accumulate
                for k in sum_metrics:
                    sum_metrics[k] += metrics[k]
                
                # Calculate contour metrics if available
                if contours is not None and contour_map is not None:
                    batch_contour_metrics = compute_metrics_batch(contour_map, contours)
                    for k in contour_metrics:
                        contour_metrics[k] += batch_contour_metrics[k]
                    has_contours = True
                
                running_loss += loss.item()
                running_merge_loss += merge_loss.item()

                pbar.set_postfix({'loss': loss.item(), 'merge': merge_loss.item(), 'iou': metrics['iou']})

                # Store first batch for visualization later
                if batch_idx == 0 and self.use_wandb:
                    first_batch_data = {
                        'pixel_values': pixel_values[:4].detach().cpu(),
                        'masks': masks[:4].detach().cpu(),
                        'mask_pred': mask_pred[:4].detach().cpu(),
                        'contours': contours[:4].detach().cpu() if contours is not None else None,
                        'contour_map': contour_map[:4].detach().cpu() if contour_map is not None else None
                    }
        
        # Calculate average loss and metrics
        avg_loss = running_loss / len(self.val_loader)
        # Average metrics over validation set
        avg_metrics = {k: (v / len(self.val_loader)) for k, v in sum_metrics.items()}
        if has_contours:
            for k in contour_metrics:
                contour_metrics[k] /= len(self.val_loader)
            avg_metrics.update({f'contour_{k}': v for k, v in contour_metrics.items()})
        
        # Log validation metrics and visualizations
        if self.use_wandb:
            print(f"Logging validation results to wandb (use_wandb={self.use_wandb})...")
            
            # Log metrics
            val_log = {
                'val/epoch_loss': avg_loss,
                'val/iou': avg_metrics['iou'],
                'val/dice': avg_metrics['dice'],
                'val/precision': avg_metrics['precision'],
                'val/recall': avg_metrics['recall'],
                'val/f1': avg_metrics['f1'],
                'epoch': epoch
            }
            if running_merge_loss > 0:
                val_log['val/epoch_merge_loss'] = running_merge_loss / len(self.val_loader)
            if has_contours:
                val_log.update({
                    'val/contour_iou': avg_metrics.get('contour_iou', 0.0),
                    'val/contour_dice': avg_metrics.get('contour_dice', 0.0),
                    'val/contour_precision': avg_metrics.get('contour_precision', 0.0),
                    'val/contour_recall': avg_metrics.get('contour_recall', 0.0),
                    'val/contour_f1': avg_metrics.get('contour_f1', 0.0)
                })
            wandb.log(val_log)
            
            # Log visualizations if we have data
            if first_batch_data is not None:
                try:
                    imgs = first_batch_data['pixel_values']
                    preds = first_batch_data['mask_pred']
                    gts = first_batch_data['masks']
                    contours = first_batch_data['contours']
                    contour_map = first_batch_data['contour_map']

                    wandb_imgs = []
                    for i in range(imgs.shape[0]):
                        img = imgs[i]
                        if img.shape[0] in (1, 3):
                            np_img = img.numpy()
                            if np_img.shape[0] == 1:
                                np_img = np_img[0]
                            else:
                                np_img = np.transpose(np_img, (1, 2, 0))
                            np_min, np_max = np_img.min(), np_img.max()
                            if np_max > np_min:
                                np_img = (np_img - np_min) / (np_max - np_min)
                            np_img = (np_img * 255).astype(np.uint8)
                        else:
                            np_img = imgs[i, :3].numpy()
                            np_img = np.transpose(np_img, (1, 2, 0))
                            np_min, np_max = np_img.min(), np_img.max()
                            if np_max > np_min:
                                np_img = (np_img - np_min) / (np_max - np_min)
                            np_img = (np_img * 255).astype(np.uint8)

                        if contours is not None and contour_map is not None:
                            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                            axes[0,0].imshow(np_img, cmap='gray' if np_img.ndim==2 else None); axes[0,0].set_title('Input'); axes[0,0].axis('off')
                            axes[0,1].imshow(gts[i].squeeze(), cmap='Blues'); axes[0,1].set_title('Mask GT'); axes[0,1].axis('off')
                            axes[0,2].imshow(preds[i].squeeze(), cmap='Blues'); axes[0,2].set_title('Mask Pred'); axes[0,2].axis('off')
                            axes[1,1].imshow(contours[i].squeeze(), cmap='Reds'); axes[1,1].set_title('Contour GT'); axes[1,1].axis('off')
                            axes[1,2].imshow(contour_map[i].squeeze(), cmap='Reds'); axes[1,2].set_title('Contour Pred'); axes[1,2].axis('off')
                            axes[1,0].axis('off')
                        else:
                            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                            axes[0].imshow(np_img, cmap='gray' if np_img.ndim==2 else None); axes[0].set_title('Input'); axes[0].axis('off')
                            axes[1].imshow(gts[i].squeeze(), cmap='Blues'); axes[1].set_title('Mask GT'); axes[1].axis('off')
                            axes[2].imshow(preds[i].squeeze(), cmap='Blues'); axes[2].set_title('Mask Pred'); axes[2].axis('off')

                        plt.tight_layout()
                        wandb_imgs.append(wandb.Image(fig, caption=f'Sample {i+1}'))
                        plt.close(fig)

                    wandb.log({'val/predictions': wandb_imgs, 'epoch': epoch})
                    print(f"✓ Logged {len(wandb_imgs)} validation visualizations to wandb")
                except Exception as e:
                    print(f"✗ Warning: Failed to log validation visualizations to wandb: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("✗ No validation batch data available for visualization")
        
        return avg_loss, avg_metrics
    
    def save_model(self, epoch, final=False):
        """
        Save SegFormer model using the Hugging Face save_pretrained method.
        
        Args:
            epoch: Current epoch number
            final: Whether this is the final model after training
        """
        if final:
            save_path = f"{self.model_save_dir}/final_model"
        else:
            save_path = f"{self.model_save_dir}/epoch_{epoch}"

        # Support either HF SegformerForSemanticSegmentation or our DualHeadSegFormer with .backbone
        try:
            if hasattr(self.model, 'save_pretrained'):
                self.model.save_pretrained(save_path)
            elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'save_pretrained'):
                self.model.backbone.save_pretrained(save_path)
            else:
                # Fallback to state_dict
                import torch
                torch.save(self.model.state_dict(), f"{save_path}.pt")
        finally:
            print(f"Model saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load SegFormer model using the Hugging Face from_pretrained method.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        from transformers import SegformerForSemanticSegmentation

        # Try to load as HF checkpoint first
        try:
            loaded = SegformerForSemanticSegmentation.from_pretrained(checkpoint_path)
            # If current model has a .backbone (our wrapper), swap it in
            if hasattr(self.model, 'backbone'):
                self.model.backbone = loaded.to(self.device)
            else:
                self.model = loaded.to(self.device)
        except Exception:
            # Fallback to plain state_dict
            import torch
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state)

        print(f"Loaded model from {checkpoint_path}")

    def train(self, num_epochs=100, save_every=1):
        """
        Train the SegFormer model with model saving after each epoch.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save model after every epoch (set to 1 for every epoch)

        Returns:
            Training history
        """
        print(f"Starting SegFormer training for {num_epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self._train_epoch(epoch)
            self.history['train_loss'].append(train_loss)

            # Validate
            val_loss, metrics = self._validate_epoch(epoch)
            self.history['val_loss'].append(val_loss)
            self.history['metrics'].append(metrics)

            # Print progress
            print(f"Epoch {epoch}/{num_epochs}, "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"IoU: {metrics['iou']:.4f}, "
                  f"F1: {metrics['f1']:.4f}")

            # Save model after each epoch
            self.save_model(epoch, final=False)

            # Plot losses at regular intervals or at the end
            if epoch % max(10, num_epochs // 10) == 0 or epoch == num_epochs:
                self._plot_losses()

        # Save final model
        self.save_model(num_epochs, final=True)

        # Calculate training time
        training_time = time.time() - start_time
        print(f"Training completed in {training_time/60:.2f} minutes")

        return self.history
