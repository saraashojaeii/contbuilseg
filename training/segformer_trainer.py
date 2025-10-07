"""
SegFormer trainer for building segmentation tasks.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import wandb
import matplotlib.pyplot as plt

from .train import BaseTrainer
# from ..evaluation.metrics import compute_metrics


class SegFormerTrainer(BaseTrainer):
    """
    Trainer class for SegFormer model with semantic segmentation for buildings.
    """
    def __init__(self, model, train_loader, val_loader, device='cuda', 
                learning_rate=2e-5, model_save_dir='./checkpoints',
                use_wandb=True, wandb_project='building_seg', wandb_run_name=None,
                dataset_name='dataset'):
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
        super().__init__(model, train_loader, val_loader, device, learning_rate, model_save_dir)
        self.use_wandb = use_wandb
        self.dataset_name = dataset_name

        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    'learning_rate': learning_rate,
                    'device': device,
                    'model_type': 'SegFormer',
                    'dataset': dataset_name
                }
            )
            wandb.watch(self.model, log='all', log_freq=100)
        
    def _get_optimizer(self):
        """
        Get AdamW optimizer for SegFormer.
        
        Returns:
            AdamW optimizer
        """
        return AdamW(self.model.parameters(), lr=self.learning_rate)
    
    def _get_loss_fn(self):
        """
        Get cross entropy loss for SegFormer.
        
        Returns:
            Cross entropy loss function
        """
        # SegFormer uses CrossEntropyLoss internally in its forward method
        # when labels are provided, so we just return None here
        return None
    
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
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        step = 0
        for batch in pbar:
            # Move batch to device
            for k, v in batch.items():
                batch[k] = v.to(self.device)
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Forward pass (SegFormer computes loss internally when labels are provided)
            outputs = self.model(
                pixel_values=batch['pixel_values'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Step logging
            if self.use_wandb and step % 10 == 0:
                wandb.log({
                    'train/step_loss': loss.item(),
                    'train/step': step,
                    'epoch': epoch
                })
            step += 1
        
        # Calculate average loss
        avg_loss = running_loss / len(self.train_loader)
        
        # Epoch-level logging
        if self.use_wandb:
            wandb.log({
                'train/epoch_loss': avg_loss,
                'epoch': epoch
            })
        
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
        all_metrics = {
            'iou': 0.0,
            'dice': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
        
        # Class-specific metrics (for background, building, boundary)
        class_metrics = {
            'iou_bg': 0.0,
            'iou_building': 0.0,
            'iou_boundary': 0.0
        }
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        val_visual_logged = False
        with torch.no_grad():
            for batch in pbar:
                # Move batch to device
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    pixel_values=batch['pixel_values'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Get predictions
                predictions = torch.argmax(logits, dim=1)
                
                # Calculate metrics for each class
                for class_id in range(3):  # assuming 3 classes: background, building, boundary
                    # Create binary masks for this class
                    pred_mask = (predictions == class_id).float()
                    gt_mask = (batch['labels'] == class_id).float()
                    
                    # Calculate IoU for this class
                    intersection = torch.logical_and(pred_mask, gt_mask).sum().item()
                    union = torch.logical_or(pred_mask, gt_mask).sum().item()
                    iou = intersection / (union + 1e-6)
                    
                    # Store class-specific IoU
                    class_key = f'iou_{"bg" if class_id == 0 else "building" if class_id == 1 else "boundary"}'
                    class_metrics[class_key] += iou
                
                # Calculate overall metrics (considering all classes)
                # Mean IoU across classes
                miou = (class_metrics['iou_bg'] + class_metrics['iou_building'] + class_metrics['iou_boundary']) / 3
                all_metrics['iou'] += miou
                
                # For other metrics, focus on the building class (usually class_id=1)
                pred_building = (predictions == 1).float()
                gt_building = (batch['labels'] == 1).float()
                
                # Precision and recall for building class
                tp = torch.logical_and(pred_building, gt_building).sum().item()
                fp = torch.logical_and(pred_building, torch.logical_not(gt_building)).sum().item()
                fn = torch.logical_and(torch.logical_not(pred_building), gt_building).sum().item()
                
                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 = 2 * precision * recall / (precision + recall + 1e-6)
                
                all_metrics['precision'] += precision
                all_metrics['recall'] += recall
                all_metrics['f1'] += f1
                all_metrics['dice'] += f1  # Dice coefficient is equivalent to F1 score for binary classification
                
                # Update running loss
                running_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'miou': miou
                })

                # Log a small set of visuals from the first validation batch
                if self.use_wandb and not val_visual_logged:
                    try:
                        imgs = batch['pixel_values'][:4].detach().cpu()
                        preds = predictions[:4].detach().cpu()
                        gts = batch['labels'][:4].detach().cpu()

                        wandb_imgs = []
                        for i in range(imgs.shape[0]):
                            img = imgs[i]
                            # Convert CHW to HWC and normalize to 0-255 for display
                            if img.shape[0] in (1, 3):
                                np_img = img.numpy()
                                if np_img.shape[0] == 1:
                                    np_img = np_img[0]
                                else:
                                    np_img = np.transpose(np_img, (1, 2, 0))
                                # Simple min-max normalization for visualization
                                np_min, np_max = np_img.min(), np_img.max()
                                if np_max > np_min:
                                    np_img = (np_img - np_min) / (np_max - np_min)
                                np_img = (np_img * 255).astype(np.uint8)
                            else:
                                # Fallback to first 3 channels if present
                                np_img = imgs[i, :3].numpy()
                                np_img = np.transpose(np_img, (1, 2, 0))
                                np_min, np_max = np_img.min(), np_img.max()
                                if np_max > np_min:
                                    np_img = (np_img - np_min) / (np_max - np_min)
                                np_img = (np_img * 255).astype(np.uint8)

                            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                            axes[0].imshow(np_img, cmap='gray' if np_img.ndim == 2 else None)
                            axes[0].set_title('Input')
                            axes[0].axis('off')

                            axes[1].imshow(gts[i].numpy(), vmin=0, vmax=2, cmap='tab20')
                            axes[1].set_title('GT')
                            axes[1].axis('off')

                            axes[2].imshow(preds[i].numpy(), vmin=0, vmax=2, cmap='tab20')
                            axes[2].set_title('Pred')
                            axes[2].axis('off')

                            plt.tight_layout()
                            wandb_imgs.append(wandb.Image(fig, caption=f'Sample {i+1}'))
                            plt.close(fig)

                        wandb.log({'val/predictions': wandb_imgs, 'epoch': epoch})
                    except Exception:
                        pass
                    val_visual_logged = True
        
        # Calculate average loss and metrics
        avg_loss = running_loss / len(self.val_loader)
        
        for k in all_metrics:
            all_metrics[k] /= len(self.val_loader)
        
        for k in class_metrics:
            class_metrics[k] /= len(self.val_loader)
        
        # Add class-specific metrics to the metrics dictionary
        all_metrics.update(class_metrics)
        
        # Log validation metrics
        if self.use_wandb:
            log_dict = {
                'val/epoch_loss': avg_loss,
                'val/iou': all_metrics['iou'],
                'val/dice': all_metrics['dice'],
                'val/precision': all_metrics['precision'],
                'val/recall': all_metrics['recall'],
                'val/f1': all_metrics['f1'],
                'val/iou_bg': all_metrics['iou_bg'],
                'val/iou_building': all_metrics['iou_building'],
                'val/iou_boundary': all_metrics['iou_boundary'],
                'epoch': epoch
            }
            wandb.log(log_dict)

        return avg_loss, all_metrics
    
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
        
        self.model.save_pretrained(save_path)
        
        print(f"Model saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load SegFormer model using the Hugging Face from_pretrained method.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        from transformers import SegformerForSemanticSegmentation
        
        self.model = SegformerForSemanticSegmentation.from_pretrained(checkpoint_path)
        self.model.to(self.device)
        
        print(f"Loaded model from {checkpoint_path}")
