"""
SegFormer trainer for building segmentation tasks.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

from .train import BaseTrainer
from ..evaluation.metrics import compute_metrics


class SegFormerTrainer(BaseTrainer):
    """
    Trainer class for SegFormer model with semantic segmentation for buildings.
    """
    def __init__(self, model, train_loader, val_loader, device='cuda', 
                learning_rate=2e-5, model_save_dir='./checkpoints'):
        """
        Initialize the SegFormerTrainer.
        
        Args:
            model: The SegFormer model to train (from Hugging Face transformers)
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to use for training ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            model_save_dir: Directory to save model checkpoints
        """
        super().__init__(model, train_loader, val_loader, device, learning_rate, model_save_dir)
        
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
        
        # Calculate average loss
        avg_loss = running_loss / len(self.train_loader)
        
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
        
        # Calculate average loss and metrics
        avg_loss = running_loss / len(self.val_loader)
        
        for k in all_metrics:
            all_metrics[k] /= len(self.val_loader)
        
        for k in class_metrics:
            class_metrics[k] /= len(self.val_loader)
        
        # Add class-specific metrics to the metrics dictionary
        all_metrics.update(class_metrics)
        
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
