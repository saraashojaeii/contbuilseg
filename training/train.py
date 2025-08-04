"""
Base trainer class for building segmentation models.
"""

import os
import time
import numpy as np
import torch
from torch.optim import Adam, AdamW
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..evaluation.metrics import compute_metrics, compute_metrics_batch


class BaseTrainer:
    """
    Base trainer class for segmentation models.
    """
    def __init__(self, model, train_loader, val_loader, device='cuda', 
                 learning_rate=1e-4, model_save_dir='./checkpoints'):
        """
        Initialize the BaseTrainer.
        
        Args:
            model: The PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to use for training ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            model_save_dir: Directory to save model checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.model_save_dir = model_save_dir
        
        # Create model save directory if it doesn't exist
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._get_optimizer()
        
        # Initialize loss function
        self.loss_fn = self._get_loss_fn()
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': []
        }
    
    def _get_optimizer(self):
        """
        Get optimizer for training.
        This method should be overridden by subclasses.
        
        Returns:
            PyTorch optimizer
        """
        return AdamW(self.model.parameters(), lr=self.learning_rate)
    
    def _get_loss_fn(self):
        """
        Get loss function for training.
        This method should be overridden by subclasses.
        
        Returns:
            PyTorch loss function
        """
        raise NotImplementedError("Subclasses must implement _get_loss_fn")
    
    def _train_epoch(self, epoch):
        """
        Train for one epoch.
        This method should be overridden by subclasses.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        raise NotImplementedError("Subclasses must implement _train_epoch")
    
    def _validate_epoch(self, epoch):
        """
        Validate for one epoch.
        This method should be overridden by subclasses.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average validation loss, metrics dictionary)
        """
        raise NotImplementedError("Subclasses must implement _validate_epoch")
    
    def train(self, num_epochs=100, save_every=10):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Frequency of epochs to save model and plot losses
            
        Returns:
            Training history
        """
        print(f"Starting training for {num_epochs} epochs...")
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
            
            # Save model and plot losses at regular intervals
            if epoch % save_every == 0:
                self._save_model(epoch)
                self._plot_losses()
        
        # Save final model
        self._save_model(num_epochs, final=True)
        
        # Calculate training time
        training_time = time.time() - start_time
        print(f"Training completed in {training_time/60:.2f} minutes")
        
        return self.history
    
    def _save_model(self, epoch, final=False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            final: Whether this is the final model after training
        """
        if final:
            save_path = os.path.join(self.model_save_dir, "final_model.pth")
        else:
            save_path = os.path.join(self.model_save_dir, f"epoch_{epoch}.pth")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'metrics': self.history['metrics']
        }, save_path)
        
        print(f"Model saved to {save_path}")
    
    def _plot_losses(self):
        """
        Plot training and validation losses.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.model_save_dir, 'loss_plot.png'))
        plt.close()
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load history if available
        if 'train_loss' in checkpoint:
            self.history['train_loss'] = checkpoint['train_loss']
        if 'val_loss' in checkpoint:
            self.history['val_loss'] = checkpoint['val_loss']
        if 'metrics' in checkpoint:
            self.history['metrics'] = checkpoint['metrics']
        
        print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")
