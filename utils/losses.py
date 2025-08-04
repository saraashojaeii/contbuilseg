"""
Loss functions for building segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks. Computes the Dice Loss between predictions and ground truth.
    Dice coefficient = 2*|X∩Y|/(|X|+|Y|) where X is the prediction and Y is the ground truth.
    """
    def __init__(self, smooth=1.0):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to prevent division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        """
        Forward pass.
        
        Args:
            predictions: Model predictions, shape [batch_size, channels, height, width]
            targets: Ground truth, shape [batch_size, channels, height, width]
            
        Returns:
            Dice loss
        """
        # Flatten the predictions and targets
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss
        return 1. - dice


class BCEWithDiceLoss(nn.Module):
    """
    Combined loss of Binary Cross Entropy and Dice Loss.
    This loss function is useful for segmentation tasks as it combines pixel-wise classification
    with region-based segmentation quality.
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        """
        Initialize the combined loss.
        
        Args:
            dice_weight: Weight for the Dice Loss component
            bce_weight: Weight for the BCE Loss component
        """
        super(BCEWithDiceLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
    def forward(self, predictions, targets):
        """
        Forward pass.
        
        Args:
            predictions: Model predictions, shape [batch_size, channels, height, width]
            targets: Ground truth, shape [batch_size, channels, height, width]
            
        Returns:
            Combined BCE and Dice loss
        """
        # Calculate both losses
        dice_loss = self.dice_loss(predictions, targets)
        bce_loss = self.bce_loss(predictions, targets)
        
        # Return weighted combination
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for dealing with class imbalance.
    FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weight factor for the rare class
            gamma: Focusing parameter to reduce loss for well-classified examples
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6
        
    def forward(self, inputs, targets):
        """
        Forward pass.
        
        Args:
            inputs: Model predictions, shape [batch_size, channels, height, width]
            targets: Ground truth, shape [batch_size, channels, height, width]
            
        Returns:
            Focal loss
        """
        # Apply sigmoid if needed
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs)
        
        if not torch.is_tensor(targets):
            targets = torch.tensor(targets)
            
        # Get probabilities
        p = torch.sigmoid(inputs)
        # Clip values to prevent log(0) errors
        p = torch.clamp(p, min=self.eps, max=1.0-self.eps)
        
        # Calculate binary cross entropy loss
        ce_loss = F.binary_cross_entropy(p, targets, reduction='none')
        
        # Calculate weights for focal loss
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha is not None:
            alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_weight * focal_weight
            
        # Calculate focal loss
        loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
