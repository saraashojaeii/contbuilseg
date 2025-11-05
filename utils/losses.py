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


class BCEWithLogitsDiceLoss(nn.Module):
    """
    Combined BCEWithLogits + Dice loss. Safer numerically than BCE on probabilities.
    Applies sigmoid internally for the Dice component and to compute probs.
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.bce_logits = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        # BCE on logits
        bce = self.bce_logits(logits, targets)
        # Dice on probabilities from logits
        probs = torch.sigmoid(logits)
        dice = self.dice(probs, targets)
        return self.bce_weight * bce + self.dice_weight * dice


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


class MergeSeparationLoss(nn.Module):
    """
    Differentiable proxy loss for merge rate.
    Penalizes high mask probabilities along ground-truth boundaries to encourage
    separation between adjacent instances (reducing merges).

    Implementation:
    - Compute a thin GT boundary mask from binary targets using a convolutional
      gradient kernel and optional dilation.
    - Apply BCE on predicted probabilities against zeros ONLY on boundary pixels.
    - Normalizes by the number of boundary pixels to keep scale stable.
    """
    def __init__(self, boundary_width: int = 1):
        super().__init__()
        self.boundary_width = max(int(boundary_width), 1)
        # 3x3 gradient kernel to detect edges
        k = torch.tensor([[0., 1., 0.],
                          [1., -4., 1.],
                          [0., 1., 0.]], dtype=torch.float32)
        self.register_buffer('laplace_kernel', k.view(1, 1, 3, 3))
        self.eps = 1e-6

    def _make_boundary_mask(self, targets: torch.Tensor) -> torch.Tensor:
        """Create a boundary mask from GT binary masks.
        targets: [B, 1 or C, H, W] or [B, H, W]
        Returns: [B, 1, H, W] float mask in {0,1}
        """
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        # Ensure single-channel for boundary derivation
        if targets.size(1) > 1:
            # If multi-channel, assume binary mask in channel 0
            t = targets[:, :1, ...]
        else:
            t = targets
        t = t.float()
        # Normalize to [0,1]
        t = (t > 0.5).float()

        # Convolve with Laplacian to detect edges (ensure kernel is on same device)
        kernel = self.laplace_kernel.to(t.device)
        edges = F.conv2d(t, kernel, padding=1)
        edges = edges.abs()
        # Binarize edges
        boundary = (edges > 0).float()

        # Optional dilation by max-pooling to thicken boundary
        # Use an odd kernel size to preserve spatial dimensions exactly
        if self.boundary_width > 1:
            radius = max(self.boundary_width // 2, 1)
            k = 2 * radius + 1  # force odd
            pad = radius
            boundary = F.max_pool2d(boundary, kernel_size=k, stride=1, padding=pad)
        return boundary

    def forward(self, logits_or_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits_or_probs: raw mask logits or probabilities [B, 1 or C, H, W]
            targets: GT mask [B, 1 or C, H, W] or [B, H, W]
        Returns:
            Scalar loss. Lower is better (fewer merges).
        """
        x = logits_or_probs
        # Accept either logits or probabilities; detect by range
        with torch.no_grad():
            xmin = x.min().item()
            xmax = x.max().item()
        boundary = self._make_boundary_mask(targets)

        # Create target zeros on boundary pixels; ignore elsewhere
        # Prefer logits-safe BCE under AMP when input is logits
        if xmin < 0.0 or xmax > 1.0:
            # logits path (safe for autocast)
            bce = F.binary_cross_entropy_with_logits(x, torch.zeros_like(x), reduction='none')
            masked = bce * boundary
        else:
            # probability path (avoid numerical issues if not in AMP path)
            probs = torch.clamp(x, 0.0, 1.0)
            bce = F.binary_cross_entropy(probs, torch.zeros_like(probs), reduction='none')
            masked = bce * boundary
        denom = boundary.sum() + self.eps
        return masked.sum() / denom


class L1Loss(nn.Module):
    """
    L1 Loss (Mean Absolute Error) for regression tasks.
    Used for inverted saturated distance transform contour prediction.
    """
    def __init__(self, reduction='mean'):
        """
        Initialize L1 Loss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super(L1Loss, self).__init__()
        self.reduction = reduction
        
    def forward(self, predictions, targets):
        """
        Forward pass.
        
        Args:
            predictions: Model predictions, shape [batch_size, channels, height, width]
            targets: Ground truth, shape [batch_size, channels, height, width]
            
        Returns:
            L1 loss
        """
        loss = torch.abs(predictions - targets)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
