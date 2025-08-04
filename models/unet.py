"""
UNet model implementation for building segmentation with contour awareness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, EncoderBlock, DecoderBlock


class UNet(nn.Module):
    """
    UNet architecture for segmentation with optional contour prediction.
    This model has an encoder-decoder structure with skip connections,
    and can output both mask and contour predictions.
    """
    def __init__(self, in_channels=3, out_channels_mask=1, out_channels_contour=1):
        """
        Initialize UNet model.
        
        Args:
            in_channels: Number of input channels (typically 3 for RGB)
            out_channels_mask: Number of output channels for the mask
            out_channels_contour: Number of output channels for the contour
        """
        super(UNet, self).__init__()
        
        # Encoder
        self.encoder1 = EncoderBlock(in_channels, 32)
        self.encoder2 = EncoderBlock(32, 64)
        self.encoder3 = EncoderBlock(64, 128)
        
        # Center
        self.center = ConvBlock(128, 256)
        
        # Decoder
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 32, 32)
        
        # Output layers
        self.out_conv_mask = nn.Conv2d(32, out_channels_mask, kernel_size=1)
        self.out_conv_contour = nn.Conv2d(32, out_channels_contour, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass through the UNet.
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]
            
        Returns:
            Tuple of (mask_output, contour_output)
        """
        # Encoder path
        e1_pool, e1 = self.encoder1(x)
        e2_pool, e2 = self.encoder2(e1_pool)
        e3_pool, e3 = self.encoder3(e2_pool)
        
        # Center
        center = self.center(e3_pool)
        
        # Decoder path
        d3 = self.decoder3(center, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        
        # Output
        mask_output = torch.sigmoid(self.out_conv_mask(d1))
        contour_output = torch.sigmoid(self.out_conv_contour(d1))
        
        return mask_output, contour_output


def get_unet_model(in_channels=3, out_channels_mask=1, out_channels_contour=1):
    """
    Factory function to create a UNet model.
    
    Args:
        in_channels: Number of input channels
        out_channels_mask: Number of output channels for the mask
        out_channels_contour: Number of output channels for the contour
        
    Returns:
        Initialized UNet model
    """
    model = UNet(
        in_channels=in_channels,
        out_channels_mask=out_channels_mask,
        out_channels_contour=out_channels_contour
    )
    return model
