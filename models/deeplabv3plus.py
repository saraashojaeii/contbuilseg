"""
DeepLabV3+ model implementation for building segmentation with contour awareness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.
    """
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        
        # 1x1 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Atrous convolutions with different rates
        self.atrous_convs = nn.ModuleList()
        for rate in atrous_rates:
            self.atrous_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Global average pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Projection layer
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
    def forward(self, x):
        size = x.shape[-2:]
        
        # Apply all branches
        feat1 = self.conv1(x)
        
        atrous_feats = []
        for atrous_conv in self.atrous_convs:
            atrous_feats.append(atrous_conv(x))
        
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=False)
        
        # Concatenate all features
        x = torch.cat([feat1] + atrous_feats + [global_feat], dim=1)
        
        # Project to output channels
        x = self.project(x)
        
        return x


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ architecture for segmentation with optional contour prediction.
    Uses ResNet50 as backbone with ASPP module and decoder.
    """
    def __init__(self, in_channels=3, out_channels_mask=1, out_channels_contour=1, backbone='resnet50'):
        """
        Initialize DeepLabV3+ model.
        
        Args:
            in_channels: Number of input channels (typically 3 for RGB)
            out_channels_mask: Number of output channels for the mask
            out_channels_contour: Number of output channels for the contour
            backbone: Backbone network (default: 'resnet50')
        """
        super(DeepLabV3Plus, self).__init__()
        
        # Load pretrained ResNet50 backbone
        if backbone == 'resnet50':
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Encoder (ResNet50 layers)
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1  # 256 channels, stride 4
        self.layer2 = resnet.layer2  # 512 channels, stride 8
        self.layer3 = resnet.layer3  # 1024 channels, stride 16
        self.layer4 = resnet.layer4  # 2048 channels, stride 16 (with dilation)
        
        # Modify layer4 to use dilation instead of stride
        self._modify_resnet_dilation(self.layer4, dilation=2)
        
        # ASPP module
        self.aspp = ASPP(2048, 256)
        
        # Decoder
        # Low-level feature projection
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder convolutions
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Output heads
        self.mask_head = nn.Conv2d(256, out_channels_mask, 1)
        self.contour_head = nn.Conv2d(256, out_channels_contour, 1)
        
    def _modify_resnet_dilation(self, layer, dilation):
        """Modify ResNet layer to use dilation instead of stride."""
        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride = (1, 1)
                if module.kernel_size == (3, 3):
                    module.dilation = (dilation, dilation)
                    module.padding = (dilation, dilation)
    
    def forward(self, x):
        """
        Forward pass through DeepLabV3+.
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]
            
        Returns:
            Tuple of (mask_output, contour_output)
        """
        input_size = x.shape[-2:]
        
        # Encoder
        x = self.layer0(x)
        low_level_feat = self.layer1(x)  # For skip connection
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # ASPP
        x = self.aspp(x)
        
        # Upsample ASPP output
        x = F.interpolate(x, size=low_level_feat.shape[-2:], mode='bilinear', align_corners=False)
        
        # Process low-level features
        low_level_feat = self.low_level_conv(low_level_feat)
        
        # Concatenate with low-level features
        x = torch.cat([x, low_level_feat], dim=1)
        
        # Decoder
        x = self.decoder(x)
        
        # Upsample to input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        # Output heads
        mask_output = torch.sigmoid(self.mask_head(x))
        # Contour output is distance transform (continuous values), no sigmoid
        contour_output = self.contour_head(x)
        
        return mask_output, contour_output


def get_deeplabv3plus_model(in_channels=3, out_channels_mask=1, out_channels_contour=1, backbone='resnet50'):
    """
    Factory function to create a DeepLabV3+ model.
    
    Args:
        in_channels: Number of input channels
        out_channels_mask: Number of output channels for the mask
        out_channels_contour: Number of output channels for the contour
        backbone: Backbone network (default: 'resnet50')
        
    Returns:
        Initialized DeepLabV3+ model
    """
    model = DeepLabV3Plus(
        in_channels=in_channels,
        out_channels_mask=out_channels_mask,
        out_channels_contour=out_channels_contour,
        backbone=backbone
    )
    return model
