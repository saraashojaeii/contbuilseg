"""
Mask2Former model implementation for building segmentation with contour awareness.
Uses transformer-based architecture for universal image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig


class Mask2FormerSegmentation(nn.Module):
    """
    Mask2Former architecture adapted for building segmentation with optional contour prediction.
    Uses pretrained Mask2Former from HuggingFace transformers.
    """
    def __init__(self, in_channels=3, out_channels_mask=1, out_channels_contour=1, 
                 pretrained=True, model_name="facebook/mask2former-swin-tiny-ade-semantic"):
        """
        Initialize Mask2Former model.
        
        Args:
            in_channels: Number of input channels (typically 3 for RGB)
            out_channels_mask: Number of output channels for the mask
            out_channels_contour: Number of output channels for the contour
            pretrained: Whether to use pretrained weights
            model_name: Name of the pretrained model to use
        """
        super(Mask2FormerSegmentation, self).__init__()
        
        self.out_channels_mask = out_channels_mask
        self.out_channels_contour = out_channels_contour
        
        if pretrained:
            # Load pretrained Mask2Former
            self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
            
            # Modify the final classification head for binary segmentation
            # The original model has a class predictor, we'll adapt it
            config = self.mask2former.config
            hidden_dim = config.hidden_dim
            
            # Create custom heads for mask and contour
            self.mask_head = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, out_channels_mask, kernel_size=1)
            )
            
            self.contour_head = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, out_channels_contour, kernel_size=1)
            )
        else:
            # Create from scratch with custom configuration
            config = Mask2FormerConfig(
                num_labels=2,  # Binary segmentation
                hidden_dim=256,
                num_queries=100
            )
            self.mask2former = Mask2FormerForUniversalSegmentation(config)
            
            hidden_dim = config.hidden_dim
            
            self.mask_head = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, out_channels_mask, kernel_size=1)
            )
            
            self.contour_head = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, out_channels_contour, kernel_size=1)
            )
        
        # Freeze backbone if using pretrained (optional - can be unfrozen for fine-tuning)
        # Uncomment the following lines to freeze the backbone
        # for param in self.mask2former.model.pixel_level_module.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through Mask2Former.
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]
            
        Returns:
            Tuple of (mask_output, contour_output)
        """
        B, C, H, W = x.shape
        
        # Get outputs from Mask2Former
        outputs = self.mask2former(pixel_values=x, output_hidden_states=True)
        
        # Extract mask predictions
        # Mask2Former outputs class predictions and mask predictions
        # We'll use the mask predictions and process them
        masks_queries_logits = outputs.masks_queries_logits  # [B, num_queries, H/4, W/4]
        
        # Aggregate queries into a single segmentation map
        # Take the mean or max across queries
        if masks_queries_logits.dim() == 4:
            # Average across queries
            aggregated_masks = masks_queries_logits.mean(dim=1, keepdim=True)  # [B, 1, H/4, W/4]
        else:
            aggregated_masks = masks_queries_logits
        
        # Upsample to original size
        aggregated_masks = F.interpolate(aggregated_masks, size=(H, W), mode='bilinear', align_corners=False)
        
        # Get pixel-level features for our custom heads
        # Use the encoder features from the backbone
        if hasattr(outputs, 'encoder_hidden_states') and outputs.encoder_hidden_states is not None:
            # Use the last encoder hidden state
            encoder_features = outputs.encoder_hidden_states[-1]
            
            # Reshape if needed
            if encoder_features.dim() == 3:
                # [B, N, C] -> [B, C, H', W']
                N = encoder_features.shape[1]
                C_feat = encoder_features.shape[2]
                H_feat = W_feat = int(N ** 0.5)
                encoder_features = encoder_features.transpose(1, 2).reshape(B, C_feat, H_feat, W_feat)
            
            # Upsample encoder features to match input size
            encoder_features = F.interpolate(encoder_features, size=(H, W), mode='bilinear', align_corners=False)
            
            # Apply custom heads
            mask_output = torch.sigmoid(self.mask_head(encoder_features))
            # Contour output is distance transform (continuous values), no sigmoid
            contour_output = self.contour_head(encoder_features)
        else:
            # Fallback: use aggregated masks directly
            mask_output = torch.sigmoid(aggregated_masks)
            
            # For contour, we can use edge detection on the mask
            # or just use the same features (no sigmoid for distance transform)
            contour_output = aggregated_masks
        
        return mask_output, contour_output


class SimplifiedMask2Former(nn.Module):
    """
    Simplified Mask2Former-inspired model for building segmentation.
    This is a lighter version that doesn't rely on the full Mask2Former complexity.
    """
    def __init__(self, in_channels=3, out_channels_mask=1, out_channels_contour=1, hidden_dim=256):
        """
        Initialize simplified Mask2Former model.
        
        Args:
            in_channels: Number of input channels
            out_channels_mask: Number of output channels for mask
            out_channels_contour: Number of output channels for contour
            hidden_dim: Hidden dimension for features
        """
        super(SimplifiedMask2Former, self).__init__()
        
        # Simple encoder (ResNet-like)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 2),
            self._make_layer(256, hidden_dim, 2)
        )
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        # Query embeddings
        self.num_queries = 100
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        
        # Decoder for upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Output heads
        self.mask_head = nn.Conv2d(32, out_channels_mask, kernel_size=1)
        self.contour_head = nn.Conv2d(32, out_channels_contour, kernel_size=1)
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        """Create a residual layer."""
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Tuple of (mask_output, contour_output)
        """
        B, C, H, W = x.shape
        
        # Encode
        features = self.encoder(x)  # [B, hidden_dim, H/8, W/8]
        
        # Flatten features for transformer
        B, C_feat, H_feat, W_feat = features.shape
        features_flat = features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        # Query embeddings
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, hidden_dim]
        
        # Transformer decoder
        decoded = self.transformer_decoder(queries, features_flat)  # [B, num_queries, hidden_dim]
        
        # Aggregate queries back to spatial format
        # Simple approach: take mean and reshape
        decoded_spatial = decoded.mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]
        decoded_spatial = decoded_spatial.transpose(1, 2).unsqueeze(-1)  # [B, hidden_dim, 1, 1]
        decoded_spatial = decoded_spatial.expand(-1, -1, H_feat, W_feat)  # [B, hidden_dim, H/8, W/8]
        
        # Decode
        decoded_features = self.decoder(decoded_spatial)  # [B, 32, H, W]
        
        # Output heads
        mask_output = torch.sigmoid(self.mask_head(decoded_features))
        # Contour output is distance transform (continuous values), no sigmoid
        contour_output = self.contour_head(decoded_features)
        
        return mask_output, contour_output


def get_mask2former_model(in_channels=3, out_channels_mask=1, out_channels_contour=1, 
                          pretrained=True, use_simplified=False):
    """
    Factory function to create a Mask2Former model.
    
    Args:
        in_channels: Number of input channels
        out_channels_mask: Number of output channels for the mask
        out_channels_contour: Number of output channels for the contour
        pretrained: Whether to use pretrained weights (only for full Mask2Former)
        use_simplified: Whether to use simplified version
        
    Returns:
        Initialized Mask2Former model
    """
    if use_simplified:
        model = SimplifiedMask2Former(
            in_channels=in_channels,
            out_channels_mask=out_channels_mask,
            out_channels_contour=out_channels_contour
        )
    else:
        model = Mask2FormerSegmentation(
            in_channels=in_channels,
            out_channels_mask=out_channels_mask,
            out_channels_contour=out_channels_contour,
            pretrained=pretrained
        )
    return model
