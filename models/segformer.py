"""
SegFormer model wrapper for building segmentation.
Uses Hugging Face's decoder head to avoid extremely low-res outputs.
"""

import torch
import torch.nn as nn
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)


class DualHeadSegFormer(nn.Module):
    """
    Dual-head SegFormer: outputs mask logits and a continuous contour map.
    The mask branch uses the official SegFormer decode head. The contour branch
    is a light 1x1+3x3 conv head stacked on top of the mask logits so that it
    benefits from the decoder's multi-scale features.
    """
    def __init__(self, pretrained_model_name="nvidia/mit-b0", num_labels=1, hf_token=None, revision=None):
        super().__init__()
        # Full model with decode head so we get high-resolution logits
        self.backbone = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            revision=revision,
            use_auth_token=hf_token,
        )
        # Simple contour head operating on mask logits
        self.contour_head = nn.Sequential(
            nn.Conv2d(num_labels, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        # outputs.logits: (B, num_labels, H, W)
        mask_logits = outputs.logits
        contour_map = self.contour_head(mask_logits)
        return mask_logits, contour_map


class SegFormerModel:
    """
    Wrapper that holds dual-head model and image processor for convenience.
    """
    def __init__(self, pretrained_model_name="nvidia/mit-b0", num_labels=1, id2label=None, hf_token=None, revision=None):
        # Keep processor for consistent preprocessing
        self.image_processor = SegformerImageProcessor.from_pretrained(
            pretrained_model_name,
            revision=revision,
            use_auth_token=hf_token,
        )
        self.model = DualHeadSegFormer(pretrained_model_name, num_labels=num_labels, hf_token=hf_token, revision=revision)
    
    def get_model(self):
        return self.model
    
    def get_image_processor(self):
        return self.image_processor
    
    def save_model(self, save_path):
        # Save using HF APIs where possible so config is preserved
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "save_pretrained"):
            self.model.backbone.save_pretrained(save_path)
        self.image_processor.save_pretrained(save_path)
    
    def load_model(self, load_path):
        self.model.backbone = SegformerForSemanticSegmentation.from_pretrained(load_path)
        self.image_processor = SegformerImageProcessor.from_pretrained(load_path)


def get_segformer_model(model_name="nvidia/mit-b0", num_labels=1, id2label=None, hf_token=None, revision=None):
    """
    Factory function to create a SegFormerModel.
    
    Args:
        model_name: Name of the pretrained model from HuggingFace
        num_labels: Number of segmentation classes (1 for binary)
        id2label: Dictionary mapping class IDs to labels
        
    Returns:
        Initialized SegFormerModel wrapper
    """
    return SegFormerModel(
        pretrained_model_name=model_name,
        num_labels=num_labels,
        id2label=id2label,
        hf_token=hf_token,
        revision=revision,
    )
