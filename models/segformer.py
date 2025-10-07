"""
SegFormer model wrapper for building segmentation.
"""

import torch
import torch.nn as nn
from transformers import SegformerModel, SegformerImageProcessor


class DualHeadSegFormer(nn.Module):
    """
    Dual-head SegFormer: outputs binary mask (sigmoid) and continuous contour map.
    """
    def __init__(self, pretrained_model_name="nvidia/mit-b0", hf_token=None, revision=None):
        super().__init__()
        # Backbone (no decode head)
        self.backbone = SegformerModel.from_pretrained(
            pretrained_model_name,
            revision=revision,
            use_auth_token=hf_token,
        )
        hidden_size = self.backbone.config.hidden_sizes[-1]
        # Simple heads operating on last hidden state
        self.mask_head = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size // 2, 1, kernel_size=1)
        )
        self.contour_head = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size // 2, 1, kernel_size=1)
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        last_hidden = outputs.last_hidden_state  # (B, C, H/4, W/4) depending on config
        # Upsample heads to input size
        input_h, input_w = pixel_values.shape[-2], pixel_values.shape[-1]
        mask_logits = self.mask_head(last_hidden)
        contour_map = self.contour_head(last_hidden)
        mask_logits = torch.nn.functional.interpolate(mask_logits, size=(input_h, input_w), mode='bilinear', align_corners=False)
        contour_map = torch.nn.functional.interpolate(contour_map, size=(input_h, input_w), mode='bilinear', align_corners=False)
        return mask_logits, contour_map


class SegFormerModel:
    """
    Wrapper that holds dual-head model and image processor for convenience.
    """
    def __init__(self, pretrained_model_name="nvidia/mit-b0", num_labels=3, id2label=None, hf_token=None, revision=None):
        # Keep processor for consistent preprocessing
        self.image_processor = SegformerImageProcessor.from_pretrained(
            pretrained_model_name,
            revision=revision,
            use_auth_token=hf_token,
        )
        self.model = DualHeadSegFormer(pretrained_model_name, hf_token=hf_token, revision=revision)
    
    def get_model(self):
        return self.model
    
    def get_image_processor(self):
        return self.image_processor
    
    def save_model(self, save_path):
        # Save backbone config via processor; state_dict via torch
        torch.save(self.model.state_dict(), f"{save_path}/pytorch_model.bin")
        self.image_processor.save_pretrained(save_path)
    
    def load_model(self, load_path):
        state = torch.load(f"{load_path}/pytorch_model.bin", map_location="cpu")
        self.model.load_state_dict(state)
        self.image_processor = SegformerImageProcessor.from_pretrained(load_path)


def get_segformer_model(model_name="nvidia/mit-b0", num_labels=3, id2label=None, hf_token=None, revision=None):
    """
    Factory function to create a SegFormerModel.
    
    Args:
        model_name: Name of the pretrained model from HuggingFace
        num_labels: Number of segmentation classes
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
