"""
SegFormer model wrapper for building segmentation.
"""

import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


class SegFormerModel:
    """
    Wrapper class for SegFormer model from HuggingFace transformers.
    """
    def __init__(self, pretrained_model_name="nvidia/mit-b0", num_labels=3, id2label=None):
        """
        Initialize SegFormer model.
        
        Args:
            pretrained_model_name: Name of the pretrained model from HuggingFace
            num_labels: Number of segmentation classes
            id2label: Dictionary mapping class IDs to labels
        """
        # Default label mapping for building segmentation (background, building, boundary)
        if id2label is None:
            self.id2label = {
                0: "background",
                1: "building",
                2: "boundary"
            }
        else:
            self.id2label = id2label
            
        # Create label2id mapping
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        # Initialize the model
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        
        # Initialize the image processor
        self.image_processor = SegformerImageProcessor.from_pretrained(pretrained_model_name)
        
    def get_model(self):
        """
        Get the SegFormer model.
        
        Returns:
            SegFormer model
        """
        return self.model
    
    def get_image_processor(self):
        """
        Get the image processor.
        
        Returns:
            SegFormer image processor
        """
        return self.image_processor
    
    def save_model(self, save_path):
        """
        Save the model to the specified path.
        
        Args:
            save_path: Path to save the model
        """
        self.model.save_pretrained(save_path)
        self.image_processor.save_pretrained(save_path)
        
    def load_model(self, load_path):
        """
        Load the model from the specified path.
        
        Args:
            load_path: Path to load the model from
        """
        self.model = SegformerForSemanticSegmentation.from_pretrained(load_path)
        self.image_processor = SegformerImageProcessor.from_pretrained(load_path)


def get_segformer_model(model_name="nvidia/mit-b0", num_labels=3, id2label=None):
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
        id2label=id2label
    )
