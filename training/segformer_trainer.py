"""
SegFormer trainer for building segmentation tasks.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import wandb
import matplotlib.pyplot as plt

from .train import BaseTrainer
from utils.losses import BCEWithDiceLoss, BCEWithLogitsDiceLoss, L1Loss
from evaluation.metrics import compute_metrics_batch


class SegFormerTrainer(BaseTrainer):
    """
    Trainer class for SegFormer model with semantic segmentation for buildings.
    """
    def __init__(self, model, train_loader, val_loader, device='cuda', 
                learning_rate=2e-5, model_save_dir='./checkpoints',
                use_wandb=True, wandb_project='building_seg', wandb_run_name=None,
                dataset_name='dataset', mask_weight=0.7, contour_weight=0.3):
        """
        Initialize the SegFormerTrainer.
        
        Args:
            model: The SegFormer model to train (from Hugging Face transformers)
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to use for training ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            model_save_dir: Directory to save model checkpoints
            use_wandb: Whether to enable Weights & Biases logging
            wandb_project: W&B project name
            wandb_run_name: Optional W&B run name
            dataset_name: Dataset name for logging context
        """
        super().__init__(model, train_loader, val_loader, device, learning_rate, model_save_dir)
        self.use_wandb = use_wandb
        self.dataset_name = dataset_name
        self.mask_weight = mask_weight
        self.contour_weight = contour_weight

        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    'learning_rate': learning_rate,
                    'device': device,
                    'model_type': 'SegFormer',
                    'dataset': dataset_name,
                    'mask_weight': mask_weight,
                    'contour_weight': contour_weight,
                }
            )
            wandb.watch(self.model, log='all', log_freq=100)
        
    def _get_optimizer(self):
        """
        Get AdamW optimizer for SegFormer.
        
        Returns:
            AdamW optimizer
        """
        # Two-tier learning rates: smaller for encoder, larger for heads
        lr_backbone = self.learning_rate
        lr_head = self.learning_rate * 10.0

        encoder_params = []
        head_params = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # Parameters inside the HF encoder live under 'backbone.segformer'
            if 'backbone.segformer' in name:
                encoder_params.append(p)
            else:
                # includes backbone.decode_head and our contour_head
                head_params.append(p)

        param_groups = [
            { 'params': encoder_params, 'lr': lr_backbone },
            { 'params': head_params, 'lr': lr_head },
        ]
        return AdamW(param_groups, lr=lr_backbone)
    
    def _get_loss_fn(self):
        """Return combined loss fns for mask and contour."""
        return {
            'mask': BCEWithLogitsDiceLoss(dice_weight=0.5, bce_weight=0.5),
            'contour': L1Loss(reduction='mean')
        }
    
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
        running_mask_loss = 0.0
        running_contour_loss = 0.0
        step = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch in pbar:
            pixel_values = batch['pixel_values'].to(self.device)
            masks = batch['mask'].to(self.device)
            contours = batch.get('contours')
            if contours is not None:
                contours = contours.to(self.device)

            self.optimizer.zero_grad()

            mask_logits, contour_map = self.model(pixel_values)
            # Align predictions to GT spatial size if needed
            if mask_logits.shape[-2:] != masks.shape[-2:]:
                mask_logits = torch.nn.functional.interpolate(
                    mask_logits, size=masks.shape[-2:], mode='bilinear', align_corners=False
                )
                if contour_map is not None and contour_map.shape[-2:] != masks.shape[-2:]:
                    contour_map = torch.nn.functional.interpolate(
                        contour_map, size=masks.shape[-2:], mode='bilinear', align_corners=False
                    )
            # loss on logits
            mask_loss = self.loss_fn['mask'](mask_logits, masks)
            # probabilities for metrics only
            mask_pred = torch.sigmoid(mask_logits)
            if contours is not None:
                contour_loss = self.loss_fn['contour'](contour_map, contours)
                loss = self.mask_weight * mask_loss + self.contour_weight * contour_loss
            else:
                contour_loss = torch.tensor(0.0, device=self.device)
                loss = mask_loss

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_mask_loss += mask_loss.item()
            running_contour_loss += contour_loss.item() if contours is not None else 0.0

            # Metrics
            with torch.no_grad():
                batch_metrics = compute_metrics_batch(mask_pred, masks)

            pbar.set_postfix({
                'loss': loss.item(),
                'mask_loss': mask_loss.item(),
                'contour_loss': contour_loss.item() if contours is not None else 0.0,
                'iou': batch_metrics['iou']
            })

            if self.use_wandb and step % 10 == 0:
                log = {
                    'train/step_loss': loss.item(),
                    'train/step_mask_loss': mask_loss.item(),
                    'train/step': step,
                    'epoch': epoch
                }
                if contours is not None:
                    log['train/step_contour_loss'] = contour_loss.item()
                wandb.log(log)
            step += 1
        
        # Calculate average loss
        avg_loss = running_loss / len(self.train_loader)
        avg_mask_loss = running_mask_loss / len(self.train_loader)
        if self.use_wandb:
            log = {
                'train/epoch_loss': avg_loss,
                'train/epoch_mask_loss': avg_mask_loss,
                'epoch': epoch
            }
            if running_contour_loss > 0:
                log['train/epoch_contour_loss'] = running_contour_loss / len(self.train_loader)
            wandb.log(log)
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
        # Accumulators for metrics
        sum_metrics = { 'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0 }
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        val_visual_logged = False
        with torch.no_grad():
            for batch in pbar:
                pixel_values = batch['pixel_values'].to(self.device)
                masks = batch['mask'].to(self.device)
                contours = batch.get('contours')
                if contours is not None:
                    contours = contours.to(self.device)

                mask_logits, contour_map = self.model(pixel_values)
                # Align predictions to GT spatial size if needed
                if mask_logits.shape[-2:] != masks.shape[-2:]:
                    mask_logits = torch.nn.functional.interpolate(
                        mask_logits, size=masks.shape[-2:], mode='bilinear', align_corners=False
                    )
                    if contour_map is not None and contour_map.shape[-2:] != masks.shape[-2:]:
                        contour_map = torch.nn.functional.interpolate(
                            contour_map, size=masks.shape[-2:], mode='bilinear', align_corners=False
                        )
                # loss on logits
                mask_loss = self.loss_fn['mask'](mask_logits, masks)
                mask_pred = torch.sigmoid(mask_logits)
                if contours is not None:
                    contour_loss = self.loss_fn['contour'](contour_map, contours)
                    loss = self.mask_weight * mask_loss + self.contour_weight * contour_loss
                else:
                    loss = mask_loss

                metrics = compute_metrics_batch(mask_pred, masks)
                # accumulate
                for k in sum_metrics:
                    sum_metrics[k] += metrics[k]
                running_loss += loss.item()

                pbar.set_postfix({'loss': loss.item(), 'iou': metrics['iou']})

                if self.use_wandb and not val_visual_logged:
                    try:
                        imgs = pixel_values[:4].detach().cpu()
                        preds = mask_pred[:4].detach().cpu()
                        gts = masks[:4].detach().cpu()

                        wandb_imgs = []
                        for i in range(imgs.shape[0]):
                            img = imgs[i]
                            if img.shape[0] in (1, 3):
                                np_img = img.numpy()
                                if np_img.shape[0] == 1:
                                    np_img = np_img[0]
                                else:
                                    np_img = np.transpose(np_img, (1, 2, 0))
                                np_min, np_max = np_img.min(), np_img.max()
                                if np_max > np_min:
                                    np_img = (np_img - np_min) / (np_max - np_min)
                                np_img = (np_img * 255).astype(np.uint8)
                            else:
                                np_img = imgs[i, :3].numpy()
                                np_img = np.transpose(np_img, (1, 2, 0))
                                np_min, np_max = np_img.min(), np_img.max()
                                if np_max > np_min:
                                    np_img = (np_img - np_min) / (np_max - np_min)
                                np_img = (np_img * 255).astype(np.uint8)

                            if contours is not None:
                                cont_gt = contours[:4].detach().cpu()
                                cont_pred = contour_map[:4].detach().cpu()
                                fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                                axes[0,0].imshow(np_img, cmap='gray' if np_img.ndim==2 else None); axes[0,0].set_title('Input'); axes[0,0].axis('off')
                                axes[0,1].imshow(gts[i].squeeze(), cmap='Blues'); axes[0,1].set_title('Mask GT'); axes[0,1].axis('off')
                                axes[0,2].imshow(preds[i].squeeze(), cmap='Blues'); axes[0,2].set_title('Mask Pred'); axes[0,2].axis('off')
                                axes[1,1].imshow(cont_gt[i].squeeze(), cmap='Reds'); axes[1,1].set_title('Contour GT'); axes[1,1].axis('off')
                                axes[1,2].imshow(cont_pred[i].squeeze(), cmap='Reds'); axes[1,2].set_title('Contour Pred'); axes[1,2].axis('off')
                                axes[1,0].axis('off')
                            else:
                                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                                axes[0].imshow(np_img, cmap='gray' if np_img.ndim==2 else None); axes[0].set_title('Input'); axes[0].axis('off')
                                axes[1].imshow(gts[i].squeeze(), cmap='Blues'); axes[1].set_title('Mask GT'); axes[1].axis('off')
                                axes[2].imshow(preds[i].squeeze(), cmap='Blues'); axes[2].set_title('Mask Pred'); axes[2].axis('off')

                            plt.tight_layout()
                            wandb_imgs.append(wandb.Image(fig, caption=f'Sample {i+1}'))
                            plt.close(fig)

                        wandb.log({'val/predictions': wandb_imgs, 'epoch': epoch})
                    except Exception:
                        pass
                    val_visual_logged = True
        
        # Calculate average loss and metrics
        avg_loss = running_loss / len(self.val_loader)
        # Average metrics over validation set
        avg_metrics = {k: (v / len(self.val_loader)) for k, v in sum_metrics.items()}
        if self.use_wandb:
            wandb.log({
                'val/epoch_loss': avg_loss,
                'val/iou': avg_metrics['iou'],
                'val/dice': avg_metrics['dice'],
                'val/precision': avg_metrics['precision'],
                'val/recall': avg_metrics['recall'],
                'val/f1': avg_metrics['f1'],
                'epoch': epoch
            })
        return avg_loss, avg_metrics
    
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

        # Support either HF SegformerForSemanticSegmentation or our DualHeadSegFormer with .backbone
        try:
            if hasattr(self.model, 'save_pretrained'):
                self.model.save_pretrained(save_path)
            elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'save_pretrained'):
                self.model.backbone.save_pretrained(save_path)
            else:
                # Fallback to state_dict
                import torch
                torch.save(self.model.state_dict(), f"{save_path}.pt")
        finally:
            print(f"Model saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load SegFormer model using the Hugging Face from_pretrained method.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        from transformers import SegformerForSemanticSegmentation

        # Try to load as HF checkpoint first
        try:
            loaded = SegformerForSemanticSegmentation.from_pretrained(checkpoint_path)
            # If current model has a .backbone (our wrapper), swap it in
            if hasattr(self.model, 'backbone'):
                self.model.backbone = loaded.to(self.device)
            else:
                self.model = loaded.to(self.device)
        except Exception:
            # Fallback to plain state_dict
            import torch
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state)

        print(f"Loaded model from {checkpoint_path}")
