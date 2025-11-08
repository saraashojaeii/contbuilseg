# Checkpoint Resuming Guide

This guide explains how to resume training from saved checkpoints for SegFormer and BuildFormer models.

## Overview

Both training scripts now support resuming from checkpoints. This allows you to:
- Continue training from a previously saved epoch
- Avoid losing progress if training is interrupted
- Train incrementally (e.g., train for 50 epochs, then continue for 50 more)

## What Gets Saved

Each checkpoint file (`.pth`) now contains:
- **Model weights** (`model_state_dict`)
- **Optimizer state** (`optimizer_state_dict`) - includes learning rate schedules, momentum, etc.
- **Training history** (`train_loss`, `val_loss`, `metrics`)
- **Epoch number** (`epoch`)

## How to Resume Training

### SegFormer Example

If you trained for 50 epochs and want to continue to 100 epochs:

```bash
# Your original 50-epoch training created: epoch_50.pth
# Now resume from epoch 51 to 100:

python scripts/train_segformer.py \
    --data_dir /path/to/data \
    --dataset_name your_dataset \
    --model_save_dir /path/to/checkpoints/segformer \
    --epochs 100 \
    --resume_from /path/to/checkpoints/segformer/epoch_50.pth
```

### BuildFormer Example

```bash
# Resume BuildFormer training from epoch 50 to 100:

python scripts/train_buildformer.py \
    --data_dir /path/to/data \
    --dataset_name your_dataset \
    --model_save_dir /path/to/checkpoints/buildformer \
    --epochs 100 \
    --resume_from /path/to/checkpoints/buildformer/epoch_50.pth
```

## Important Notes

### 1. Epochs Parameter
- `--epochs` specifies the **target epoch** (not additional epochs)
- If you trained to epoch 50 and set `--epochs 100`, it will train from epoch 51 to 100
- To train for 50 MORE epochs, set `--epochs 100` (50 + 50 = 100)

### 2. Checkpoint File Format
- **New format** (recommended): Contains model, optimizer, and history
- **Old format**: Contains only model weights (will load but won't resume optimizer state)
- All new checkpoints are saved in the new format

### 3. Checkpoint Files Saved
- **SegFormer**: Saves after **every epoch** (`epoch_1.pth`, `epoch_2.pth`, ...)
- **BuildFormer**: Saves every `--save_every` epochs (default: every 10 epochs)
- Both save a `final_model.pth` at the end

### 4. Resuming Best Practices
- Use the exact same `--model_save_dir` when resuming
- Use the same hyperparameters (learning rate, batch size, etc.)
- The optimizer state will be restored, maintaining momentum and learning rate schedules

## Checking Your Checkpoints

To see what epoch a checkpoint contains:

```python
import torch

checkpoint = torch.load('path/to/epoch_50.pth')
print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Training history length: {len(checkpoint['train_loss'])} epochs")
```

## Example Workflow

### Scenario: Train SegFormer for 100 total epochs (currently at 50)

```bash
# Step 1: Check your existing checkpoint
ls /root/home/pvc/conbuildseg_results/checkpoints/segformer/
# Output: epoch_1.pth, epoch_2.pth, ..., epoch_50.pth

# Step 2: Resume training to epoch 100
python scripts/train_segformer.py \
    --data_dir /root/home/pvc/datasets \
    --dataset_name inria \
    --model_name nvidia/mit-b0 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --epochs 100 \
    --model_save_dir /root/home/pvc/conbuildseg_results/checkpoints/segformer \
    --resume_from /root/home/pvc/conbuildseg_results/checkpoints/segformer/epoch_50.pth \
    --use_wandb

# This will:
# - Load model weights from epoch 50
# - Restore optimizer state (learning rates, momentum)
# - Continue training from epoch 51 to 100
# - Append new results to training history
```

## Troubleshooting

### Issue: "Checkpoint not found"
- Verify the path with `ls` or `find`
- Use absolute paths instead of relative paths

### Issue: "Training starts from epoch 1"
- Make sure you're using the `--resume_from` argument
- Check that the checkpoint file is valid

### Issue: "Different results after resuming"
- Ensure you're using the same hyperparameters
- Random seed may differ - set `torch.manual_seed()` for reproducibility

## Migration from Old Checkpoints

If you have old checkpoint files (only model weights), you can still load them:

```python
# The load_checkpoint method handles both formats:
# - New format: Loads everything
# - Old format: Loads only weights, starts from epoch 1
```

However, you won't get the optimizer state, so learning may not be optimal. It's recommended to start fresh with the new checkpoint format.
