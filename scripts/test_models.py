#!/usr/bin/env python
"""
Universal test script for building segmentation models (UNet, SegFormer).
Reports the following metrics per image and on average:
- accuracy, precision, recall, f1, IoU
- merge rate
- centroid-based precision/recall/F1 (a detection is correct if its centroid
  lies inside any GT building instance)

Usage example:
  python scripts/test_models.py \
      --model_type segformer \
      --checkpoint /path/to/checkpoint_or_dir \
      --data_dir /path/to/datasets \
      --dataset_name massachusetts \
      --split test

Outputs a CSV at output_dir and prints averaged metrics.
"""

import os
import argparse
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import CustomDataset, DataPrep
from evaluation.metrics import compute_all_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Test segmentation models and report advanced metrics")
    p.add_argument('--model_type', choices=['unet', 'hrnet', 'deeplabv3plus', 'swin_unet', 'mask2former', 'segformer'], required=True)
    p.add_argument('--checkpoint', type=str, required=True, help='Path to .pth (UNet/dual-head) or HF dir (SegFormer)')
    p.add_argument('--data_dir', type=str, required=True, help='Root datasets directory')
    p.add_argument('--dataset_name', type=str, required=True, help='Dataset folder name under data_dir')
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=1)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output_dir', type=str, default='./test_results')
    p.add_argument('--resize', action='store_true', help='Resize inputs via processor (SegFormer only)')
    p.add_argument('--height', type=int, default=512, help='Resize height when --resize used (SegFormer only)')
    p.add_argument('--width', type=int, default=512, help='Resize width when --resize used (SegFormer only)')
    p.add_argument('--model_name', type=str, default='nvidia/mit-b0', help='SegFormer HF id when loading .pth checkpoints')
    return p.parse_args()


def get_paths(root, dataset_name, split):
    base = os.path.join(root, dataset_name)
    # Accept multiple common extensions
    img_exts = ['png', 'jpg', 'jpeg', 'tif', 'tiff']
    def listdir(d, exts):
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(base, d, f'*.{e}')))
        return sorted(files)
    return {
        'images': listdir(split, img_exts),
        'masks': listdir(f'{split}_labels', img_exts),
    }


def upsample_like(x, ref):
    if x.shape[-2:] != ref.shape[-2:]:
        x = torch.nn.functional.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)
    return x


def test_unet(args, paths):
    from models.unet import get_unet_model
    os.makedirs(args.output_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CustomDataset(paths['images'], paths['masks'], transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = get_unet_model()
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    # Support either wrapped dict or raw state_dict
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()

    import pandas as pd
    rows = []
    with torch.no_grad():
        idx = 0
        for batch in loader:
            images, masks = batch
            images = images.to(args.device)
            masks = masks.to(args.device)
            outputs = model(images)
            mask_pred = outputs[0] if isinstance(outputs, tuple) else outputs
            mask_pred = upsample_like(mask_pred, masks)

            # compute metrics on CPU numpy via helper
            m = compute_all_metrics(mask_pred.cpu(), masks.cpu())
            m['sample_idx'] = idx
            rows.append(m)
            idx += 1

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, f'unet_{args.dataset_name}_{args.split}.csv'), index=False)

    avg = df.drop(columns=['sample_idx'], errors='ignore').mean().to_dict()
    print('\nAverages (UNet):')
    for k, v in avg.items():
        print(f'{k}: {v:.4f}')


def _test_generic_dual_head(args, paths, get_model_fn, model_name: str):
    """
    Helper for models that output (mask, contour), already at input resolution,
    e.g., HRNet, DeepLabV3+, Swin-UNet, Mask2Former wrapper.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CustomDataset(paths['images'], paths['masks'], transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = get_model_fn()
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()

    import pandas as pd
    rows = []
    with torch.no_grad():
        idx = 0
        for batch in loader:
            images, masks = batch
            images = images.to(args.device)
            masks = masks.to(args.device)
            outputs = model(images)
            mask_pred = outputs[0] if isinstance(outputs, tuple) else outputs
            mask_pred = upsample_like(mask_pred, masks)
            m = compute_all_metrics(mask_pred.cpu(), masks.cpu())
            m['sample_idx'] = idx
            rows.append(m)
            idx += 1

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, f'{model_name}_{args.dataset_name}_{args.split}.csv'), index=False)

    avg = df.drop(columns=['sample_idx'], errors='ignore').mean().to_dict()
    print(f'\nAverages ({model_name}):')
    for k, v in avg.items():
        print(f'{k}: {v:.4f}')


def test_hrnet(args, paths):
    from models.hrnet import get_hrnet_model
    return _test_generic_dual_head(args, paths, get_hrnet_model, 'hrnet')


def test_deeplabv3plus(args, paths):
    from models.deeplabv3plus import get_deeplabv3plus_model
    return _test_generic_dual_head(args, paths, get_deeplabv3plus_model, 'deeplabv3plus')


def test_swin_unet(args, paths):
    from models.swin_unet import get_swin_unet_model
    return _test_generic_dual_head(args, paths, get_swin_unet_model, 'swin_unet')


def test_mask2former(args, paths):
    from models.mask2former import get_mask2former_model
    return _test_generic_dual_head(args, paths, get_mask2former_model, 'mask2former')


def test_segformer(args, paths):
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    from models.segformer import DualHeadSegFormer
    os.makedirs(args.output_dir, exist_ok=True)

    is_file_ckpt = os.path.isfile(args.checkpoint)

    if is_file_ckpt:
        # Load our dual-head wrapper and its processor using provided model_name
        processor = SegformerImageProcessor.from_pretrained(args.model_name)
        model = DualHeadSegFormer(pretrained_model_name=args.model_name, num_labels=1)
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state)
        # When using our wrapper, forward returns (mask_logits, contour_map)
        use_wrapper = True
    else:
        # HF directory
        model = SegformerForSemanticSegmentation.from_pretrained(args.checkpoint)
        processor = SegformerImageProcessor.from_pretrained(args.checkpoint)
        use_wrapper = False

    proc_kwargs = {'do_resize': args.resize}
    if args.resize:
        proc_kwargs['size'] = {'height': args.height, 'width': args.width}

    dataset = DataPrep(paths['images'], paths['masks'], processor, processor_kwargs=proc_kwargs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model.to(args.device)
    model.eval()

    import pandas as pd
    rows = []
    with torch.no_grad():
        idx = 0
        for batch in loader:
            pixel_values = batch['pixel_values'].to(args.device)
            masks = batch['mask'].to(args.device)
            if use_wrapper:
                logits, _ = model(pixel_values)
            else:
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
            logits = upsample_like(logits, masks)
            probs = torch.sigmoid(logits)

            m = compute_all_metrics(probs.cpu(), masks.cpu())
            m['sample_idx'] = idx
            rows.append(m)
            idx += 1

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, f'segformer_{args.dataset_name}_{args.split}.csv'), index=False)

    avg = df.drop(columns=['sample_idx'], errors='ignore').mean().to_dict()
    print('\nAverages (SegFormer):')
    for k, v in avg.items():
        print(f'{k}: {v:.4f}')



def main():
    args = parse_args()
    out_dir = os.path.join(args.output_dir, args.model_type, args.dataset_name, args.split)
    os.makedirs(out_dir, exist_ok=True)
    args.output_dir = out_dir

    paths = get_paths(args.data_dir, args.dataset_name, args.split)
    print(f"Found {len(paths['images'])} {args.split} images")

    if args.model_type == 'unet':
        test_unet(args, paths)
    elif args.model_type == 'hrnet':
        test_hrnet(args, paths)
    elif args.model_type == 'deeplabv3plus':
        test_deeplabv3plus(args, paths)
    elif args.model_type == 'swin_unet':
        test_swin_unet(args, paths)
    elif args.model_type == 'mask2former':
        test_mask2former(args, paths)
    elif args.model_type == 'segformer':
        test_segformer(args, paths)


if __name__ == '__main__':
    main()
