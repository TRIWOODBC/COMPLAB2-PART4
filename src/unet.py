# -*- coding: utf-8 -*-
"""
UNet (with in-model preprocessing) for OASIS MRI segmentation.
- Dataset: PNG slices under data/OASIS/keras_png_slices_{train,validate,test} and *_seg_*.
- Labels: categorical indices (0..K-1). If masks are grayscale like {0, 63, 127, ...}, we remap to 0..K-1.
- Loss: CrossEntropyLoss on class indices (categorical). Output: logits [N, C, H, W].
- Metrics: mean Dice and per-class Dice on val/test.
- Visualizations: prediction grids saved under outputs/.
- Checkpoint: best model by val mean Dice at outputs/best_unet.pt.
- Demo: single-image inference saves colorized mask.

Run examples:
  python src/unet.py --mode train --num_classes 4 --image_size 256
  python src/unet.py --mode test  --ckpt outputs/best_unet.pt
  python src/unet.py --mode infer --ckpt outputs/best_unet.pt --image_path data/OASIS/keras_png_slices_test/xxx.png
"""

import os
import glob
import argparse
import random
import platform
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image


# ----------------------------
# Configuration
# ----------------------------
@dataclass
class Cfg:
    data_root: str = "data/OASIS"
    outputs: str = "outputs"
    image_size: int = 256             # target H=W; if images differ, we resize inside the model
    in_channels: int = 1              # grayscale MRI slices
    num_classes: int = 4              # set to your label count
    base_ch: int = 64                 # UNet base channels
    batch_size: int = 16
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 4              # will be clamped for Windows/CPU for stability
    seed: int = 42
    amp: bool = True                  # automatic mixed precision (CUDA only)


# ----------------------------
# Repro & FS helpers
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if CUDA not available
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ----------------------------
# Safe image conversions / resizing
# ----------------------------
def pil_open_gray(path: str) -> Image.Image:
    """Open image in grayscale mode 'L'."""
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    return img

def _to_pil_safe(arr: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL safely to avoid dtype errors (e.g., int64).
    - Float => float32
    - Int => uint8 if within [0, 255], otherwise int32
    """
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    else:
        if arr.size == 0:
            arr = arr.astype(np.uint8)
        else:
            vmin, vmax = arr.min(), arr.max()
            if vmin >= 0 and vmax <= 255:
                arr = arr.astype(np.uint8)
            else:
                arr = arr.astype(np.int32)
    return Image.fromarray(arr)

def _resize_nearest(arr: np.ndarray, size: int) -> np.ndarray:
    pil = _to_pil_safe(arr)
    pil = pil.resize((size, size), resample=Image.NEAREST)
    return np.array(pil)

def _resize_bilinear(arr: np.ndarray, size: int) -> np.ndarray:
    pil = _to_pil_safe(arr)
    pil = pil.resize((size, size), resample=Image.BILINEAR)
    return np.array(pil)


# ----------------------------
# Dataset
# ----------------------------
class OasisSlices(Dataset):
    """
    Expected layout under data_root:
      keras_png_slices_train/          (input)
      keras_png_slices_seg_train/      (masks)
      keras_png_slices_validate/
      keras_png_slices_seg_validate/
      keras_png_slices_test/
      keras_png_slices_seg_test/

    Notes:
    - We map any grayscale-coded mask levels (e.g., 0, 63, 127, ...) to dense class indices 0..K-1.
    - We convert masks to uint8 before PIL resize to avoid TypeError: (1,1), <i8>.
    - Basic flips as lightweight augmentation for train split.
    - We keep heavy preprocessing (resize+normalize) inside the model.
    """
    def __init__(self, root: str, split: str = "train", image_size: int = 256, aug: bool = False):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.aug = aug

        img_dir = os.path.join(root, f"keras_png_slices_{'train' if split=='train' else split}")
        seg_dir = os.path.join(root, f"keras_png_slices_seg_{'train' if split=='train' else split}")
        self.images = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.masks  = sorted(glob.glob(os.path.join(seg_dir, "*.png")))
        assert len(self.images) == len(self.masks) and len(self.images) > 0, f"No data for split='{split}'"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        # Load as numpy arrays
        img  = np.array(pil_open_gray(self.images[idx]))   # uint8 [0,255]
        mask = np.array(pil_open_gray(self.masks[idx]))    # may be levels not equal to 0..K-1

        # Map grayscale-coded mask levels to 0..K-1 indices
        uniq = np.unique(mask)
        if len(uniq) > 1:
            lut = {v: i for i, v in enumerate(sorted(uniq.tolist()))}
            mask = np.vectorize(lut.get)(mask)

        # Important: ensure mask is uint8 BEFORE PIL resize to avoid int64 errors
        mask = mask.astype(np.uint8)

        # Resize here for robustness (image bilinear, mask nearest-neighbor)
        if self.image_size is not None:
            img  = _resize_bilinear(img, self.image_size)
            mask = _resize_nearest(mask,  self.image_size)

        # Light augmentations (train only)
        if self.aug:
            if random.random() < 0.5:  # horizontal flip
                img  = np.ascontiguousarray(img[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if random.random() < 0.2:  # vertical flip
                img  = np.ascontiguousarray(img[::-1, :])
                mask = np.ascontiguousarray(mask[::-1, :])

        # Convert to tensors:
        # - image remains 0..255 float32; normalization done inside the model forward
        # - mask is long indices for CrossEntropyLoss
        img_t  = torch.from_numpy(img).float().unsqueeze(0)   # (1, H, W)
        mask_t = torch.from_numpy(mask.astype(np.int64))      # (H, W)
        return img_t, mask_t


# ----------------------------
# Model (UNet core + in-model preprocessing)
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Dynamic padding for safe concatenation (works for non-2^n sizes)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetCore(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, base_ch=64):
        super().__init__()
        self.inc   = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.down4 = Down(base_ch*8, base_ch*16)
        self.up1   = Up(base_ch*16, base_ch*8)
        self.up2   = Up(base_ch*8,  base_ch*4)
        self.up3   = Up(base_ch*4,  base_ch*2)
        self.up4   = Up(base_ch*2,  base_ch)
        self.outc  = nn.Conv2d(base_ch, num_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)  # logits (N, C, H, W)

class UNetWithPreproc(nn.Module):
    """
    In-model preprocessing:
      - Resize to target size if needed (safety, but dataset already resizes).
      - Normalize to [-1, 1] from 0..255 uint8 inputs.
    """
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.target_size = cfg.image_size
        self.core = UNetCore(cfg.in_channels, cfg.num_classes, cfg.base_ch)

    def forward(self, x):
        # x: (N, 1, H, W), uint8/float in 0..255
        if x.dtype != torch.float32:
            x = x.float()
        if self.target_size is not None and (x.shape[-2] != self.target_size or x.shape[-1] != self.target_size):
            x = F.interpolate(x, size=(self.target_size, self.target_size),
                              mode="bilinear", align_corners=False)
        # Normalize to [-1, 1]
        x = (x / 255.0 - 0.5) / 0.5
        return self.core(x)


# ----------------------------
# Metrics & visualization
# ----------------------------
@torch.no_grad()
def dice_score(pred_logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> Tuple[float, List[float]]:
    """
    Compute mean Dice and per-class Dice.
    pred_logits: (N, C, H, W)
    target:      (N, H, W) long
    """
    pred = pred_logits.argmax(1)
    eps = 1e-6
    dices: List[float] = []
    for c in range(num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p * t).sum(dim=(1, 2))
        union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        d = ((2 * inter + eps) / (union + eps)).mean().item()
        dices.append(d)
    return float(np.mean(dices)), dices

def colorize_mask(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert a single-channel class index mask (H, W) or (1, H, W)
    to a 3xHxW pseudo-color tensor in [0,1] for visualization.
    """
    # Remember original device to return result on same device
    device = mask.device
    
    # For compatibility, move processing to CPU
    mask = mask.cpu()
    
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    h, w = mask.shape
    out = torch.zeros(3, h, w, dtype=torch.float32)
    palette = [
        (0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0),
        (1, 0.5, 0), (0.5, 1, 0), (0.5, 0, 1), (0, 0.5, 1)
    ]
    for c in range(num_classes):
        clr = palette[c % len(palette)]
        m = (mask == c).float()
        out[0] += m * clr[0]
        out[1] += m * clr[1]
        out[2] += m * clr[2]
    
    # Return result on original device
    return out.clamp(0, 1).to(device)

@torch.no_grad()
def save_val_visuals(model: nn.Module, dl: DataLoader, cfg: Cfg, device: str, tag: str = "val"):
    """Save a grid of [input, prediction, target] triplets for quick inspection."""
    model.eval()
    img, mask = next(iter(dl))
    img, mask = img.to(device), mask.to(device)
    logits = model(img)
    pred = logits.argmax(1)

    tiles = []
    n_show = min(8, img.size(0))
    for i in range(n_show):
        # Convert single-channel image to three-channel format, ensuring all images are [3,H,W] format
        g = (img[i].float() / 255.0).clamp(0, 1)         # input (1,H,W) scaled to 0..1 for saving
        g_rgb = torch.cat([g, g, g], dim=0)              # Convert to [3,H,W] to match other images
        
        # Keep processing on the same device (all on GPU or all on CPU)
        # Process in device memory, only move to CPU for final saving
        p = colorize_mask(pred[i], cfg.num_classes).to(device)  # Keep on same device
        t = colorize_mask(mask[i], cfg.num_classes).to(device)  # Keep on same device
        
        tiles += [g_rgb, p, t]

    # Create grid on device
    grid = make_grid(tiles, nrow=3, padding=2)
    
    # Only move to CPU for final saving
    grid = grid.cpu()
    
    ensure_dir(cfg.outputs)
    out_path = os.path.join(cfg.outputs, f"visual_{tag}.png")
    save_image(grid, out_path)
    print(f"  -> Saved visuals: {out_path}")


# ----------------------------
# Training / Evaluation / Inference
# ----------------------------
def train(cfg: Cfg):
    set_seed(cfg.seed)
    use_cuda = torch.cuda.is_available()
    dev_type = "cuda" if use_cuda else "cpu"

    # Windows/CPU stability
    if not use_cuda or platform.system().lower().startswith("win"):
        cfg.num_workers = 0
    if not use_cuda:
        cfg.amp = False  # Enable AMP only with CUDA

    device = torch.device(dev_type)
    ensure_dir(cfg.outputs)

    ds_tr = OasisSlices(cfg.data_root, "train", cfg.image_size, aug=True)
    ds_va = OasisSlices(cfg.data_root, "validate", cfg.image_size, aug=False)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers, pin_memory=use_cuda)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=use_cuda)

    model = UNetWithPreproc(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    # Initialize GradScaler without device_type parameter
    scaler = amp.GradScaler(enabled=cfg.amp)

    # device_type is only used for autocast context manager
    def autocast_ctx():
        return amp.autocast(device_type="cuda" if use_cuda else "cpu",
                            enabled=cfg.amp and use_cuda)

    best = -1.0
    best_path = None

    for ep in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0

        for img, mask in dl_tr:
            img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx():
                logits = model(img)
                loss = criterion(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * img.size(0)

        sched.step()
        tr_loss = running / len(ds_tr)

        # ----- validation (no need for AMP/Scaler)
        model.eval()
        va_loss = 0.0
        va_dsc = 0.0
        per_class = np.zeros(cfg.num_classes, dtype=np.float64)
        with torch.no_grad():
            for img, mask in dl_va:
                img, mask = img.to(device), mask.to(device)
                logits = model(img)
                va_loss += criterion(logits, mask).item() * img.size(0)
                d_mean, d_each = dice_score(logits, mask, cfg.num_classes)
                va_dsc += d_mean * img.size(0)
                per_class += np.array(d_each) * img.size(0)
        va_loss /= len(ds_va)
        va_dsc  /= len(ds_va)
        per_class = (per_class / len(ds_va)).tolist()

        print(f"[Epoch {ep:03d}] train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
              f"val_DSC={va_dsc:.4f}  per_class={['%.3f'%x for x in per_class]}")

        if ep == 1 or ep % 5 == 0:
            save_val_visuals(model, dl_va, cfg, dev_type, tag=f"ep{ep:03d}")

        if va_dsc > best:
            best = va_dsc
            best_path = os.path.join(cfg.outputs, "best_unet.pt")
            torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict(), "val_dsc": best}, best_path)
            print(f"  -> Saved best to {best_path} (DSC={best:.4f})")

    print(f"Training done. Best DSC={best:.4f}, ckpt={best_path}")


@torch.no_grad()
def evaluate_test(cfg: Cfg, ckpt: str):
    use_cuda = torch.cuda.is_available()
    dev_type = "cuda" if use_cuda else "cpu"
    if not use_cuda:
        cfg.amp = False
        cfg.num_workers = 0
    device = torch.device(dev_type)

    ds_te = OasisSlices(cfg.data_root, "test", cfg.image_size, aug=False)
    dl_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=use_cuda)

    model = UNetWithPreproc(cfg).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total = len(ds_te)
    loss = 0.0
    dsc = 0.0
    per_class = np.zeros(cfg.num_classes, dtype=np.float64)

    for img, mask in dl_te:
        img, mask = img.to(device), mask.to(device)
        logits = model(img)
        loss += criterion(logits, mask).item() * img.size(0)
        d_mean, d_each = dice_score(logits, mask, cfg.num_classes)
        dsc += d_mean * img.size(0)
        per_class += np.array(d_each) * img.size(0)

    loss /= total
    dsc  /= total
    per_class = (per_class / total).tolist()
    print(f"[TEST] loss={loss:.4f}  DSC={dsc:.4f}  per_class={['%.3f'%x for x in per_class]}")

    # Save a visualization grid on test data
    save_val_visuals(model, dl_te, cfg, dev_type, tag="test")


@torch.no_grad()
def infer_single(cfg: Cfg, ckpt: str, image_path: str, save_path: str = None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if not use_cuda:
        cfg.amp = False

    # Load a single image (grayscale)
    img_np = np.array(pil_open_gray(image_path))  # (H, W), 0..255
    img_t = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Build model and load checkpoint
    model = UNetWithPreproc(cfg).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    logits = model(img_t.to(device))
    pred = logits.argmax(1).cpu().squeeze(0)  # (H, W)
    color = colorize_mask(pred, cfg.num_classes)

    if save_path is None:
        base = os.path.basename(image_path).replace(".png", "")
        save_path = os.path.join(cfg.outputs, f"pred_{base}.png")
    ensure_dir(cfg.outputs)
    save_image(color, save_path)
    print(f"Saved prediction to {save_path}")


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=Cfg.data_root)
    parser.add_argument("--outputs", type=str, default=Cfg.outputs)
    parser.add_argument("--image_size", type=int, default=Cfg.image_size)
    parser.add_argument("--num_classes", type=int, default=Cfg.num_classes)
    parser.add_argument("--batch_size", type=int, default=Cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=Cfg.epochs)
    parser.add_argument("--lr", type=float, default=Cfg.lr)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "infer"])
    parser.add_argument("--ckpt", type=str, default=os.path.join(Cfg.outputs, "best_unet.pt"))
    parser.add_argument("--image_path", type=str, help="path to a single PNG for --mode infer")
    args = parser.parse_args()

    cfg = Cfg(
        data_root=args.data_root, outputs=args.outputs, image_size=args.image_size,
        num_classes=args.num_classes, batch_size=args.batch_size,
        epochs=args.epochs, lr=args.lr
    )
    ensure_dir(cfg.outputs)

    if args.mode == "train":
        train(cfg)
    elif args.mode == "test":
        evaluate_test(cfg, args.ckpt)
    elif args.mode == "infer":
        assert args.image_path is not None, "--image_path is required for infer"
        infer_single(cfg, args.ckpt, args.image_path)

if __name__ == "__main__":
    main()
