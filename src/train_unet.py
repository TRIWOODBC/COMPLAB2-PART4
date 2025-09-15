import os
import argparse
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset_png import OasisMRIDataset   # ✅ fixed import (lowercase)

# ---------------------- Simple 2D UNet ----------------------
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
    def forward(self, x):
        return self.net(x)

class UNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.inc   = DoubleConv(in_ch, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*2, base*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*4, base*8))
        self.bot   = DoubleConv(base*8, base*16)
        self.up3   = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec3  = DoubleConv(base*16, base*8)
        self.up2   = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec2  = DoubleConv(base*8, base*4)
        self.up1   = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec1  = DoubleConv(base*4, base*2)
        self.outc  = nn.Conv2d(base*2, out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xb = self.bot(x4)
        x  = self.up3(xb)
        x  = torch.cat([x, x4], dim=1)
        x  = self.dec3(x)
        x  = self.up2(x)
        x  = torch.cat([x, x3], dim=1)
        x  = self.dec2(x)
        x  = self.up1(x)
        x  = torch.cat([x, x2], dim=1)
        x  = self.dec1(x)
        return self.outc(x)  # logits

# ---------------------- Loss & Metrics ----------------------
class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets, dims)
        union = torch.sum(probs, dims) + torch.sum(targets, dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

@torch.no_grad()
def dice_score(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    dims = (0, 2, 3)
    intersection = torch.sum(preds * targets, dims)
    union = torch.sum(preds, dims) + torch.sum(targets, dims)
    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    return dice.mean().item()

# ---------------------- Data ----------------------
def get_loaders(root: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    """Create train/val loaders using your PNG folder layout under data/OASIS/"""
    train_img = os.path.join(root, "keras_png_slices_train")
    train_msk = os.path.join(root, "keras_png_slices_seg_train")
    val_img   = os.path.join(root, "keras_png_slices_validate")
    val_msk   = os.path.join(root, "keras_png_slices_seg_validate")

    train_ds = OasisMRIDataset(train_img, train_msk)
    val_ds   = OasisMRIDataset(val_img,   val_msk)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

# ---------------------- Visualization ----------------------
def save_vis(img, gt, pred, path: str):
    """Save side-by-side visualization for quick sanity check."""
    img = img.squeeze(0).cpu().numpy()
    gt  = gt.squeeze(0).cpu().numpy()
    pr  = pred.squeeze(0).cpu().numpy()
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1); plt.imshow(img, cmap="gray"); plt.title("image"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(gt, cmap="gray");  plt.title("mask");  plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(img, cmap="gray"); plt.imshow(pr, alpha=0.4); plt.title("pred"); plt.axis("off")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data/OASIS")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    os.makedirs(os.path.join(args.outdir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "visualizations"), exist_ok=True)

    train_loader, val_loader = get_loaders(args.data_root, args.batch_size, args.num_workers)

    model = UNet2D(in_ch=1, out_ch=1, base=32).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True

    bce  = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    opt  = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(device="cuda", enabled=args.amp)  # ✅ updated

    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        # -------- train --------
        model.train()
        running = []
        for img, msk in train_loader:
            img = img.to(device, non_blocking=True)
            msk = msk.to(device, non_blocking=True).float().unsqueeze(1)
            if device.type == "cuda":
                img = img.to(memory_format=torch.channels_last)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=args.amp):
                logits = model(img)
                loss = 0.5 * bce(logits, msk) + 0.5 * dice(logits, msk)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running.append(loss.item())

        train_loss = float(np.mean(running))

        # -------- validate --------
        model.eval()
        dices = []
        with torch.no_grad():
            for img, msk in val_loader:
                img = img.to(device, non_blocking=True)
                msk = msk.to(device, non_blocking=True).float().unsqueeze(1)
                with autocast(device_type="cuda", enabled=args.amp):
                    logits = model(img)
                dices.append(dice_score(logits, msk))
            mean_dice = float(np.mean(dices))

        print(f"[Epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} | val_dice={mean_dice:.4f}")

        # Save best
        if mean_dice > best_dice:
            best_dice = mean_dice
            ckpt_path = os.path.join(args.outdir, "checkpoints", "best_unet2d.pth")
            torch.save(model.state_dict(), ckpt_path)

            # save multiple samples
            for i in range(min(3, img.shape[0])):  # save first 3 samples
                img0 = img[i].detach().cpu()
                msk0 = msk[i].detach().cpu()
                pr0  = (torch.sigmoid(logits[i]) > 0.5).float().detach().cpu()
                vis_path = os.path.join(args.outdir, "visualizations",
                                        f"val_epoch{epoch}_sample{i}_dice{best_dice:.3f}.png")
                save_vis(img0, msk0, pr0, vis_path)
            print(f"  -> saved best to {ckpt_path}")

    print(f"Training done. Best Val Dice = {best_dice:.4f}")

if __name__ == "__main__":
    main()
