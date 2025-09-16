import os, argparse, numpy as np
from dataclasses import dataclass
from PIL import Image

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid


# ------------------------------
# Dataset
# ------------------------------
class OasisSlices(Dataset):
    """OASIS MRI dataset loader (image, segmentation mask)."""
    def __init__(self, root, split="train", image_size=128):
        super().__init__()
        self.image_size = image_size
        img_dir = os.path.join(root, f"keras_png_slices_{split}")
        mask_dir = os.path.join(root, f"keras_png_slices_seg_{split}")

        self.images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        self.masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])

    def __len__(self):
        return len(self.images)

    def _resize(self, arr, size, nearest=False):
        """Resize using bilinear (image) or nearest (mask)."""
        mode = Image.NEAREST if nearest else Image.BILINEAR
        arr = arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
        return np.array(Image.fromarray(arr).resize((size, size), mode))

    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]).convert("L"))
        mask = np.array(Image.open(self.masks[idx]).convert("L"))

        img = self._resize(img, self.image_size) / 255.0
        mask = self._resize(mask, self.image_size, nearest=True)
        mask = (mask > 0).astype(np.int64)

        img = torch.from_numpy(img).unsqueeze(0).float()
        mask = torch.from_numpy(mask).long()
        return img, mask


# ------------------------------
# UNet building blocks
# ------------------------------
class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Standard UNet for binary segmentation."""
    def __init__(self, in_ch=1, out_ch=2):
        super().__init__()
        self.d1 = DoubleConv(in_ch, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)
        self.bridge = DoubleConv(512, 1024)

        self.u1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.u1_conv = DoubleConv(1024, 512)
        self.u2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.u2_conv = DoubleConv(512, 256)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.u3_conv = DoubleConv(256, 128)
        self.u4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.u4_conv = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_ch, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))
        bridge = self.bridge(self.pool(d4))

        u1 = self.u1(bridge)
        u1 = self.u1_conv(torch.cat([u1, d4], 1))
        u2 = self.u2(u1)
        u2 = self.u2_conv(torch.cat([u2, d3], 1))
        u3 = self.u3(u2)
        u3 = self.u3_conv(torch.cat([u3, d2], 1))
        u4 = self.u4(u3)
        u4 = self.u4_conv(torch.cat([u4, d1], 1))

        return self.out_conv(u4)


# ------------------------------
# Variational Autoencoder (VAE)
# ------------------------------
class VAE(nn.Module):
    """Convolutional VAE for MRI reconstruction."""
    def __init__(self, in_ch=1, latent_dim=128, image_size=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU()
        )
        feat_size = image_size // 16
        self.fc_mu = nn.Linear(256 * feat_size * feat_size, latent_dim)
        self.fc_logvar = nn.Linear(256 * feat_size * feat_size, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 256 * feat_size * feat_size)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, in_ch, 4, 2, 1), nn.Sigmoid()
        )
        self.feat_size = feat_size

    def encode(self, x):
        h = self.enc(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 256, self.feat_size, self.feat_size)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar


# ------------------------------
# Losses and Metrics
# ------------------------------
def dice_loss(pred, target):
    pred = F.softmax(pred, dim=1)[:, 1]
    target = (target == 1).float()
    inter = (pred * target).sum()
    return 1 - (2. * inter + 1e-8) / (pred.sum() + target.sum() + 1e-8)

def foreground_dice(pred, target):
    pred = F.softmax(pred, dim=1)[:, 1]
    target = (target == 1).float()
    inter = (pred * target).sum()
    return (2. * inter + 1e-8) / (pred.sum() + target.sum() + 1e-8)

def iou_score(pred, target, thr=0.5):
    pred = (F.softmax(pred, dim=1)[:, 1] > thr).float()
    target = (target == 1).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return inter / (union + 1e-8)

def vae_loss_fn(recon, x, mu, logvar):
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl

ce_loss = nn.CrossEntropyLoss()
def seg_loss_combo(logits, y):
    """Combined segmentation loss: Dice + CE"""
    return dice_loss(logits, y) + 0.5 * ce_loss(logits, y)


# ------------------------------
# Config
# ------------------------------
@dataclass
class CFG:
    data_root: str = "data/OASIS"
    outputs: str = "outputs"
    image_size: int = 128
    batch_size: int = 8
    epochs: int = 10
    lr: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------
# Training
# ------------------------------
def train(cfg):
    # datasets
    ds_tr = OasisSlices(cfg.data_root, "train", cfg.image_size)
    ds_va = OasisSlices(cfg.data_root, "validate", cfg.image_size)
    ds_te = OasisSlices(cfg.data_root, "test", cfg.image_size)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    dl_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    # models
    unet = UNet().to(cfg.device)
    vae = VAE(image_size=cfg.image_size).to(cfg.device)
    opt = torch.optim.Adam(list(unet.parameters()) + list(vae.parameters()), lr=cfg.lr)

    # dirs
    img_dir = os.path.join(cfg.outputs, "images"); os.makedirs(img_dir, exist_ok=True)
    model_dir = os.path.join(cfg.outputs, "models"); os.makedirs(model_dir, exist_ok=True)

    # log file
    log_file = os.path.join(cfg.outputs, "train.log")
    with open(log_file, "w") as f:
        f.write("Epoch,TrainLoss,ValLoss,ValDice,ValFgDice,ValIoU,Best\n")

    best_dice = 0.0

    for ep in range(1, cfg.epochs + 1):
        # --- Train ---
        unet.train(); vae.train()
        total_loss = 0
        for img, mask in dl_tr:
            img, mask = img.to(cfg.device), mask.to(cfg.device)
            logits = unet(img)
            seg_loss = seg_loss_combo(logits, mask)
            recon, mu, logvar = vae(img)
            vae_loss = vae_loss_fn(recon, img, mu, logvar)
            loss = seg_loss + 0.5 * vae_loss
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        train_loss = total_loss / len(dl_tr)

        # --- Validation ---
        unet.eval(); vae.eval()
        val_losses, dices, fg_dices, ious = [], [], [], []
        with torch.no_grad():
            for img, mask in dl_va:
                img, mask = img.to(cfg.device), mask.to(cfg.device)
                logits = unet(img)
                seg_loss = seg_loss_combo(logits, mask)
                recon, mu, logvar = vae(img)
                vloss = seg_loss + 0.5 * vae_loss_fn(recon, img, mu, logvar)

                val_losses.append(vloss.item())
                dices.append(1 - dice_loss(logits, mask).item())
                fg_dices.append(foreground_dice(logits, mask).item())
                ious.append(iou_score(logits, mask).item())

            val_loss = np.mean(val_losses)
            val_dice = np.mean(dices)
            val_fg = np.mean(fg_dices)
            val_iou = np.mean(ious)

        print(f"[Epoch {ep}] TrainLoss={train_loss:.4f} | "
              f"ValLoss={val_loss:.4f} | ValDice={val_dice:.4f} | "
              f"FgDice={val_fg:.4f} | IoU={val_iou:.4f}")

        with open(log_file, "a") as f:
            f.write(f"{ep},{train_loss:.4f},{val_loss:.4f},{val_dice:.4f},"
                    f"{val_fg:.4f},{val_iou:.4f},{val_dice>best_dice}\n")

        # save best
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"unet": unet.state_dict(), "vae": vae.state_dict()},
                       os.path.join(model_dir, "best_unet_vae.pt"))

        # save visuals
        if ep % 5 == 0 or ep == 1:
            pred = torch.argmax(logits, dim=1, keepdim=True).float()
            grid = make_grid(torch.cat([img[:4],
                                        mask[:4].unsqueeze(1).float(),
                                        pred[:4]], 0),
                             nrow=4, normalize=True)
            save_image(grid, os.path.join(img_dir, f"val_seg_ep{ep:03d}.png"))

            grid2 = make_grid(torch.cat([img[:4], recon[:4]], 0),
                              nrow=4, normalize=True)
            save_image(grid2, os.path.join(img_dir, f"val_vae_ep{ep:03d}.png"))

    # --- Test ---
    print("Running test inference...")
    unet.eval(); vae.eval()
    with torch.no_grad():
        for img, mask in dl_te:
            img, mask = img.to(cfg.device), mask.to(cfg.device)
            logits = unet(img)
            recon, _, _ = vae(img)
            pred = torch.argmax(logits, dim=1, keepdim=True).float()
            grid = make_grid(torch.cat([img[:4],
                                        mask[:4].unsqueeze(1).float(),
                                        pred[:4]], 0),
                             nrow=4, normalize=True)
            save_image(grid, os.path.join(img_dir, "test_seg.png"))
            grid2 = make_grid(torch.cat([img[:4], recon[:4]], 0),
                              nrow=4, normalize=True)
            save_image(grid2, os.path.join(img_dir, "test_vae.png"))
            break


# ------------------------------
# Entry
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=128)
    args = parser.parse_args()
    cfg = CFG(epochs=args.epochs, batch_size=args.batch_size, image_size=args.image_size)
    train(cfg)

if __name__ == "__main__":
    main()
