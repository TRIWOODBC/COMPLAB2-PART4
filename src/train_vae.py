# src/vae_train.py
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import argparse

# -----------------------------
# Dataset (OASIS PNG only)
# -----------------------------
SPLIT2DIR = {
    "train": "keras_png_slices_train",
    "validate": "keras_png_slices_validate",
}

class MRIIntensityNorm(object):
    """Percentile clip (1-99) + per-image z-score + min-max to [0,1]."""
    def __init__(self, lo=1, hi=99):
        self.lo = lo
        self.hi = hi
    def __call__(self, img: Image.Image):
        x = np.array(img.convert('L'), dtype=np.float32)
        lo = np.percentile(x, self.lo)
        hi = np.percentile(x, self.hi)
        x = np.clip(x, lo, hi)
        mean = x.mean()
        std = x.std() + 1e-6
        x = (x - mean) / std
        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min + 1e-6)
        return Image.fromarray((x * 255.0).astype(np.uint8))

class ResizeIfNeeded(object):
    """Only resize when size mismatch; keeps native resolution when already correct."""
    def __init__(self, size):
        self.size = size if isinstance(size, (tuple, list)) else (size, size)
    def __call__(self, img: Image.Image):
        if img.size != self.size[::-1]:  # PIL: (W,H)
            return transforms.functional.resize(img, self.size, antialias=True)
        return img

class OASISPNGSlices(Dataset):
    def __init__(self, data_root: str, split: str = "train", img_size: int = 256):
        self.root = Path(data_root) / SPLIT2DIR[split]
        self.files = sorted([p for p in self.root.glob("*.png")])
        if not self.files:
            raise FileNotFoundError(f"No PNG files found in {self.root}")
        self.tf = transforms.Compose([
            MRIIntensityNorm(),
            ResizeIfNeeded((img_size, img_size)),
            transforms.ToTensor(),  # -> [0,1], shape (1,H,W)
        ])

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        return self.tf(img)  # 1xHxW

# -----------------------------
# Helpers
# -----------------------------
def GN(c, g=8):
    return nn.GroupNorm(num_groups=min(g, c), num_channels=c)

# -----------------------------
# VAE Model (6 down blocks)
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=64, img_size=256, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 4, 2, 1), nn.ReLU(True),                    # 256->128
            nn.Conv2d(base, base*2, 4, 2, 1), GN(base*2), nn.ReLU(True),   # 128->64
            nn.Conv2d(base*2, base*4, 4, 2, 1), GN(base*4), nn.ReLU(True), # 64->32
            nn.Conv2d(base*4, base*8, 4, 2, 1), GN(base*8), nn.ReLU(True), # 32->16
            nn.Conv2d(base*8, base*16, 4, 2, 1), GN(base*16), nn.ReLU(True),#16->8
            nn.Conv2d(base*16, base*32, 4, 2, 1), GN(base*32), nn.ReLU(True),#8->4
        )
        feat = img_size // (2**6)  # 256/64=4
        self.fc_mu = nn.Linear(base*32*feat*feat, latent_dim)
        self.fc_logvar = nn.Linear(base*32*feat*feat, latent_dim)

    def forward(self, x):
        h = self.net(x)
        h = torch.flatten(h, 1)
        return self.fc_mu(h), self.fc_logvar(h)

class UpBlock(nn.Module):
    """Nearest-neighbor upsample + Conv(3x3)x2 (GN+ReLU)."""
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(cin, cout, 3, 1, 1), GN(cout), nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, 1, 1), GN(cout), nn.ReLU(True),
        )
    def forward(self, x): return self.block(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=64, img_size=256, base=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, base*32*4*4)
        self.unflatten = nn.Unflatten(1, (base*32, 4, 4))
        self.up1 = UpBlock(base*32, base*16) # 4->8
        self.up2 = UpBlock(base*16, base*8)  # 8->16
        self.up3 = UpBlock(base*8, base*4)   # 16->32
        self.up4 = UpBlock(base*4, base*2)   # 32->64
        self.up5 = UpBlock(base*2, base)     # 64->128
        self.conv_out = nn.Conv2d(base, 1, 3, 1, 1)  # 128->128 (then next up brings to 256)
        # 再补一次上采样把 128->256
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(1, 1, 3, 1, 1)
        )
        self.act = nn.Sigmoid()

    def forward(self, z):
        h = self.unflatten(self.fc(z))
        h = self.up1(h); h = self.up2(h); h = self.up3(h); h = self.up4(h); h = self.up5(h)
        h = self.conv_out(h)
        h = self.final_up(h)
        return self.act(h)

class VAE(nn.Module):
    def __init__(self, latent_dim=64, img_size=256, base=32):
        super().__init__()
        self.enc = Encoder(latent_dim, img_size, base)
        self.dec = Decoder(latent_dim, img_size, base)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar

# -----------------------------
# Loss (L1+SSIM, MSE, BCE)
# -----------------------------
def ssim(x, y, C1=0.01**2, C2=0.03**2):
    # x,y: (B,1,H,W) in [0,1]
    mu_x = F.avg_pool2d(x, 11, 1, 5)
    mu_y = F.avg_pool2d(y, 11, 1, 5)
    sigma_x = F.avg_pool2d(x*x, 11, 1, 5) - mu_x**2
    sigma_y = F.avg_pool2d(y*y, 11, 1, 5) - mu_y**2
    sigma_xy = F.avg_pool2d(x*y, 11, 1, 5) - mu_x*mu_y
    ssim_n = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    ssim_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = ssim_n / (ssim_d + 1e-8)
    return ssim_map.mean()

def vae_loss(x, x_rec, mu, logvar, recon_type="l1_ssim", beta=1.0, ssim_w=0.5):
    if recon_type == "mse":
        recon = F.mse_loss(x_rec, x, reduction="mean")
    elif recon_type == "bce":
        recon = F.binary_cross_entropy(x_rec, x, reduction="mean")
    else:
        l1 = F.l1_loss(x_rec, x, reduction="mean")
        ssim_loss = 1.0 - ssim(x_rec, x)
        recon = (1 - ssim_w) * l1 + ssim_w * ssim_loss
    logvar = logvar.clamp(-10, 10)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon + beta * kl
    return loss, recon.item(), kl.item()

# -----------------------------
# Utils
# -----------------------------
def save_grid(tensor, path, nrow=8):
    grid = utils.make_grid(tensor, nrow=nrow)
    nd = (grid.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
    if nd.shape[-1] == 1: nd = nd.squeeze(-1)
    Image.fromarray(nd).save(path)

def plot_curves(history, path):
    plt.figure()
    for k,v in history.items():
        plt.plot(v, label=k)
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()

# -----------------------------
# Train
# -----------------------------
def train(data_root="data/OASIS", out_dir="outputs/vae", img_size=256,
          batch_size=8, epochs=80, lr=2e-4, latent_dim=128, seed=42,
          save_every=10, recon="l1_ssim", kl_warmup_epochs=40, beta_cap=0.5,
          ssim_w=0.5, max_grad_norm=5.0, base=32):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_tr = OASISPNGSlices(data_root, "train", img_size)
    ds_va = OASISPNGSlices(data_root, "validate", img_size)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, num_workers=2, pin_memory=True)

    model = VAE(latent_dim, img_size, base=base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    out = Path(out_dir); (out/"checkpoints").mkdir(parents=True, exist_ok=True); (out/"figs").mkdir(parents=True, exist_ok=True)
    history = {"train": [], "val": []}; best_val = float("inf")

    for ep in range(1, epochs+1):
        beta = min(beta_cap, (ep / max(1, kl_warmup_epochs)) * beta_cap)
        model.train(); tr_loss=[]
        pbar = tqdm(dl_tr, desc=f"Epoch {ep}/{epochs}")
        for x in pbar:
            x = x.to(device); opt.zero_grad(set_to_none=True)
            x_rec, mu, logvar = model(x)
            loss, _, _ = vae_loss(x, x_rec, mu, logvar, recon_type=recon, beta=beta, ssim_w=ssim_w)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            tr_loss.append(loss.item())
            pbar.set_postfix(loss=np.mean(tr_loss))

        history["train"].append(np.mean(tr_loss))

        # ----- Deterministic eval: z = mu -----
        model.eval(); va_loss=[]
        with torch.no_grad():
            for x in dl_va:
                x = x.to(device)
                mu, logvar = model.enc(x)
                x_rec = model.dec(mu)
                l, _, _ = vae_loss(x, x_rec, mu, logvar, recon_type=recon, beta=beta_cap, ssim_w=ssim_w)
                va_loss.append(l.item())
        val_mean = np.mean(va_loss); history["val"].append(val_mean)

        # Save sample grids (deterministic recon)
        if ep % save_every == 0 or ep == 1:
            with torch.no_grad():
                x = next(iter(dl_va))[:32].to(device)
                mu, logvar = model.enc(x)
                x_rec = model.dec(mu)
                save_grid(x.cpu(), out/"figs"/f"ep{ep:03d}_inputs.png")
                save_grid(x_rec.cpu(), out/"figs"/f"ep{ep:03d}_recon.png")

        ckpt={"model":model.state_dict(),"epoch":ep,"val_loss":val_mean,
              "cfg":{"latent_dim":latent_dim,"img_size":img_size,"base":base}}
        torch.save(ckpt, out/"checkpoints"/"last.pt")
        if val_mean < best_val:
            best_val=val_mean; torch.save(ckpt, out/"checkpoints"/"best.pt")

        plot_curves(history, out/"figs"/"curves.png")
        print(f"Epoch {ep} done. Val loss={val_mean:.6f} (beta={beta:.3f})")

    print("Training finished. Best val loss:", best_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/OASIS")
    parser.add_argument("--out_dir", type=str, default="outputs/vae")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--recon", type=str, default="l1_ssim", choices=["mse","bce","l1_ssim"])
    parser.add_argument("--kl_warmup_epochs", type=int, default=40)
    parser.add_argument("--beta_cap", type=float, default=0.5, help="final beta after warmup (e.g., 0.3~0.5)")
    parser.add_argument("--ssim_w", type=float, default=0.5, help="weight for SSIM in l1_ssim")
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--base", type=int, default=32, help="channel base width; try 48/64 if VRAM allows")
    args = parser.parse_args()

    train(**vars(args))
