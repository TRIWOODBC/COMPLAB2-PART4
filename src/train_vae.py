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

class OASISPNGSlices(Dataset):
    def __init__(self, data_root: str, split: str = "train", img_size: int = 256):
        self.root = Path(data_root) / SPLIT2DIR[split]
        self.files = sorted([p for p in self.root.glob("*.png")])
        if not self.files:
            raise FileNotFoundError(f"No PNG files found in {self.root}")
        self.tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
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
# VAE Model
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=32, img_size=256, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(base, base*2, 4, 2, 1), GN(base*2), nn.ReLU(True),
            nn.Conv2d(base*2, base*4, 4, 2, 1), GN(base*4), nn.ReLU(True),
            nn.Conv2d(base*4, base*8, 4, 2, 1), GN(base*8), nn.ReLU(True),
        )
        feat = img_size // 16
        self.fc_mu = nn.Linear(base*8*feat*feat, latent_dim)
        self.fc_logvar = nn.Linear(base*8*feat*feat, latent_dim)

    def forward(self, x):
        h = self.net(x)
        h = torch.flatten(h, 1)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim=32, img_size=256, base=32):
        super().__init__()
        feat = img_size // 16
        self.fc = nn.Linear(latent_dim, base*8*feat*feat)
        self.unflatten = nn.Unflatten(1, (base*8, feat, feat))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base*8, base*4, 4, 2, 1), GN(base*4), nn.ReLU(True),
            nn.ConvTranspose2d(base*4, base*2, 4, 2, 1), GN(base*2), nn.ReLU(True),
            nn.ConvTranspose2d(base*2, base, 4, 2, 1),   GN(base),   nn.ReLU(True),
            nn.ConvTranspose2d(base, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, z): return self.net(self.unflatten(self.fc(z)))

class VAE(nn.Module):
    def __init__(self, latent_dim=32, img_size=256):
        super().__init__()
        self.enc = Encoder(latent_dim, img_size)
        self.dec = Decoder(latent_dim, img_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar

# -----------------------------
# Loss
# -----------------------------
def vae_loss(x, x_rec, mu, logvar, recon_type="mse", beta=1.0):
    if recon_type == "mse":
        recon = F.mse_loss(x_rec, x, reduction="mean")
    else:
        recon = F.binary_cross_entropy(x_rec, x, reduction="mean")
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
    for k,v in history.items(): plt.plot(v, label=k)
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss")
    plt.savefig(path); plt.close()

# -----------------------------
# Train
# -----------------------------
def train(data_root="data/OASIS", out_dir="outputs/vae", img_size=256,
          batch_size=16, epochs=50, lr=1e-4, latent_dim=32, seed=42,
          save_every=10, recon="mse", kl_warmup_epochs=15, max_grad_norm=1.0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_tr = OASISPNGSlices(data_root, "train", img_size)
    ds_va = OASISPNGSlices(data_root, "validate", img_size)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size)

    model = VAE(latent_dim, img_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    out = Path(out_dir); (out/"checkpoints").mkdir(parents=True, exist_ok=True); (out/"figs").mkdir(parents=True, exist_ok=True)
    history = {"train": [], "val": []}; best_val = float("inf")

    for ep in range(1, epochs+1):
        beta = min(1.0, ep / max(1, kl_warmup_epochs))
        model.train(); tr_loss=[]
        for x in tqdm(dl_tr, desc=f"Epoch {ep}/{epochs}"):
            x = x.to(device); opt.zero_grad(set_to_none=True)
            x_rec, mu, logvar = model(x)
            loss, _, _ = vae_loss(x, x_rec, mu, logvar, recon_type=recon, beta=beta)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            tr_loss.append(loss.item())
        history["train"].append(np.mean(tr_loss))

        model.eval(); va_loss=[]
        with torch.no_grad():
            for x in dl_va:
                x = x.to(device); x_rec, mu, logvar = model(x)
                l, _, _ = vae_loss(x, x_rec, mu, logvar, recon_type=recon, beta=1.0)
                va_loss.append(l.item())
        val_mean = np.mean(va_loss); history["val"].append(val_mean)

        if ep % save_every == 0 or ep == 1:
            with torch.no_grad():
                x = next(iter(dl_va))[:32].to(device)
                x_rec,_,_ = model(x)
                save_grid(x.cpu(), out/"figs"/f"ep{ep:03d}_inputs.png")
                save_grid(x_rec.cpu(), out/"figs"/f"ep{ep:03d}_recon.png")

        ckpt={"model":model.state_dict(),"epoch":ep,"val_loss":val_mean}
        torch.save(ckpt, out/"checkpoints"/"last.pt")
        if val_mean < best_val:
            best_val=val_mean; torch.save(ckpt, out/"checkpoints"/"best.pt")

        plot_curves(history, out/"figs"/"curves.png")
        print(f"Epoch {ep} done. Val loss={val_mean:.4f}")

    print("Training finished. Best val loss:", best_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/OASIS")
    parser.add_argument("--out_dir", type=str, default="outputs/vae")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--recon", type=str, default="mse", choices=["mse","bce"])
    parser.add_argument("--kl_warmup_epochs", type=int, default=15)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    args = parser.parse_args()

    train(**vars(args))
