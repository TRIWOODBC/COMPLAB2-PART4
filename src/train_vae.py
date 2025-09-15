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

# -----------------------------
# Dataset (OASIS PNG only)
# -----------------------------
SPLIT2DIR = {
    "train": "keras_png_slices_train",
    "validate": "keras_png_slices_validate",
}

class OASISPNGSlices(Dataset):
    def __init__(self, data_root: str, split: str = "train", img_size: int = 128):
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
# VAE Model
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=16, img_size=128, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(base, base*2, 4, 2, 1), nn.BatchNorm2d(base*2), nn.ReLU(True),
            nn.Conv2d(base*2, base*4, 4, 2, 1), nn.BatchNorm2d(base*4), nn.ReLU(True),
            nn.Conv2d(base*4, base*8, 4, 2, 1), nn.BatchNorm2d(base*8), nn.ReLU(True),
        )
        feat = img_size // 16
        self.fc_mu = nn.Linear(base*8*feat*feat, latent_dim)
        self.fc_logvar = nn.Linear(base*8*feat*feat, latent_dim)

    def forward(self, x):
        h = self.net(x)
        h = torch.flatten(h, 1)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim=16, img_size=128, base=32):
        super().__init__()
        feat = img_size // 16
        self.fc = nn.Linear(latent_dim, base*8*feat*feat)
        self.unflatten = nn.Unflatten(1, (base*8, feat, feat))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base*8, base*4, 4, 2, 1), nn.BatchNorm2d(base*4), nn.ReLU(True),
            nn.ConvTranspose2d(base*4, base*2, 4, 2, 1), nn.BatchNorm2d(base*2), nn.ReLU(True),
            nn.ConvTranspose2d(base*2, base, 4, 2, 1),   nn.BatchNorm2d(base),   nn.ReLU(True),
            nn.ConvTranspose2d(base, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, z): return self.net(self.unflatten(self.fc(z)))

class VAE(nn.Module):
    def __init__(self, latent_dim=16, img_size=128):
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
def vae_loss(x, x_rec, mu, logvar):
    recon = F.binary_cross_entropy(x_rec, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl, recon.item(), kl.item()

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
def train(data_root="data/OASIS", out_dir="outputs/vae", img_size=128,
          batch_size=64, epochs=50, lr=2e-4, latent_dim=16, seed=42):
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
        model.train(); tr_loss=[]
        for x in tqdm(dl_tr, desc=f"Epoch {ep}/{epochs}"):
            x = x.to(device); opt.zero_grad()
            x_rec, mu, logvar = model(x)
            loss,_,_ = vae_loss(x, x_rec, mu, logvar)
            loss.backward(); opt.step(); tr_loss.append(loss.item())
        history["train"].append(np.mean(tr_loss))

        model.eval(); va_loss=[]
        with torch.no_grad():
            for x in dl_va:
                x = x.to(device); x_rec, mu, logvar = model(x)
                l,_,_ = vae_loss(x, x_rec, mu, logvar)
                va_loss.append(l.item())
        val_mean = np.mean(va_loss); history["val"].append(val_mean)

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
    train()
