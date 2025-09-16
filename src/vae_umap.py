# src/vae_umap.py
import argparse
from pathlib import Path
import re
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from train_vae import VAE, OASISPNGSlices

# ---------- helpers ----------
def parse_subject_from_filename(p: Path) -> str:
    """
    Try to parse subject ID from typical OASIS-style filenames.
    Examples it will handle:
      OAS1_0001_...png        -> OAS1_0001
      OASIS-TR-001_slice.png  -> OASIS-TR-001
      OASIS_OAS1_0123_...png  -> OAS1_0123
    If no pattern is found, returns 'Unknown'.
    """
    s = p.name
    # Match OAS1_0123
    m = re.search(r"(OAS1[_-]\d{3,4})", s, re.IGNORECASE)
    if m: return m.group(1).upper()
    # Match OASIS-TR-001 or similar
    m = re.search(r"(OASIS[-_][A-Z]{2}[-_]\d{3,4})", s, re.IGNORECASE)
    if m: return m.group(1).upper()
    # Match OASIS_0123
    m = re.search(r"(OASIS[-_]*\d{3,4})", s, re.IGNORECASE)
    if m: return m.group(1).upper()
    return "Unknown"

def extract_latents_and_labels(ds: OASISPNGSlices, model: VAE, dl, device,
                               use_mu=True, color_by="subject",
                               split_name="validate", csv_map=None):
    """
    Encode dataset images into latent vectors (mu or z) and assign labels
    for coloring in the UMAP plot.
    """
    model.eval()
    latents = []
    labels = []

    with torch.no_grad():
        for x in dl:
            x = x.to(device)
            mu, logvar = model.enc(x)
            feats = mu if use_mu else (mu + torch.randn_like(mu) * torch.exp(0.5 * logvar))
            latents.append(feats.cpu().numpy())

    # Build labels according to the chosen color scheme
    if color_by == "subject":
        labels = [parse_subject_from_filename(p) for p in ds.files]
    elif color_by == "split":
        labels = [split_name] * len(ds)
    elif color_by == "csv" and csv_map is not None:
        labels = []
        for p in ds.files:
            labels.append(csv_map.get(p.name, "Unknown"))
    else:
        labels = ["Unknown"] * len(ds)

    return np.concatenate(latents, axis=0), labels

def plot_umap(latents_2d, labels, out_png: Path, title="VAE latent space (UMAP)"):
    """
    Plot a UMAP scatter with color by label.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))

    unique_labels = sorted(set(labels))
    colors = plt.cm.get_cmap("tab20", len(unique_labels))

    for i, lab in enumerate(unique_labels):
        idx = [j for j, l in enumerate(labels) if l == lab]
        plt.scatter(latents_2d[idx, 0], latents_2d[idx, 1],
                    s=6, alpha=0.7, label=lab, color=colors(i))

    plt.title(title)
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="outputs/vae/checkpoints/best.pt",
                        help="Path to checkpoint (best.pt or last.pt)")
    parser.add_argument("--data_root", type=str, default="data/OASIS")
    parser.add_argument("--split", type=str, default="validate", choices=["validate", "test", "train"])
    parser.add_argument("--img_size", type=int, default=256, help="Image size (must match training)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=None, help="Leave empty to auto-detect")
    parser.add_argument("--out_dir", type=str, default="outputs/vae/figs")
    parser.add_argument("--use_z", action="store_true", help="Use sampled z instead of mu")
    parser.add_argument("--color_by", type=str, default="subject",
                        choices=["subject", "split", "csv"], help="How to color the points")
    parser.add_argument("--csv_labels", type=str, default=None,
                        help="Optional CSV with filename,label for coloring")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and loader
    ds = OASISPNGSlices(args.data_root, split=args.split, img_size=args.img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)

    # Detect latent_dim
    latent_dim_ckpt = None
    img_size_ckpt = args.img_size
    if isinstance(ckpt, dict) and "args" in ckpt and isinstance(ckpt["args"], dict):
        latent_dim_ckpt = ckpt["args"].get("latent_dim", None)
        img_size_ckpt = ckpt["args"].get("img_size", img_size_ckpt)
    if latent_dim_ckpt is None:
        try:
            latent_dim_ckpt = ckpt["model"]["enc.fc_mu.weight"].shape[0]
        except Exception:
            latent_dim_ckpt = args.latent_dim
    if args.latent_dim is not None:
        latent_dim_ckpt = args.latent_dim

    # Build model
    model = VAE(latent_dim=latent_dim_ckpt, img_size=img_size_ckpt).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[info] Loaded ckpt with latent_dim={latent_dim_ckpt}, img_size={img_size_ckpt}")

    # Optional CSV labels
    csv_map = None
    if args.color_by == "csv" and args.csv_labels:
        csv_map = {}
        with open(args.csv_labels, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    csv_map[row[0]] = row[1]

    # Extract latents and labels
    latents, labels = extract_latents_and_labels(
        ds, model, dl, device,
        use_mu=not args.use_z,
        color_by=args.color_by,
        split_name=args.split,
        csv_map=csv_map
    )

    # UMAP
    import umap
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
    latents_2d = reducer.fit_transform(latents)

    out_png = Path(args.out_dir) / f"umap_colorby-{args.color_by}.png"
    plot_umap(latents_2d, labels, out_png,
              title=f"VAE latent space (UMAP) — split={args.split}, {'mu' if not args.use_z else 'z'}")
    print(f"UMAP saved to {out_png}")

if __name__ == "__main__":
    main()
