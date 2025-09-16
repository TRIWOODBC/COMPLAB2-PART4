# src/vae_umap.py
import argparse
from pathlib import Path
import re
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import your training code (encoder/decoder + dataset)
from train_vae import VAE, OASISPNGSlices


# -------------------- helpers --------------------
def parse_subject_from_filename(p: Path) -> str:
    """
    Parse a subject/patient ID from common filename patterns.

    Supported examples:
      - case_402_slice_10.nii.png  -> case_402
      - case-123-slice-7.png       -> case-123
      - OAS1_0001_...png           -> OAS1_0001
      - OASIS-TR-001_slice.png     -> OASIS-TR-001
      - OASIS_OAS1_0123_...png     -> OAS1_0123
      - Fallbacks:
          * strip the trailing "slice_xxx" and extensions (including .nii.png)
          * use parent folder name if it looks like a subject folder
          * finally use filename stem
    """
    s = p.name

    # (A) Kaggle-style / medical exports: "case_402_slice_10.nii.png"
    m = re.search(r"(case[_-]\d+)", s, re.IGNORECASE)
    if m:
        return m.group(1).lower()  # normalize to lower for grouping

    # (B) OASIS-like patterns
    for pat in [
        r"(OAS1[_-]\d{3,4})",
        r"(OASIS[-_][A-Z]{2}[-_]\d{3,4})",
        r"(OASIS[-_]*\d{3,4})",
    ]:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # (C) Generic fallback: remove trailing "slice_###" and double extensions
    #     e.g. "case_402_slice_10.nii.png" -> "case_402"
    m = re.search(r"^(.*?)(?:[_-]?slice[_-]?\d+)?(?:\.nii)?\.png$", s, re.IGNORECASE)
    if m and len(m.group(1)) >= 3:
        return m.group(1)

    # (D) If files are organized as subject folders, use the folder name
    parent = p.parent.name
    if parent and parent.lower() not in {
        "keras_png_slices_train", "keras_png_slices_validate",
        "train", "validate", "test"
    }:
        return parent

    # (E) Last resort: filename without extension
    return p.stem


def build_labels(ds: OASISPNGSlices, mode: str, split_name: str, csv_map):
    """Return a label per file according to the chosen coloring scheme."""
    if mode == "subject":
        return [parse_subject_from_filename(p) for p in ds.files]
    if mode == "split":
        return [split_name] * len(ds)
    if mode == "filename":
        return [p.name for p in ds.files]
    if mode == "csv" and csv_map is not None:
        return [csv_map.get(p.name, "Unknown") for p in ds.files]
    return ["Unknown"] * len(ds)


def extract_latents(ds, model: VAE, dl, device, use_mu=True):
    """Encode images to latent vectors (mu by default, z if requested)."""
    model.eval()
    latents = []
    with torch.no_grad():
        for x in tqdm(dl, desc="Encoding"):
            x = x.to(device, non_blocking=True)
            mu, logvar = model.enc(x)
            feats = mu if use_mu else (mu + torch.randn_like(mu) * torch.exp(0.5 * logvar))
            latents.append(feats.cpu().numpy())
    return np.concatenate(latents, axis=0)


def plot_umap(latents_2d, labels, out_png: Path, title="VAE latent space (UMAP)"):
    """Scatter plot of UMAP embeddings with labels as colors."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))

    uniq = list(dict.fromkeys(labels))  # preserve first-seen order
    too_many = len(uniq) > 20
    cmap = plt.cm.get_cmap("tab20", min(len(uniq), 20))

    for i, lab in enumerate(uniq):
        idx = [j for j, l in enumerate(labels) if l == lab]
        color = cmap(i % 20) if not too_many else None
        plt.scatter(
            latents_2d[idx, 0], latents_2d[idx, 1],
            s=6, alpha=0.75, label=lab, c=[color] if color else None
        )

    plt.title(title)
    if not too_many:
        plt.legend(markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        plt.xlabel(f"{len(uniq)} labels (legend omitted)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="outputs/vae/checkpoints/best.pt",
                        help="Path to checkpoint (best.pt or last.pt)")
    parser.add_argument("--data_root", type=str, default="data/OASIS")
    parser.add_argument("--split", type=str, default="test", choices=["validate", "test", "train"])
    parser.add_argument("--img_size", type=int, default=256, help="Must match training size")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=None, help="Override; otherwise infer from ckpt")
    parser.add_argument("--base", type=int, default=None, help="Override base width; otherwise infer from ckpt")
    parser.add_argument("--out_dir", type=str, default="outputs/vae/figs")
    parser.add_argument("--use_z", action="store_true", help="Use sampled z instead of mu")
    parser.add_argument("--color_by", type=str, default="subject",
                        choices=["subject", "split", "csv", "filename"])
    parser.add_argument("--csv_labels", type=str, default=None, help="CSV: filename,label")
    parser.add_argument("--save_csv", action="store_true", help="Export 2D coords and label as CSV")
    parser.add_argument("--pca_dim", type=int, default=0, help="Run PCA to this dim before UMAP (0=off)")
    parser.add_argument("--standardize", action="store_true", help="Standardize features before PCA/UMAP")
    parser.add_argument("--seed", type=int, default=42)
    # UMAP knobs
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--metric", type=str, default="euclidean")
    args = parser.parse_args()

    # Repro + device
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset/loader (no shuffle!)
    ds = OASISPNGSlices(args.data_root, split=args.split, img_size=args.img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Load checkpoint and infer model config
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = {}
    if isinstance(ckpt, dict):
        cfg = ckpt.get("cfg", ckpt.get("args", {})) or {}
    latent_dim = args.latent_dim or cfg.get("latent_dim", None)
    img_size_ckpt = cfg.get("img_size", args.img_size)
    base = args.base or cfg.get("base", 32)

    if latent_dim is None:
        try:
            latent_dim = ckpt["model"]["enc.fc_mu.weight"].shape[0]
        except Exception:
            raise RuntimeError("Cannot infer latent_dim; pass --latent_dim explicitly.")

    model = VAE(latent_dim=latent_dim, img_size=img_size_ckpt, base=base).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    print(f"[info] Loaded ckpt: latent_dim={latent_dim}, img_size={img_size_ckpt}, base={base}")

    # Labels (optional CSV)
    csv_map = None
    if args.color_by == "csv" and args.csv_labels:
        csv_map = {}
        with open(args.csv_labels, newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    csv_map[row[0]] = row[1]
    labels = build_labels(ds, args.color_by, args.split, csv_map)

    # Encode to latents
    latents = extract_latents(ds, model, dl, device, use_mu=not args.use_z)

    # Optional: Standardize + PCA
    X = latents
    if args.standardize:
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)
    if args.pca_dim and args.pca_dim > 0:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=args.pca_dim, random_state=args.seed)
        X = pca.fit_transform(X)
        print(f"[info] PCA -> {X.shape[1]} dims (explained var sum: {pca.explained_variance_ratio_.sum():.3f})")

    # UMAP
    try:
        import umap
    except ImportError:
        raise SystemExit("Please install umap-learn: pip install umap-learn")
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
    )
    latents_2d = reducer.fit_transform(X)

    # Save figure
    out_dir = Path(args.out_dir)
    out_png = out_dir / f"umap_colorby-{args.color_by}_{'z' if args.use_z else 'mu'}.png"
    plot_umap(latents_2d, labels, out_png,
              title=f"UMAP — split={args.split}, {'z' if args.use_z else 'mu'}")
    print(f"[ok] UMAP saved to {out_png}")

    # Optional CSV export
    if args.save_csv:
        out_csv = out_dir / f"umap_{args.color_by}_{'z' if args.use_z else 'mu'}.csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "y", "label"])
            for (x, y), lab in zip(latents_2d, labels):
                w.writerow([float(x), float(y), lab])
        print(f"[ok] CSV saved to {out_csv}")


if __name__ == "__main__":
    main()
