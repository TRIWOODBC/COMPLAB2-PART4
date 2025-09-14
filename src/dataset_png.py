import os
from typing import Optional, Callable, List, Tuple
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

class OasisMRIDataset(Dataset):
    """
    Dataset for OASIS PNG slices (image-mask pairs).
    Supports file names like:
      images: case_001_slice_0.nii.png / img_001_slice_0.nii.png / 001_slice_0.nii.png
      masks : seg_001_slice_0.nii.png  / 001_slice_0.nii.png
    We align by a normalized "core name" (prefix removed, extension normalized).
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Optional[Callable] = None,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        img_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(".png")])
        msk_files = sorted([f for f in os.listdir(masks_dir)  if f.lower().endswith(".png")])

        def core_name(fname: str) -> str:
            """Normalize name for matching:
            - strip .nii.png or .png
            - drop leading prefixes like case_/img_/seg_
            """
            name = fname
            low = name.lower()
            if low.endswith(".nii.png"):
                name = name[:-8]  # remove ".nii.png"
            elif low.endswith(".png"):
                name = name[:-4]  # remove ".png"

            for pref in ("case_", "img_", "seg_"):
                if name.startswith(pref):
                    name = name[len(pref):]
                    break
            return name

        # Map core_name -> original filename
        img_map = {core_name(f): f for f in img_files}
        msk_map = {core_name(f): f for f in msk_files}

        common_keys = sorted(set(img_map.keys()) & set(msk_map.keys()))
        if not common_keys:
            raise RuntimeError(
                "No matching image/mask pairs found.\n"
                f"images_dir={images_dir}\n"
                f"masks_dir={masks_dir}\n"
                "Check filenames and folder structure."
            )

        # Build aligned pairs
        self.pairs: List[Tuple[str, str]] = [
            (os.path.join(images_dir, img_map[k]), os.path.join(masks_dir, msk_map[k]))
            for k in common_keys
        ]

        # Optional: sanity print first pair
        print(f"[OasisMRIDataset] Found {len(self.pairs)} pairs. Example:\n"
              f"  image: {self.pairs[0][0]}\n  mask : {self.pairs[0][1]}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]

        image = Image.open(img_path).convert("L")
        mask  = Image.open(msk_path).convert("L")

        image = np.asarray(image, dtype=np.float32) / 255.0  # [0,1]
        mask  = np.asarray(mask,  dtype=np.uint8)
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)

        image_t = torch.from_numpy(image).unsqueeze(0)        # [1,H,W] float32
        mask_t  = torch.from_numpy(mask.astype(np.int64))     # [H,W]   int64

        if self.transform is not None:
            image_t, mask_t = self.transform(image_t, mask_t)

        return image_t, mask_t

if __name__ == "__main__":
    from pathlib import Path
    import torch

    ROOT = Path(__file__).resolve().parents[1]
    DATA_ROOT = ROOT / "data" / "OASIS"

    img_dir = DATA_ROOT / "keras_png_slices_train"
    msk_dir = DATA_ROOT / "keras_png_slices_seg_train"

    ds = OasisMRIDataset(str(img_dir), str(msk_dir))
    print("Dataset length:", len(ds))

    img, msk = ds[0]
    print("Image shape:", img.shape, "dtype:", img.dtype)
    print("Mask shape:", msk.shape, "dtype:", msk.dtype, "unique:", torch.unique(msk))
