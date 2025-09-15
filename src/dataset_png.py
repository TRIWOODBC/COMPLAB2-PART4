import os
from typing import Optional, Callable, List, Tuple
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np
import torch

# Allow truncated PNGs to load instead of raising an error
ImageFile.LOAD_TRUNCATED_IMAGES = True


class OasisMRIDataset(Dataset):
    """
    Dataset class for OASIS PNG slices (image-mask pairs).

    Supported file naming conventions:
      - Images: case_001_slice_0.nii.png / img_001_slice_0.nii.png / 001_slice_0.nii.png
      - Masks : seg_001_slice_0.nii.png  / 001_slice_0.nii.png

    Matching is based on a normalized "core name" (removes prefixes and extensions).
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

        # Collect all .png files
        img_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(".png")])
        msk_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith(".png")])

        def core_name(fname: str) -> str:
            """
            Normalize filenames for matching:
              - Strip .nii.png or .png
              - Remove leading prefixes such as case_/img_/seg_
            """
            name = fname
            low = name.lower()
            if low.endswith(".nii.png"):
                name = name[:-8]
            elif low.endswith(".png"):
                name = name[:-4]

            for pref in ("case_", "img_", "seg_"):
                if name.startswith(pref):
                    name = name[len(pref):]
                    break
            return name

        # Map normalized names to original filenames
        img_map = {core_name(f): f for f in img_files}
        msk_map = {core_name(f): f for f in msk_files}

        # Find common keys between images and masks
        common_keys = sorted(set(img_map.keys()) & set(msk_map.keys()))
        if not common_keys:
            raise RuntimeError(
                "No matching image/mask pairs found.\n"
                f"images_dir={images_dir}\n"
                f"masks_dir={masks_dir}\n"
                "Check filenames and folder structure."
            )

        # Build aligned image-mask pairs (fast, no validation)
        self.pairs: List[Tuple[str, str]] = [
            (os.path.join(images_dir, img_map[k]), os.path.join(masks_dir, msk_map[k]))
            for k in common_keys
        ]

        print(f"[OasisMRIDataset] Found {len(self.pairs)} pairs. Example:\n"
              f"  image: {self.pairs[0][0]}\n  mask : {self.pairs[0][1]}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]
        try:
            image = Image.open(img_path).convert("L")
            mask = Image.open(msk_path).convert("L")
        except Exception:
            # If reading fails, fallback to the next pair
            alt = (idx + 1) % len(self.pairs)
            img_path, msk_path = self.pairs[alt]
            image = Image.open(img_path).convert("L")
            mask = Image.open(msk_path).convert("L")

        # Convert to numpy arrays
        image = np.asarray(image, dtype=np.float32) / 255.0
        mask = np.asarray(mask, dtype=np.uint8)

        # Ensure mask is binary {0,1}
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)

        # Convert to tensors
        image_t = torch.from_numpy(image).unsqueeze(0)  # [1,H,W]
        mask_t = torch.from_numpy(mask.astype(np.int64))  # [H,W]

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

    if len(ds) > 0:
        img, msk = ds[0]
        print("Image shape:", img.shape, "dtype:", img.dtype)
        print("Mask shape:", msk.shape, "dtype:", msk.dtype, "unique:", torch.unique(msk))
