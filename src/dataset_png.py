import os
from typing import Optional, Callable
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

class OasisMRIDataset(Dataset):
    """
    Dataset for OASIS PNG slices (image-mask pairs).
    It expects two directories with matching file names:
      - images_dir: PNG slices (grayscale)
      - masks_dir:  PNG masks (grayscale, 0 background, >0 foreground)
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

        self.image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(".png")])
        self.mask_files  = sorted([f for f in os.listdir(masks_dir)  if f.lower().endswith(".png")])

        assert len(self.image_files) == len(self.mask_files), \
            f"Mismatch between images ({len(self.image_files)}) and masks ({len(self.mask_files)})"
        # Optional: ensure name alignment
        for a, b in zip(self.image_files, self.mask_files):
            assert a == b, f"File name mismatch: {a} vs {b}"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        msk_path = os.path.join(self.masks_dir,  self.mask_files[idx])

        # Load as grayscale
        image = Image.open(img_path).convert("L")
        mask  = Image.open(msk_path).convert("L")

        # To numpy
        image = np.asarray(image, dtype=np.float32) / 255.0  # [0,1]
        mask  = np.asarray(mask,  dtype=np.uint8)

        # Binarize mask: anything >0 becomes 1
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)

        # To tensor: image -> [1,H,W], mask -> [H,W]
        image_t = torch.from_numpy(image).unsqueeze(0)          # float32
        mask_t  = torch.from_numpy(mask.astype(np.int64))       # int64 class ids (0/1)

        if self.transform is not None:
            # If you use transforms (e.g., albumentations), apply here
            image_t, mask_t = self.transform(image_t, mask_t)

        return image_t, mask_t
