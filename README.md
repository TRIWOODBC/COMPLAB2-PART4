src/ # code scripts
data/ # dataset (ignored in git)
outputs/ # model checkpoints & visualizations
data/OASIS/
# COMP3710 Part 4 – OASIS MRI Project

This repository contains the implementation of Part 4 tasks:

- Variational Autoencoder (VAE)
- UNet for MRI segmentation
- Generative Adversarial Network (GAN) (optional)

## Project Structure

```
src/      # code scripts
data/     # dataset (ignored in git)
outputs/  # model checkpoints & visualizations
```

## Setup

### Option 1: Using environment.yml (recommended)

Create a new conda environment:

```bash
conda env create -f environment.yml
conda activate comp3710
```

### Option 2: Using requirements.txt

Install dependencies into an existing environment:

```bash
pip install -r requirements.txt
```

## Dataset

Place the dataset under:

```
data/OASIS/
    keras_png_slices_train/
    keras_png_slices_seg_train/
    keras_png_slices_validate/
    keras_png_slices_seg_validate/
    keras_png_slices_test/
    keras_png_slices_seg_test/
```