#!/usr/bin/env python3
"""
Create balanced dataset folders from existing camelyon_images.

- If filenames contain 'tumor' or 'normal', they will be used directly.
- Otherwise, the script will show a small sample (saved to sample_preview/) for quick manual labeling,
  then propagate labels by filename patterns if possible.

Usage:
    python scripts/create_dataset_from_camelyon_images.py --source camelyon_images --out data/processed/train --n 250

"""
import argparse
from pathlib import Path
import shutil
import random
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--source', default='camelyon_images')
parser.add_argument('--out', default='data/processed/train')
parser.add_argument('--n', type=int, default=250, help='number of samples per class')
parser.add_argument('--preview', type=int, default=20, help='how many random images to preview')
args = parser.parse_args()

src = Path(args.source)
out = Path(args.out)
preview_dir = out / 'sample_preview'
train_tumor = out / 'tumor'
train_normal = out / 'normal'

preview_dir.mkdir(parents=True, exist_ok=True)
train_tumor.mkdir(parents=True, exist_ok=True)
train_normal.mkdir(parents=True, exist_ok=True)

all_files = sorted([p for p in src.iterdir() if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg')])
print(f'Found {len(all_files)} images in {src}')

# Heuristic labeling by filename
labeled = {'tumor': [], 'normal': [], 'unknown': []}
for p in all_files:
    name = p.name.lower()
    if 'tumor' in name:
        labeled['tumor'].append(p)
    elif 'normal' in name:
        labeled['normal'].append(p)
    else:
        labeled['unknown'].append(p)

print(f"Heuristic counts -> tumor: {len(labeled['tumor'])}, normal: {len(labeled['normal'])}, unknown: {len(labeled['unknown'])}")

# Create previews for manual labeling if needed
if len(labeled['tumor']) < args.n or len(labeled['normal']) < args.n:
    sample = random.sample(all_files, min(args.preview, len(all_files)))
    print(f'Saving {len(sample)} sample preview images to {preview_dir} for quick inspection')
    for i,p in enumerate(sample):
        dst = preview_dir / f'sample_{i:03d}{p.suffix}'
        shutil.copy(p, dst)
    print('Open the images in the preview folder and check whether filenames need manual mapping.')
    print('If filenames are not labeled (no tumor/normal in name), you can manually move or rename files to include class, or rerun with --preview=0 to skip.')

# If both classes have sufficient counts, sample and copy
if len(labeled['tumor']) >= args.n and len(labeled['normal']) >= args.n:
    print('Sufficient labeled data found by heuristic — sampling and copying now...')
    sel_t = random.sample(labeled['tumor'], args.n)
    sel_n = random.sample(labeled['normal'], args.n)
    for i,p in enumerate(sel_t):
        shutil.copy(p, train_tumor / p.name)
    for i,p in enumerate(sel_n):
        shutil.copy(p, train_normal / p.name)
    print('Done. Dataset created at', out)
else:
    print('Not enough heuristic-labeled files to auto-create dataset. Please either:')
    print("  1) Rename/move files to include 'tumor' or 'normal' in filename, or")
    print("  2) Manually move desired images into the output folders, or")
    print("  3) Ask me to implement interactive labeling CLI to label unknown files")

print('Script finished.')
