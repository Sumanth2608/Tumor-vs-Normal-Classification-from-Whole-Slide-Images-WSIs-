#!/usr/bin/env python3
"""
Cluster images into two groups (tumor/normal) using feature embeddings and create a balanced dataset.

Usage:
  # Step 1: generate cluster previews (will save previews to out/cluster_preview)
  python scripts/cluster_and_create_dataset.py --source camelyon_images --out data/processed/train --sample 2000 --preview 20

  # Inspect previews in data/processed/train/cluster_preview/cluster_0 and cluster_1,
  # decide which cluster is 'tumor' (0 or 1), then assign and build final dataset:
  python scripts/cluster_and_create_dataset.py --source camelyon_images --out data/processed/train --assign 1 --n 250

Notes:
- By default the script will try to use a pretrained ResNet-18 to extract features (better),
  but falls back to color-histogram features if PyTorch isn't available or you pass --no-torch.
- Requires: numpy, pillow, scikit-learn; optional: torch, torchvision
"""
import argparse
from pathlib import Path
import random
import shutil
import math
import sys

import numpy as np
from PIL import Image

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
except Exception as e:
    print('scikit-learn is required. Install with: pip install scikit-learn')
    raise

parser = argparse.ArgumentParser()
parser.add_argument('--source', default='camelyon_images')
parser.add_argument('--out', default='data/processed/train')
parser.add_argument('--sample', type=int, default=2000, help='max images to use for clustering')
parser.add_argument('--preview', type=int, default=20, help='preview images per cluster')
parser.add_argument('--n', type=int, default=250, help='number of samples per class when assigning')
parser.add_argument('--assign', type=int, choices=[0,1], help='After inspecting previews, set which cluster index is tumor')
parser.add_argument('--no-torch', action='store_true', help='Force using color histogram features instead of ResNet')
args = parser.parse_args()

src = Path(args.source)
out = Path(args.out)
preview_dir = out / 'cluster_preview'
train_tumor = out / 'tumor'
train_normal = out / 'normal'

preview_dir.mkdir(parents=True, exist_ok=True)
train_tumor.mkdir(parents=True, exist_ok=True)
train_normal.mkdir(parents=True, exist_ok=True)

all_files = sorted([p for p in src.iterdir() if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg')])
if len(all_files) == 0:
    print('No images found in', src)
    sys.exit(1)

print(f'Found {len(all_files)} images. Sampling up to {args.sample} for clustering...')

if len(all_files) > args.sample:
    sampled = random.sample(all_files, args.sample)
else:
    sampled = all_files

# Feature extraction
use_torch = False
if not args.no_torch:
    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        use_torch = True
    except Exception:
        print('PyTorch not available or import failed. Falling back to color-histogram features.')
        use_torch = False

features = []
paths = []

if use_torch:
    print('Using pretrained ResNet-18 to extract features (this may be slow).')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.resnet18(pretrained=True)
    # remove final fc
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    batch = []
    batch_paths = []
    BATCH_SIZE = 64
    with torch.no_grad():
        for p in sampled:
            try:
                img = Image.open(p).convert('RGB')
                t = preprocess(img)
                batch.append(t)
                batch_paths.append(p)
                if len(batch) == BATCH_SIZE:
                    x = torch.stack(batch).to(device)
                    feat = model(x).cpu().numpy().reshape(len(batch), -1)
                    features.append(feat)
                    paths.extend(batch_paths)
                    batch = []
                    batch_paths = []
            except Exception as e:
                print('Error reading', p, e)
        # final batch
        if batch:
            x = torch.stack(batch).to(device)
            feat = model(x).cpu().numpy().reshape(len(batch), -1)
            features.append(feat)
            paths.extend(batch_paths)

    if features:
        features = np.vstack(features)
    else:
        features = np.zeros((0,512))

else:
    print('Using color-histogram features (fast fallback).')
    def color_hist_feature(img, bins=32):
        # img: PIL Image RGB
        arr = np.array(img.resize((128,128))).astype(np.uint8)
        feats = []
        for c in range(3):
            h,_ = np.histogram(arr[:,:,c], bins=bins, range=(0,255), density=True)
            feats.append(h)
        return np.concatenate(feats)

    feats = []
    pths = []
    for p in sampled:
        try:
            img = Image.open(p).convert('RGB')
            f = color_hist_feature(img, bins=32)
            feats.append(f)
            pths.append(p)
        except Exception as e:
            print('Error reading', p, e)
    features = np.vstack(feats)
    paths = pths

print('Extracted features for', features.shape[0], 'images.')

# Dimensionality reduction
n_components = min(50, features.shape[1])
print('Running PCA ->', n_components, 'components')
try:
    pca = PCA(n_components=n_components)
    feats_p = pca.fit_transform(features)
except Exception as e:
    print('PCA failed:', e)
    feats_p = features

# KMeans clustering
print('Clustering into 2 clusters (KMeans)')
km = KMeans(n_clusters=2, random_state=42)
labels = km.fit_predict(feats_p)

# Save preview images for each cluster
cluster_dir = preview_dir
for i in [0,1]:
    (cluster_dir / f'cluster_{i}').mkdir(parents=True, exist_ok=True)

indices = {0: [], 1: []}
for idx, lab in enumerate(labels):
    indices[lab].append(paths[idx])

for i in [0,1]:
    n_preview = min(args.preview, len(indices[i]))
    sel = random.sample(indices[i], n_preview) if n_preview>0 else []
    for j,p in enumerate(sel):
        dest = cluster_dir / f'cluster_{i}' / p.name
        shutil.copy(p, dest)

print('Saved previews to', cluster_dir)
print('Cluster sizes:', {i: len(indices[i]) for i in [0,1]})

# If assign flag provided, copy samples into tumor/normal
if args.assign is not None:
    tumor_cluster = args.assign
    normal_cluster = 1 - tumor_cluster
    tumor_list = indices[tumor_cluster]
    normal_list = indices[normal_cluster]

    if len(tumor_list) < args.n:
        print(f'WARNING: Cluster {tumor_cluster} has only {len(tumor_list)} images, fewer than requested {args.n}.')
    if len(normal_list) < args.n:
        print(f'WARNING: Cluster {normal_cluster} has only {len(normal_list)} images, fewer than requested {args.n}.')

    sel_t = random.sample(tumor_list, min(len(tumor_list), args.n))
    sel_n = random.sample(normal_list, min(len(normal_list), args.n))

    for p in sel_t:
        shutil.copy(p, train_tumor / p.name)
    for p in sel_n:
        shutil.copy(p, train_normal / p.name)

    print('Copied', len(sel_t), 'tumor and', len(sel_n), 'normal images to', out)
    print('Done.')

print('\nNext steps:')
print(' - Inspect the preview images in', preview_dir)
print(' - Decide which cluster index (0 or 1) corresponds to tumor, then run script again with --assign <index>')
print('Example: python scripts/cluster_and_create_dataset.py --assign 1 --n 250')
