#!/usr/bin/env python3
"""
Data Preprocessing Script for CAMELYON16

This script extracts 250 tumor and 250 normal patches from CAMELYON16 WSIs
to create a balanced dataset for training.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import openslide
from PIL import Image
import random
from pathlib import Path
import tqdm

class CAMELYON16Preprocessor:
    """Preprocesses CAMELYON16 data to extract tumor and normal patches."""

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw" / "camelyon16"
        self.processed_dir = self.data_dir / "processed"
        self.train_dir = self.processed_dir / "train"
        self.tumor_dir = self.train_dir / "tumor"
        self.normal_dir = self.train_dir / "normal"

        # Create directories
        for dir_path in [self.tumor_dir, self.normal_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.patch_size = 224  # ResNet-18 input size
        self.target_patches = 250  # per class

    def parse_xml_annotations(self, xml_file):
        """Parse XML annotation file to get tumor coordinates."""
        tree = ET.parse(xml_file)
        root = tree.getroot()

        tumor_coords = []
        for annotation in root.findall('.//Annotation'):
            if annotation.get('Type') == '4':  # Tumor annotation type
                for coordinate in annotation.findall('.//Coordinate'):
                    x = float(coordinate.get('X'))
                    y = float(coordinate.get('Y'))
                    tumor_coords.append((x, y))

        return tumor_coords

    def is_point_in_tumor(self, x, y, tumor_coords, threshold=50):
        """Check if a point is within tumor region (simplified distance check)."""
        if not tumor_coords:
            return False

        # Check distance to nearest tumor coordinate
        min_distance = min(np.sqrt((x - tx)**2 + (y - ty)**2) for tx, ty in tumor_coords)
        return min_distance < threshold

    def extract_patches_from_wsi(self, slide_path, xml_path, level=0):
        """Extract tumor and normal patches from a single WSI."""
        print(f"Processing: {slide_path.name}")

        # Open slide
        slide = openslide.OpenSlide(str(slide_path))
        dimensions = slide.dimensions

        # Parse annotations
        tumor_coords = self.parse_xml_annotations(xml_path) if xml_path.exists() else []

        tumor_patches = []
        normal_patches = []

        # Sample patches across the slide
        num_samples = 1000  # Sample more than needed, then filter

        for _ in range(num_samples):
            # Random position (avoid edges)
            margin = self.patch_size * 4  # At level 0
            x = random.randint(margin, dimensions[0] - margin - self.patch_size)
            y = random.randint(margin, dimensions[1] - margin - self.patch_size)

            # Extract patch
            patch = slide.read_region((x, y), level, (self.patch_size, self.patch_size))
            patch_rgb = np.array(patch.convert('RGB'))

            # Skip if patch is mostly white (background)
            if self.is_background_patch(patch_rgb):
                continue

            # Classify patch
            if self.is_point_in_tumor(x, y, tumor_coords):
                if len(tumor_patches) < self.target_patches * 2:  # Collect extra
                    tumor_patches.append(patch_rgb)
            else:
                if len(normal_patches) < self.target_patches * 2:  # Collect extra
                    normal_patches.append(patch_rgb)

            # Stop if we have enough
            if len(tumor_patches) >= self.target_patches * 2 and len(normal_patches) >= self.target_patches * 2:
                break

        slide.close()
        return tumor_patches, normal_patches

    def is_background_patch(self, patch, threshold=0.8):
        """Check if patch is mostly background (white)."""
        # Convert to grayscale and check if mostly white
        gray = np.mean(patch, axis=2)
        white_pixels = np.sum(gray > 220)  # White threshold
        return white_pixels / (patch.shape[0] * patch.shape[1]) > threshold

    def save_patches(self, patches, class_name, start_idx=0):
        """Save patches to disk."""
        class_dir = self.tumor_dir if class_name == "tumor" else self.normal_dir

        for i, patch in enumerate(patches[:self.target_patches]):
            patch_img = Image.fromarray(patch)
            filename = f"{class_name}_patch_{start_idx + i:03d}.png"
            patch_img.save(class_dir / filename)

    def process_all_slides(self):
        """Process all available CAMELYON16 slides."""
        training_dir = self.raw_dir / "training"
        annotations_dir = self.raw_dir / "lesion_annotations"

        if not training_dir.exists():
            print(f"❌ Training directory not found: {training_dir}")
            print("Please download CAMELYON16 training data first.")
            return

        svs_files = list(training_dir.glob("*.svs"))
        print(f"Found {len(svs_files)} SVS files")

        if len(svs_files) == 0:
            print("❌ No SVS files found. Please download CAMELYON16 data.")
            return

        all_tumor_patches = []
        all_normal_patches = []

        # Process each slide
        for svs_file in tqdm.tqdm(svs_files[:5], desc="Processing slides"):  # Limit to first 5 slides
            # Find corresponding XML file
            slide_name = svs_file.stem
            xml_file = annotations_dir / f"{slide_name}.xml"

            try:
                tumor_patches, normal_patches = self.extract_patches_from_wsi(svs_file, xml_file)
                all_tumor_patches.extend(tumor_patches)
                all_normal_patches.extend(normal_patches)

                print(f"  {slide_name}: {len(tumor_patches)} tumor, {len(normal_patches)} normal patches")

            except Exception as e:
                print(f"  Error processing {slide_name}: {e}")
                continue

        # Shuffle and select target number of patches
        random.shuffle(all_tumor_patches)
        random.shuffle(all_normal_patches)

        tumor_patches = all_tumor_patches[:self.target_patches]
        normal_patches = all_normal_patches[:self.target_patches]

        print(f"\nExtracted {len(tumor_patches)} tumor and {len(normal_patches)} normal patches")

        # Save patches
        print("Saving patches...")
        self.save_patches(tumor_patches, "tumor")
        self.save_patches(normal_patches, "normal")

        print("✅ Preprocessing complete!")
        print(f"  Tumor patches saved to: {self.tumor_dir}")
        print(f"  Normal patches saved to: {self.normal_dir}")

    def show_statistics(self):
        """Show dataset statistics."""
        tumor_count = len(list(self.tumor_dir.glob("*.png")))
        normal_count = len(list(self.normal_dir.glob("*.png")))

        print("
📊 Dataset Statistics:"        print(f"  Tumor patches: {tumor_count}")
        print(f"  Normal patches: {normal_count}")
        print(f"  Total patches: {tumor_count + normal_count}")

        if tumor_count == self.target_patches and normal_count == self.target_patches:
            print("✅ Dataset is ready for training!")
        else:
            print("⚠ Dataset incomplete. Run preprocessing again.")

def main():
    """Main preprocessing workflow."""
    print("CAMELYON16 Data Preprocessor")
    print("=" * 30)
    print("This will extract 250 tumor and 250 normal patches from CAMELYON16 WSIs")
    print()

    preprocessor = CAMELYON16Preprocessor()

    # Check if data exists
    training_dir = preprocessor.raw_dir / "training"
    if not training_dir.exists() or not list(training_dir.glob("*.svs")):
        print("❌ CAMELYON16 training data not found!")
        print("Please run: python download_data.py")
        print("Then download CAMELYON16 training slides to data/raw/camelyon16/training/")
        return

    # Process the data
    preprocessor.process_all_slides()

    # Show final statistics
    preprocessor.show_statistics()

if __name__ == "__main__":
    main()