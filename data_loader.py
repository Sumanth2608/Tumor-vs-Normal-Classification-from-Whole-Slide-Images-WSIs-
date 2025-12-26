import openslide
import numpy as np
from PIL import Image

class WSIDataLoader:
    """
    Data loader for Whole-Slide Images (WSIs).
    Handles loading and extracting patches from WSIs.
    """
    def __init__(self, wsi_path):
        """
        Initialize the data loader with a WSI file path.

        Args:
            wsi_path (str): Path to the WSI file (.svs, .tif, etc.)
        """
        self.slide = openslide.OpenSlide(wsi_path)
        self.level_count = self.slide.level_count
        self.dimensions = self.slide.dimensions

    def get_patch(self, x, y, level, size):
        """
        Extract a patch from the WSI at the specified level and position.

        Args:
            x (int): X coordinate
            y (int): Y coordinate
            level (int): Magnification level (0 is highest)
            size (int): Patch size (square)

        Returns:
            np.ndarray: RGB image patch
        """
        patch = self.slide.read_region((x, y), level, (size, size))
        return np.array(patch.convert('RGB'))

    def get_thumbnail(self, level=None):
        """
        Get a thumbnail of the entire slide at a specified level.

        Args:
            level (int): Level for thumbnail (default: lowest level)

        Returns:
            np.ndarray: Thumbnail image
        """
        if level is None:
            level = self.level_count - 1
        thumbnail = self.slide.get_thumbnail(self.slide.level_dimensions[level])
        return np.array(thumbnail.convert('RGB'))

    def close(self):
        """Close the slide file."""
        self.slide.close()