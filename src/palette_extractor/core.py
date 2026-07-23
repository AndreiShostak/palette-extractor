import numpy as np
from skimage import io, transform


class PaletteExtractor:
    def __init__(self, n_colors: int = 5, target_size: int = 300) -> None:
        self.n_colors = n_colors
        self.target_size = target_size

    def _load_image(self, image_path: str) -> np.ndarray:
        img = io.imread(image_path)

        H, W = img.shape[:2]

        min_side = min(H, W)

        new_H = (H * self.target_size) // min_side
        new_W = (W * self.target_size) // min_side

        resized = transform.resize(img, (new_H, new_W), order=0)

        return resized
