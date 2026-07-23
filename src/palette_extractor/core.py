import numpy as np
from skimage import io, transform
from sklearn.cluster import KMeans

from .utils import rgb_to_hex


class PaletteExtractor:
    def __init__(
        self, n_colors: int = 5, target_size: int = 500, random_state: int = 8
    ) -> None:
        self.n_colors = n_colors
        self.target_size = target_size
        self.random_state = random_state

    def _load_image(self, image_path: str) -> np.ndarray:
        img = io.imread(image_path)

        H, W = img.shape[:2]

        min_side = min(H, W)

        new_H = (H * self.target_size) // min_side
        new_W = (W * self.target_size) // min_side

        resized = (transform.resize(img, (new_H, new_W), order=0) / 255).astype(
            np.float32
        )

        return resized

    def extract(self, image_path: str):
        img = self._load_image(image_path)

        X = img.reshape((-1, 3))

        kmeans = KMeans(self.n_colors, random_state=self.random_state)

        kmeans.fit(X)

        centers = kmeans.cluster_centers_

        colors = (centers * 255).astype(np.uint8)

        labels = kmeans.labels_
        counts = np.bincount(labels)

        return {
            "colors": colors,
            "counts": counts,
            "hex": list(map(lambda color: rgb_to_hex(*color), colors)),
        }
