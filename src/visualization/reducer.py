import numpy as np
import umap
from typing import Optional
from loguru import logger
import pickle
from pathlib import Path

class UMAPReducer:
    """
    Reduce high-dimensional embeddings to 2D/3D using UMAP.

    UMAP preserves both local and global structure, making it
    ideal for visualization and clustering.

    Args:
        n_components: Output dimensions (2 or 3)
        n_neighbors: Neighbors for local structure (5-50)
        min_dist: Minimum distance between points (0.0-1.0)
        metric: Distance metric ('cosine' or 'euclidean')
        random_state: Random seed for reproducibility

    Example:
        >>> reducer = UMAPReducer(n_components=2)
        >>> reduced = reducer.fit_transform(embeddings)
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'cosine',
        random_state: int = 42
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state

        # Initialize UMAP
        self.reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            verbose=False
        )

    def fit(self, embeddings: np.ndarray):
        """
        Fit UMAP on embeddings.

        Args:
            embeddings: High-dimensional embeddings (N, D)
        """
        logger.info(
            f"Fitting UMAP: {embeddings.shape} → {self.n_components}D"
        )

        self.reducer.fit(embeddings)

        logger.info("✓ UMAP fitted")
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings using fitted UMAP.

        Args:
            embeddings: High-dimensional embeddings

        Returns:
            Reduced embeddings (N, n_components)
        """
        reduced = self.reducer.transform(embeddings)
        return reduced

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit UMAP and transform embeddings in one step.

        Args:
            embeddings: High-dimensional embeddings

        Returns:
            Reduced embeddings
        """
        logger.info(
            f"Reducing dimensions: {embeddings.shape} → "
            f"({embeddings.shape[0]}, {self.n_components})"
        )

        reduced = self.reducer.fit_transform(embeddings)

        logger.info(f"✓ Reduction complete: {reduced.shape}")
        return reduced

    def save(self, path: str):
        """Save fitted reducer to disk."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(self, f)

        logger.info(f"✓ Saved reducer to {path}")

    @staticmethod
    def load(path: str) -> 'UMAPReducer':
        """Load fitted reducer from disk."""
        with open(path, 'rb') as f:
            reducer = pickle.load(f)

        logger.info(f"✓ Loaded reducer from {path}")
        return reducer
