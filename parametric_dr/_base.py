"""Base class for parametric dimensionality reduction methods.

Provides shared encoder management, transform, serialization, PCA
preprocessing, and an _extra_loss hook for composable extensions
(e.g., TemporalMixin).
"""

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError(
        "PyTorch is required but not installed. Install CPU-only (~200MB) with:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "Or with CUDA support:\n"
        "  pip install torch"
    )

from sklearn.base import BaseEstimator, TransformerMixin


def _build_default_encoder(num_inputs: int, num_outputs: int) -> nn.Sequential:
    """Build the default encoder (van der Maaten 2009 architecture with ReLU)."""
    return nn.Sequential(
        nn.Linear(num_inputs, 500),
        nn.ReLU(),
        nn.Linear(500, 500),
        nn.ReLU(),
        nn.Linear(500, 2000),
        nn.ReLU(),
        nn.Linear(2000, num_outputs),
    )


class ParametricDR(BaseEstimator, TransformerMixin):
    """Base class for parametric dimensionality reduction.

    Subclasses must implement fit(). The base class provides shared encoder
    management, transform, serialization, and an _extra_loss hook.

    Parameters
    ----------
    num_inputs : int
        Dimensionality of input data.
    num_outputs : int
        Dimensionality of embedding (typically 2).
    n_pca : int or None
        Number of PCA components to reduce input data to before feeding
        the encoder. Set to None to disable PCA. PCA is automatically
        skipped when n_pca >= num_inputs. When PCA is active, the default
        encoder (or any custom encoder) must accept n_pca-dimensional
        input.
    learning_rate : float
        Learning rate for the optimizer.
    encoder : nn.Module or None
        Custom encoder network. If None, uses the default architecture
        [500, 500, 2000, output] with ReLU. When n_pca is set, the
        default encoder is built with n_pca input dimensions.
    seed : int
        Random seed.
    batch_size : int
        Training batch size.
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        n_pca: Optional[int] = 50,
        learning_rate: float = 1e-3,
        encoder: Optional[nn.Module] = None,
        seed: int = 0,
        batch_size: int = 64,
    ):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.n_pca = n_pca
        self.learning_rate = learning_rate
        self.seed = seed
        self.batch_size = batch_size

        # PCA is used only when it actually reduces dimensionality
        self._use_pca = n_pca is not None and n_pca < num_inputs
        self._encoder_dim = n_pca if self._use_pca else num_inputs
        self._pca = None

        torch.manual_seed(seed)
        np.random.seed(seed)

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = _build_default_encoder(self._encoder_dim, num_outputs)

        self._is_fitted = False

    def _process_training_data(self, X: np.ndarray) -> np.ndarray:
        """Validate input and optionally fit/apply PCA for training.

        Called at the start of each subclass's fit(). On the first call
        with raw data, fits PCA and transforms. On subsequent calls
        (e.g., from a parent class after TemporalMixin), detects that
        the data is already transformed and passes through.

        Parameters
        ----------
        X : 2-d array

        Returns
        -------
        X : 2-d array (possibly PCA-reduced)
        """
        X = np.asarray(X, dtype=np.float32)
        if self._use_pca:
            if self._pca is None:
                # First call: fit PCA on raw data
                assert X.shape[1] == self.num_inputs, (
                    f"Expected {self.num_inputs} input features, got {X.shape[1]}"
                )
                from sklearn.decomposition import PCA
                self._pca = PCA(n_components=self.n_pca)
                X = self._pca.fit_transform(X).astype(np.float32)
            elif X.shape[1] == self.num_inputs:
                # PCA already fitted, raw data passed: just transform
                X = self._pca.transform(X).astype(np.float32)
            else:
                # Already PCA-transformed (from TemporalMixin or similar)
                assert X.shape[1] == self._encoder_dim, (
                    f"Expected {self.num_inputs} or {self._encoder_dim} features, "
                    f"got {X.shape[1]}"
                )
        else:
            assert X.shape[1] == self.num_inputs, (
                f"Expected {self.num_inputs} input features, got {X.shape[1]}"
            )
        return X

    def _setup_training(self):
        """Create optimizer and set encoder to training mode.

        Returns
        -------
        optimizer : torch.optim.Adam
        """
        opt = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        if hasattr(self.encoder, "train"):
            self.encoder.train()
        return opt

    def _extra_loss(self, output, batch_indices):
        """Hook for additional loss terms (e.g., temporal smoothness).

        Called by each subclass's training loop. Override in mixins
        to add composable loss terms.

        Parameters
        ----------
        output : torch.Tensor (batch_size, num_outputs)
            Encoder output for the current batch.
        batch_indices : np.ndarray (batch_size,)
            Indices of the current batch in the full dataset.

        Returns
        -------
        loss : torch.Tensor (scalar)
            Additional loss term. Default: 0.
        """
        return torch.tensor(0.0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data into the low-dimensional space.

        Parameters
        ----------
        X : 2-d array (M, num_inputs)

        Returns
        -------
        embedding : 2-d array (M, num_outputs)
        """
        assert self._is_fitted, "Must call fit() before transform()"
        X = np.asarray(X, dtype=np.float32)
        assert X.shape[1] == self.num_inputs, (
            f"Expected {self.num_inputs} input features, got {X.shape[1]}"
        )

        if self._use_pca and self._pca is not None:
            X = self._pca.transform(X).astype(np.float32)

        if hasattr(self.encoder, "eval"):
            self.encoder.eval()
        with torch.no_grad():
            output = self.encoder(torch.tensor(X, dtype=torch.float32))
        if hasattr(output, "numpy"):
            return output.numpy()
        return np.asarray(output)

    def save_model(self, model_path: str):
        """Save encoder weights (and PCA state if applicable) to disk.

        Parameters
        ----------
        model_path : str
            File path (conventionally ending in .pt).
        """
        if not hasattr(self.encoder, "state_dict"):
            raise TypeError("Encoder does not support state_dict(); cannot save")
        save_data = {"encoder_state_dict": self.encoder.state_dict()}
        if self._use_pca and self._pca is not None:
            save_data["pca_components_"] = torch.from_numpy(
                self._pca.components_.copy()
            )
            save_data["pca_mean_"] = torch.from_numpy(self._pca.mean_.copy())
            save_data["pca_explained_variance_"] = torch.from_numpy(
                self._pca.explained_variance_.copy()
            )
        torch.save(save_data, model_path)

    def restore_model(self, model_path: str):
        """Load encoder weights (and PCA state if applicable) from disk.

        Parameters
        ----------
        model_path : str
        """
        if not hasattr(self.encoder, "load_state_dict"):
            raise TypeError(
                "Encoder does not support load_state_dict(); cannot restore"
            )
        data = torch.load(model_path, weights_only=True)
        if isinstance(data, dict) and "encoder_state_dict" in data:
            # New format: dict with encoder + optional PCA
            self.encoder.load_state_dict(data["encoder_state_dict"])
            if "pca_components_" in data and self._use_pca:
                from sklearn.decomposition import PCA
                n_comp = data["pca_components_"].shape[0]
                self._pca = PCA(n_components=n_comp)
                self._pca.components_ = data["pca_components_"].numpy()
                self._pca.mean_ = data["pca_mean_"].numpy()
                self._pca.explained_variance_ = data[
                    "pca_explained_variance_"
                ].numpy()
                self._pca.n_features_in_ = self._pca.mean_.shape[0]
        else:
            # Legacy format: bare state_dict
            self.encoder.load_state_dict(data)
        self._is_fitted = True
