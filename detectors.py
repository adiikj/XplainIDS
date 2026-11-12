"""
Second stage: anomaly detectors with a common interface.

    det = make_detector(name, seed)
    det.fit(X_fit, X_es)      # X_es: held-out rows for early stopping (AEs only); never test data
    s = det.score(X)          # higher = more anomalous

Inputs are transformed by `InputTransform` before reaching the detector (fit on detector-training rows only).
"""

import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

_TORCH = None


def _torch():
    """
    Import PyTorch only when an autoencoder is used. On Windows every SHAP worker process re-imports the main script;
    a top-level `import torch` there loaded PyTorch once per worker and ran out of memory (UNSW-NB15 Exploits).
    A fixed thread count (XZ_TORCH_THREADS) makes AE training repeat exactly on one machine.
    """
    global _TORCH
    if _TORCH is None:
        import torch
        torch.set_num_threads(int(os.environ.get('XZ_TORCH_THREADS', 0)) or os.cpu_count())
        _TORCH = torch
    return _TORCH


class InputTransform:
    """
    Builds the detector input from raw features and/or SHAP vectors.

    raw  -> signed log1p (network features are heavy-tailed) then MinMax to (-1, 1)
    shap -> MinMax to (-1, 1), as in Barnard et al.
    concat -> both, side by side
    Scalers are fit on the detector-training rows only; out-of-range values are clipped.
    """

    def __init__(self, kind):
        assert kind in ('raw', 'shap', 'concat')
        self.kind = kind

    @staticmethod
    def _slog(X):
        return np.sign(X) * np.log1p(np.abs(X))

    def _stack(self, X_raw, X_shap):
        parts = []
        if self.kind in ('raw', 'concat'):
            parts.append(self._slog(X_raw))
        if self.kind in ('shap', 'concat'):
            parts.append(X_shap)
        return np.hstack(parts).astype(np.float32)

    def fit(self, X_raw, X_shap):
        self.n_shap_ = X_shap.shape[1] if X_shap is not None and self.kind != 'raw' else 0
        self.scaler = MinMaxScaler(feature_range=(-1, 1), clip=True).fit(self._stack(X_raw, X_shap))
        return self

    def transform(self, X_raw, X_shap):
        return self.scaler.transform(self._stack(X_raw, X_shap)).astype(np.float32)

    def inverse_shap(self, Z):
        """
        Map detector-space rows (e.g. an AE reconstruction) back to SHAP units: returns the SHAP block only.
        Exact for values inside (-1, 1); transform() clips, so for evidence use the original (unclipped) SHAP
        vector as phi and only invert the reconstruction phi_hat.
        """
        assert self.kind in ('shap', 'concat'), "no SHAP block in a raw-only transform"
        X = self.scaler.inverse_transform(np.asarray(Z, dtype=np.float64))
        n_shap = self.n_shap_
        return X[:, -n_shap:]


# ------------------------------------------------------------------ autoencoders

_AE_CLASS = None


def _ae_class():
    """The AE module class, built on first use (same layers and init order as before, so results are unchanged)."""
    global _AE_CLASS
    if _AE_CLASS is None:
        torch = _torch()

        class _AE(torch.nn.Module):
            def __init__(self, d, hidden, dropout):
                super().__init__()
                layers, sizes = [], [d] + hidden
                for i in range(len(hidden)):
                    layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
                    layers.append(torch.nn.ReLU())
                    if dropout and i in dropout:
                        layers.append(torch.nn.Dropout(0.2))
                layers += [torch.nn.Linear(sizes[-1], d), torch.nn.Tanh()]   # inputs live in (-1, 1)
                self.net = torch.nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

        _AE.__module__, _AE.__qualname__ = __name__, '_AE'   # so pickle finds it as detectors._AE
        _AE_CLASS = _AE
    return _AE_CLASS


def __getattr__(name):
    # pickle looks up detectors._AE when loading a saved pipeline
    if name == '_AE':
        return _ae_class()
    raise AttributeError(name)


class AutoencoderDetector:
    """
    Dense AE trained with mean-absolute-error loss; anomaly score = mean absolute reconstruction error (ARE).
    small: d -> 64 -> 16 -> 64 -> d
    p1:    d -> 1456 -> 724 -> 14 -> 632 -> 1644 -> d with dropout after the 1st and 4th layer (Barnard et al. 2022)
    """
    ARCH = {'small': ([64, 16, 64], None), 'p1': ([1456, 724, 14, 632, 1644], {0, 3})}

    def __init__(self, arch='small', seed=0, max_epochs=200, patience=10, batch_size=512, lr=1e-3):
        self.arch, self.seed = arch, seed
        self.max_epochs, self.patience, self.batch_size, self.lr = max_epochs, patience, batch_size, lr

    def fit(self, X, X_es):
        torch = _torch()
        torch.manual_seed(self.seed)
        hidden, dropout = self.ARCH[self.arch]
        self.model = _ae_class()(X.shape[1], hidden, dropout)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        Xt, Xe = torch.from_numpy(X), torch.from_numpy(X_es)
        g = torch.Generator().manual_seed(self.seed)
        best, best_state, wait = np.inf, None, 0
        for epoch in range(self.max_epochs):
            self.model.train()
            for idx in torch.randperm(len(Xt), generator=g).split(self.batch_size):
                xb = Xt[idx]
                loss = (self.model(xb) - xb).abs().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            val = self.score(X_es).mean()
            if val < best - 1e-5:
                best, wait = val, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        self.epochs_ = epoch + 1
        return self

    def score(self, X):
        torch = _torch()
        self.model.eval()
        out = []
        with torch.no_grad():
            for xb in torch.from_numpy(X).split(8192):
                out.append((self.model(xb) - xb).abs().mean(1))
        return torch.cat(out).numpy()

    def reconstruct(self, X):
        """The AE's reconstruction of each row, in the same (detector) space as X."""
        torch = _torch()
        self.model.eval()
        with torch.no_grad():
            return torch.cat([self.model(xb) for xb in torch.from_numpy(np.asarray(X, np.float32)).split(8192)]).numpy()


# ------------------------------------------------------------------ classical detectors

class IForestDetector:
    def __init__(self, seed=0):
        self.m = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)

    def fit(self, X, X_es=None):
        self.m.fit(X)
        return self

    def score(self, X):
        return -self.m.score_samples(X)


class KNNDetector:
    """Mean distance to the k nearest detector-training rows (training set subsampled for speed)."""

    def __init__(self, seed=0, k=5, max_fit=20000):
        self.k, self.seed, self.max_fit = k, seed, max_fit

    def fit(self, X, X_es=None):
        if len(X) > self.max_fit:
            X = X[np.random.default_rng(self.seed).choice(len(X), self.max_fit, replace=False)]
        self.nn = NearestNeighbors(n_neighbors=self.k, algorithm='brute', n_jobs=-1).fit(X)
        return self

    def score(self, X):
        out = [self.nn.kneighbors(X[i:i + 20000])[0].mean(1) for i in range(0, len(X), 20000)]
        return np.concatenate(out)


class PCADetector:
    """Reconstruction error after projecting onto the components that keep 95% of the variance."""

    def __init__(self, seed=0, var=0.95):
        self.m = PCA(n_components=var, random_state=seed)

    def fit(self, X, X_es=None):
        self.m.fit(X)
        return self

    def score(self, X):
        return np.abs(X - self.reconstruct(X)).mean(1)

    def reconstruct(self, X):
        return self.m.inverse_transform(self.m.transform(X))


def make_detector(name, seed):
    if name == 'ae_small':
        return AutoencoderDetector('small', seed)
    if name == 'ae_p1':
        return AutoencoderDetector('p1', seed)
    if name == 'iforest':
        return IForestDetector(seed)
    if name == 'knn':
        return KNNDetector(seed)
    if name == 'pca':
        return PCADetector(seed)
    raise ValueError(name)
