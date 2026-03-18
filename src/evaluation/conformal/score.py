"""Nonconformity score functions for conformal prediction (Strategy Pattern).

To add a new score function, subclass ``NonconformityScore`` and implement
``compute`` and ``compute_matrix``.

Available scores
----------------
* ``logit_margin``      – :math:`s = \\max(z) - z_y`
* ``softmax_response``  – :math:`s = 1 - \\hat\\pi_y`
* ``softmax_margin``    – :math:`s = \\hat\\pi_{\\max} - \\hat\\pi_y`
* ``aps``               – Adaptive Prediction Sets (Romano et al. 2020)
* ``raps``              – Regularised APS (Angelopoulos et al. 2021)
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ======================================================================
# Abstract base
# ======================================================================

class NonconformityScore(ABC):
    """Abstract base for nonconformity score computation.

    Every subclass operates on **logit** arrays ``(N, C)``.  If the
    concrete score needs softmax probabilities it should call
    :func:`_softmax` internally.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the score function."""
        ...

    @abstractmethod
    def compute(self, logits: np.ndarray, true_indices: np.ndarray) -> np.ndarray:
        """Compute nonconformity scores for samples with known true labels.

        Args:
            logits: ``(N, C)`` logit array.
            true_indices: ``(N,)`` integer array of true class indices.

        Returns:
            ``(N,)`` array of nonconformity scores.
        """
        ...

    @abstractmethod
    def compute_matrix(self, logits: np.ndarray) -> np.ndarray:
        """Compute nonconformity scores for **all** candidate labels.

        Args:
            logits: ``(N, C)`` logit array.

        Returns:
            ``(N, C)`` matrix where entry ``[i, c]`` is the score when
            assuming class *c* is the true label.
        """
        ...


# ======================================================================
# Concrete implementations
# ======================================================================

class LogitMarginScore(NonconformityScore):
    r"""Logit-margin nonconformity score: :math:`s = \max(z) - z_y`."""

    @property
    def name(self) -> str:
        return "logit_margin"

    def compute(self, logits: np.ndarray, true_indices: np.ndarray) -> np.ndarray:
        max_logits = np.max(logits, axis=1)
        true_logits = logits[np.arange(len(true_indices)), true_indices]
        return max_logits - true_logits

    def compute_matrix(self, logits: np.ndarray) -> np.ndarray:
        max_logits = np.max(logits, axis=1)
        return max_logits[:, None] - logits


class SoftmaxResponseScore(NonconformityScore):
    r"""Softmax-response score: :math:`s = 1 - \hat\pi_y`."""

    @property
    def name(self) -> str:
        return "softmax_response"

    def compute(self, logits: np.ndarray, true_indices: np.ndarray) -> np.ndarray:
        probs = _softmax(logits)
        # return 1.0 - probs[np.arange(len(true_indices)), true_indices]
        return -np.log(probs[np.arange(len(true_indices)), true_indices])

    def compute_matrix(self, logits: np.ndarray) -> np.ndarray:
        probs = _softmax(logits)
        # return 1.0 - probs
        return -np.log(probs)

class SoftmaxMarginScore(NonconformityScore):
    r"""Softmax-margin score: :math:`s = \hat\pi_{\max} - \hat\pi_y`."""

    @property
    def name(self) -> str:
        return "softmax_margin"

    def compute(self, logits: np.ndarray, true_indices: np.ndarray) -> np.ndarray:
        probs = _softmax(logits)
        max_probs = np.max(probs, axis=1)
        true_probs = probs[np.arange(len(true_indices)), true_indices]
        return max_probs - true_probs

    def compute_matrix(self, logits: np.ndarray) -> np.ndarray:
        probs = _softmax(logits)
        max_probs = np.max(probs, axis=1)
        return max_probs[:, None] - probs


class APSScore(NonconformityScore):
    r"""Adaptive Prediction Sets (Romano et al. 2020).

    :math:`s_i = \sum_{j:\,\hat\pi_j \ge \hat\pi_{y_i}} \hat\pi_j`

    For the matrix variant (all candidate labels), entry ``[i, c]`` is the
    cumulative probability mass up to and including class *c* in the
    descending-probability order.
    """

    @property
    def name(self) -> str:
        return "aps"

    def compute(self, logits: np.ndarray, true_indices: np.ndarray) -> np.ndarray:
        probs = _softmax(logits)
        n = probs.shape[0]
        # Descending sort
        sorted_idx = np.argsort(-probs, axis=1)
        sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
        cumsum = np.cumsum(sorted_probs, axis=1)
        # Rank of the true class in the descending order
        ranks = np.zeros(n, dtype=np.intp)
        for i in range(n):
            ranks[i] = int(np.where(sorted_idx[i] == true_indices[i])[0][0])
        return cumsum[np.arange(n), ranks]

    def compute_matrix(self, logits: np.ndarray) -> np.ndarray:
        probs = _softmax(logits)
        n, c = probs.shape
        sorted_idx = np.argsort(-probs, axis=1)
        sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
        cumsum = np.cumsum(sorted_probs, axis=1)
        # Map cumulative sums back to original class order
        result = np.empty_like(probs)
        np.put_along_axis(result, sorted_idx, cumsum, axis=1)
        return result


class RAPSScore(NonconformityScore):
    r"""Regularised Adaptive Prediction Sets (Angelopoulos et al. 2021).

    :math:`s_i = \text{APS}_i + \lambda \cdot \max(0,\, \text{rank}(y_i) - k_{\text{reg}})`

    Parameters *lambda_reg* and *k_reg* are passed at construction time
    (read from config).
    """

    def __init__(self, lambda_reg: float = 0.01, k_reg: int = 1):
        self._lambda = lambda_reg
        self._k_reg = k_reg

    @property
    def name(self) -> str:
        return "raps"

    def compute(self, logits: np.ndarray, true_indices: np.ndarray) -> np.ndarray:
        probs = _softmax(logits)
        n = probs.shape[0]
        sorted_idx = np.argsort(-probs, axis=1)
        sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
        cumsum = np.cumsum(sorted_probs, axis=1)
        ranks = np.zeros(n, dtype=np.intp)
        for i in range(n):
            ranks[i] = int(np.where(sorted_idx[i] == true_indices[i])[0][0])
        aps_scores = cumsum[np.arange(n), ranks]
        penalty = self._lambda * np.maximum(0, ranks - self._k_reg)
        return aps_scores + penalty

    def compute_matrix(self, logits: np.ndarray) -> np.ndarray:
        probs = _softmax(logits)
        n, c = probs.shape
        sorted_idx = np.argsort(-probs, axis=1)
        sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
        cumsum = np.cumsum(sorted_probs, axis=1)
        # Rank penalties in sorted order: rank 0,1,...,C-1
        rank_arr = np.arange(c)[None, :]  # (1, C)
        penalty = self._lambda * np.maximum(0, rank_arr - self._k_reg)
        cumsum_reg = cumsum + penalty
        # Map back to original class order
        result = np.empty_like(probs)
        np.put_along_axis(result, sorted_idx, cumsum_reg, axis=1)
        return result


# ======================================================================
# Factory
# ======================================================================

_SCORE_REGISTRY: Dict[str, type] = {
    "logit_margin": LogitMarginScore,
    "softmax_response": SoftmaxResponseScore,
    "softmax_margin": SoftmaxMarginScore,
    "aps": APSScore,
    "raps": RAPSScore,
}


def create_score_fn(config: dict) -> NonconformityScore:
    """Instantiate a :class:`NonconformityScore` from the config dict.

    Reads ``calibration.conformal.score`` (default ``"logit_margin"``).
    For RAPS, also reads ``calibration.conformal.raps_lambda`` and
    ``calibration.conformal.raps_k_reg``.
    """
    conformal_cfg = config.get("calibration", {}).get("conformal", {})
    name = conformal_cfg.get("score", "logit_margin")

    cls = _SCORE_REGISTRY.get(name)
    if cls is None:
        supported = ", ".join(sorted(_SCORE_REGISTRY))
        raise ValueError(
            f"Unknown nonconformity score '{name}'. "
            f"Supported: {supported}"
        )

    if name == "raps":
        return cls(
            lambda_reg=conformal_cfg.get("raps_lambda", 0.01),
            k_reg=conformal_cfg.get("raps_k_reg", 1),
        )

    return cls()


# ======================================================================
# Shared utilities
# ======================================================================

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over the last axis."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def to_logit_space(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert a probability array to (log-)logit space."""
    probs = np.clip(probs, eps, 1 - eps)
    return np.log(probs)
