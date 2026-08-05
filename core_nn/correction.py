"""Correction strategies shared by the adaptive and fixed-interval pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor


# Each event stores:
# (rollout step, mean error at correction, number of corrected samples)
CorrectionEvent = tuple[int, float, int]


class CorrectionStrategy(ABC):
    """
    Base strategy for correcting the recurrent hidden state.

    Subclasses decide *when* correction happens. The trainer depends only on
    this common interface, so both strategies must provide ``maybe_correct()``,
    ``correction_log``, and ``reset_log()``.

    ``enabled`` (new): when False, ``maybe_correct()`` is a no-op regardless
    of what the subclass would otherwise decide. This is used for a training
    warm-up period -- a freshly-initialised model has high per-step error
    almost everywhere, so a threshold tuned for a *trained* model fires on
    nearly every sample from epoch 1, severing the GRU's gradient chain at
    every step and preventing it from ever learning multi-step dynamics.
    Disabling correction for the first few epochs lets the model first learn
    reasonable free-running predictions, so "drift" means something once
    correction is switched on.
    """

    def __init__(self) -> None:
        self.correction_log: list[CorrectionEvent] = []
        self.enabled: bool = True

    @abstractmethod
    def _maybe_correct(
        self,
        hidden: Tensor,
        real_frame: Tensor,
        encode_fn: Callable[[Tensor], Tensor],
        error: Tensor,
        step: int,
    ) -> Tensor:
        """Subclass-specific correction logic. Not called when disabled."""
        raise NotImplementedError

    def maybe_correct(
        self,
        hidden: Tensor,
        real_frame: Tensor,
        encode_fn: Callable[[Tensor], Tensor],
        error: Tensor,
        step: int,
    ) -> Tensor:
        """Return either the original hidden state or a corrected one."""
        if not self.enabled:
            return hidden

        return self._maybe_correct(hidden, real_frame, encode_fn, error, step)

    def reset_log(self) -> None:
        """Clear correction events before a new training epoch."""
        self.correction_log.clear()

    def _encode_real_frame(
        self,
        real_frame: Tensor,
        encode_fn: Callable[[Tensor], Tensor],
    ) -> Tensor:
        """Encode a real observation without building an autograd graph."""
        with torch.no_grad():
            return encode_fn(real_frame)

    def _record_event(
        self,
        step: int,
        mean_error: float,
        corrected_samples: int,
    ) -> None:
        """Record one correction event in the format expected by the trainer."""
        self.correction_log.append(
            (int(step), float(mean_error), int(corrected_samples))
        )


class AdaptiveCorrector(CorrectionStrategy):
    """Correct samples whose prediction error exceeds a threshold."""

    def __init__(self, threshold: float = 0.08) -> None:
        super().__init__()

        if threshold < 0:
            raise ValueError("Adaptive correction threshold cannot be negative.")

        self.threshold = float(threshold)

    def _maybe_correct(
        self,
        hidden: Tensor,
        real_frame: Tensor,
        encode_fn: Callable[[Tensor], Tensor],
        error: Tensor,
        step: int,
    ) -> Tensor:
        if error.ndim != 1:
            raise ValueError(
                "AdaptiveCorrector expects one error value per sample, "
                f"but received shape {tuple(error.shape)}."
            )

        mask = error > self.threshold

        if not bool(mask.any()):
            return hidden

        corrected_latent = self._encode_real_frame(real_frame, encode_fn)

        if corrected_latent.shape != hidden.shape:
            raise ValueError(
                "The encoded real frame and hidden state must have the same "
                "shape for hard correction: "
                f"encoded={tuple(corrected_latent.shape)}, "
                f"hidden={tuple(hidden.shape)}."
            )

        corrected_hidden = torch.where(
            mask.unsqueeze(1),
            corrected_latent,
            hidden,
        )

        corrected_samples = int(mask.sum().item())
        mean_error = float(error[mask].mean().item())
        self._record_event(step, mean_error, corrected_samples)

        return corrected_hidden


class FixedIntervalCorrector(CorrectionStrategy):
    """Correct the full batch after every configured number of rollout steps."""

    def __init__(self, interval: int = 10) -> None:
        super().__init__()

        if interval <= 0:
            raise ValueError("Fixed correction interval must be greater than zero.")

        self.interval = int(interval)

    def _maybe_correct(
        self,
        hidden: Tensor,
        real_frame: Tensor,
        encode_fn: Callable[[Tensor], Tensor],
        error: Tensor,
        step: int,
    ) -> Tensor:
        if (step + 1) % self.interval != 0:
            return hidden

        corrected_hidden = self._encode_real_frame(real_frame, encode_fn)

        if corrected_hidden.shape != hidden.shape:
            raise ValueError(
                "The encoded real frame and hidden state must have the same "
                "shape for hard correction: "
                f"encoded={tuple(corrected_hidden.shape)}, "
                f"hidden={tuple(hidden.shape)}."
            )

        batch_size = int(hidden.shape[0])
        mean_error = float(error.mean().item())
        self._record_event(step, mean_error, batch_size)

        return corrected_hidden


# Backward-compatible alias for older files that imported ``Corrector``.
Corrector = CorrectionStrategy