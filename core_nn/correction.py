from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Callable

import torch

EncodeFunction = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class CorrectionEvent:
    step: int
    mean_error: float
    samples_corrected: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class CorrectionStrategy(ABC):
    """Strategy interface used by all self-correcting pipelines."""

    def __init__(self) -> None:
        self._events: list[CorrectionEvent] = []

    @property
    def events(self) -> tuple[CorrectionEvent, ...]:
        return tuple(self._events)

    def reset(self) -> None:
        self._events.clear()

    def _record(self, step: int, error: torch.Tensor, sample_count: int) -> None:
        self._events.append(
            CorrectionEvent(
                step=step,
                mean_error=float(error.mean().item()),
                samples_corrected=sample_count,
            )
        )

    @staticmethod
    def _encode_anchor(
        real_frame: torch.Tensor,
        encode_fn: EncodeFunction,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        # The correction anchor is deliberately detached. Correction is an
        # intervention, not another backpropagation path through the encoder.
        with torch.no_grad():
            corrected_hidden = encode_fn(real_frame)

        if corrected_hidden.shape != hidden.shape:
            raise ValueError(
                "Encoded correction anchor must match hidden-state shape; "
                f"received {tuple(corrected_hidden.shape)} and {tuple(hidden.shape)}"
            )
        return corrected_hidden

    @abstractmethod
    def maybe_correct(
        self,
        hidden: torch.Tensor,
        real_frame: torch.Tensor,
        encode_fn: EncodeFunction,
        error: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Return either the original or corrected recurrent hidden state."""
        raise NotImplementedError


class FixedIntervalCorrector(CorrectionStrategy):
    """Pipeline 2: correct the whole batch after every N rollout steps."""

    def __init__(self, interval: int = 10) -> None:
        super().__init__()
        if interval < 1:
            raise ValueError("interval must be at least 1")
        self.interval = interval

    def maybe_correct(
        self,
        hidden: torch.Tensor,
        real_frame: torch.Tensor,
        encode_fn: EncodeFunction,
        error: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        if (step + 1) % self.interval != 0:
            return hidden

        corrected_hidden = self._encode_anchor(real_frame, encode_fn, hidden)
        self._record(step, error, hidden.shape[0])
        return corrected_hidden


class AdaptiveCorrector(CorrectionStrategy):
    """Pipeline 3: correct only samples whose measured drift exceeds a threshold."""

    def __init__(self, threshold: float = 0.01) -> None:
        super().__init__()
        if threshold < 0:
            raise ValueError("threshold cannot be negative")
        self.threshold = threshold

    def maybe_correct(
        self,
        hidden: torch.Tensor,
        real_frame: torch.Tensor,
        encode_fn: EncodeFunction,
        error: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        if error.ndim != 1 or error.shape[0] != hidden.shape[0]:
            raise ValueError("error must contain one value per batch sample")

        mask = error > self.threshold
        if not bool(mask.any().item()):
            return hidden

        corrected_hidden = self._encode_anchor(real_frame, encode_fn, hidden)
        selected_error = error[mask]
        self._record(step, selected_error, int(mask.sum().item()))

        return torch.where(mask.unsqueeze(1), corrected_hidden, hidden)
