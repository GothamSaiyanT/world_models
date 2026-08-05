"""Shared trainer for the adaptive and fixed-interval pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class EpochResult:
    """Summary of one completed training epoch."""

    average_loss: float
    correction_events: int
    corrected_samples: int
    mean_correction_error: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the result."""
        return asdict(self)


class SelfCorrectingTrainer:
    """
    Shared autoregressive trainer for both self-correcting pipelines.

    The model starts with the real first frame. At later steps, its previous
    prediction is fed back as the next input. A correction strategy may then
    replace the recurrent hidden state using the matching real observation.

    New in this version: ``warmup_epochs``. ``train_epoch()`` now accepts an
    optional ``epoch`` index; while ``epoch < warmup_epochs``, the corrector
    is disabled for that epoch (see CorrectionStrategy.enabled), so the model
    trains purely autoregressively before any hidden-state correction is
    introduced. Calling ``train_epoch()`` with no ``epoch`` argument keeps
    the old behaviour (corrector always enabled) for any code that hasn't
    been updated to pass the epoch index yet.
    """

    def __init__(
        self,
        model,
        dataset: Dataset,
        corrector,
        drift_detector,
        config=None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        optimizer: str | None = None,
        device: str | torch.device | None = None,
        warmup_epochs: int | None = None,
    ) -> None:
        """
        Create the trainer.

        ``config`` is the preferred OOP interface. The individual keyword
        arguments are retained so older entry points remain compatible.
        """
        self.model = model
        self.corrector = corrector
        self.drift_detector = drift_detector

        learning_rate = self._resolve_setting(
            explicit=learning_rate,
            config=config,
            name="learning_rate",
            default=0.001,
        )
        batch_size = self._resolve_setting(
            explicit=batch_size,
            config=config,
            name="batch_size",
            default=32,
        )
        num_workers = self._resolve_setting(
            explicit=num_workers,
            config=config,
            name="num_workers",
            default=2,
        )
        optimizer_name = self._resolve_setting(
            explicit=optimizer,
            config=config,
            name="optimizer",
            default="sgd",
        )
        configured_device = self._resolve_setting(
            explicit=device,
            config=config,
            name="device",
            default=None,
        )

        # warmup_epochs isn't a field on the current TrainingConfig --
        # falls back to the explicit kwarg, then a config attribute if
        # you add one later, then a default of 5.
        self.warmup_epochs = self._resolve_setting(
            explicit=warmup_epochs,
            config=config,
            name="warmup_epochs",
            default=5,
        )

        self.device = self._select_device(configured_device)
        self.model.to(self.device)

        self.dataloader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=int(num_workers),
            pin_memory=self.device.type == "cuda",
        )

        self.optimizer = self._build_optimizer(
            name=str(optimizer_name),
            learning_rate=float(learning_rate),
        )

    @staticmethod
    def _resolve_setting(*, explicit, config, name: str, default):
        """Prefer an explicit value, then a config attribute, then a default."""
        if explicit is not None:
            return explicit
        if config is not None and hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
        return default

    @staticmethod
    def _select_device(requested_device) -> torch.device:
        if requested_device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        selected = torch.device(requested_device)
        if selected.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested in TrainingConfig, but CUDA is not available."
            )
        return selected

    def _build_optimizer(self, name: str, learning_rate: float):
        normalised_name = name.strip().lower()

        if normalised_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
            )

        if normalised_name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
            )

        raise ValueError(
            f"Unsupported optimizer '{name}'. Choose either 'sgd' or 'adam'."
        )

    def train_epoch(self, epoch: int | None = None) -> EpochResult:
        """
        Train for one epoch and return loss and correction statistics.

        ``epoch`` should be the current epoch index (0-based) so the
        warm-up period can be applied. If omitted, correction stays
        enabled the whole time, matching the previous behaviour.
        """
        self.model.train()

        self.corrector.enabled = (
            epoch is None or epoch >= self.warmup_epochs
        )
        self.corrector.reset_log()

        total_loss = 0.0
        batch_count = 0

        for frames, actions in self.dataloader:
            frames = frames.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True).long()

            if frames.ndim != 5:
                raise ValueError(
                    "Expected frames with shape (batch, sequence, channels, "
                    f"height, width), but received {tuple(frames.shape)}."
                )

            if actions.ndim != 2:
                raise ValueError(
                    "Expected actions with shape (batch, sequence_length), "
                    f"but received {tuple(actions.shape)}."
                )

            batch_size, sequence_length = actions.shape

            if frames.shape[1] != sequence_length + 1:
                raise ValueError(
                    "Each action sequence needs one more frame than actions: "
                    f"received {frames.shape[1]} frames and "
                    f"{sequence_length} actions."
                )

            hidden = self.model.init_hidden(
                batch_size=batch_size,
                device=self.device,
            )

            self.optimizer.zero_grad(set_to_none=True)

            # The first model input is real. Subsequent inputs are predictions.
            current_frame = frames[:, 0]
            sequence_loss = torch.zeros((), device=self.device)

            for step in range(sequence_length):
                latent = self.model.encode(current_frame)

                prediction, hidden = self.model.step(
                    latent,
                    actions[:, step],
                    hidden,
                )

                real_next_frame = frames[:, step + 1]
                sequence_loss = sequence_loss + F.mse_loss(
                    prediction,
                    real_next_frame,
                )

                with torch.no_grad():
                    error = self.drift_detector.compute_error(
                        prediction,
                        real_next_frame,
                    )

                hidden = self.corrector.maybe_correct(
                    hidden=hidden,
                    real_frame=real_next_frame,
                    encode_fn=self.model.encode,
                    error=error,
                    step=step,
                )

                # Prevent gradients from passing through the predicted-image
                # feedback path. The recurrent hidden state still carries the
                # temporal computation graph until a hard correction occurs.
                current_frame = prediction.detach()

            sequence_loss = sequence_loss / sequence_length
            sequence_loss.backward()
            self.optimizer.step()

            total_loss += float(sequence_loss.detach().item())
            batch_count += 1

        if batch_count == 0:
            raise RuntimeError(
                "The training DataLoader produced no batches. Check that the "
                "dataset contains enough frames for the configured sequence length."
            )

        return self._build_epoch_result(
            average_loss=total_loss / batch_count,
        )

    def _build_epoch_result(self, average_loss: float) -> EpochResult:
        """Aggregate the corrector log into an epoch-level result."""
        correction_log = self.corrector.correction_log
        correction_events = len(correction_log)

        corrected_samples = sum(
            int(event[2])
            for event in correction_log
        )

        if corrected_samples == 0:
            mean_correction_error = None
        else:
            weighted_error_sum = sum(
                float(event[1]) * int(event[2])
                for event in correction_log
            )
            mean_correction_error = weighted_error_sum / corrected_samples

        return EpochResult(
            average_loss=float(average_loss),
            correction_events=correction_events,
            corrected_samples=corrected_samples,
            mean_correction_error=mean_correction_error,
        )