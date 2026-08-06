"""Shared trainer for adaptive and fixed-interval correction pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class EpochResult:
    """Summary of one completed training epoch."""

    average_loss: float
    correction_events: int
    corrected_samples: int
    mean_correction_error: float | None
    average_base_loss: float
    average_motion_loss: float
    average_foreground_loss: float
    average_motion_fraction: float
    mean_drift_error: float
    p90_drift_error: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SelfCorrectingTrainer:

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
        motion_weight: float | None = None,
        foreground_weight: float | None = None,
        motion_threshold: float | None = None,
        foreground_threshold: float | None = None,
        motion_dilation: int | None = None,
        gradient_clip: float | None = None,
    ) -> None:
        self.model = model
        self.corrector = corrector
        self.drift_detector = drift_detector

        learning_rate = self._resolve(
            learning_rate, config, "learning_rate", 0.001
        )
        batch_size = self._resolve(
            batch_size, config, "batch_size", 32
        )
        num_workers = self._resolve(
            num_workers, config, "num_workers", 2
        )
        optimizer_name = self._resolve(
            optimizer, config, "optimizer", "adam"
        )
        requested_device = self._resolve(
            device, config, "device", None
        )

        self.warmup_epochs = int(self._resolve(
            warmup_epochs, config, "warmup_epochs", 3
        ))
        self.motion_weight = float(self._resolve(
            motion_weight, config, "motion_weight", 10.0
        ))
        self.foreground_weight = float(self._resolve(
            foreground_weight, config, "foreground_weight", 2.0
        ))
        self.motion_threshold = float(self._resolve(
            motion_threshold, config, "motion_threshold", 0.02
        ))
        self.foreground_threshold = float(self._resolve(
            foreground_threshold, config, "foreground_threshold", 0.05
        ))
        self.motion_dilation = int(self._resolve(
            motion_dilation, config, "motion_dilation", 3
        ))
        self.gradient_clip = float(self._resolve(
            gradient_clip, config, "gradient_clip", 1.0
        ))

        self._validate_settings()

        self.device = self._select_device(requested_device)
        self.model.to(self.device)

        self.dataloader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=int(num_workers),
            pin_memory=self.device.type == "cuda",
        )

        self.optimizer = self._build_optimizer(
            str(optimizer_name),
            float(learning_rate),
        )

    @staticmethod
    def _resolve(explicit, config, name: str, default):
        if explicit is not None:
            return explicit
        if config is not None and hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
        return default

    def _validate_settings(self) -> None:
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative.")
        if self.motion_weight < 0 or self.foreground_weight < 0:
            raise ValueError("Loss weights cannot be negative.")
        if self.motion_threshold < 0 or self.foreground_threshold < 0:
            raise ValueError("Mask thresholds cannot be negative.")
        if self.motion_dilation < 1 or self.motion_dilation % 2 == 0:
            raise ValueError("motion_dilation must be a positive odd integer.")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive.")

    @staticmethod
    def _select_device(requested) -> torch.device:
        if requested is None:
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        selected = torch.device(requested)
        if selected.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return selected

    def _build_optimizer(self, name: str, learning_rate: float):
        name = name.strip().lower()
        if name == "adam":
            return torch.optim.Adam(
                self.model.parameters(), lr=learning_rate
            )
        if name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(), lr=learning_rate
            )
        raise ValueError(
            f"Unsupported optimizer '{name}'. Use 'adam' or 'sgd'."
        )

    def _build_motion_mask(
        self,
        real_current: Tensor,
        real_next: Tensor,
    ) -> Tensor:
        """Return a dilated binary mask of pixels that changed."""
        mask = (
            (real_next - real_current).abs() > self.motion_threshold
        ).to(real_next.dtype)

        if self.motion_dilation > 1:
            mask = F.max_pool2d(
                mask,
                kernel_size=self.motion_dilation,
                stride=1,
                padding=self.motion_dilation // 2,
            )

        return mask

    def _object_aware_loss(
        self,
        prediction: Tensor,
        real_current: Tensor,
        real_next: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Return total loss and diagnostic components.

        Weighted MSE is normalised by the sum of pixel weights, preventing the
        numerical loss scale from exploding when motion_weight is increased.
        """
        if prediction.shape != real_next.shape:
            raise ValueError(
                f"Prediction shape {tuple(prediction.shape)} does not match "
                f"target shape {tuple(real_next.shape)}."
            )

        squared_error = (prediction - real_next).pow(2)
        motion_mask = self._build_motion_mask(
            real_current, real_next
        )
        foreground_mask = (
            real_next.abs() > self.foreground_threshold
        ).to(real_next.dtype)

        weights = (
            torch.ones_like(squared_error)
            + self.motion_weight * motion_mask
            + self.foreground_weight * foreground_mask
        )

        total_loss = (
            (weights * squared_error).sum()
            / weights.sum().clamp_min(1.0)
        )

        base_loss = squared_error.mean()

        motion_count = motion_mask.sum()
        motion_loss = (
            (motion_mask * squared_error).sum()
            / motion_count.clamp_min(1.0)
        )

        foreground_count = foreground_mask.sum()
        foreground_loss = (
            (foreground_mask * squared_error).sum()
            / foreground_count.clamp_min(1.0)
        )

        motion_fraction = motion_mask.mean()

        return (
            total_loss,
            base_loss.detach(),
            motion_loss.detach(),
            foreground_loss.detach(),
            motion_fraction.detach(),
        )

    def train_epoch(self, epoch: int | None = None) -> EpochResult:
        """
        Train one epoch.

        ``epoch`` must be zero-based. Corrections are disabled during the
        configured warm-up period, but object-aware reconstruction is active
        from the first epoch.
        """
        self.model.train()

        self.corrector.enabled = (
            epoch is None or epoch >= self.warmup_epochs
        )
        self.corrector.reset_log()

        total_loss = 0.0
        total_base_loss = 0.0
        total_motion_loss = 0.0
        total_foreground_loss = 0.0
        total_motion_fraction = 0.0
        batch_count = 0
        drift_values: list[Tensor] = []

        for frames, actions in self.dataloader:
            frames = frames.to(self.device, non_blocking=True)
            actions = actions.to(
                self.device, non_blocking=True
            ).long()

            if frames.ndim != 5:
                raise ValueError(
                    "Expected frames shaped "
                    "(batch, sequence, channels, height, width), "
                    f"got {tuple(frames.shape)}."
                )
            if actions.ndim != 2:
                raise ValueError(
                    "Expected actions shaped (batch, sequence), "
                    f"got {tuple(actions.shape)}."
                )

            batch_size, sequence_length = actions.shape
            if frames.shape[1] != sequence_length + 1:
                raise ValueError(
                    "The frame sequence must contain one more item "
                    "than the action sequence."
                )

            hidden = self.model.init_hidden(
                batch_size=batch_size,
                device=self.device,
            )
            self.optimizer.zero_grad(set_to_none=True)

            current_input = frames[:, 0]
            sequence_loss = torch.zeros((), device=self.device)
            sequence_base = 0.0
            sequence_motion = 0.0
            sequence_foreground = 0.0
            sequence_motion_fraction = 0.0

            for step in range(sequence_length):
                latent = self.model.encode(current_input)
                prediction, hidden = self.model.step(
                    latent,
                    actions[:, step],
                    hidden,
                )

                real_current = frames[:, step]
                real_next = frames[:, step + 1]

                (
                    step_loss,
                    base_loss,
                    motion_loss,
                    foreground_loss,
                    motion_fraction,
                ) = self._object_aware_loss(
                    prediction,
                    real_current,
                    real_next,
                )

                sequence_loss = sequence_loss + step_loss
                sequence_base += float(base_loss.item())
                sequence_motion += float(motion_loss.item())
                sequence_foreground += float(
                    foreground_loss.item()
                )
                sequence_motion_fraction += float(
                    motion_fraction.item()
                )

                with torch.no_grad():
                    drift_error = (
                        self.drift_detector.compute_error(
                            prediction,
                            real_next,
                        )
                    )
                    drift_values.append(
                        drift_error.detach().cpu()
                    )

                hidden = self.corrector.maybe_correct(
                    hidden=hidden,
                    real_frame=real_next,
                    encode_fn=self.model.encode,
                    error=drift_error,
                    step=step,
                )

                current_input = prediction.detach()

            sequence_loss = sequence_loss / sequence_length
            sequence_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.gradient_clip,
            )
            self.optimizer.step()

            total_loss += float(sequence_loss.detach().item())
            total_base_loss += sequence_base / sequence_length
            total_motion_loss += sequence_motion / sequence_length
            total_foreground_loss += (
                sequence_foreground / sequence_length
            )
            total_motion_fraction += (
                sequence_motion_fraction / sequence_length
            )
            batch_count += 1

        if batch_count == 0:
            raise RuntimeError(
                "The DataLoader produced no batches."
            )

        all_drift = torch.cat(drift_values)

        return self._build_epoch_result(
            average_loss=total_loss / batch_count,
            average_base_loss=total_base_loss / batch_count,
            average_motion_loss=total_motion_loss / batch_count,
            average_foreground_loss=(
                total_foreground_loss / batch_count
            ),
            average_motion_fraction=(
                total_motion_fraction / batch_count
            ),
            mean_drift_error=float(all_drift.mean().item()),
            p90_drift_error=float(
                torch.quantile(all_drift, 0.90).item()
            ),
        )

    def _build_epoch_result(
        self,
        average_loss: float,
        average_base_loss: float,
        average_motion_loss: float,
        average_foreground_loss: float,
        average_motion_fraction: float,
        mean_drift_error: float,
        p90_drift_error: float,
    ) -> EpochResult:
        correction_log = self.corrector.correction_log
        corrected_samples = sum(
            int(event[2]) for event in correction_log
        )

        if corrected_samples == 0:
            mean_correction_error = None
        else:
            weighted_error = sum(
                float(event[1]) * int(event[2])
                for event in correction_log
            )
            mean_correction_error = (
                weighted_error / corrected_samples
            )

        return EpochResult(
            average_loss=float(average_loss),
            correction_events=len(correction_log),
            corrected_samples=corrected_samples,
            mean_correction_error=mean_correction_error,
            average_base_loss=float(average_base_loss),
            average_motion_loss=float(average_motion_loss),
            average_foreground_loss=float(
                average_foreground_loss
            ),
            average_motion_fraction=float(
                average_motion_fraction
            ),
            mean_drift_error=float(mean_drift_error),
            p90_drift_error=float(p90_drift_error),
        )
