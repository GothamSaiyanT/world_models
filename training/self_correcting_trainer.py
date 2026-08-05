"""Shared trainer for adaptive and fixed-interval self-correcting pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor
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

    Reconstruction uses adaptive motion-weighted MSE so small moving objects
    such as the Breakout ball and paddle are not overwhelmed by static
    background pixels.

    The motion mask is computed from two consecutive REAL frames:
        abs(real_next - real_current) > motion_threshold
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
        motion_base_weight: float | None = None,
        motion_weight: float | None = None,
        motion_threshold: float | None = None,
        motion_rho_max: float | None = None,
        motion_epsilon: float | None = None,
    ) -> None:
        self.model = model
        self.corrector = corrector
        self.drift_detector = drift_detector

        learning_rate = self._resolve_setting(
            explicit=learning_rate, config=config,
            name="learning_rate", default=0.001,
        )
        batch_size = self._resolve_setting(
            explicit=batch_size, config=config,
            name="batch_size", default=32,
        )
        num_workers = self._resolve_setting(
            explicit=num_workers, config=config,
            name="num_workers", default=2,
        )
        optimizer_name = self._resolve_setting(
            explicit=optimizer, config=config,
            name="optimizer", default="sgd",
        )
        configured_device = self._resolve_setting(
            explicit=device, config=config,
            name="device", default=None,
        )

        self.motion_base_weight = float(self._resolve_setting(
            explicit=motion_base_weight, config=config,
            name="motion_base_weight", default=1.0,
        ))
        self.motion_weight = float(self._resolve_setting(
            explicit=motion_weight, config=config,
            name="motion_weight", default=10.0,
        ))
        self.motion_threshold = float(self._resolve_setting(
            explicit=motion_threshold, config=config,
            name="motion_threshold", default=0.02,
        ))
        self.motion_rho_max = float(self._resolve_setting(
            explicit=motion_rho_max, config=config,
            name="motion_rho_max", default=0.35,
        ))
        self.motion_epsilon = float(self._resolve_setting(
            explicit=motion_epsilon, config=config,
            name="motion_epsilon", default=1e-8,
        ))

        self._validate_motion_loss_settings()

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
        if explicit is not None:
            return explicit
        if config is not None and hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
        return default

    def _validate_motion_loss_settings(self) -> None:
        if self.motion_base_weight < 0:
            raise ValueError("motion_base_weight cannot be negative.")
        if self.motion_weight < 0:
            raise ValueError("motion_weight cannot be negative.")
        if self.motion_threshold < 0:
            raise ValueError("motion_threshold cannot be negative.")
        if not 0 < self.motion_rho_max <= 1:
            raise ValueError("motion_rho_max must be in (0, 1].")
        if self.motion_epsilon <= 0:
            raise ValueError("motion_epsilon must be greater than zero.")

    @staticmethod
    def _select_device(requested_device) -> torch.device:
        if requested_device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        selected = torch.device(requested_device)
        if selected.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested in TrainingConfig, but CUDA is unavailable."
            )
        return selected

    def _build_optimizer(self, name: str, learning_rate: float):
        normalised_name = name.strip().lower()
        if normalised_name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        if normalised_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        raise ValueError(
            f"Unsupported optimizer '{name}'. Choose either 'sgd' or 'adam'."
        )

    def _motion_weighted_loss(
        self,
        prediction: Tensor,
        real_current_frame: Tensor,
        real_next_frame: Tensor,
    ) -> Tensor:
        """Compute adaptive motion-weighted reconstruction loss."""
        if prediction.shape != real_next_frame.shape:
            raise ValueError(
                "Prediction and target shapes must match: "
                f"{tuple(prediction.shape)} vs {tuple(real_next_frame.shape)}."
            )
        if real_current_frame.shape != real_next_frame.shape:
            raise ValueError(
                "Consecutive real-frame shapes must match: "
                f"{tuple(real_current_frame.shape)} vs "
                f"{tuple(real_next_frame.shape)}."
            )

        squared_error = (prediction - real_next_frame).pow(2)

        motion_mask = (
            (real_next_frame - real_current_frame).abs()
            > self.motion_threshold
        ).to(dtype=squared_error.dtype)

        reduction_dims = tuple(range(1, squared_error.ndim))

        base_loss_per_sample = squared_error.mean(dim=reduction_dims)

        motion_pixel_count = motion_mask.sum(dim=reduction_dims)
        motion_error_sum = (
            motion_mask * squared_error
        ).sum(dim=reduction_dims)

        motion_loss_per_sample = (
            motion_error_sum / motion_pixel_count.clamp_min(1.0)
        )

        motion_fraction = motion_mask.mean(dim=reduction_dims)
        safe_fraction = motion_fraction.clamp_min(self.motion_epsilon)

        motion_scale = torch.clamp(
            self.motion_rho_max / safe_fraction,
            max=1.0,
        )

        combined_per_sample = (
            self.motion_base_weight * base_loss_per_sample
            + self.motion_weight * motion_scale * motion_loss_per_sample
        )

        return combined_per_sample.mean()

    def train_epoch(self) -> EpochResult:
        self.model.train()
        self.corrector.reset_log()

        total_loss = 0.0
        batch_count = 0

        for frames, actions in self.dataloader:
            frames = frames.to(self.device, non_blocking=True)
            actions = actions.to(
                self.device, non_blocking=True
            ).long()

            if frames.ndim != 5:
                raise ValueError(
                    "Expected frames with shape "
                    "(batch, sequence, channels, height, width), "
                    f"received {tuple(frames.shape)}."
                )
            if actions.ndim != 2:
                raise ValueError(
                    "Expected actions with shape (batch, sequence_length), "
                    f"received {tuple(actions.shape)}."
                )

            batch_size, sequence_length = actions.shape

            if frames.shape[1] != sequence_length + 1:
                raise ValueError(
                    "Each action sequence needs one more frame than actions: "
                    f"{frames.shape[1]} frames vs {sequence_length} actions."
                )

            hidden = self.model.init_hidden(
                batch_size=batch_size,
                device=self.device,
            )

            self.optimizer.zero_grad(set_to_none=True)

            current_frame = frames[:, 0]
            sequence_loss = torch.zeros((), device=self.device)

            for step in range(sequence_length):
                latent = self.model.encode(current_frame)

                prediction, hidden = self.model.step(
                    latent,
                    actions[:, step],
                    hidden,
                )

                real_current_frame = frames[:, step]
                real_next_frame = frames[:, step + 1]

                step_loss = self._motion_weighted_loss(
                    prediction=prediction,
                    real_current_frame=real_current_frame,
                    real_next_frame=real_next_frame,
                )
                sequence_loss = sequence_loss + step_loss

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

                current_frame = prediction.detach()

            sequence_loss = sequence_loss / sequence_length
            sequence_loss.backward()
            self.optimizer.step()

            total_loss += float(sequence_loss.detach().item())
            batch_count += 1

        if batch_count == 0:
            raise RuntimeError(
                "The DataLoader produced no batches. "
                "Check the dataset and sequence length."
            )

        return self._build_epoch_result(
            average_loss=total_loss / batch_count,
        )

    def _build_epoch_result(self, average_loss: float) -> EpochResult:
        correction_log = self.corrector.correction_log
        correction_events = len(correction_log)

        corrected_samples = sum(int(event[2]) for event in correction_log)

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