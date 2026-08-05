import torch
import torch.nn.functional as F


class DriftDetector:
    """Measure per-sample divergence between predicted and real frames."""

    SUPPORTED_METRICS = {"mse", "ssim"}

    def __init__(self, metric: str = "mse") -> None:
        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric '{metric}'. Choose from {self.SUPPORTED_METRICS}."
            )
        self.metric = metric

    def compute_error(
        self,
        predicted: torch.Tensor,
        real: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(predicted, real)
        if self.metric == "mse":
            return F.mse_loss(predicted, real, reduction="none").mean(
                dim=tuple(range(1, predicted.ndim))
            )
        return self._compute_ssim_error(predicted, real)

    @staticmethod
    def _validate_inputs(predicted: torch.Tensor, real: torch.Tensor) -> None:
        if predicted.shape != real.shape:
            raise ValueError(
                "predicted and real frames must have the same shape; "
                f"received {tuple(predicted.shape)} and {tuple(real.shape)}"
            )
        if predicted.ndim != 4:
            raise ValueError(
                "Expected frame tensors shaped (batch, channels, height, width)"
            )

    @staticmethod
    def _compute_ssim_error(
        predicted: torch.Tensor,
        real: torch.Tensor,
    ) -> torch.Tensor:
        try:
            from skimage.metrics import structural_similarity as ssim
        except ImportError as exc:
            raise ImportError(
                "SSIM requires scikit-image. Install it with: pip install scikit-image"
            ) from exc

        predicted_np = predicted.detach().cpu().numpy()
        real_np = real.detach().cpu().numpy()
        errors: list[float] = []

        for predicted_sample, real_sample in zip(predicted_np, real_np):
            channel_scores = [
                ssim(
                    predicted_channel,
                    real_channel,
                    data_range=1.0,
                )
                for predicted_channel, real_channel in zip(
                    predicted_sample,
                    real_sample,
                )
            ]
            errors.append(1.0 - float(sum(channel_scores) / len(channel_scores)))

        return torch.tensor(
            errors,
            device=predicted.device,
            dtype=predicted.dtype,
        )
