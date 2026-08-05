from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional


@dataclass(frozen=True)
class DataConfig:
    folder: str = "data"
    sequence_length: int = 16

    def __post_init__(self) -> None:
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")


@dataclass(frozen=True)
class ModelConfig:
    latent_size: int = 128
    hidden_size: int = 128
    image_size: int = 64
    input_channels: int = 1
    num_actions: int = 4
    action_embedding_size: int = 32

    def __post_init__(self) -> None:
        if self.latent_size != self.hidden_size:
            raise ValueError(
                "latent_size must equal hidden_size because correction replaces "
                "the recurrent hidden state with an encoded observation"
            )
        if self.image_size < 22:
            raise ValueError(
                "image_size is too small for three 4x4 stride-2 convolutions"
            )
        if self.input_channels < 1:
            raise ValueError("input_channels must be at least 1")
        if self.num_actions < 1:
            raise ValueError("num_actions must be at least 1")


@dataclass(frozen=True)
class DriftConfig:
    metric: Literal["mse", "ssim"] = "mse"
    adaptive_threshold: float = 0.01
    fixed_interval: int = 10

    def __post_init__(self) -> None:
        if self.adaptive_threshold < 0:
            raise ValueError("adaptive_threshold cannot be negative")
        if self.fixed_interval < 1:
            raise ValueError("fixed_interval must be at least 1")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    learning_rate: float = 0.001
    batch_size: int = 32
    num_workers: int = 2
    optimizer: Literal["sgd", "adam"] = "sgd"
    device: Optional[str] = None
    checkpoint_folder: str = "models"
    history_folder: str = "results"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")


def serialise_configs(**configs: Any) -> Dict[str, Dict[str, Any]]:
    """Convert dataclass configuration objects into checkpoint-safe dictionaries."""
    return {name: asdict(config) for name, config in configs.items()}
