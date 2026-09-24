from dataclasses import asdict, dataclass


@dataclass
class V3Config:
    image_size: int = 128
    input_channels: int = 1
    latent_size: int = 256
    hidden_size: int = 256
    num_actions: int = 4
    action_embedding_size: int = 32
    sequence_length: int = 32

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)
