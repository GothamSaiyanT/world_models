import torch


class RolloutGenerator:
    """
    Generates future frames autoregressively.

    Each predicted frame becomes the input for the following
    prediction step.
    """

    def __init__(self, model):
        self.model = model

    @torch.inference_mode()
    def generate(self, initial_frame, actions):

        self.model.eval()

        model_device = self.model.parameters()[0].data.device

        frame = initial_frame.to(
            device=model_device,
            dtype=torch.float32
        )

        # Accept either (1, H, W) or (1, 1, H, W)
        if frame.ndim == 3:
            frame = frame.unsqueeze(0)

        if frame.ndim != 4:
            raise ValueError(
                "initial_frame must have shape "
                "(1, 1, H, W) or (1, H, W)"
            )

        # Normalise uint8-style images into 0-1
        if frame.max().item() > 1.0:
            frame = frame / 255.0

        actions = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=model_device
        ).reshape(-1)

        if actions.numel() == 0:
            raise ValueError(
                "The actions tensor is empty. "
                "No future frames can be generated."
            )

        hidden = self.model.init_hidden(
            batch_size=1,
            device=model_device
        )

        predicted_frames = [
            frame[0].detach().cpu()
        ]

        for action in actions:

            prediction, hidden = self.model(
                frame,
                action.reshape(1),
                hidden
            )

            prediction = prediction.clamp(
                min=0.0,
                max=1.0
            )

            predicted_frames.append(
                prediction[0].detach().cpu()
            )

            # Autoregressive feedback
            frame = prediction

        return torch.stack(
            predicted_frames,
            dim=0
        )