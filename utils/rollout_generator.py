import torch


class RolloutGenerator:
    """
    Drives any world model forward autoregressively -- works for
    the baseline (core.world_model.WorldModel) and both self-
    correcting pipelines (core_nn.world_model.WorldModel) unchanged,
    since all three expose the same forward(frame, action, hidden)
    and init_hidden(batch_size) interface.

    This file replaces core/rollout_generator.py -- update the
    baseline's imports to point here instead, so there is only one
    copy of this logic shared across all three pipelines.
    """

    def __init__(self, model):

        self.model = model

    @torch.no_grad()
    def generate(self, initial_frame, actions):
        """
        initial_frame: (1, 1, H, W) tensor -- the real starting frame
        actions: (T,) tensor of actions to imagine forward with

        Returns: (T + 1, 1, H, W) tensor of frames, where index 0 is
        the real initial frame and indices 1..T are imagined.
        """

        self.model.eval()

        device = initial_frame.device

        hidden = self.model.init_hidden(
            batch_size=1
        ).to(device)

        frame = initial_frame

        predicted_frames = [initial_frame.squeeze(0)]

        for action in actions:

            prediction, hidden = self.model(
                frame,
                action.view(1),
                hidden
            )

            predicted_frames.append(
                prediction.squeeze(0)
            )

            frame = prediction

        return torch.stack(predicted_frames)