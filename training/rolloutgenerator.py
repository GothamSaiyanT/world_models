import torch


class RolloutGenerator:
    """
    Drives the world model forward autoregressively: each predicted
    frame is fed back in as the input for the next step, instead of
    always being given the real frame (which is what the training
    loop and evaluate.py currently do). This is what actually
    exposes long-horizon prediction drift, and it is the same
    generation loop the adaptive and fixed-interval correctors will
    hook into later.
    """

    def __init__(self, model):

        self.model = model

    @torch.no_grad()
    def generate(self, initial_frame, actions):
        """
        initial_frame: (1, 1, H, W) tensor -> the real starting frame
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

            # Feed the prediction back in as the next input --
            # this is the autoregressive step that causes drift.
            frame = prediction

        return torch.stack(predicted_frames)