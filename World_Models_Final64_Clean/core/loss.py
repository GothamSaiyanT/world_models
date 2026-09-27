import torch


class StableMotionWeightedMSELoss:

    def __init__(
        self,
        motion_weight=2.0,
        motion_threshold=0.05
    ):

        self.motion_weight = motion_weight
        self.motion_threshold = motion_threshold

    def forward(
        self,
        prediction,
        target,
        current_frame
    ):

        squared_error = (
            prediction - target
        ) ** 2

        frame_difference = torch.abs(
            target - current_frame
        )

        motion_mask = (
            frame_difference
            > self.motion_threshold
        ).to(prediction.dtype)

        pixel_weights = (
            1.0
            + self.motion_weight * motion_mask
        )

        weighted_error = (
            pixel_weights * squared_error
        )

        return (
            weighted_error.sum()
            / pixel_weights.sum().clamp_min(1.0)
        )

    def __call__(
        self,
        prediction,
        target,
        current_frame
    ):

        return self.forward(
            prediction,
            target,
            current_frame
        )