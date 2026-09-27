import torch


class StableMotionWeightedMSELoss:
    """
    Weighted MSE loss for sparse visual environments.

    The loss gives additional importance to:

    1. Motion pixels:
       Pixels whose values changed between the current
       real frame and the target frame.

    2. Foreground pixels:
       Visible/non-background pixels in the target frame.

    This prevents the large black background from dominating
    the training objective while small objects such as the
    ball and paddle receive too little attention.
    """

    def __init__(
        self,
        motion_weight=2.0,
        motion_threshold=0.05,
        foreground_weight=2.0,
        foreground_threshold=0.05,
    ):
        self.motion_weight = motion_weight
        self.motion_threshold = motion_threshold

        self.foreground_weight = foreground_weight
        self.foreground_threshold = foreground_threshold

    def forward(
        self,
        prediction,
        target,
        current_frame,
    ):
        # -----------------------------------------------------
        # Standard pixel-wise squared error
        # -----------------------------------------------------

        squared_error = (
            prediction - target
        ) ** 2

        # -----------------------------------------------------
        # MOTION MASK
        # -----------------------------------------------------
        #
        # Compare the real current frame with the real target.
        #
        # Pixels that changed by more than the threshold are
        # considered motion pixels.
        # -----------------------------------------------------

        frame_difference = torch.abs(
            target - current_frame
        )

        motion_mask = (
            frame_difference
            > self.motion_threshold
        ).to(prediction.dtype)

        # -----------------------------------------------------
        # FOREGROUND MASK
        # -----------------------------------------------------
        #
        # Breakout has a large black background.
        #
        # Pixels above the foreground threshold are considered
        # visible game objects such as:
        #
        #   - ball
        #   - paddle
        #   - bricks
        #
        # These pixels receive additional importance so that
        # the network cannot obtain a deceptively low loss by
        # mainly learning the black background.
        # -----------------------------------------------------

        foreground_mask = (
            torch.abs(target)
            > self.foreground_threshold
        ).to(prediction.dtype)

        # -----------------------------------------------------
        # PIXEL WEIGHTS
        # -----------------------------------------------------
        #
        # Normal background pixel:
        #       weight = 1
        #
        # Motion pixel:
        #       weight = 1 + motion_weight
        #
        # Foreground pixel:
        #       weight = 1 + foreground_weight
        #
        # Moving foreground pixel:
        #       weight =
        #       1 + motion_weight + foreground_weight
        #
        # With both weights = 2:
        #
        # background          = 1x
        # motion only         = 3x
        # foreground only     = 3x
        # moving foreground   = 5x
        # -----------------------------------------------------

        pixel_weights = (
            1.0
            + self.motion_weight * motion_mask
            + self.foreground_weight * foreground_mask
        )

        # -----------------------------------------------------
        # Apply weighting
        # -----------------------------------------------------

        weighted_error = (
            pixel_weights
            * squared_error
        )

        # -----------------------------------------------------
        # Normalize by total pixel weight
        # -----------------------------------------------------

        return (
            weighted_error.sum()
            / pixel_weights.sum().clamp_min(1.0)
        )

    def __call__(
        self,
        prediction,
        target,
        current_frame,
    ):
        return self.forward(
            prediction,
            target,
            current_frame,
        )