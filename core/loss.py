import torch


class AdaptiveMotionWeightedMSELoss:
    """
    Combines normal pixel MSE with additional emphasis on pixels
    that change between the current frame and the target frame.

    No torch.nn or torch.nn.functional is used.

    The motion contribution is automatically reduced when a large
    part of the screen changes, such as in scrolling games.
    """

    def __init__(
        self,
        motion_weight=10.0,
        motion_threshold=0.02,
        base_loss_weight=1.0,
        max_motion_fraction=0.35
    ):

        if motion_weight < 0:
            raise ValueError(
                "motion_weight must be zero or greater."
            )

        if motion_threshold < 0:
            raise ValueError(
                "motion_threshold must be zero or greater."
            )

        if base_loss_weight <= 0:
            raise ValueError(
                "base_loss_weight must be greater than zero."
            )

        if not 0 < max_motion_fraction <= 1:
            raise ValueError(
                "max_motion_fraction must be between 0 and 1."
            )

        self.motion_weight = motion_weight
        self.motion_threshold = motion_threshold
        self.base_loss_weight = base_loss_weight
        self.max_motion_fraction = max_motion_fraction

    def forward(
        self,
        prediction,
        target,
        current_frame
    ):

        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must have the same shape."
            )

        if current_frame.shape != target.shape:
            raise ValueError(
                "current_frame and target must have the same shape."
            )

        # -----------------------------------
        # Standard MSE for the complete image
        # -----------------------------------

        squared_error = (
            prediction - target
        ) ** 2

        base_loss = torch.mean(
            squared_error
        )

        # -----------------------------------
        # Detect changing pixels
        # -----------------------------------

        frame_difference = torch.abs(
            target - current_frame
        )

        motion_mask = (
            frame_difference
            > self.motion_threshold
        ).float()

        motion_fraction = torch.mean(
            motion_mask
        )

        # If almost the entire frame changes, the change may come
        # from camera movement or scrolling rather than small,
        # important game objects.
        motion_scale = torch.clamp(
            self.max_motion_fraction
            / motion_fraction.clamp_min(1e-8),
            max=1.0
        )

        effective_motion_weight = (
            self.motion_weight
            * motion_scale
        )

        # -----------------------------------
        # Motion-weighted error
        # -----------------------------------

        motion_pixel_count = (
            motion_mask.sum()
        )

        motion_loss = (
            squared_error
            * motion_mask
        ).sum() / motion_pixel_count.clamp_min(1.0)

        # If no motion is detected, motion_loss becomes zero.
        has_motion = (
            motion_pixel_count > 0
        ).float()

        motion_loss = (
            motion_loss
            * has_motion
        )

        # -----------------------------------
        # Combined loss
        # -----------------------------------

        total_loss = (
            self.base_loss_weight
            * base_loss
            + effective_motion_weight
            * motion_loss
        )

        return total_loss

    def __call__(
        self,
        prediction,
        target,
        current_frame
    ):

        return self.forward(
            prediction=prediction,
            target=target,
            current_frame=current_frame
        )