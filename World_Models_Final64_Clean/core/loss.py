import torch


def motion_weighted_error(
    prediction,
    target,
    current_frame,
    motion_weight=2.0,
    motion_threshold=0.05,
):
    """Same weighting scheme as StableMotionWeightedMSELoss, but as a plain
    function so it can be called outside training (e.g. during rollout,
    under torch.no_grad()) without needing a loss object.

    Pixels where the real scene actually changed (the ball, the paddle)
    get weighted more heavily than static background pixels, so the
    returned number reflects how well the moving objects are tracked,
    not just the overall frame.
    """

    squared_error = (prediction - target) ** 2

    frame_difference = torch.abs(target - current_frame)
    motion_mask = (frame_difference > motion_threshold).to(prediction.dtype)

    pixel_weights = 1.0 + motion_weight * motion_mask
    weighted_error = pixel_weights * squared_error

    return (
        weighted_error.sum() / pixel_weights.sum().clamp_min(1.0)
    ).item()


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


def detect_ball_mask(
    frame,
    wall_margin=4,
    paddle_band=10,
    brightness_threshold=0.3,
):
    """Heuristic mask that isolates the ball, independent of motion.

    Assumes the cropped Breakout play area (see CROP_TOP / CROP_BOTTOM in
    config_final64.py) has: a bright wall border a few pixels wide on the
    left/right/top, and the paddle confined to a band of rows near the
    bottom. Excluding both of those and thresholding what's left catches the
    ball (a small bright blob) without catching the paddle or the walls.

    This is a heuristic, not ground truth — it can misfire if the wall/paddle
    geometry doesn't match these assumptions. Check it with
    scripts/inspect_ball_mask.py against a handful of real frames before
    trusting it in a training run, and adjust wall_margin / paddle_band /
    brightness_threshold if the paddle or walls leak into the mask.

    frame: (B, 1, H, W) tensor in [0, 1].
    Returns a bool tensor of the same shape.
    """

    batch_size, channels, height, width = frame.shape
    mask = torch.zeros_like(frame, dtype=torch.bool)

    row_end = max(height - paddle_band, wall_margin + 1)
    col_end = max(width - wall_margin, wall_margin + 1)

    interior = frame[:, :, wall_margin:row_end, wall_margin:col_end]
    mask[:, :, wall_margin:row_end, wall_margin:col_end] = (
        interior > brightness_threshold
    )

    return mask


class BallAwareMotionWeightedMSELoss:
    """StableMotionWeightedMSELoss, plus a dedicated weight for the ball.

    Motion weighting alone isn't enough for an object this small: in a
    typical frame-to-frame transition only ~1% of pixels move at all (mostly
    the paddle, since it's much bigger than the ball), so even a large
    motion_weight barely changes the ball's share of the total loss. This
    adds a second mask — detect_ball_mask, spatial rather than motion-based —
    with its own (larger) weight, so the ball gets meaningful gradient signal
    even in the (rare) frames where it isn't classified as "moving", and
    regardless of how many other pixels are moving in that frame.
    """

    def __init__(
        self,
        motion_weight=8.0,
        motion_threshold=0.05,
        ball_weight=40.0,
        wall_margin=4,
        paddle_band=10,
        ball_brightness_threshold=0.3,
    ):

        self.motion_weight = motion_weight
        self.motion_threshold = motion_threshold
        self.ball_weight = ball_weight
        self.wall_margin = wall_margin
        self.paddle_band = paddle_band
        self.ball_brightness_threshold = ball_brightness_threshold

    def forward(
        self,
        prediction,
        target,
        current_frame
    ):

        squared_error = (prediction - target) ** 2

        frame_difference = torch.abs(target - current_frame)
        motion_mask = (frame_difference > self.motion_threshold).to(prediction.dtype)

        ball_mask = detect_ball_mask(
            target,
            wall_margin=self.wall_margin,
            paddle_band=self.paddle_band,
            brightness_threshold=self.ball_brightness_threshold,
        ).to(prediction.dtype)

        pixel_weights = (
            1.0
            + self.motion_weight * motion_mask
            + self.ball_weight * ball_mask
        )

        weighted_error = pixel_weights * squared_error

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