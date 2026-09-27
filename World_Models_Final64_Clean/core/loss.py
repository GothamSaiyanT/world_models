import numpy as np
import torch

from scipy import ndimage


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
    brightness_threshold=0.3,
    min_ball_area=1,
    max_ball_area=12,
):
    """Heuristic mask that isolates the ball by SIZE, not by location.

    An earlier version of this function tried to exclude the paddle and
    walls by their position (a margin from the edges, a band at the
    bottom). That assumed the crop had no bricks in view — wrong for this
    dataset: the brick rows near the top are bright too, and a
    position-based exclusion has no way to tell "brick block" apart from
    "ball" (both end up inside the same allowed region). The result was
    the ball_weight below being spent almost entirely on bricks (in one
    check, ~900 flagged pixels, of which only 4 were actually the ball).

    Instead: threshold on brightness, then connected-component label the
    result. The ball is a small isolated blob; the paddle and any brick
    block are large contiguous regions. Keeping only components in
    [min_ball_area, max_ball_area] pixels finds the ball regardless of
    where it is in the frame, and regardless of the crop's wall/brick/
    paddle layout. Still a heuristic — check it with
    scripts/inspect_ball_mask.py against real frames, and widen
    max_ball_area if the ball renders larger than ~12 pixels at your
    resolution, or narrow it if paddle fragments start getting caught.

    frame: (B, 1, H, W) tensor in [0, 1].
    Returns a bool tensor of the same shape.
    """

    frame_np = frame.detach().to("cpu").numpy()
    batch_size, channels, height, width = frame_np.shape
    mask_np = np.zeros_like(frame_np, dtype=bool)

    for b in range(batch_size):
        for c in range(channels):
            bright = frame_np[b, c] > brightness_threshold
            labeled, num_components = ndimage.label(bright)
            if num_components == 0:
                continue

            sizes = ndimage.sum(bright, labeled, range(1, num_components + 1))
            for component_id, size in enumerate(sizes, start=1):
                if min_ball_area <= size <= max_ball_area:
                    mask_np[b, c][labeled == component_id] = True

    return torch.from_numpy(mask_np).to(device=frame.device)


class BallAwareMotionWeightedMSELoss:
    """StableMotionWeightedMSELoss, plus a dedicated weight for the ball.

    Motion weighting alone isn't enough for an object this small: in a
    typical frame-to-frame transition only ~1% of pixels move at all (mostly
    the paddle, since it's much bigger than the ball), so even a large
    motion_weight barely changes the ball's share of the total loss. This
    adds a second mask — detect_ball_mask, size-based rather than
    motion-based — with its own (larger) weight, so the ball gets
    meaningful gradient signal even in the (rare) frames where it isn't
    classified as "moving", and regardless of how many other pixels are
    moving in that frame.

    ball_false_positive_weight closes a gap the above doesn't: nothing so
    far penalizes the model for drawing a ball-sized blob somewhere the
    ball ISN'T. Since a false guess there only ever cost the same as an
    ordinary background pixel, the cheapest way to lower the loss was to
    draw a low-confidence blob at whatever position was most often close
    to correct across the training set — a fixed decoy, not real tracking
    (this is what showed up in practice: the same blob, same location,
    regardless of the true game state). Detecting ball-sized blobs in the
    PREDICTION too, and adding extra weight wherever one appears with no
    matching real ball there, makes that guessing strategy costly instead
    of free.
    """

    def __init__(
        self,
        motion_weight=8.0,
        motion_threshold=0.05,
        ball_weight=40.0,
        ball_brightness_threshold=0.3,
        ball_min_area=1,
        ball_max_area=12,
        ball_false_positive_weight=20.0,
    ):

        self.motion_weight = motion_weight
        self.motion_threshold = motion_threshold
        self.ball_weight = ball_weight
        self.ball_brightness_threshold = ball_brightness_threshold
        self.ball_min_area = ball_min_area
        self.ball_max_area = ball_max_area
        self.ball_false_positive_weight = ball_false_positive_weight

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
            brightness_threshold=self.ball_brightness_threshold,
            min_ball_area=self.ball_min_area,
            max_ball_area=self.ball_max_area,
        ).to(prediction.dtype)

        predicted_ball_mask = detect_ball_mask(
            prediction,
            brightness_threshold=self.ball_brightness_threshold,
            min_ball_area=self.ball_min_area,
            max_ball_area=self.ball_max_area,
        ).to(prediction.dtype)

        # A predicted ball-sized blob that doesn't overlap a real one.
        false_positive_mask = predicted_ball_mask * (1.0 - ball_mask)

        pixel_weights = (
            1.0
            + self.motion_weight * motion_mask
            + self.ball_weight * ball_mask
            + self.ball_false_positive_weight * false_positive_mask
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