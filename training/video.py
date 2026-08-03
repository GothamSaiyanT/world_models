import cv2
import numpy as np


def tensor_to_uint8(frame):
    """
    Converts a (1, H, W) float tensor in [0, 1] to a uint8 (H, W)
    array suitable for OpenCV.
    """

    array = frame.squeeze(0).detach().cpu().numpy()

    return (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)


def render_comparison_video(
    real_frames,
    predicted_frames,
    out_path,
    fps=10
):
    """
    real_frames / predicted_frames: sequences of (1, H, W) tensors
    of the same length. Writes a single greyscale MP4 with the real
    rollout on the left and the predicted rollout on the right,
    separated by a thin gap, so drift is visible frame by frame.
    """

    height, width = real_frames[0].shape[-2:]

    gap = 4

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2 + gap, height),
        isColor=False
    )

    for real, predicted in zip(real_frames, predicted_frames):

        divider = np.zeros(
            (height, gap),
            dtype=np.uint8
        )

        combined = np.hstack((
            tensor_to_uint8(real),
            divider,
            tensor_to_uint8(predicted)
        ))

        writer.write(combined)

    writer.release()

    print(f"Saved comparison video to {out_path}")