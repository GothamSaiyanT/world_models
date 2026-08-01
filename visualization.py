import matplotlib.pyplot as plt


def show_prediction(current,
                    prediction,
                    target):
    """
    Display the current frame,
    predicted next frame,
    and actual next frame.
    """

    current = current.squeeze().cpu().numpy()
    prediction = prediction.squeeze().detach().cpu().numpy()
    target = target.squeeze().cpu().numpy()

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )

    axes[0].imshow(
        current,
        cmap="gray"
    )
    axes[0].set_title("Current Frame")
    axes[0].axis("off")

    axes[1].imshow(
        prediction,
        cmap="gray"
    )
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    axes[2].imshow(
        target,
        cmap="gray"
    )
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()