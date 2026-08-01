import matplotlib.pyplot as plt

def show_prediction(current, prediction, target):

    current = current.squeeze().cpu().numpy()
    prediction = prediction.squeeze().detach().cpu().numpy()
    target = target.squeeze().detach().cpu().numpy()

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))

    figure.suptitle(
        "Baseline World Model Prediction",
        fontsize=14
    )

    axes[0].imshow(current, cmap="gray")
    axes[0].set_title("Current Frame")
    axes[0].axis("off")

    axes[1].imshow(prediction, cmap="gray")
    axes[1].set_title("Predicted Next Frame")
    axes[1].axis("off")

    axes[2].imshow(target, cmap="gray")
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=300)
    plt.show()