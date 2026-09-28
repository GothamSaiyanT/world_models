"""Train / validation split for the sequence dataset."""

from __future__ import annotations

from torch.utils.data import Dataset, Subset


def split_train_validation(
    dataset: Dataset,
    validation_fraction: float,
    gap: int,
) -> tuple[Dataset, Dataset | None]:
    """Split sequence windows into a training set and a validation set.

    Each item in the dataset is a window of (gap + 1) consecutive frames, and
    neighbouring windows overlap almost completely. A random split would put
    nearly identical windows on both sides and make validation look better
    than it really is.

    So the split is done in time order instead:

        [ train windows ... | skipped windows | validation windows ]

    The last ``gap`` training windows are skipped so that no frame appears in
    both sets. ``gap`` should be the sequence length.

    A ``validation_fraction`` of 0 turns validation off and returns
    ``(dataset, None)``.
    """
    if validation_fraction <= 0:
        return dataset, None

    total = len(dataset)
    validation_count = int(total * validation_fraction)
    train_count = total - validation_count - gap

    if validation_count < 1 or train_count < 1:
        raise ValueError(
            f"Not enough data to split: {total} sequences with "
            f"validation_fraction={validation_fraction} and a gap of {gap} "
            f"leaves {train_count} for training and {validation_count} for "
            "validation. Collect more data or lower validation_fraction."
        )

    train_indices = range(0, train_count)
    validation_indices = range(total - validation_count, total)

    return Subset(dataset, train_indices), Subset(dataset, validation_indices)
