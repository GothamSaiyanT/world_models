from __future__ import annotations

import torch

from training.v3_loss import per_sample_drift_error


class RolloutGeneratorV3:
    """
    Evaluate all three strategies using the same trained architecture.

    Baseline is pure open-loop. Fixed and Adaptive are observation-corrected
    rollouts and therefore require the matching real sequence. A correction
    affects the *next* step; the prediction that triggered it is still kept in
    the returned sequence and can be scored fairly.
    """

    def __init__(
        self,
        model,
        *,
        strategy="baseline",
        fixed_interval=8,
        adaptive_threshold=None,
        runtime=None,
    ):
        if strategy not in {"baseline", "fixed_interval", "adaptive"}:
            raise ValueError("Unknown V3 rollout strategy.")
        self.model = model
        self.strategy = strategy
        self.fixed_interval = int(fixed_interval)
        self.adaptive_threshold = adaptive_threshold
        self.runtime = runtime
        self.correction_steps = []

    @torch.no_grad()
    def generate(self, initial_frame, actions, real_sequence=None):
        self.model.eval()
        device = initial_frame.device
        hidden = self.model.init_hidden(1, device)
        frame = initial_frame
        predictions = [initial_frame.squeeze(0)]
        correction_masks = []

        if self.strategy != "baseline" and real_sequence is None:
            raise ValueError(
                f"{self.strategy} evaluation needs real_sequence for observation correction."
            )
        if real_sequence is not None:
            real_sequence = real_sequence.to(device)

        for step, action in enumerate(actions):
            prediction, hidden = self.model(frame, action.view(1), hidden)
            predictions.append(prediction.squeeze(0))
            next_input = prediction
            correction_mask = torch.zeros(1, dtype=torch.bool, device=device)

            if self.strategy == "fixed_interval":
                if (step + 1) % self.fixed_interval == 0:
                    real_next = real_sequence[step + 1].unsqueeze(0)
                    hidden = self.model.encode(real_next)
                    next_input = real_next
                    correction_mask = torch.ones(1, dtype=torch.bool, device=device)

            elif self.strategy == "adaptive":
                if self.adaptive_threshold is None:
                    raise ValueError("Adaptive rollout requires adaptive_threshold.")
                real_current = real_sequence[step].unsqueeze(0)
                real_next = real_sequence[step + 1].unsqueeze(0)
                error = per_sample_drift_error(
                    prediction, real_current, real_next
                )
                correction_mask = error > float(self.adaptive_threshold)

                # Keep the correction decision on the accelerator rather than
                # converting an XLA tensor to a Python bool every frame.
                encoded_real = self.model.encode(real_next)
                hidden = torch.where(
                    correction_mask.unsqueeze(1), encoded_real, hidden
                )
                next_input = torch.where(
                    correction_mask[:, None, None, None],
                    real_next,
                    prediction,
                )

            correction_masks.append(correction_mask)
            frame = next_input

        if self.runtime is not None:
            self.runtime.sync()

        if correction_masks:
            masks = torch.cat(correction_masks).detach().cpu().numpy()
            self.correction_steps = [
                index + 1 for index, corrected in enumerate(masks) if bool(corrected)
            ]
        else:
            self.correction_steps = []

        return torch.stack(predictions)
