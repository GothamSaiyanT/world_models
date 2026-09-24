from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.v3_loss import motion_aware_loss, per_sample_drift_error


class V3Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        *,
        pipeline: str,
        output_folder: str = "models_v3",
        results_folder: str = "results_v3",
        batch_size: int = 4,
        learning_rate: float = 3e-4,
        fixed_interval: int = 8,
        warmup_epochs: int = 3,
        device: str | None = None,
    ):
        if pipeline not in {"baseline", "fixed_interval", "adaptive"}:
            raise ValueError("pipeline must be baseline, fixed_interval, or adaptive")
        self.model = model
        self.pipeline = pipeline
        self.output_folder = Path(output_folder)
        self.results_folder = Path(results_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )
        self.fixed_interval = int(fixed_interval)
        self.warmup_epochs = int(warmup_epochs)
        self.adaptive_threshold: float | None = None
        self.history = []

    @staticmethod
    def teacher_forcing_probability(epoch: int, epochs: int) -> float:
        # Begin with some real-frame anchoring, then become mostly autoregressive.
        if epochs <= 1:
            return 0.10
        progress = epoch / (epochs - 1)
        return max(0.10, 0.50 * (1.0 - progress))

    def _correct_state(
        self,
        *,
        hidden,
        prediction,
        real_next,
        drift_error,
        step,
        corrections_enabled,
    ):
        batch = prediction.shape[0]
        mask = torch.zeros(batch, dtype=torch.bool, device=self.device)

        if corrections_enabled and self.pipeline == "fixed_interval":
            if (step + 1) % self.fixed_interval == 0:
                mask[:] = True

        elif corrections_enabled and self.pipeline == "adaptive":
            if self.adaptive_threshold is not None:
                mask = drift_error > self.adaptive_threshold

        if bool(mask.any()):
            with torch.no_grad():
                encoded_real = self.model.encode(real_next)
            hidden = torch.where(mask.unsqueeze(1), encoded_real, hidden)
            next_input = torch.where(
                mask[:, None, None, None],
                real_next,
                prediction.detach(),
            )
            return hidden, next_input, int(mask.sum().item())

        return hidden, prediction.detach(), 0

    def _run_train_epoch(self, epoch: int, epochs: int):
        self.model.train()
        tf_prob = self.teacher_forcing_probability(epoch, epochs)
        corrections_enabled = epoch >= self.warmup_epochs

        totals = {
            "loss": 0.0,
            "base": 0.0,
            "motion": 0.0,
            "foreground": 0.0,
            "delta": 0.0,
            "edge": 0.0,
            "motion_fraction": 0.0,
        }
        batches = 0
        corrected_samples = 0
        drift_values = []

        for frames, actions in self.train_loader:
            frames = frames.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True)
            batch_size, sequence_length = actions.shape
            hidden = self.model.init_hidden(batch_size, self.device)
            current_input = frames[:, 0]
            sequence_loss = torch.zeros((), device=self.device)
            batch_parts = {key: 0.0 for key in totals if key != "loss"}

            self.optimizer.zero_grad(set_to_none=True)

            for step in range(sequence_length):
                real_current = frames[:, step]
                real_next = frames[:, step + 1]
                prediction, hidden = self.model(
                    current_input,
                    actions[:, step],
                    hidden,
                )
                parts = motion_aware_loss(
                    prediction,
                    real_current,
                    real_next,
                )
                sequence_loss = sequence_loss + parts.total
                batch_parts["base"] += float(parts.base)
                batch_parts["motion"] += float(parts.motion)
                batch_parts["foreground"] += float(parts.foreground)
                batch_parts["delta"] += float(parts.delta)
                batch_parts["edge"] += float(parts.edge)
                batch_parts["motion_fraction"] += float(parts.motion_fraction)

                with torch.no_grad():
                    drift = per_sample_drift_error(
                        prediction,
                        real_current,
                        real_next,
                    )
                    drift_values.extend(drift.detach().cpu().tolist())

                hidden, next_input, corrected = self._correct_state(
                    hidden=hidden,
                    prediction=prediction,
                    real_next=real_next,
                    drift_error=drift,
                    step=step,
                    corrections_enabled=corrections_enabled,
                )
                corrected_samples += corrected

                # Scheduled sampling only applies where a correction did not
                # already anchor the model to the observation.
                if corrected == 0 and tf_prob > 0:
                    teacher_mask = (
                        torch.rand(batch_size, device=self.device) < tf_prob
                    )
                    current_input = torch.where(
                        teacher_mask[:, None, None, None],
                        real_next,
                        next_input,
                    )
                else:
                    current_input = next_input

            sequence_loss = sequence_loss / sequence_length
            sequence_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            totals["loss"] += float(sequence_loss.detach().cpu())
            for key in batch_parts:
                totals[key] += batch_parts[key] / sequence_length
            batches += 1

        if batches == 0:
            raise RuntimeError("No training batches were produced.")

        if drift_values:
            drift_array = np.asarray(drift_values, dtype=np.float64)
            p90 = float(np.percentile(drift_array, 90))
            mean_drift = float(drift_array.mean())
        else:
            p90 = 0.0
            mean_drift = 0.0

        # Adaptive threshold for the next epoch is calibrated to this model's
        # current error scale rather than using the old 0.05 constant.
        if self.pipeline == "adaptive":
            self.adaptive_threshold = p90

        result = {key: value / batches for key, value in totals.items()}
        result.update(
            teacher_forcing=tf_prob,
            corrected_samples=corrected_samples,
            mean_drift_error=mean_drift,
            p90_drift_error=p90,
            adaptive_threshold=self.adaptive_threshold,
        )
        return result

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total = 0.0
        batches = 0
        drift_values = []

        # Validation is pure open-loop for all pipelines. This gives a common
        # model-quality criterion for checkpoint selection.
        for frames, actions in self.val_loader:
            frames = frames.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True)
            batch_size, sequence_length = actions.shape
            hidden = self.model.init_hidden(batch_size, self.device)
            current_input = frames[:, 0]
            sequence_loss = torch.zeros((), device=self.device)

            for step in range(sequence_length):
                real_current = frames[:, step]
                real_next = frames[:, step + 1]
                prediction, hidden = self.model(
                    current_input, actions[:, step], hidden
                )
                parts = motion_aware_loss(prediction, real_current, real_next)
                sequence_loss = sequence_loss + parts.total
                drift = per_sample_drift_error(
                    prediction, real_current, real_next
                )
                drift_values.extend(drift.detach().cpu().tolist())
                current_input = prediction

            total += float((sequence_loss / sequence_length).cpu())
            batches += 1

        drift_array = np.asarray(drift_values, dtype=np.float64)
        return {
            "validation_loss": total / max(batches, 1),
            "validation_mean_drift": float(drift_array.mean()),
            "validation_p90_drift": float(np.percentile(drift_array, 90)),
        }

    def train(self, epochs: int):
        best_val = math.inf
        model_path = self.output_folder / f"best_{self.pipeline}_model.pt"
        checkpoint_path = self.output_folder / f"best_{self.pipeline}_checkpoint.pt"
        history_path = self.results_folder / f"{self.pipeline}_history.json"

        for epoch in range(int(epochs)):
            train_result = self._run_train_epoch(epoch, epochs)
            val_result = self.validate()
            record = {
                "epoch": epoch + 1,
                **train_result,
                **val_result,
            }
            self.history.append(record)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train={train_result['loss']:.6f} | "
                f"val={val_result['validation_loss']:.6f} | "
                f"motion={train_result['motion']:.6f} | "
                f"delta={train_result['delta']:.6f} | "
                f"corrected={train_result['corrected_samples']} | "
                f"adaptive_threshold={train_result['adaptive_threshold']}"
            )

            if val_result["validation_loss"] < best_val:
                best_val = val_result["validation_loss"]
                torch.save(self.model.state_dict(), model_path)
                torch.save(
                    {
                        "pipeline": self.pipeline,
                        "epoch": epoch + 1,
                        "best_validation_loss": best_val,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "model_config": self.model.config.to_dict(),
                        "fixed_interval": self.fixed_interval,
                        "adaptive_threshold": self.adaptive_threshold,
                        "validation_p90_drift": val_result["validation_p90_drift"],
                    },
                    checkpoint_path,
                )
                print("  Best V3 checkpoint updated.")

            with history_path.open("w", encoding="utf-8") as file:
                json.dump(self.history, file, indent=2)

        print(f"Training complete. Best validation loss: {best_val:.6f}")
        print("Model:", model_path)
        print("History:", history_path)
        return self.history
