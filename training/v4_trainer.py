from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.v4_loss import balanced_motion_loss, per_sample_drift_error
from v3.device import resolve_accelerator


class V4Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        *,
        pipeline: str,
        output_folder: str = "models_v4_64",
        results_folder: str = "results_v4_64",
        batch_size: int = 16,
        learning_rate: float = 2e-4,
        fixed_interval: int = 8,
        warmup_epochs: int = 3,
        device: str = "auto",
        num_workers: int = 2,
        patience: int = 8,
        min_delta: float = 1e-5,
    ):
        if pipeline not in {"baseline", "fixed_interval", "adaptive"}:
            raise ValueError("pipeline must be baseline, fixed_interval, or adaptive")

        self.model = model
        self.pipeline = pipeline
        self.output_folder = Path(output_folder)
        self.results_folder = Path(results_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.runtime = resolve_accelerator(device)
        self.device = self.runtime.device
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )

        drop_last = self.runtime.is_xla
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.runtime.kind == "cuda",
            drop_last=drop_last,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.runtime.kind == "cuda",
            drop_last=drop_last,
        )

        self.fixed_interval = int(fixed_interval)
        self.warmup_epochs = int(warmup_epochs)
        self.adaptive_threshold: float | None = None
        self.history = []
        self.patience = max(1, int(patience))
        self.min_delta = float(min_delta)

        self.drift_sample_every = 10 if self.runtime.is_xla else 5

        print("Accelerator:", self.runtime.label)
        print("Torch device:", self.device)
        print("XLA enabled:", self.runtime.is_xla)
        print("Batch size:", batch_size)
        print("Early stopping patience:", self.patience)

    @staticmethod
    def teacher_forcing_probability(epoch: int, epochs: int) -> float:
        # Slightly more anchoring than V3 early on, but still learns open-loop use.
        if epochs <= 1:
            return 0.15
        progress = epoch / (epochs - 1)
        return max(0.15, 0.60 * (1.0 - progress))

    def _move_batch(self, frames, actions):
        if self.runtime.is_xla:
            return frames, actions
        return (
            frames.to(self.device, non_blocking=self.runtime.kind == "cuda"),
            actions.to(self.device, non_blocking=self.runtime.kind == "cuda"),
        )

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

        if not corrections_enabled:
            return hidden, prediction.detach(), mask

        if self.pipeline == "fixed_interval":
            if (step + 1) % self.fixed_interval != 0:
                return hidden, prediction.detach(), mask
            mask = torch.ones(batch, dtype=torch.bool, device=self.device)

        elif self.pipeline == "adaptive":
            if self.adaptive_threshold is None:
                return hidden, prediction.detach(), mask
            mask = drift_error > float(self.adaptive_threshold)

        else:
            return hidden, prediction.detach(), mask

        encoded_real = self.model.encode(real_next)
        hidden = torch.where(mask.unsqueeze(1), encoded_real, hidden)
        next_input = torch.where(
            mask[:, None, None, None],
            real_next,
            prediction.detach(),
        )
        return hidden, next_input, mask

    def _metric_tensor_dict(self):
        return {
            key: torch.zeros((), device=self.device)
            for key in [
                "loss",
                "base",
                "motion",
                "foreground",
                "delta",
                "ghost",
                "edge",
                "motion_fraction",
                "ghost_fraction",
            ]
        }

    @staticmethod
    def _sample_p90(samples):
        if not samples:
            return 0.0
        values = np.concatenate(samples).astype(np.float64, copy=False)
        return float(np.percentile(values, 90))

    def _run_train_epoch(self, epoch: int, epochs: int):
        self.model.train()
        tf_prob = self.teacher_forcing_probability(epoch, epochs)
        corrections_enabled = epoch >= self.warmup_epochs

        totals = self._metric_tensor_dict()
        corrected_total = torch.zeros((), device=self.device)
        drift_sum = torch.zeros((), device=self.device)
        drift_count = torch.zeros((), device=self.device)
        drift_samples = []
        batches = 0

        loader = self.runtime.wrap_loader(self.train_loader)

        for batch_index, (frames, actions) in enumerate(loader):
            frames, actions = self._move_batch(frames, actions)
            batch_size, sequence_length = actions.shape
            hidden = self.model.init_hidden(batch_size, self.device)
            current_input = frames[:, 0]
            sequence_loss = torch.zeros((), device=self.device)
            batch_parts = {
                key: torch.zeros((), device=self.device)
                for key in totals
                if key != "loss"
            }
            batch_corrected = torch.zeros((), device=self.device)
            batch_drift_chunks = []

            self.optimizer.zero_grad(set_to_none=True)

            for step in range(sequence_length):
                real_current = frames[:, step]
                real_next = frames[:, step + 1]
                prediction, hidden = self.model(
                    current_input,
                    actions[:, step],
                    hidden,
                )

                parts = balanced_motion_loss(
                    prediction,
                    real_current,
                    real_next,
                )
                sequence_loss = sequence_loss + parts.total

                for key in batch_parts:
                    batch_parts[key] = batch_parts[key] + getattr(parts, key)

                with torch.no_grad():
                    drift = per_sample_drift_error(
                        prediction,
                        real_current,
                        real_next,
                    )
                    drift_sum = drift_sum + drift.sum()
                    drift_count = drift_count + drift.numel()
                    batch_drift_chunks.append(drift.detach())

                hidden, next_input, correction_mask = self._correct_state(
                    hidden=hidden,
                    prediction=prediction,
                    real_next=real_next,
                    drift_error=drift,
                    step=step,
                    corrections_enabled=corrections_enabled,
                )
                batch_corrected = batch_corrected + correction_mask.sum()

                if tf_prob > 0:
                    teacher_mask = (
                        torch.rand(batch_size, device=self.device) < tf_prob
                    )
                    use_real = correction_mask | ((~correction_mask) & teacher_mask)
                    current_input = torch.where(
                        use_real[:, None, None, None],
                        real_next,
                        next_input,
                    )
                else:
                    current_input = next_input

            sequence_loss = sequence_loss / sequence_length
            sequence_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                1.0,
                foreach=False,
            )
            self.runtime.optimizer_step(self.optimizer)

            totals["loss"] = totals["loss"] + sequence_loss.detach()
            for key in batch_parts:
                totals[key] = totals[key] + batch_parts[key] / sequence_length
            corrected_total = corrected_total + batch_corrected
            batches += 1

            if batch_index % self.drift_sample_every == 0:
                sampled = torch.cat(batch_drift_chunks).detach().cpu().numpy()
                drift_samples.append(sampled)

        if batches == 0:
            raise RuntimeError("No training batches were produced.")

        self.runtime.sync()

        names = list(totals.keys())
        metric_values = torch.stack(
            [totals[name] for name in names]
        ).detach().cpu().tolist()
        result = {
            name: float(value) / batches
            for name, value in zip(names, metric_values)
        }

        drift_pair = torch.stack([drift_sum, drift_count]).detach().cpu().tolist()
        mean_drift = float(drift_pair[0] / max(drift_pair[1], 1.0))
        p90 = self._sample_p90(drift_samples)
        corrected_samples = int(corrected_total.detach().cpu().item())

        if self.pipeline == "adaptive":
            self.adaptive_threshold = p90

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
        totals = self._metric_tensor_dict()
        drift_sum = torch.zeros((), device=self.device)
        drift_count = torch.zeros((), device=self.device)
        drift_samples = []
        batches = 0

        loader = self.runtime.wrap_loader(self.val_loader)

        # Common pure open-loop validation for all three pipelines.
        for batch_index, (frames, actions) in enumerate(loader):
            frames, actions = self._move_batch(frames, actions)
            batch_size, sequence_length = actions.shape
            hidden = self.model.init_hidden(batch_size, self.device)
            current_input = frames[:, 0]
            sequence_loss = torch.zeros((), device=self.device)
            batch_parts = {
                key: torch.zeros((), device=self.device)
                for key in totals
                if key != "loss"
            }
            batch_drifts = []

            for step in range(sequence_length):
                real_current = frames[:, step]
                real_next = frames[:, step + 1]
                prediction, hidden = self.model(
                    current_input,
                    actions[:, step],
                    hidden,
                )

                parts = balanced_motion_loss(
                    prediction,
                    real_current,
                    real_next,
                )
                sequence_loss = sequence_loss + parts.total
                for key in batch_parts:
                    batch_parts[key] = batch_parts[key] + getattr(parts, key)

                drift = per_sample_drift_error(
                    prediction,
                    real_current,
                    real_next,
                )
                drift_sum = drift_sum + drift.sum()
                drift_count = drift_count + drift.numel()
                batch_drifts.append(drift.detach())
                current_input = prediction

            totals["loss"] = totals["loss"] + sequence_loss / sequence_length
            for key in batch_parts:
                totals[key] = totals[key] + batch_parts[key] / sequence_length
            batches += 1

            if batch_index % self.drift_sample_every == 0:
                sampled = torch.cat(batch_drifts).detach().cpu().numpy()
                drift_samples.append(sampled)

        self.runtime.sync()

        names = list(totals.keys())
        metric_values = torch.stack(
            [totals[name] for name in names]
        ).detach().cpu().tolist()
        result = {
            f"validation_{name}": float(value) / max(batches, 1)
            for name, value in zip(names, metric_values)
        }

        drift_pair = torch.stack([drift_sum, drift_count]).detach().cpu().tolist()
        result["validation_mean_drift"] = float(
            drift_pair[0] / max(drift_pair[1], 1.0)
        )
        result["validation_p90_drift"] = self._sample_p90(drift_samples)
        return result

    def _save_best(self, model_path, checkpoint_path, epoch, best_val, val_result):
        self.runtime.sync()
        self.runtime.save(self.model.state_dict(), model_path)
        self.runtime.save(
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
                "training_accelerator": self.runtime.kind,
                "v4_loss": {
                    "motion_weight": 1.5,
                    "foreground_weight": 0.5,
                    "delta_weight": 0.75,
                    "ghost_weight": 2.0,
                    "edge_weight": 0.10,
                },
            },
            checkpoint_path,
        )

    def train(self, epochs: int):
        best_val = math.inf
        epochs_without_improvement = 0

        model_path = self.output_folder / f"best_{self.pipeline}_model.pt"
        checkpoint_path = self.output_folder / f"best_{self.pipeline}_checkpoint.pt"
        history_path = self.results_folder / f"{self.pipeline}_history.json"

        for epoch in range(int(epochs)):
            train_result = self._run_train_epoch(epoch, epochs)
            val_result = self.validate()

            record = {
                "epoch": epoch + 1,
                "accelerator": self.runtime.kind,
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
                f"ghost={train_result['ghost']:.6f} | "
                f"corrected={train_result['corrected_samples']} | "
                f"adaptive_threshold={train_result['adaptive_threshold']}"
            )

            current_val = val_result["validation_loss"]
            if current_val < best_val - self.min_delta:
                best_val = current_val
                epochs_without_improvement = 0
                self._save_best(
                    model_path,
                    checkpoint_path,
                    epoch,
                    best_val,
                    val_result,
                )
                print("  Best V4 checkpoint updated.")
            else:
                epochs_without_improvement += 1
                print(
                    f"  No validation improvement: "
                    f"{epochs_without_improvement}/{self.patience}"
                )

            with history_path.open("w", encoding="utf-8") as file:
                json.dump(self.history, file, indent=2)

            if epochs_without_improvement >= self.patience:
                print(
                    f"Early stopping at epoch {epoch + 1}. "
                    f"Best validation loss: {best_val:.6f}"
                )
                break

        self.runtime.sync()
        print(f"Training complete. Best validation loss: {best_val:.6f}")
        print("Model:", model_path)
        print("History:", history_path)
        return self.history
