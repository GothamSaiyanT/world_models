from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class AcceleratorRuntime:
    """Small device abstraction for CPU, CUDA GPU, and single-device TPU/XLA."""

    kind: str
    device: torch.device
    label: str
    is_xla: bool = False
    torch_xla: Any = None
    xm: Any = None
    parallel_loader: Any = None

    def wrap_loader(self, loader):
        if self.is_xla:
            return self.parallel_loader.MpDeviceLoader(loader, self.device)
        return loader

    def optimizer_step(self, optimizer):
        if self.is_xla:
            # MpDeviceLoader handles the execution barrier as batches advance.
            # optimizer_step performs the XLA-aware parameter update.
            self.xm.optimizer_step(optimizer)
        else:
            optimizer.step()

    def sync(self):
        if self.is_xla:
            # Current PyTorch/XLA exposes torch_xla.sync(). Keep a fallback for
            # older Kaggle images that still rely on xm.mark_step().
            if hasattr(self.torch_xla, "sync"):
                self.torch_xla.sync()
            elif hasattr(self.xm, "mark_step"):
                self.xm.mark_step()

    def save(self, obj, path):
        path = str(Path(path))
        if self.is_xla:
            # xm.save moves XLA tensors to CPU-compatible storage, so these
            # checkpoints can later be loaded on CPU/GPU for the dashboard.
            self.xm.save(obj, path)
        else:
            torch.save(obj, path)


def _tpu_runtime(required: bool = False) -> AcceleratorRuntime | None:
    """Return a TPU runtime when PyTorch/XLA can see a TPU."""

    if required:
        # Must be set before XLA runtime initialization. setdefault avoids
        # overriding a valid Kaggle PJRT configuration.
        os.environ.setdefault("PJRT_DEVICE", "TPU")

    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.runtime as xr

        device_type = xr.device_type()
        if str(device_type).upper() != "TPU":
            if required:
                raise RuntimeError(
                    f"PyTorch/XLA is installed, but the active XLA device is "
                    f"{device_type!r}, not TPU. Select TPU in Kaggle settings."
                )
            return None

        if hasattr(torch_xla, "device"):
            device = torch_xla.device()
        else:  # Compatibility with older Kaggle XLA images.
            device = xm.xla_device()

        try:
            real_devices = torch_xla.real_devices()
            hardware = ", ".join(real_devices[:4])
            if len(real_devices) > 4:
                hardware += ", ..."
        except Exception:
            hardware = "TPU"

        return AcceleratorRuntime(
            kind="tpu",
            device=device,
            label=f"TPU/XLA ({hardware})",
            is_xla=True,
            torch_xla=torch_xla,
            xm=xm,
            parallel_loader=pl,
        )

    except Exception:
        if required:
            raise
        return None


def resolve_accelerator(requested: str = "auto") -> AcceleratorRuntime:
    """
    Resolve ``auto``, ``cuda``, ``tpu`` or ``cpu``.

    Auto intentionally prefers a normal CUDA GPU when one is available because
    standard PyTorch CUDA does not require XLA graph compilation. If CUDA is not
    present, it checks for an attached TPU before falling back to CPU.
    """

    requested = str(requested).lower().strip()
    if requested not in {"auto", "cuda", "tpu", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, tpu, cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Select a GPU accelerator in Kaggle or use --device tpu/cpu."
            )
        index = torch.cuda.current_device()
        return AcceleratorRuntime(
            kind="cuda",
            device=torch.device(f"cuda:{index}"),
            label=f"CUDA GPU ({torch.cuda.get_device_name(index)})",
        )

    if requested == "tpu":
        return _tpu_runtime(required=True)

    if requested == "cpu":
        return AcceleratorRuntime(
            kind="cpu",
            device=torch.device("cpu"),
            label="CPU",
        )

    # auto: CUDA first, then TPU/XLA, then CPU.
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        return AcceleratorRuntime(
            kind="cuda",
            device=torch.device(f"cuda:{index}"),
            label=f"CUDA GPU ({torch.cuda.get_device_name(index)})",
        )

    tpu = _tpu_runtime(required=False)
    if tpu is not None:
        return tpu

    return AcceleratorRuntime(
        kind="cpu",
        device=torch.device("cpu"),
        label="CPU",
    )
