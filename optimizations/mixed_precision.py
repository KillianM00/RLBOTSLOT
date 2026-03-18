"""
AMPContext: automatic mixed precision with BF16/FP16/FP32 auto-detection.

Decision tree:
  - CUDA available + SM >= 8.0 (Ampere/Ada)  → BF16 (no GradScaler needed)
  - CUDA available + SM < 8.0 (Turing/Volta) → FP16 (GradScaler required)
  - DirectML (AMD on Windows)                 → FP32 (AMP not supported by DirectML)
  - CPU only                                  → FP32 (AMP disabled)
"""
from __future__ import annotations

import contextlib
import logging
from typing import Optional

import torch
import torch.nn as nn

import config as cfg

logger = logging.getLogger(__name__)


class AMPContext:
    """
    Manages autocast context and gradient scaling for mixed precision training.

    Usage:
        amp = AMPContext()
        with amp.autocast():
            loss = model(obs)
        amp.scale_and_step(loss, optimizer)
    """

    def __init__(self, enabled: bool = cfg.USE_AMP) -> None:
        self.enabled = enabled and torch.cuda.is_available()
        self.dtype   = self._detect_dtype()
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"
        # GradScaler only needed for FP16 (BF16 has sufficient dynamic range)
        self.scaler: Optional[torch.amp.GradScaler] = None
        if self.enabled and self.dtype == torch.float16:
            self.scaler = torch.amp.GradScaler(device=self.device)
        self._log_config()

    # ------------------------------------------------------------------
    def _detect_dtype(self) -> torch.dtype:
        if not self.enabled:
            return torch.float32
        if not torch.cuda.is_available():
            return torch.float32
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:   # Ampere (RTX 30xx) and above
            logger.info("GPU SM %d >= 8: using BF16 precision.", major)
            return torch.bfloat16
        else:
            logger.info("GPU SM %d < 8: using FP16 precision.", major)
            return torch.float16

    def _log_config(self) -> None:
        dtype_name = {
            torch.float32: "FP32 (AMP disabled)",
            torch.float16: "FP16 (with GradScaler)",
            torch.bfloat16: "BF16 (no GradScaler)",
        }.get(self.dtype, str(self.dtype))
        logger.info("AMPContext: %s", dtype_name)

    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def autocast(self):
        """Context manager for mixed precision forward passes."""
        if self.enabled:
            with torch.amp.autocast(device_type=self.device, dtype=self.dtype):
                yield
        else:
            yield

    # ------------------------------------------------------------------
    def scale_and_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        max_norm: float = cfg.MAX_GRAD_NORM,
        model: Optional[nn.Module] = None,
    ) -> None:
        """
        Scale loss (if FP16), clip gradients, and step the optimizer.
        """
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            if model is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if model is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
