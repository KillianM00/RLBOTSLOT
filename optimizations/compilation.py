"""
torch.compile wrapper with safe fallback.

Applies compile with the configured mode.  Falls back gracefully on:
  - CPU-only systems (torch.compile still works but with less gain)
  - PyTorch < 2.0 where compile is unavailable
  - fullgraph=True failures (reduces to partial graph compilation)
"""
from __future__ import annotations

import logging
import torch.nn as nn

import config as cfg

logger = logging.getLogger(__name__)


def compile_model(
    model: nn.Module,
    mode: str = cfg.COMPILE_MODE,
    fullgraph: bool = False,
) -> nn.Module:
    """
    Wrap a model with torch.compile.

    Parameters
    ----------
    model     : PyTorch Module to compile
    mode      : "reduce-overhead" (default), "max-autotune", or "default"
    fullgraph : If True, fail loudly on graph breaks; if False, compile subgraphs

    Returns
    -------
    Compiled module (or original if compilation is unavailable).
    """
    try:
        import torch  # type: ignore
        if not hasattr(torch, "compile"):
            logger.warning("torch.compile unavailable (torch < 2.0). Skipping.")
            return model

        compiled = torch.compile(model, mode=mode, fullgraph=fullgraph)
        logger.info("Model compiled with torch.compile(mode=%r, fullgraph=%s)",
                    mode, fullgraph)
        return compiled
    except Exception as e:
        logger.warning("torch.compile failed (%s). Using uncompiled model.", e)
        return model
