"""
ProfilerContext: thin wrapper around torch.profiler for hot-path analysis.

Usage:
    ctx = ProfilerContext(active=True, output_dir="./prof_traces")
    with ctx.profile("debate_engine"):
        verdict = debate_engine(obs_seq, action_seq, concept_ids)
    ctx.export()   # writes Chrome trace JSON

Set active=False in production to zero-overhead no-op.
"""
from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ProfilerContext:
    """
    Wraps torch.profiler for measuring hot-path performance.
    Falls back to a no-op context if torch.profiler is unavailable.
    """

    def __init__(
        self,
        active: bool       = False,
        output_dir: str    = "./prof_traces",
        warmup_steps: int  = 10,
        active_steps: int  = 20,
    ) -> None:
        self.active       = active
        self.output_dir   = Path(output_dir)
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self._prof        = None
        self._timings: dict = {}

        if active:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._init_profiler()

    def _init_profiler(self) -> None:
        try:
            import torch
            from torch.profiler import (  # type: ignore
                profile, record_function, ProfilerActivity, schedule
            )
            self._prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=2, warmup=self.warmup_steps, active=self.active_steps),
                on_trace_ready=self._on_trace_ready,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            self._prof.__enter__()
            logger.info("ProfilerContext: torch.profiler active")
        except Exception as e:
            logger.warning("torch.profiler unavailable (%s). Using timing fallback.", e)
            self._prof = None

    def _on_trace_ready(self, prof) -> None:
        trace_path = self.output_dir / f"trace_{int(time.time())}.json"
        prof.export_chrome_trace(str(trace_path))
        logger.info("Profiler trace written to %s", trace_path)

    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def profile(self, name: str):
        """Profile a named code block."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._timings[name] = self._timings.get(name, []) + [elapsed]

    def step(self) -> None:
        """Call once per training iteration to advance profiler schedule."""
        if self._prof is not None:
            try:
                self._prof.step()
            except Exception:
                pass

    def summary(self) -> dict:
        """Return mean timings per section."""
        return {
            k: sum(v) / len(v) * 1000  # ms
            for k, v in self._timings.items()
        }

    def export(self) -> None:
        """Stop the profiler and export final trace."""
        if self._prof is not None:
            try:
                self._prof.__exit__(None, None, None)
            except Exception:
                pass
