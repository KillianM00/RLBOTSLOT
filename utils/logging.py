"""
Unified logger that writes to console and optionally to Weights & Biases.
"""
import logging
import time
from typing import Any, Dict, Optional

_CONSOLE_FMT = "[%(asctime)s | %(levelname)s] %(message)s"
_DATE_FMT    = "%H:%M:%S"


class BotLogger:
    """
    Wraps Python's logging module and optionally a wandb run.
    Usage:
        logger = BotLogger(use_wandb=True, project="rlbot-professor")
        logger.log({"reward": 1.2, "goal_rate": 0.4}, step=1000)
        logger.info("Training started")
    """

    def __init__(
        self,
        use_wandb: bool = False,
        project: str = "rlbot-professor",
        entity: str = "",
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        logging.basicConfig(format=_CONSOLE_FMT, datefmt=_DATE_FMT, level=logging.INFO)
        self._log = logging.getLogger("BotLogger")
        self._wandb_run = None
        self._start_time = time.time()

        if use_wandb:
            try:
                import wandb  # type: ignore
                self._wandb_run = wandb.init(
                    project=project,
                    entity=entity or None,
                    config=config,
                    name=name or f"run_{int(self._start_time)}",
                )
                self._log.info("WandB run initialised: %s", self._wandb_run.url)
            except Exception as exc:
                self._log.warning("WandB initialisation failed: %s", exc)

    # ------------------------------------------------------------------
    # Standard log levels
    # ------------------------------------------------------------------

    def info(self, msg: str, *args: Any) -> None:
        self._log.info(msg, *args)

    def warning(self, msg: str, *args: Any) -> None:
        self._log.warning(msg, *args)

    def error(self, msg: str, *args: Any) -> None:
        self._log.error(msg, *args)

    def debug(self, msg: str, *args: Any) -> None:
        self._log.debug(msg, *args)

    # ------------------------------------------------------------------
    # Metrics logging
    # ------------------------------------------------------------------

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log a dict of metrics to console and WandB."""
        msg_parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in metrics.items()]
        self._log.info("[step=%s] %s", step, "  ".join(msg_parts))
        if self._wandb_run is not None:
            try:
                self._wandb_run.log(metrics, step=step)
            except Exception:
                pass

    def finish(self) -> None:
        """Finalise WandB run (call at end of training)."""
        if self._wandb_run is not None:
            self._wandb_run.finish()
