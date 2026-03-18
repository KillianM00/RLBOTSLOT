"""
Parses research_program.md to extract focus areas, frozen params,
parameter bounds, and stop conditions for the autoresearch system.
"""
from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

logger = logging.getLogger("autoresearch.research_program")

_DEFAULT_PATH = Path(__file__).parent / "research_program.md"


class ResearchProgram:
    """Parsed research agenda from research_program.md."""

    def __init__(self, path: Path = _DEFAULT_PATH) -> None:
        self._path = path
        self._focus_areas: List[str] = []
        self._frozen_params: Set[str] = set()
        self._param_bounds: Dict[str, Tuple[float, float]] = {}
        self._max_experiments: int = 100
        self._parse()

    def _parse(self) -> None:
        if not self._path.exists():
            logger.warning("Research program not found: %s", self._path)
            return

        text = self._path.read_text(encoding="utf-8")
        section = None

        for line in text.splitlines():
            stripped = line.strip()

            # Detect section headers
            if stripped.startswith("## "):
                header = stripped[3:].strip().lower()
                if "focus" in header:
                    section = "focus"
                elif "frozen" in header:
                    section = "frozen"
                elif "bound" in header:
                    section = "bounds"
                elif "stop" in header:
                    section = "stop"
                else:
                    section = None
                continue

            if not stripped or stripped.startswith("#"):
                continue

            if section == "focus" and stripped.startswith("- "):
                self._focus_areas.append(stripped[2:].strip())

            elif section == "frozen" and stripped.startswith("- "):
                self._frozen_params.add(stripped[2:].strip())

            elif section == "bounds" and stripped.startswith("- "):
                # Format: "- PARAM_NAME: [min, max]"
                match = re.match(
                    r"-\s+(\w+)\s*:\s*\[([^,]+),\s*([^\]]+)\]",
                    stripped,
                )
                if match:
                    name = match.group(1)
                    lo = float(match.group(2))
                    hi = float(match.group(3))
                    self._param_bounds[name] = (lo, hi)

            elif section == "stop":
                match = re.match(r"-\s*max_experiments\s*:\s*(\d+)", stripped)
                if match:
                    self._max_experiments = int(match.group(1))

    @property
    def focus_areas(self) -> List[str]:
        return self._focus_areas

    @property
    def frozen_params(self) -> Set[str]:
        return self._frozen_params

    @property
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        return self._param_bounds

    @property
    def max_experiments(self) -> int:
        return self._max_experiments
