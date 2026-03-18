"""
BallchasingFetcher: asynchronous replay downloader from ballchasing.com.

Key design decisions:
  - Uses aiohttp for non-blocking HTTP requests
  - Maintains a SQLite index (replay_index.db) to avoid re-downloading
  - Only fetches replays at (or slightly above) the bot's current rank tier
  - Rate-limited to cfg.REPLAY_RATE_LIMIT_RPS requests per second
  - Runs in a background thread managed by AsyncReplayFetcher (optimizations/)
"""
from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import List, Optional

import config as cfg
from utils.rank_enum import RankTier
from replay_system.rank_tracker import RankTracker

logger = logging.getLogger(__name__)


class BallchasingFetcher:
    """
    Downloads Rocket League replays from ballchasing.com filtered by rank.
    """

    BASE_URL = "https://ballchasing.com/api"

    def __init__(
        self,
        api_key: str = "",
        rank_tracker: Optional[RankTracker] = None,
        output_dir: Path = cfg.REPLAY_CACHE_DIR,
        db_path: Path    = cfg.REPLAY_DB_PATH,
        rate_limit_rps: float = cfg.REPLAY_RATE_LIMIT_RPS,
    ) -> None:
        self.api_key       = api_key or os.getenv("BALLCHASING_API_KEY", "")
        self.rank_tracker  = rank_tracker or RankTracker()
        self.output_dir    = Path(output_dir)
        self.db_path       = Path(db_path)
        self.rate_limit    = rate_limit_rps
        self._last_request = 0.0

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS replays (
                    replay_id TEXT PRIMARY KEY,
                    rank_tier INTEGER,
                    downloaded_at REAL,
                    file_path TEXT
                )
            """)
            conn.commit()

    def _is_downloaded(self, replay_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM replays WHERE replay_id = ?", (replay_id,)
            ).fetchone()
        return row is not None

    def _record_download(self, replay_id: str, tier: int, file_path: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO replays VALUES (?, ?, ?, ?)",
                (replay_id, tier, time.time(), file_path),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Rank → ballchasing filter
    # ------------------------------------------------------------------

    def _build_filters(self) -> dict:
        tier     = self.rank_tracker.current_tier
        max_tier = tier.clamped_above(cfg.MAX_RANK_GAP)
        return {
            "min-rank": tier.to_ballchasing_min(),
            "max-rank": max_tier.to_ballchasing_max(),
            "playlist": "ranked-duels",   # 1v1 for cleaner learning signal
            "count": cfg.REPLAY_BATCH_SIZE,
        }

    # ------------------------------------------------------------------
    # Async fetch
    # ------------------------------------------------------------------

    async def fetch_batch(self, count: int = cfg.REPLAY_BATCH_SIZE) -> List[Path]:
        """
        Download up to `count` replays matching the current rank.
        Returns list of local .replay file paths.
        """
        try:
            import aiohttp  # type: ignore
        except ImportError:
            logger.warning("aiohttp not installed — replay fetching disabled.")
            return []

        if not self.api_key:
            logger.warning("No BALLCHASING_API_KEY set — replay fetching disabled.")
            return []

        headers = {"Authorization": self.api_key}
        params  = self._build_filters()
        params["count"] = count

        downloaded: List[Path] = []

        async with aiohttp.ClientSession(headers=headers) as session:
            # Step 1: List replays
            try:
                await self._rate_limit_wait()
                async with session.get(f"{self.BASE_URL}/replays", params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            except Exception as e:
                logger.error("Failed to list replays: %s", e)
                return []

            replays = data.get("list", [])
            logger.info("Ballchasing: found %d replays at rank %s",
                        len(replays), self.rank_tracker.current_tier.name)

            # Step 2: Download each replay file
            for entry in replays:
                rid = entry.get("id", "")
                if not rid or self._is_downloaded(rid):
                    continue

                out_path = self.output_dir / f"{rid}.replay"
                try:
                    await self._rate_limit_wait()
                    dl_url = f"{self.BASE_URL}/replays/{rid}/file"
                    async with session.get(dl_url) as dl_resp:
                        dl_resp.raise_for_status()
                        content = await dl_resp.read()
                    out_path.write_bytes(content)
                    self._record_download(
                        rid, int(self.rank_tracker.current_tier), str(out_path)
                    )
                    downloaded.append(out_path)
                    logger.debug("Downloaded replay %s", rid)
                except Exception as e:
                    logger.warning("Failed to download replay %s: %s", rid, e)

        logger.info("Fetched %d new replays", len(downloaded))
        return downloaded

    async def _rate_limit_wait(self) -> None:
        """Enforce cfg.REPLAY_RATE_LIMIT_RPS by sleeping if needed."""
        min_interval = 1.0 / self.rate_limit
        elapsed      = time.monotonic() - self._last_request
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_request = time.monotonic()

    # ------------------------------------------------------------------
    # Sync wrapper (for use outside async context)
    # ------------------------------------------------------------------

    def fetch_batch_sync(self, count: int = cfg.REPLAY_BATCH_SIZE) -> List[Path]:
        return asyncio.run(self.fetch_batch(count))
