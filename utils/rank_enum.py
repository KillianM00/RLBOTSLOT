"""
RankTier enum mapping between internal tier indices and ballchasing.com rank strings.
"""
from enum import IntEnum


class RankTier(IntEnum):
    BRONZE          = 0
    SILVER          = 1
    GOLD            = 2
    PLATINUM        = 3
    DIAMOND         = 4
    CHAMPION        = 5
    GRAND_CHAMPION  = 6
    SUPERSONIC_LEGEND = 7

    @classmethod
    def from_string(cls, s: str) -> "RankTier":
        """Parse a ballchasing-style rank string, e.g. 'silver-1' -> SILVER."""
        mapping = {
            "bronze": cls.BRONZE,
            "silver": cls.SILVER,
            "gold": cls.GOLD,
            "platinum": cls.PLATINUM,
            "diamond": cls.DIAMOND,
            "champion": cls.CHAMPION,
            "grand-champion": cls.GRAND_CHAMPION,
            "supersonic-legend": cls.SUPERSONIC_LEGEND,
        }
        base = s.split("-")[0].lower()
        if base not in mapping:
            raise ValueError(f"Unknown rank string: {s!r}")
        return mapping[base]

    def to_ballchasing_min(self) -> str:
        """Return the ballchasing min_rank query string for this tier."""
        _map = {
            RankTier.BRONZE:           "bronze-1",
            RankTier.SILVER:           "silver-1",
            RankTier.GOLD:             "gold-1",
            RankTier.PLATINUM:         "platinum-1",
            RankTier.DIAMOND:          "diamond-1",
            RankTier.CHAMPION:         "champion-1",
            RankTier.GRAND_CHAMPION:   "grand-champion-1",
            RankTier.SUPERSONIC_LEGEND:"supersonic-legend",
        }
        return _map[self]

    def to_ballchasing_max(self) -> str:
        """Return the ballchasing max_rank query string for this tier."""
        _map = {
            RankTier.BRONZE:           "bronze-3",
            RankTier.SILVER:           "silver-3",
            RankTier.GOLD:             "gold-3",
            RankTier.PLATINUM:         "platinum-3",
            RankTier.DIAMOND:          "diamond-3",
            RankTier.CHAMPION:         "champion-3",
            RankTier.GRAND_CHAMPION:   "grand-champion-3",
            RankTier.SUPERSONIC_LEGEND:"supersonic-legend",
        }
        return _map[self]

    def clamped_above(self, n_tiers: int = 2) -> "RankTier":
        """Return self + n_tiers, clamped to SUPERSONIC_LEGEND."""
        return RankTier(min(int(self) + n_tiers, int(RankTier.SUPERSONIC_LEGEND)))
