"""
Low-level math helpers used across reward functions and obs builders.
All functions operate on plain numpy arrays for speed.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Vector ops
# ---------------------------------------------------------------------------

def vec3_norm(v: np.ndarray) -> float:
    """Euclidean norm of a 3-vector."""
    return float(np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2))


def vec3_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = vec3_norm(v)
    return v / (n + eps)


def vec3_dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def vec3_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ], dtype=np.float32)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in radians between two 3-vectors."""
    na = vec3_normalize(a)
    nb = vec3_normalize(b)
    cos_theta = np.clip(vec3_dot(na, nb), -1.0, 1.0)
    return float(np.arccos(cos_theta))


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def rotation_matrix_from_euler(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """
    Returns a 3x3 rotation matrix from Rocket League Euler angles (radians).
    Convention: yaw around Z, pitch around Y', roll around X''.
    """
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    cr, sr = np.cos(roll),  np.sin(roll)

    R = np.array([
        [cp * cy,  cy * sp * sr - cr * sy,  sr * sy + cr * cy * sp],
        [cp * sy,  cr * cy + sp * sr * sy,  cr * sp * sy - cy * sr],
        [-sp,      cp * sr,                 cp * cr               ],
    ], dtype=np.float32)
    return R


def forward_vec(pitch: float, yaw: float) -> np.ndarray:
    """
    Returns unit forward vector from pitch and yaw (radians).
    Equivalent to first column of rotation matrix when roll=0.
    """
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    return np.array([cp * cy, cp * sy, sp], dtype=np.float32)


# ---------------------------------------------------------------------------
# Game-specific helpers
# ---------------------------------------------------------------------------

FIELD_WIDTH  = 4096.0   # uu, half-width (X axis)
FIELD_LENGTH = 5120.0   # uu, half-length (Y axis)
CEILING      = 2044.0   # uu
GOAL_WIDTH   = 1786.0   # uu, half-width of goal
GOAL_HEIGHT  = 642.78   # uu
MAX_CAR_SPEED = 2300.0  # uu/s
MAX_BALL_SPEED = 6000.0 # uu/s
SUPERSONIC_THRESHOLD = 2200.0  # uu/s

BLUE_GOAL_POS  = np.array([0.0, -FIELD_LENGTH, 0.0], dtype=np.float32)
ORANGE_GOAL_POS = np.array([0.0,  FIELD_LENGTH, 0.0], dtype=np.float32)


def is_on_ground(car_pos: np.ndarray, threshold: float = 20.0) -> bool:
    """Returns True if car z-position is near the ground."""
    return bool(car_pos[2] < threshold)


def is_in_air(car_pos: np.ndarray, threshold: float = 20.0) -> bool:
    return not is_on_ground(car_pos, threshold)


def ball_in_attacking_half(ball_pos: np.ndarray, team_is_blue: bool) -> bool:
    """True if ball is in opponent's half."""
    if team_is_blue:
        return bool(ball_pos[1] > 0)
    return bool(ball_pos[1] < 0)


def goal_pos_for_team(team_is_blue: bool) -> np.ndarray:
    """Return opponent's goal centre for attacker perspective."""
    return ORANGE_GOAL_POS if team_is_blue else BLUE_GOAL_POS


def own_goal_pos(team_is_blue: bool) -> np.ndarray:
    return BLUE_GOAL_POS if team_is_blue else ORANGE_GOAL_POS
