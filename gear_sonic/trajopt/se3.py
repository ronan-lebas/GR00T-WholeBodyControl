"""Minimal SE(3) helpers shared by the trajectory optimizer.

``rot_exp`` / ``rot_log`` / ``Pose.quat_wxyz`` sit in the optimizer's inner loop (they run
once per body per residual evaluation, and a finite-difference Jacobian needs tens of
thousands of those), so they use closed-form numpy rather than scipy's ``Rotation``, which
costs ~30 us per call. scipy is kept as the fallback near the pi singularity, where the
closed form loses precision.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


@dataclass
class Pose:
    """Rigid pose: world position + rotation matrix."""

    p: np.ndarray
    Rm: np.ndarray

    @staticmethod
    def identity() -> "Pose":
        return Pose(np.zeros(3), np.eye(3))

    @staticmethod
    def from_xyz_yaw(x: float, y: float, z: float, yaw: float) -> "Pose":
        return Pose(np.array([x, y, z], dtype=float), R.from_euler("z", yaw).as_matrix())

    @staticmethod
    def from_quat_wxyz(p, quat_wxyz) -> "Pose":
        q = np.asarray(quat_wxyz, dtype=float)
        return Pose(np.asarray(p, dtype=float), R.from_quat(q[[1, 2, 3, 0]]).as_matrix())

    def quat_wxyz(self) -> np.ndarray:
        return mat_to_quat(self.Rm)

    def yaw(self) -> float:
        return float(R.from_matrix(self.Rm).as_euler("zyx")[0])

    def apply(self, v: np.ndarray) -> np.ndarray:
        """Transform a point (or an (N, 3) stack of points) from local to world."""
        v = np.asarray(v, dtype=float)
        return v @ self.Rm.T + self.p

    def rotate(self, v: np.ndarray) -> np.ndarray:
        """Rotate a direction (or an (N, 3) stack) from local to world."""
        return np.asarray(v, dtype=float) @ self.Rm.T

    def inv(self) -> "Pose":
        return Pose(-self.Rm.T @ self.p, self.Rm.T)

    def __mul__(self, other: "Pose") -> "Pose":
        return Pose(self.Rm @ other.p + self.p, self.Rm @ other.Rm)

    def copy(self) -> "Pose":
        return Pose(self.p.copy(), self.Rm.copy())


def mat_to_quat(Rm: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w, x, y, z), via MuJoCo's C implementation."""
    import mujoco  # local import: se3 stays usable without mujoco

    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(Rm, dtype=float).reshape(9))
    return q


def rot_log(Rm: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (closed form; scipy near the pi singularity)."""
    c = (np.trace(Rm) - 1.0) * 0.5
    if c < -0.9:  # angle > ~154 deg: the closed form gets ill-conditioned
        return R.from_matrix(Rm).as_rotvec()
    theta = np.arccos(min(1.0, max(-1.0, c)))
    v = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    if theta < 1e-7:
        return 0.5 * v
    return (theta / (2.0 * np.sin(theta))) * v


def rot_exp(rv: np.ndarray) -> np.ndarray:
    """Rotation vector -> rotation matrix (Rodrigues)."""
    rv = np.asarray(rv, dtype=float)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-9:
        return np.eye(3) + _skew(rv)
    K = _skew(rv / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def rot_error(Rm: np.ndarray, R_ref: np.ndarray) -> np.ndarray:
    """Rotation-vector error of ``Rm`` w.r.t. ``R_ref`` (zero iff equal)."""
    return rot_log(R_ref.T @ Rm)


def interp_pose(a: Pose, b: Pose, s: float) -> Pose:
    """Linear position / geodesic rotation interpolation, ``s`` in [0, 1]."""
    dr = rot_log(a.Rm.T @ b.Rm)
    return Pose(a.p + s * (b.p - a.p), a.Rm @ rot_exp(s * dr))
