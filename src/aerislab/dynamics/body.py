"""
Rigid body dynamics with 6 degrees of freedom using quaternion representation.

All physical quantities use SI units:
- Position: meters [m]
- Velocity: meters per second [m/s]
- Angular velocity: radians per second [rad/s]
- Mass: kilograms [kg]
- Inertia: kilogram-meter-squared [kg·m²]
"""
from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as ScR

# Constants
QUATERNION_EPSILON = 1e-12
MIN_MASS = 1e-10  # Minimum mass to avoid division by zero


def quat_normalize(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Return unit quaternion (float64).

    Parameters
    ----------
    q : NDArray[np.float64]
        Quaternion in scalar-last format [x, y, z, w].

    Returns
    -------
    NDArray[np.float64]
        Normalized unit quaternion. Returns [0, 0, 0, 1] if input norm is zero.

    Notes
    -----
    Uses scalar-last convention: [qx, qy, qz, qw] where qw is the real part.
    """
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < QUATERNION_EPSILON:
        warnings.warn(
            "Zero-norm quaternion detected. Returning identity quaternion [0,0,0,1].",
            RuntimeWarning,
            stacklevel=2
        )
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def quat_derivative(q: NDArray[np.float64], omega: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute quaternion derivative from angular velocity.

    Uses the formula: qdot = 0.5 * q ⊗ [ω, 0] (scalar-last convention)

    Parameters
    ----------
    q : NDArray[np.float64]
        Unit quaternion [x, y, z, w] (4,)
    omega : NDArray[np.float64]
        Angular velocity in world frame [ωx, ωy, ωz] (3,) [rad/s]

    Returns
    -------
    NDArray[np.float64]
        Quaternion time derivative (4,) [rad/s]

    References
    ----------
    .. [1] Kuipers, J. B. (1999). Quaternions and Rotation Sequences.
           Princeton University Press.
    """
    qx, qy, qz, qw = q
    ox, oy, oz = omega
    return 0.5 * np.array([
        qw*ox + qy*oz - qz*oy,
        qw*oy - qx*oz + qz*ox,
        qw*oz + qx*oy - qy*ox,
        -qx*ox - qy*oy - qz*oz
    ], dtype=np.float64)


def quat_integrate_exponential_map(
    q: NDArray[np.float64],
    omega: NDArray[np.float64],
    dt: float
) -> NDArray[np.float64]:
    """
    Integrate quaternion using exponential map (more accurate than Euler).

    This method preserves quaternion unit norm better than simple Euler integration
    and is recommended for long simulations or large angular velocities.

    Parameters
    ----------
    q : NDArray[np.float64]
        Current unit quaternion (4,)
    omega : NDArray[np.float64]
        Angular velocity [rad/s] (3,)
    dt : float
        Time step [s]

    Returns
    -------
    NDArray[np.float64]
        Updated unit quaternion (4,)

    Notes
    -----
    Uses scipy's Rotation for robust implementation.
    """
    if np.linalg.norm(omega) < QUATERNION_EPSILON:
        return q  # No rotation

    # Create rotation from current quaternion
    R_current = ScR.from_quat(q)

    # Create incremental rotation from angular velocity
    angle = np.linalg.norm(omega) * dt
    if angle < QUATERNION_EPSILON:
        return q
    axis = omega / np.linalg.norm(omega)
    R_delta = ScR.from_rotvec(axis * angle)

    # Compose rotations
    R_new = R_delta * R_current
    return R_new.as_quat()


class RigidBody6DOF:
    """
    6-DoF rigid body in world frame with quaternion orientation.

    Coordinate Frames
    -----------------
    - World frame: Inertial reference frame (fixed)
    - Body frame: Principal axes frame attached to rigid body

    State Variables
    ---------------
    - p : NDArray[np.float64]
        Position of body origin in world frame [m] (3,)
    - q : NDArray[np.float64]
        Unit quaternion body->world (scalar-last [x,y,z,w]) (4,)
    - v : NDArray[np.float64]
        Linear velocity in world frame [m/s] (3,)
    - w : NDArray[np.float64]
        Angular velocity in world frame [rad/s] (3,)

    Properties
    ----------
    - mass : float
        Body mass [kg]
    - I_body : NDArray[np.float64]
        Inertia tensor in body principal frame [kg·m²] (3,3)

    Force/Torque Accumulators (cleared each step)
    ----------------------------------------------
    - f : NDArray[np.float64]
        Accumulated force in world frame [N] (3,)
    - tau : NDArray[np.float64]
        Accumulated torque in world frame [N·m] (3,)

    Notes
    -----
    Uses __slots__ to reduce memory overhead and improve cache performance.
    World inertia tensor: I_world = R(q) @ I_body @ R(q)^T
    """
    __slots__ = (
        "name", "p", "q", "v", "w",
        "mass", "I_body", "I_body_inv",
        "inv_mass", "radius",
        "f", "tau",
        "per_body_forces",
    )

    def __init__(
        self,
        name: str,
        mass: float,
        inertia_tensor_body: NDArray[np.float64],
        position: NDArray[np.float64],
        orientation: NDArray[np.float64],
        linear_velocity: NDArray[np.float64] | None = None,
        angular_velocity: NDArray[np.float64] | None = None,
        radius: float = 0.0,
    ) -> None:
        """
        Initialize a 6-DOF rigid body.

        Parameters
        ----------
        name : str
            Unique identifier for the body
        mass : float
            Body mass [kg]. Must be positive. Use large mass for quasi-static bodies.
        inertia_tensor_body : NDArray[np.float64]
            3x3 inertia tensor in body principal frame [kg·m²]
        position : NDArray[np.float64]
            Initial position in world frame [m] (3,)
        orientation : NDArray[np.float64]
            Initial orientation quaternion [x,y,z,w] (4,). Will be normalized.
        linear_velocity : NDArray[np.float64] | None
            Initial linear velocity [m/s] (3,). Defaults to zero.
        angular_velocity : NDArray[np.float64] | None
            Initial angular velocity [rad/s] (3,). Defaults to zero.
        radius : float
            Characteristic radius for visualization [m]. Optional.

        Raises
        ------
        ValueError
            If mass is negative or inertia tensor is not positive definite.
        """
        if mass < 0:
            raise ValueError(f"Mass must be non-negative, got {mass}")
        if mass < MIN_MASS:
            warnings.warn(
                f"Very small mass ({mass} kg) detected. Consider using a larger value.",
                RuntimeWarning, stacklevel=2
            )

        self.name = name
        self.mass = float(mass)
        self.inv_mass = 0.0 if self.mass < MIN_MASS else 1.0 / self.mass

        # Validate and store inertia tensor
        I = np.asarray(inertia_tensor_body, dtype=np.float64)
        if I.shape != (3, 3):
            raise ValueError(f"Inertia tensor must be 3x3, got shape {I.shape}")

        # Check if inertia is positive definite
        eigenvalues = np.linalg.eigvals(I)
        if np.any(eigenvalues <= 0):
            raise ValueError(
                f"Inertia tensor must be positive definite. Got eigenvalues: {eigenvalues}"
            )

        self.I_body = I.copy()
        try:
            self.I_body_inv = np.linalg.inv(self.I_body)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Cannot invert inertia tensor: {e}") from e

        # Initialize state
        self.p = np.asarray(position, dtype=np.float64).copy()
        self.q = quat_normalize(np.asarray(orientation, dtype=np.float64).copy())
        self.v = (np.zeros(3, dtype=np.float64) if linear_velocity is None
                  else np.asarray(linear_velocity, dtype=np.float64).copy())
        self.w = (np.zeros(3, dtype=np.float64) if angular_velocity is None
                  else np.asarray(angular_velocity, dtype=np.float64).copy())

        self.radius = float(radius)

        # Force/torque accumulators
        self.f = np.zeros(3, dtype=np.float64)
        self.tau = np.zeros(3, dtype=np.float64)

        # Per-body force list
        self.per_body_forces: list = []

    def clear_forces(self) -> None:
        """Reset force and torque accumulators to zero."""
        self.f.fill(0.0)
        self.tau.fill(0.0)

    def rotation_world(self) -> NDArray[np.float64]:
        """
        Get rotation matrix from body to world frame.

        Returns
        -------
        NDArray[np.float64]
            3x3 rotation matrix R such that v_world = R @ v_body
        """
        return ScR.from_quat(self.q).as_matrix()

    def inertia_world(self) -> NDArray[np.float64]:
        """
        Compute inertia tensor in world frame.

        Returns
        -------
        NDArray[np.float64]
            3x3 inertia tensor in world frame [kg·m²]

        Notes
        -----
        I_world = R @ I_body @ R^T where R is the rotation matrix.
        """
        R = self.rotation_world()
        return R @ self.I_body @ R.T

    def mass_matrix_world(self) -> NDArray[np.float64]:
        """
        Return block-diagonal generalized mass matrix.

        Returns
        -------
        NDArray[np.float64]
            6x6 mass matrix M = diag(m*I₃, I_world)

        Notes
        -----
        Used in KKT system assembly. Structure:
        [ m*I₃    0      ]
        [  0    I_world  ]
        """
        M = np.zeros((6, 6), dtype=np.float64)
        M[0:3, 0:3] = self.mass * np.eye(3)
        M[3:6, 3:6] = self.inertia_world()
        return M

    def inv_mass_matrix_world(self) -> NDArray[np.float64]:
        """
        Return block-diagonal inverse generalized mass matrix.

        Returns
        -------
        NDArray[np.float64]
            6x6 inverse mass matrix W = diag((1/m)*I₃, I_world⁻¹)

        Notes
        -----
        Used in KKT system for computing constraint forces efficiently.
        """
        W = np.zeros((6, 6), dtype=np.float64)
        W[0:3, 0:3] = self.inv_mass * np.eye(3)
        R = self.rotation_world()
        W[3:6, 3:6] = R @ self.I_body_inv @ R.T
        return W

    def apply_force(
        self,
        f: NDArray[np.float64],
        point_world: NDArray[np.float64] | None = None
    ) -> None:
        """
        Apply force to the body.

        Parameters
        ----------
        f : NDArray[np.float64]
            Force vector in world frame [N] (3,)
        point_world : NDArray[np.float64] | None
            Application point in world frame [m] (3,). If provided,
            generates torque τ = r × f where r = point_world - body_origin.
            If None, force is applied at center of mass (no torque).

        Notes
        -----
        Forces accumulate until clear_forces() is called.
        """
        f = np.asarray(f, dtype=np.float64)
        self.f += f
        if point_world is not None:
            r = np.asarray(point_world, dtype=np.float64) - self.p
            self.tau += np.cross(r, f)

    def apply_torque(self, tau: NDArray[np.float64]) -> None:
        """
        Apply torque to the body.

        Parameters
        ----------
        tau : NDArray[np.float64]
            Torque vector in world frame [N·m] (3,)
        """
        self.tau += np.asarray(tau, dtype=np.float64)

    def generalized_force(self) -> NDArray[np.float64]:
        """
        Return concatenated generalized force vector.

        Returns
        -------
        NDArray[np.float64]
            Generalized force [f; tau] (6,) where f is force [N]
            and tau is torque [N·m]
        """
        out = np.zeros(6, dtype=np.float64)
        out[:3] = self.f
        out[3:] = self.tau
        return out

    def integrate_semi_implicit(
        self,
        dt: float,
        a_lin: NDArray[np.float64],
        a_ang: NDArray[np.float64],
        use_exponential_map: bool = False
    ) -> None:
        """
        Semi-implicit (symplectic) Euler integration.

        Parameters
        ----------
        dt : float
            Time step [s]
        a_lin : NDArray[np.float64]
            Linear acceleration [m/s²] (3,)
        a_ang : NDArray[np.float64]
            Angular acceleration [rad/s²] (3,)
        use_exponential_map : bool
            If True, use exponential map for quaternion integration (more accurate).
            If False, use simple Euler method (faster but less accurate).

        Notes
        -----
        Integration order (symplectic):
        1. v_{n+1} = v_n + a_lin * dt
        2. w_{n+1} = w_n + a_ang * dt
        3. p_{n+1} = p_n + v_{n+1} * dt
        4. q_{n+1} = integrate_quaternion(q_n, w_{n+1}, dt)

        This ordering preserves energy better than explicit Euler.
        """
        # Update velocities first (momentum level)
        self.v += a_lin * dt
        self.w += a_ang * dt

        # Update position using new velocity
        self.p += self.v * dt

        # Update orientation
        if use_exponential_map:
            self.q = quat_integrate_exponential_map(self.q, self.w, dt)
        else:
            # Standard Euler method
            qdot = quat_derivative(self.q, self.w)
            self.q = quat_normalize(self.q + qdot * dt)

    def kinetic_energy(self) -> float:
        """
        Compute total kinetic energy.

        Returns
        -------
        float
            Kinetic energy [J] = 0.5 * m * |v|² + 0.5 * ω^T * I_world * ω
        """
        T_trans = 0.5 * self.mass * np.dot(self.v, self.v)
        I_world = self.inertia_world()
        T_rot = 0.5 * np.dot(self.w, I_world @ self.w)
        return float(T_trans + T_rot)
