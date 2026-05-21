from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as ScR

from .body import RigidBody6DOF

Array = np.ndarray

def skew(v: Array) -> Array:
    """
    Skew-symmetric matrix S(v) s.t. S(v) @ w = v × w.
    v: (3,) -> (3,3)
    """
    vx, vy, vz = v
    return np.array([
        [0.0, -vz,  vy],
        [vz,  0.0, -vx],
        [-vy, vx,  0.0]
    ], dtype=np.float64)

class Constraint:
    """Abstract base for constraints used in KKT solve."""
    def rows(self) -> int: raise NotImplementedError
    def index_map(self) -> list[int]: raise NotImplementedError  # body indices in world order
    def evaluate(self) -> Array: raise NotImplementedError       # C(q)
    def jacobian(self) -> Array: raise NotImplementedError       # J such that Cdot = J v_g (v_g stacks [v, w] per-body)

    def jdot_v(self) -> Array:
        """
        Acceleration-bias term J̇·v evaluated at the current state/velocities.

        The acceleration-level constraint is C̈ = J·a + J̇·v = 0, so the solver
        needs J·a = -J̇·v. This term carries the centripetal/Coriolis
        acceleration; omitting it makes constrained motion (e.g. a swinging
        pendulum) systematically wrong, independent of timestep.

        Returns a vector of length ``rows()``.
        """
        raise NotImplementedError

    # Helper for velocity-level residual:
    def c_dot(self, vstack: Array) -> Array:
        J = self.jacobian()
        return J @ vstack


class DistanceConstraint(Constraint):
    """
    Enforce fixed separation between two attachment points.
    Scalar constraint: C = 0.5 (||d||^2 - L^2) = 0

    Ċ = d · (vA + ωA×ra_w - vB - ωB×rb_w)
      = [d^T, (ra_w × d)^T, -d^T, -(rb_w × d)^T] [vA, ωA, vB, ωB]
    """
    def __init__(
        self,
        world_bodies: list[RigidBody6DOF],
        body_i: int,
        body_j: int,
        attach_i_local: Array,
        attach_j_local: Array,
        length: float,
    ) -> None:
        self.bodies = world_bodies
        self.i = body_i
        self.j = body_j
        self.ri_local = np.asarray(attach_i_local, dtype=np.float64)
        self.rj_local = np.asarray(attach_j_local, dtype=np.float64)
        self.L = float(length)

    def rows(self) -> int: return 1
    def index_map(self) -> list[int]: return [self.i, self.j]

    def _geom(self) -> tuple[Array, Array, Array, Array, Array]:
        bi = self.bodies[self.i]
        bj = self.bodies[self.j]
        Ri = bi.rotation_world()
        Rj = bj.rotation_world()
        ri_w = Ri @ self.ri_local
        rj_w = Rj @ self.rj_local
        pi = bi.p + ri_w
        pj = bj.p + rj_w
        d = pi - pj
        return d, ri_w, rj_w, pi, pj

    def evaluate(self) -> Array:
        d, *_ = self._geom()
        return np.array([0.5*(d @ d - self.L*self.L)], dtype=np.float64)

    def jacobian(self) -> Array:
        d, ri_w, rj_w, *_ = self._geom()
        J = np.zeros((1, 12), dtype=np.float64)  # [v_i, w_i, v_j, w_j]
        J[0, 0:3] = d
        J[0, 3:6] = np.cross(ri_w, d)  # (ri_w × d)
        J[0, 6:9] = -d
        J[0, 9:12] = -np.cross(rj_w, d)
        return J

    def jdot_v(self) -> Array:
        """
        J̇·v for the distance constraint (scalar).

        With u ≡ ḋ = (v_i + ω_i×r_i) − (v_j + ω_j×r_j):

            J̇·v = u·u + d·[ω_i×(ω_i×r_i) − ω_j×(ω_j×r_j)]

        The u·u term is the relative-speed (centripetal) contribution; the
        ω×(ω×r) terms vanish unless an attachment point is offset from the
        body origin and that body is spinning.
        """
        d, ri_w, rj_w, *_ = self._geom()
        bi = self.bodies[self.i]
        bj = self.bodies[self.j]
        ui = bi.v + np.cross(bi.w, ri_w)   # world velocity of attachment on i
        uj = bj.v + np.cross(bj.w, rj_w)   # world velocity of attachment on j
        u = ui - uj                         # = ḋ
        spin = (np.cross(bi.w, np.cross(bi.w, ri_w))
                - np.cross(bj.w, np.cross(bj.w, rj_w)))
        return np.array([u @ u + d @ spin], dtype=np.float64)


class PointWeldConstraint(Constraint):
    """
    Enforce coincidence of two attachment points (3 equations):
    C = pa - pb = 0

    Velocity-level:
      Ċ = vA + ωA×ra_w - vB - ωB×rb_w
         = [I, -skew(ra_w), -I,  +skew(rb_w)] [vA, ωA, vB, ωB]
    """
    def __init__(
        self,
        world_bodies: list[RigidBody6DOF],
        body_i: int,
        body_j: int,
        attach_i_local: Array,
        attach_j_local: Array,
    ) -> None:
        self.bodies = world_bodies
        self.i = body_i
        self.j = body_j
        self.ri_local = np.asarray(attach_i_local, dtype=np.float64)
        self.rj_local = np.asarray(attach_j_local, dtype=np.float64)

    def rows(self) -> int: return 3
    def index_map(self) -> list[int]: return [self.i, self.j]

    def _geom(self):
        bi = self.bodies[self.i]
        bj = self.bodies[self.j]
        Ri = bi.rotation_world()
        Rj = bj.rotation_world()
        ri_w = Ri @ self.ri_local
        rj_w = Rj @ self.rj_local
        pi = bi.p + ri_w
        pj = bj.p + rj_w
        return ri_w, rj_w, pi, pj

    def evaluate(self) -> Array:
        ri_w, rj_w, pi, pj = self._geom()
        return (pi - pj).astype(np.float64)

    def jacobian(self) -> Array:
        ri_w, rj_w, *_ = self._geom()
        J = np.zeros((3, 12), dtype=np.float64)
        J[:, 0:3] = np.eye(3)
        J[:, 3:6] = -skew(ri_w)
        J[:, 6:9] = -np.eye(3)
        J[:, 9:12] = skew(rj_w)
        return J

    def jdot_v(self) -> Array:
        """
        J̇·v for the point-weld constraint (3-vector).

            J̇·v = ω_i×(ω_i×r_i) − ω_j×(ω_j×r_j)

        These are the centripetal accelerations of the welded attachment
        points due to each body's spin; zero when both attachments sit at
        their body origin (r = 0) or neither body rotates.
        """
        ri_w, rj_w, *_ = self._geom()
        bi = self.bodies[self.i]
        bj = self.bodies[self.j]
        return (np.cross(bi.w, np.cross(bi.w, ri_w))
                - np.cross(bj.w, np.cross(bj.w, rj_w))).astype(np.float64)


class DOFLockConstraint(Constraint):
    """
    Lock selected world-frame DOFs of one body relative to another — one KKT
    row per locked axis. Translational rows come first, then rotational rows.

    **Translational lock** of world axis ê_k holds the relative position
    component along ê_k at a fixed offset:

        C_k = ê_k · d − offset_k,        d = (p_i + r_i) − (p_j + r_j)
        Ċ_k = ê_k · u,                   u = (v_i + ω_i×r_i) − (v_j + ω_j×r_j)
        J-row = [ê_k, r_i×ê_k, −ê_k, −r_j×ê_k]
        J̇v_k = ê_k · [ω_i×(ω_i×r_i) − ω_j×(ω_j×r_j)]   (0 for origin attachments)

    **Rotational lock** of world axis ê_k holds the relative orientation about
    ê_k. With the world-frame orientation error
        θ_world = log_SO3( R_i R_i0ᵀ R_j0 R_jᵀ ),   (= 0 at the reference pose)
    which satisfies θ̇_world = ω_i − ω_j at the satisfied state:

        C_k = ê_k · θ_world
        Ċ_k = ê_k · (ω_i − ω_j)
        J-row = [0, ê_k, 0, −ê_k]
        J̇v_k = 0                                     (world-fixed axes)

    Examples
    --------
    Confine a point mass to the x–z plane (2-D): ``locked_translation=(0,1,0)``.
    Weld a body to the world frame: both masks all-ones with ``body_j=world.WORLD``.
    Planar *rigid-body* reduction: lock y-translation + x,z-rotation
    (``locked_translation=(0,1,0)``, ``locked_rotation=(1,0,1)``).

    Notes
    -----
    Axes are world-fixed (columns of ``axes``, default world x/y/z). A
    body-attached/tumbling joint frame (where ``J̇v`` for rotational rows is
    nonzero) is a future extension; world-fixed axes are exact for welds and
    for planar motion (the free axis stays world-aligned).
    """
    def __init__(
        self,
        world_bodies: list[RigidBody6DOF],
        body_i: int,
        body_j: int,
        attach_i_local: Array,
        attach_j_local: Array,
        locked_translation: tuple[bool, bool, bool] = (False, False, False),
        locked_rotation: tuple[bool, bool, bool] = (False, False, False),
        axes: Array | None = None,
        targets: Array | None = None,
    ) -> None:
        self.bodies = world_bodies
        self.i = body_i
        self.j = body_j
        self.ri_local = np.asarray(attach_i_local, dtype=np.float64)
        self.rj_local = np.asarray(attach_j_local, dtype=np.float64)

        # World-frame lock axes (columns). Default: world x/y/z.
        self.axes = np.eye(3) if axes is None else np.asarray(axes, dtype=np.float64)
        self.locked_t = [k for k, on in enumerate(locked_translation) if on]
        self.locked_r = [k for k, on in enumerate(locked_rotation) if on]
        if not self.locked_t and not self.locked_r:
            raise ValueError("DOFLockConstraint needs at least one locked axis.")

        # Translational targets: hold each axis at its current value unless given.
        d, *_ = self._geom()
        if targets is None:
            self.targets = {k: float(self.axes[:, k] @ d) for k in self.locked_t}
        else:
            tgt = np.asarray(targets, dtype=np.float64)
            self.targets = {k: float(tgt[k]) for k in self.locked_t}

        # Rotational reference: capture both bodies' orientations at construction.
        self.Ri0 = self.bodies[self.i].rotation_world().copy()
        self.Rj0 = self.bodies[self.j].rotation_world().copy()

    def rows(self) -> int:
        return len(self.locked_t) + len(self.locked_r)

    def index_map(self) -> list[int]:
        return [self.i, self.j]

    def _geom(self) -> tuple[Array, Array, Array]:
        bi = self.bodies[self.i]
        bj = self.bodies[self.j]
        ri_w = bi.rotation_world() @ self.ri_local
        rj_w = bj.rotation_world() @ self.rj_local
        d = (bi.p + ri_w) - (bj.p + rj_w)
        return d, ri_w, rj_w

    def _orientation_error(self) -> Array:
        """World-frame rotation vector θ_world = log_SO3(R_i R_i0ᵀ R_j0 R_jᵀ)."""
        Ri = self.bodies[self.i].rotation_world()
        Rj = self.bodies[self.j].rotation_world()
        M = Ri @ self.Ri0.T @ self.Rj0 @ Rj.T
        return ScR.from_matrix(M).as_rotvec()

    def evaluate(self) -> Array:
        d, *_ = self._geom()
        out = [self.axes[:, k] @ d - self.targets[k] for k in self.locked_t]
        if self.locked_r:
            theta = self._orientation_error()
            out += [self.axes[:, k] @ theta for k in self.locked_r]
        return np.array(out, dtype=np.float64)

    def jacobian(self) -> Array:
        _, ri_w, rj_w = self._geom()
        J = np.zeros((self.rows(), 12), dtype=np.float64)
        row = 0
        for k in self.locked_t:                      # translational rows
            e = self.axes[:, k]
            J[row, 0:3] = e
            J[row, 3:6] = np.cross(ri_w, e)
            J[row, 6:9] = -e
            J[row, 9:12] = -np.cross(rj_w, e)
            row += 1
        for k in self.locked_r:                      # rotational rows
            e = self.axes[:, k]
            J[row, 3:6] = e
            J[row, 9:12] = -e
            row += 1
        return J

    def jdot_v(self) -> Array:
        _, ri_w, rj_w = self._geom()
        bi = self.bodies[self.i]
        bj = self.bodies[self.j]
        spin = (np.cross(bi.w, np.cross(bi.w, ri_w))
                - np.cross(bj.w, np.cross(bj.w, rj_w)))
        out = [self.axes[:, k] @ spin for k in self.locked_t]
        out += [0.0 for _ in self.locked_r]          # world-fixed rotation axes
        return np.array(out, dtype=np.float64)
