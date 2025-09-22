from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple
from .body import RigidBody6DOF
from .constraints import Constraint
from .mathutil import Array

def assemble_system(
    bodies: List[RigidBody6DOF],
    constraints: List[Constraint],
    alpha: float = 0.0,
    beta: float = 0.0,
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Assemble KKT blocks for current state:
      [ M  J^T ] [a] = [F]
      [ J   0  ] [λ]   [rhs]

    Returns (M, J, F, rhs, vstack)
    """
    nb = len(bodies)
    nv = 6 * nb

    # Mass & forces
    M = np.zeros((nv, nv), dtype=np.float64)
    F = np.zeros(nv, dtype=np.float64)
    v = np.zeros(nv, dtype=np.float64)
    for i, b in enumerate(bodies):
        Mi = b.mass_matrix_world()
        M[6*i:6*i+6, 6*i:6*i+6] = Mi
        F[6*i:6*i+6] = b.generalized_force()
        v[6*i:6*i+3] = b.v
        v[6*i+3:6*i+6] = b.w

    # Constraints
    m = sum(c.rows() for c in constraints)
    J = np.zeros((m, nv), dtype=np.float64)
    rhs = np.zeros(m, dtype=np.float64)

    row = 0
    for c in constraints:
        r = c.rows()
        idx = c.index_map()
        Jloc = c.jacobian()  # shape (r, 12) for two-body constraints here
        # scatter into global
        for local_body, body_index in enumerate(idx):
            col_start = 6 * local_body
            col_end = col_start + 6
            g_start = 6 * body_index
            g_end = g_start + 6
            J[row:row+r, g_start:g_end] = Jloc[:, col_start:col_end]

        # velocity-level rhs with optional Baumgarte: -J v - α C - β Ċ
        vel_stack = np.concatenate([v[6*k:6*k+6] for k in idx])
        C = c.evaluate()
        Cdot = c.c_dot(vel_stack)
        rhs[row:row+r] = -(Jloc @ vel_stack).ravel()
        rhs[row:row+r] += -alpha * C - beta * Cdot
        row += r

    return M, J, F, rhs, v


def solve_kkt(M: Array, J: Array, F: Array, rhs: Array) -> Tuple[Array, Array]:
    """
    Solve KKT system for accelerations a and multipliers λ.
    """
    nv = M.shape[0]
    m = J.shape[0]
    K = np.zeros((nv + m, nv + m), dtype=np.float64)
    rhs_full = np.zeros(nv + m, dtype=np.float64)
    # [M J^T; J 0] [a; λ] = [F; rhs]
    K[0:nv, 0:nv] = M
    K[0:nv, nv:nv+m] = J.T
    K[nv:nv+m, 0:nv] = J
    rhs_full[0:nv] = F
    rhs_full[nv:nv+m] = rhs

    sol = np.linalg.solve(K, rhs_full)
    a = sol[0:nv]
    lam = sol[nv:nv+m]
    return a, lam


class HybridSolver:
    """
    Fixed-step hybrid solver (semi-implicit Euler + KKT each step).
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)

    def step(self, bodies: List[RigidBody6DOF], constraints: List[Constraint], dt: float) -> np.ndarray:
        # Assemble forces already accumulated on bodies externally.
        M, J, F, rhs, _ = assemble_system(bodies, constraints, self.alpha, self.beta)
        a, _ = solve_kkt(M, J, F, rhs)
        # Scatter accelerations and integrate
        for i, b in enumerate(bodies):
            a_lin = a[6*i:6*i+3]
            a_ang = a[6*i+3:6*i+6]
            b.integrate_semi_implicit(dt, a_lin, a_ang)
        return a  # returned for diagnostics


class HybridIVPSolver:
    """
    Variable-step stiff integrator using scipy.integrate.solve_ivp (Radau/BDF).
    State per body: [p(3), q(4), v(3), w(3)] => 13 per body.
    Accelerations come from KKT at (t, y).
    Terminal event: payload_z - ground_z == 0 (direction -1).
    """
    def __init__(self, method: str = "Radau", rtol: float = 1e-6, atol: float = 1e-8, max_step: float | None = None,
                 alpha: float = 0.0, beta: float = 0.0) -> None:
        self.method = method
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.max_step = np.inf if max_step is None else float(max_step)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def _pack(self, bodies: List[RigidBody6DOF]) -> np.ndarray:
        y = []
        for b in bodies:
            y.extend([*b.p, *b.q, *b.v, *b.w])
        return np.array(y, dtype=np.float64)

    def _unpack_to_world(self, y: np.ndarray, bodies: List[RigidBody6DOF]) -> None:
        # Overwrite bodies' state from flat y
        for k, b in enumerate(bodies):
            off = 13*k
            b.p[:] = y[off:off+3]
            b.q[:] = y[off+3:off+7]
            # renormalization is done by dynamics via quaternion derivative integration
            b.v[:] = y[off+7:off+10]
            b.w[:] = y[off+10:off+13]

    def integrate(self, world, t_end: float):
        try:
            from scipy.integrate import solve_ivp
        except Exception as e:
            raise ImportError("SciPy is required for HybridIVPSolver. Install scipy>=1.8.") from e

        bodies = world.bodies
        constraints = world.constraints

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            # Copy y into bodies (no integration here)
            self._unpack_to_world(y, bodies)

            # Clear and re-apply forces at time t
            for b in bodies:
                b.clear_forces()
                for fb in b.per_body_forces:
                    fb.apply(b, t)
            for fg in world.global_forces:
                for b in bodies:
                    fg.apply(b, t)
            for fpair in world.interaction_forces:
                fpair.apply_pair(t)

            # Assemble and solve KKT for accelerations
            M, J, F, rhs_v, _ = assemble_system(bodies, constraints, self.alpha, self.beta)
            a, _ = solve_kkt(M, J, F, rhs_v)

            # Compose ydot = [v, qdot, a_lin, a_ang]
            ydot = np.zeros_like(y)
            for i, b in enumerate(bodies):
                a_lin = a[6*i:6*i+3]
                a_ang = a[6*i+3:6*i+6]
                off = 13*i
                ydot[off:off+3] = b.v
                # qdot = 0.5 q ⊗ [0, w]
                from .mathutil import quat_derivative, quat_normalize
                qdot = quat_derivative(b.q, b.w)
                # Keep quaternion normalized via derivative dynamics; normalize lightly:
                b.q[:] = quat_normalize(b.q)
                ydot[off+3:off+7] = qdot
                ydot[off+7:off+10] = a_lin
                ydot[off+10:off+13] = a_ang
            return ydot

        def touchdown_event(t: float, y: np.ndarray) -> float:
            # z_payload - ground_z (terminal when == 0; stopping on descending)
            idx = world.payload_index
            z = y[13*idx + 2]
            return float(z - world.ground_z)

        touchdown_event.terminal = True   # type: ignore[attr-defined]
        touchdown_event.direction = -1.0  # type: ignore[attr-defined]

        y0 = self._pack(bodies)
        sol = solve_ivp(
            rhs,
            t_span=(world.t, t_end),
            y0=y0,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.max_step,
            events=touchdown_event,
            dense_output=False,
        )
        # Update world with final state
        self._unpack_to_world(sol.y[:, -1], bodies)
        world.t = float(sol.t[-1])
        world.t_touchdown = float(sol.t_events[0][0]) if len(sol.t_events[0]) else None
        return sol
