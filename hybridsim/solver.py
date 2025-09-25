from __future__ import annotations
import numpy as np
from typing import List, Tuple
from .body import RigidBody6DOF
from .constraints import Constraint

Array = np.ndarray


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

    Returns (M, J, F, rhs, v)

    Velocity-level stabilization:
      rhs = -Jv - α C - β Ċ, with Ċ = Jv  ⇒ rhs = -(1+β) Jv - α C
    """
    nb = len(bodies)
    nv = 6 * nb

    # Mass, forces, velocities
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
        i, j = c.index_map()
        Jloc = c.jacobian()  # (r, 12) for two-body constraints

        # Scatter local J into global
        J[row:row+r, 6*i:6*i+6] = Jloc[:, 0:6]
        J[row:row+r, 6*j:6*j+6] = Jloc[:, 6:12]

        # Local Jv consistent with Jloc
        v_loc = np.concatenate([v[6*i:6*i+6], v[6*j:6*j+6]])
        Jv = Jloc @ v_loc  # (r,)

        # Residual with Baumgarte
        C = c.evaluate()
        rhs[row:row+r] = -(1.0 + beta) * Jv - alpha * C
        row += r

    return M, J, F, rhs, v


def solve_kkt(M: Array, J: Array, F: Array, rhs: Array) -> Tuple[Array, Array]:
    """
    Solve the KKT system for accelerations a and multipliers λ:

      [ M  J^T ] [a] = [F]
      [ J   0  ] [λ]   [rhs]
    """
    nv = M.shape[0]
    m = J.shape[0]

    K = np.zeros((nv + m, nv + m), dtype=np.float64)
    b = np.zeros(nv + m, dtype=np.float64)

    K[0:nv, 0:nv] = M
    K[0:nv, nv:nv+m] = J.T
    K[nv:nv+m, 0:nv] = J
    b[0:nv] = F
    b[nv:nv+m] = rhs

    sol = np.linalg.solve(K, b)
    a = sol[0:nv]
    lam = sol[nv:nv+m]
    return a, lam


class HybridSolver:
    """Fixed-step hybrid solver (semi-implicit Euler + KKT each step)."""
    def __init__(self, alpha: float = 0.0, beta: float = 0.0) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)

    def step(self, bodies: List[RigidBody6DOF], constraints: List[Constraint], dt: float) -> np.ndarray:
        M, J, F, rhs, _ = assemble_system(bodies, constraints, self.alpha, self.beta)
        a, _ = solve_kkt(M, J, F, rhs)

        for i, b in enumerate(bodies):
            a_lin = a[6*i:6*i+3]
            a_ang = a[6*i+3:6*i+6]
            b.integrate_semi_implicit(dt, a_lin, a_ang)
        return a


class HybridIVPSolver:
    """
    Variable-step stiff integrator using scipy.integrate.solve_ivp (Radau/BDF).
    State per body: [p(3), q(4), v(3), ω(3)] => 13 per body.
    Accelerations come from KKT at (t, y).
    Terminal event: payload_z - ground_z == 0 (direction -1).
    """
    def __init__(self, method: str = "Radau", rtol: float = 1e-6, atol: float = 1e-8,
                 max_step: float | None = None, alpha: float = 0.0, beta: float = 0.0) -> None:
        self.method = method
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.max_step = max_step
        self.alpha = float(alpha)
        self.beta = float(beta)

    def _pack(self, bodies: List[RigidBody6DOF]) -> np.ndarray:
        y = []
        for b in bodies:
            y.extend([*b.p, *b.q, *b.v, *b.w])
        return np.array(y, dtype=np.float64)

    def _unpack_to_world(self, y: np.ndarray, bodies: List[RigidBody6DOF]) -> None:
        for k, b in enumerate(bodies):
            off = 13*k
            b.p[:] = y[off:off+3]
            b.q[:] = y[off+3:off+7]
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
            self._unpack_to_world(y, bodies)

            # Clear & apply forces at time t
            for b in bodies:
                b.clear_forces()
                for fb in b.per_body_forces:
                    fb.apply(b, t)
            for fg in world.global_forces:
                for b in bodies:
                    fg.apply(b, t)
            for fpair in world.interaction_forces:
                fpair.apply_pair(t)

            # Assemble and solve KKT
            M, J, F, rhs_v, _ = assemble_system(bodies, constraints, self.alpha, self.beta)
            a, _ = solve_kkt(M, J, F, rhs_v)

            # Compose ydot = [v, qdot, a_lin, a_ang]
            ydot = np.zeros_like(y)
            for i, b in enumerate(bodies):
                a_lin = a[6*i:6*i+3]
                a_ang = a[6*i+3:6*i+6]
                off = 13*i
                ydot[off:off+3] = b.v
                from .mathutil import quat_derivative, quat_normalize
                b.q[:] = quat_normalize(b.q)
                ydot[off+3:off+7] = quat_derivative(b.q, b.w)
                ydot[off+7:off+10] = a_lin
                ydot[off+10:off+13] = a_ang
            return ydot

        def touchdown_event(t: float, y: np.ndarray) -> float:
            z = y[13*world.payload_index + 2]
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

        # --- Post-run CSV logging along solver time grid (if logger set) ---
        if world.logger is not None and sol.t.size > 0:
            for k, tk in enumerate(sol.t):
                # Set state to this sample
                self._unpack_to_world(sol.y[:, k], bodies)
                world.t = float(tk)

                # Re-apply forces for this sample for consistent logging
                for b in bodies:
                    b.clear_forces()
                    for fb in b.per_body_forces:
                        fb.apply(b, tk)
                for fg in world.global_forces:
                    for b in bodies:
                        fg.apply(b, tk)
                for fpair in world.interaction_forces:
                    fpair.apply_pair(tk)

                world.logger.log(world)

        return sol
