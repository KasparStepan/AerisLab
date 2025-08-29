from __future__ import annotations
import numpy as np
from typing import Optional, Tuple

from .mathutil import EPS
from .body import RigidBody6DOF

def _assemble_system(world, alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble global M, Q, J, C, Jv.
    Returns:
        M  : (6n,6n) block-diag mass/inertia
        Q  : (6n,) generalized forces with gyroscopic bias
        J  : (m,6n) velocity Jacobian
        C  : (m,) constraint values
        Jv : (m,) constraint velocity (Cdot)
    """
    n = len(world.bodies)
    dof = 6 * n
    # Mass & generalized forces
    M = np.zeros((dof, dof), dtype=np.float64)
    Q = np.zeros(dof, dtype=np.float64)
    vglob = np.zeros(dof, dtype=np.float64)

    for i, b in enumerate(world.bodies):
        # M block
        Mi = b.mass_matrix_world()
        M[6*i:6*i+6, 6*i:6*i+6] = Mi

        # Per-body forces (already accumulated)
        Qi = b.generalized_force()
        Q[6*i:6*i+6] = Qi

        # Velocity block
        vglob[6*i:6*i+6] = np.hstack([b.v, b.w])

    # Constraints
    m = sum(c.rows() for c in world.constraints)
    if m == 0:
        J = np.zeros((0, dof), dtype=np.float64)
        C = np.zeros(0, dtype=np.float64)
        Jv = np.zeros(0, dtype=np.float64)
        return M, Q, J, C, Jv

    J = np.zeros((m, dof), dtype=np.float64)
    C = np.zeros(m, dtype=np.float64)

    row = 0
    for c in world.constraints:
        r = c.rows()
        idxs = c.index_map(world)
        C[row:row+r] = c.evaluate(world)
        Jloc = c.jacobian_local(world)  # (r, 6*nb)
        # Scatter into global J
        for k, bi in enumerate(idxs):
            J[row:row+r, 6*bi:6*bi+6] = Jloc[:, 6*k:6*(k+1)]
        row += r

    Jv = J @ vglob
    return M, Q, J, C, Jv

def _solve_kkt(M: np.ndarray, Q: np.ndarray, J: np.ndarray, rhs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve KKT:
        [ M  Jᵀ ][a] = [ Q   ]
        [ J   0 ][λ]   [ rhs ]
    Returns:
        a (6n,), λ (m,)
    """
    n = M.shape[0]
    m = J.shape[0]
    if m == 0:
        # No constraints
        a = np.linalg.solve(M, Q)
        return a, np.zeros(0, dtype=np.float64)

    K = np.zeros((n+m, n+m), dtype=np.float64)
    K[:n, :n] = M
    K[:n, n:] = J.T
    K[n:, :n] = J
    b = np.zeros(n+m, dtype=np.float64)
    b[:n] = Q
    b[n:] = rhs

    sol = np.linalg.solve(K, b)
    a = sol[:n]
    lam = sol[n:]
    return a, lam

class HybridSolver:
    """Fixed-step solver: semi-implicit Euler + KKT per step.
    Baumgarte stabilization:
        rhs = -J v - (alpha * C + beta * Cdot)
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.last_lambdas: Optional[np.ndarray] = None  # optional exposure

    def step(self, world, dt: float) -> None:
        # 1) Clear & apply forces
        for b in world.bodies:
            b.clear_forces()
        for gf in world.global_forces:
            for b in world.bodies:
                gf.apply(b, world.time)
        # per-body forces
        for b in world.bodies:
            for f in b.forces:
                f.apply(b, world.time)
        # interaction forces (springs, etc.)
        for f in world.interaction_forces:
            f.apply(world.time)

        # 2) Assemble
        M, Q, J, C, Jv = _assemble_system(world, self.alpha, self.beta)

        # 3) Build RHS for constraints
        rhs = -Jv - (self.alpha * C + self.beta * Jv)

        # 4) Solve KKT -> accelerations
        a, lam = _solve_kkt(M, Q, J, rhs)
        self.last_lambdas = lam

        # 5) Scatter + integrate
        for i, b in enumerate(world.bodies):
            ai = a[6*i:6*i+6]
            b.a_lin = ai[:3]
            b.a_ang = ai[3:]
            b.integrate_semi_implicit(dt)

        world.time += dt

class HybridIVPSolver:
    """Variable-step solver using scipy.integrate.solve_ivp with stiff methods."""
    def __init__(
        self,
        method: str = "Radau",
        rtol: float = 1e-6,
        atol: float = 1e-9,
        max_step: Optional[float] = None,
        alpha: float = 0.0, beta: float = 0.0,
    ) -> None:
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.last_lambdas: Optional[np.ndarray] = None

    def integrate_to(self, world, t_end: float):
        try:
            from scipy.integrate import solve_ivp
        except Exception as e:
            raise RuntimeError("SciPy is required for HybridIVPSolver.") from e

        n = len(world.bodies)

        def pack_state() -> np.ndarray:
            y = []
            for b in world.bodies:
                y.extend([*b.p, *b.q, *b.v, *b.w])
            return np.array(y, dtype=np.float64)

        def unpack_state(y: np.ndarray) -> None:
            for i, b in enumerate(world.bodies):
                off = 13 * i
                b.p = y[off:off+3]
                b.q = y[off+3:off+7]
                b.q = b.q / np.linalg.norm(b.q)  # keep unit
                b.v = y[off+7:off+10]
                b.w = y[off+10:off+13]

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            unpack_state(y)
            # Clear & apply forces
            for b in world.bodies:
                b.clear_forces()
            for gf in world.global_forces:
                for b in world.bodies:
                    gf.apply(b, t)
            for b in world.bodies:
                for f in b.forces:
                    f.apply(b, t)
            for f in world.interaction_forces:
                f.apply(t)

            # Assemble
            M, Q, J, C, Jv = _assemble_system(world, self.alpha, self.beta)
            rhs_c = -Jv - (self.alpha * C + self.beta * Jv)
            a, lam = _solve_kkt(M, Q, J, rhs_c)
            self.last_lambdas = lam

            # Build ydot
            ydot = np.zeros_like(y)
            for i, b in enumerate(world.bodies):
                off = 13 * i
                ydot[off:off+3] = b.v
                # qdot
                w0, x, yq, z = b.q
                wx, wy, wz = b.w
                qdot = 0.5 * np.array([
                    -x*wx - yq*wy - z*wz,
                     w0*wx + yq*wz - z*wy,
                     w0*wy - x*wz + z*wx,
                     w0*wz + x*wy - yq*wx,
                ], dtype=np.float64)
                ydot[off+3:off+7] = qdot
                ai = a[6*i:6*i+6]
                ydot[off+7:off+10] = ai[:3]
                ydot[off+10:off+13] = ai[3:]
            return ydot

        # Terminal event: payload z - ground_z (downwards crossing)
        payload_idx = world.payload_index if world.payload_index is not None else 0
        def ground_event(t: float, y: np.ndarray) -> float:
            z = y[13*payload_idx + 2]  # payload p_z
            return z - world.ground_z
        ground_event.terminal = True
        ground_event.direction = -1.0

        y0 = pack_state()
        t0 = float(world.time)
        sol = solve_ivp(
            rhs, (t0, float(t_end)), y0,
            method=self.method, rtol=self.rtol, atol=self.atol, max_step=self.max_step,
            events=ground_event
        )
        # Set world to final state (no projection)
        unpack_state(sol.y[:, -1])
        world.time = float(sol.t[-1])
        return sol
