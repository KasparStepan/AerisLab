from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable

from .mathutil import normalize_quaternion, quaternion_multiply

Array = np.ndarray

# --------------------------- Fixed-step hybrid solver -------------------------

@dataclass
class SolverSettings:
    """Settings for the hybrid constraint solve and integration."""
    baumgarte_alpha: float = 0.0   # position stabilization factor (s^-1)
    baumgarte_beta: float  = 0.0   # velocity stabilization factor (s^-1)

class HybridSolver:
    """
    Fixed-step hybrid solver:
      1) Clear forces
      2) Apply forces (optionally time-aware)
      3) Assemble M, F, J
      4) Solve KKT: [M J^T; J 0][a; λ] = [F; rhs]
      5) Integrate (semi-implicit)
      6) Contact post-process
    """
    def __init__(self, settings: Optional[SolverSettings] = None):
        self.settings = settings or SolverSettings()

    # --- shared assembly utilities -------------------------------------------

    @staticmethod
    def _block_diag(blocks: list[Array]) -> Array:
        n = sum(b.shape[0] for b in blocks)
        M = np.zeros((n, n))
        i = 0
        for b in blocks:
            k = b.shape[0]
            M[i:i+k, i:i+k] = b
            i += k
        return M

    @staticmethod
    def assemble_mass_force(bodies: list["RigidBody6DOF"]) -> tuple[Array, Array]:
        M_blocks, F_list = [], []
        for b in bodies:
            M_blocks.append(b.mass_matrix_world())    # (6,6)
            F_list.append(b.generalized_force())      # (6,)
        M = HybridSolver._block_diag(M_blocks) if M_blocks else np.zeros((0, 0))
        F = np.concatenate(F_list) if F_list else np.zeros(0)
        return M, F

    @staticmethod
    def assemble_velocity(bodies: list["RigidBody6DOF"]) -> Array:
        chunks = [np.hstack([b.linear_velocity, b.angular_velocity]) for b in bodies]
        return np.concatenate(chunks) if chunks else np.zeros(0)

    def assemble_constraints(self, world: "World") -> tuple[Array | None, Array | None]:
        if not world.constraints:
            return None, None

        rows = sum(c.rows() for c in world.constraints)
        N = len(world.bodies)
        J = np.zeros((rows, 6 * N))
        rhs = np.zeros(rows)
        r0 = 0

        v = self.assemble_velocity(world.bodies)

        for c in world.constraints:
            k = c.rows()
            idxs = c.index_map(world)  # e.g., [i, j]
            Jloc = c.jacobian_local(world)  # (k, 12) for two-body constraints
            if len(idxs) != 2:
                raise NotImplementedError("Template implements 2-body constraints; extend for more.")
            i, j = idxs
            J[r0:r0+k, 6*i:6*i+3]     = Jloc[:, 0:3]
            J[r0:r0+k, 6*i+3:6*i+6]   = Jloc[:, 3:6]
            J[r0:r0+k, 6*j:6*j+3]     = Jloc[:, 6:9]
            J[r0:r0+k, 6*j+3:6*j+6]   = Jloc[:, 9:12]

            rhs[r0:r0+k] = -(J[r0:r0+k, :] @ v)  # velocity-level target

            if (self.settings.baumgarte_alpha != 0.0) or (self.settings.baumgarte_beta != 0.0):
                C = c.evaluate(world)
                Cdot = J[r0:r0+k, :] @ v
                rhs[r0:r0+k] -= (self.settings.baumgarte_alpha * C + self.settings.baumgarte_beta * Cdot)
            r0 += k

        return J, rhs

    # --- fixed-step step() ----------------------------------------------------

    def step(self, world: "World", dt: float) -> None:
        # 1) clear forces
        for b in world.bodies:
            b.clear_forces()

        # 2) apply forces (time-aware if provided)
        for gf in world.global_forces:
            for b in world.bodies:
                gf.apply(b, world.time)
        for b in world.bodies:
            for bf in b.forces:
                bf.apply(b, world.time)
        for s in world.interaction_forces:
            s.apply()

        # 3) assemble
        M, F = self.assemble_mass_force(world.bodies)
        if M.size == 0:
            return
        J, rhs = self.assemble_constraints(world)

        # 4) KKT
        if J is not None and J.size > 0:
            m = J.shape[0]
            Z = np.zeros((m, m))
            A = np.block([[M, J.T],
                          [J, Z]])
            b = np.concatenate([F, rhs])
            sol = np.linalg.solve(A, b)
            a_all = sol[:M.shape[0]]
        else:
            a_all = np.linalg.solve(M, F)

        # scatter + integrate
        idx = 0
        for b in world.bodies:
            b._a_lin = a_all[idx:idx+3]; b._a_ang = a_all[idx+3:idx+6]
            idx += 6
            b.integrate_semi_implicit(dt)

        # contact
        if world.contact_model is not None:
            world.contact_model.post_integrate(world)

# --------------------------- Variable-step IVP solver -------------------------

@dataclass
class IVPSettings:
    """
    Settings for variable-step solve_ivp integration.
    """
    method: str = "Radau"      # "Radau" (stiff, A-stable) or "BDF" (stiff)
    rtol: float = 1e-6
    atol: float = 1e-8
    max_step: float | None = None
    normalize_quaternion_each_step: bool = True
    max_contact_events: int = 128  # safety guard

class HybridIVPSolver:
    """
    Variable-step stiff integrator using scipy.integrate.solve_ivp.
    Builds a continuous ODE y' = f(t, y) by solving a KKT system at each RHS call
    to obtain accelerations consistent with constraints.

    State y per body: [px,py,pz,  qx,qy,qz,qw,  vx,vy,vz,  wx,wy,wz]  (13)
    """
    def __init__(self, settings: Optional[SolverSettings] = None, ivp: Optional[IVPSettings] = None):
        self.settings = settings or SolverSettings()
        self.ivp = ivp or IVPSettings()

    # ---- packing / unpacking -------------------------------------------------

    @staticmethod
    def pack_state(bodies: list["RigidBody6DOF"]) -> Array:
        chunks = []
        for b in bodies:
            chunks.append(np.hstack([b.position, b.orientation, b.linear_velocity, b.angular_velocity]))
        return np.concatenate(chunks) if chunks else np.zeros(0)

    @staticmethod
    def unpack_state(y: Array, bodies: list["RigidBody6DOF"], normalize_quat: bool = True) -> None:
        idx = 0
        for b in bodies:
            b.position = y[idx:idx+3]; idx += 3
            b.orientation = y[idx:idx+4]; idx += 4
            if normalize_quat:
                b.orientation = normalize_quaternion(b.orientation)
            b.linear_velocity = y[idx:idx+3]; idx += 3
            b.angular_velocity = y[idx:idx+3]; idx += 3

    # ---- assembling M,F,J and RHS (same as fixed-step) ----------------------

    def _assemble_MFJ_rhs(self, world: "World") -> tuple[Array, Array, Array | None, Array | None]:
        # forces are assumed already applied by caller (rhs) prior to this call
        M, F = HybridSolver.assemble_mass_force(world.bodies)
        J, rhs = HybridSolver(self.settings).assemble_constraints(world)
        return M, F, J, rhs

    # ---- RHS for solve_ivp ---------------------------------------------------

    def _rhs(self, t: float, y: Array, world: "World") -> Array:
        # reflect y into bodies
        self.unpack_state(y, world.bodies, normalize_quat=self.ivp.normalize_quaternion_each_step)

        # clear & apply forces with time
        for b in world.bodies:
            b.clear_forces()
        for gf in world.global_forces:
            for b in world.bodies:
                gf.apply(b, t)
        for b in world.bodies:
            for bf in b.forces:
                bf.apply(b, t)
        for s in world.interaction_forces:
            s.apply()

        # assemble and solve for accelerations
        M, F, J, rhs = self._assemble_MFJ_rhs(world)
        if J is not None and J.size > 0:
            m = J.shape[0]
            Z = np.zeros((m, m))
            A = np.block([[M, J.T],
                          [J, Z]])
            b = np.concatenate([F, rhs])
            sol = np.linalg.solve(A, b)
            a_all = sol[:M.shape[0]]
        else:
            a_all = np.linalg.solve(M, F)

        # build ydot
        ydot = np.zeros_like(y)
        idx = 0
        a_idx = 0
        for b in world.bodies:
            # p' = v
            ydot[idx:idx+3] = b.linear_velocity; idx += 3
            # q' = 0.5 * q ⊗ [0, ω]
            dq = quaternion_multiply(b.orientation, np.hstack(([0.0], b.angular_velocity))) * 0.5
            ydot[idx:idx+4] = dq; idx += 4
            # v' = a_lin,  ω' = a_ang
            ydot[idx:idx+3] = a_all[a_idx:a_idx+3]; idx += 3; a_idx += 3
            ydot[idx:idx+3] = a_all[a_idx:a_idx+3]; idx += 3; a_idx += 3

        return ydot

    # ---- events for contact (ground) ----------------------------------------

    @staticmethod
    def _make_ground_event(world: "World") -> Callable[[float, Array], float]:
        """Event when any body crosses ground level (becomes z < ground_z)."""
        if world.contact_model is None:
            # dummy event that never triggers
            def no_event(t: float, y: Array) -> float:  # pragma: no cover
                return 1.0
            no_event.terminal = False
            no_event.direction = -1
            return no_event

        ground_z = getattr(world.contact_model, "ground_z", 0.0)

        def event(t: float, y: Array) -> float:
            # min over all bodies of (z - ground_z)
            # state layout per body: [p(3), q(4), v(3), w(3)]
            min_val = np.inf
            stride = 13
            for i in range(len(world.bodies)):
                z = y[i*stride + 2]
                min_val = min(min_val, z - ground_z)
            return min_val

        event.terminal = True   # stop integration at contact
        event.direction = -1    # detect downward crossing
        return event

    # ---- integrate to target time with events & contact handling -------------

    def integrate(self, world: "World", t_end: float) -> None:
        """
        Integrate world.time -> t_end with a stiff variable-step method.
        Handles ground contact via event detection + post_integrate + restart.
        """
        try:
            from scipy.integrate import solve_ivp
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("scipy is required for HybridIVPSolver.integrate") from e

        t0 = world.time
        if t_end <= t0:
            return

        # pack initial state
        y0 = self.pack_state(world.bodies)

        # integrate in segments, restarting at each contact event
        events_handled = 0
        while t0 < t_end - 1e-15:
            # define RHS bound with current world reference
            def rhs(t, y): return self._rhs(t, y, world)

            # events (only ground)
            ev = self._make_ground_event(world)
            sol = solve_ivp(
                rhs, (t0, t_end), y0,
                method=self.ivp.method, rtol=self.ivp.rtol, atol=self.ivp.atol,
                events=ev, max_step=self.ivp.max_step
            )

            # update world to final solver state of this segment
            self.unpack_state(sol.y[:, -1], world.bodies, normalize_quat=True)
            world.time = float(sol.t[-1])
            if world.logger is not None:
                world.logger.log(world.time, world.bodies)

            # if contact occurred, apply contact model and continue immediately
            if sol.t_events and len(sol.t_events[0]) > 0:
                # set state exactly at event time
                te = sol.t_events[0][-1]
                ye = sol.y_events[0][-1]
                self.unpack_state(ye, world.bodies, normalize_quat=True)
                world.time = float(te)

                # apply contact impulse/projection/penalty
                if world.contact_model is not None:
                    world.contact_model.post_integrate(world)
                if world.logger is not None:
                    world.logger.log(world.time, world.bodies)

                # restart from here
                y0 = self.pack_state(world.bodies)
                t0 = world.time
                events_handled += 1
                if events_handled > self.ivp.max_contact_events:  # safety
                    raise RuntimeError("Exceeded max_contact_events during IVP integration.")
                continue

            # no more events; we reached t_end
            break
