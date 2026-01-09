"""
Numerical solvers for constrained rigid body dynamics.

Implements KKT (Karush-Kuhn-Tucker) system solution for velocity-level
constraints with Baumgarte stabilization. Supports both fixed-step
(semi-implicit Euler) and adaptive-step (scipy IVP) integration.

Mathematical Framework
----------------------
Constrained dynamics system:
    M * a = F + J^T * λ
    J * a = rhs

where:
    M: generalized mass matrix
    a: generalized accelerations [linear; angular]
    F: applied forces/torques
    J: constraint Jacobian matrix
    λ: Lagrange multipliers (constraint forces)
    rhs: right-hand side with Baumgarte stabilization

Physical units:
- Mass: kilograms [kg]
- Force: Newtons [N]
- Acceleration: meters per second squared [m/s²]
- Angular acceleration: radians per second squared [rad/s²]
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple
from aerislab.dynamics.body import RigidBody6DOF, quat_derivative, quat_normalize
from aerislab.dynamics.constraints import Constraint
import warnings

# Numerical tolerance for singularity detection
CONDITION_NUMBER_THRESHOLD = 1e10
MIN_CONSTRAINT_MASS = 1e-12


def assemble_system(
    bodies: List[RigidBody6DOF],
    constraints: List[Constraint],
    alpha: float = 0.0,
    beta: float = 0.0,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], 
           NDArray[np.float64], NDArray[np.float64]]:
    """
    Assemble global system matrices for KKT solver.
    
    Constructs the constrained dynamics system:
        M * a = F + J^T * λ
        J * a = rhs
    
    Parameters
    ----------
    bodies : List[RigidBody6DOF]
        List of all rigid bodies in the system
    constraints : List[Constraint]
        List of all constraints (joints, distance constraints, etc.)
    alpha : float
        Baumgarte position stabilization parameter [1/s].
        Typical range: 1-10. Higher values → stronger correction but may cause instability.
    beta : float
        Baumgarte velocity stabilization parameter [-].
        Typical range: 0.1-1.0. Usually β < α for stability.
        
    Returns
    -------
    Minv : NDArray[np.float64]
        Inverse mass matrix (6N, 6N) where N is number of bodies
    J : NDArray[np.float64]
        Constraint Jacobian (m, 6N) where m is total constraint rows
    F : NDArray[np.float64]
        Generalized forces (6N,)
    rhs : NDArray[np.float64]
        Right-hand side for constraints with Baumgarte stabilization (m,)
    v : NDArray[np.float64]
        Current generalized velocities (6N,)
        
    Notes
    -----
    Baumgarte stabilization:
        rhs = -(1 + β) * J*v - α * C
        
    where C is the constraint violation vector. This adds feedback terms
    to drive constraint violations toward zero.
    
    References
    ----------
    .. [1] Baumgarte, J. (1972). Stabilization of constraints and integrals
           of motion in dynamical systems. Computer Methods in Applied
           Mechanics and Engineering, 1(1), 1-16.
    """
    nb = len(bodies)
    nv = 6 * nb

    # Preallocate system matrices
    Minv = np.zeros((nv, nv), dtype=np.float64)
    F = np.zeros(nv, dtype=np.float64)
    v = np.zeros(nv, dtype=np.float64)
    
    # Fill mass matrix, forces, and velocities
    for i, b in enumerate(bodies):
        Wi = b.inv_mass_matrix_world()  # ← THIS IS THE CORRECT METHOD NAME!
        Minv[6*i:6*i+6, 6*i:6*i+6] = Wi
        F[6*i:6*i+6] = b.generalized_force()
        v[6*i:6*i+3] = b.v
        v[6*i+3:6*i+6] = b.w

    # Assemble constraint Jacobian and RHS
    m = sum(c.rows() for c in constraints)
    J = np.zeros((m, nv), dtype=np.float64)
    rhs = np.zeros(m, dtype=np.float64)

    row = 0
    for c in constraints:
        r = c.rows()
        i, j = c.index_map()
        Jloc = c.jacobian()  # (r, 12) for two-body constraints

        # Scatter local Jacobian into global system
        J[row:row+r, 6*i:6*i+6] = Jloc[:, 0:6]
        J[row:row+r, 6*j:6*j+6] = Jloc[:, 6:12]

        # Constraint velocity: Jv
        v_loc = np.concatenate([v[6*i:6*i+6], v[6*j:6*j+6]])
        Jv = Jloc @ v_loc  # (r,)

        # Constraint violation
        C = c.evaluate()
        
        # Baumgarte stabilization: rhs = -(1+β)*Jv - α*C
        rhs[row:row+r] = -(1.0 + beta) * Jv - alpha * C
        row += r

    return Minv, J, F, rhs, v


def solve_kkt(
    Minv: NDArray[np.float64], 
    J: NDArray[np.float64], 
    F: NDArray[np.float64], 
    rhs: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Solve KKT system using Schur complement method.
    
    Solves the constrained dynamics system:
        M * a = F + J^T * λ
        J * a = rhs
        
    using the Schur complement approach:
        1. Compute unconstrained acceleration: a₀ = M⁻¹ * F
        2. Form effective mass matrix: A = J * M⁻¹ * J^T
        3. Solve for multipliers: A * λ = rhs - J * a₀
        4. Compute constrained acceleration: a = a₀ + M⁻¹ * J^T * λ
    
    Parameters
    ----------
    Minv : NDArray[np.float64]
        Inverse mass matrix (6N, 6N)
    J : NDArray[np.float64]
        Constraint Jacobian (m, 6N)
    F : NDArray[np.float64]
        Generalized forces (6N,)
    rhs : NDArray[np.float64]
        Constraint RHS with stabilization (m,)
        
    Returns
    -------
    a : NDArray[np.float64]
        Constrained generalized accelerations (6N,)
    lam : NDArray[np.float64]
        Lagrange multipliers (constraint forces) (m,)
        
    Raises
    ------
    RuntimeError
        If the constraint mass matrix is singular or ill-conditioned
        
    Notes
    -----
    The Schur complement method is efficient for systems with many bodies
    but relatively few constraints (m << 6N). For dense constraint systems,
    consider direct sparse solvers.
    
    Condition number checking helps detect poorly defined constraint systems
    that may lead to numerical instability.
    """
    # Compute unconstrained accelerations
    a0 = Minv @ F

    m = J.shape[0]
    if m == 0:
        # No constraints - return unconstrained acceleration
        return a0, np.zeros(0, dtype=np.float64)

    # Form effective mass matrix (Schur complement)
    # A = J * M⁻¹ * J^T
    A = J @ (Minv @ J.T)
    
    # Check for singularity
    try:
        # Estimate condition number (expensive but important for debugging)
        if A.shape[0] <= 100:  # Only for small systems
            cond = np.linalg.cond(A)
            if cond > CONDITION_NUMBER_THRESHOLD:
                warnings.warn(
                    f"Constraint mass matrix is ill-conditioned (cond={cond:.2e}). "
                    "This may indicate redundant or conflicting constraints.",
                    RuntimeWarning,
                    stacklevel=2
                )
    except np.linalg.LinAlgError:
        pass  # Skip condition check if it fails

    # Form RHS for Lagrange multipliers
    b = rhs - J @ a0

    # Solve for multipliers
    try:
        lam = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        # Try least-squares solution as fallback
        warnings.warn(
            f"Constraint system is singular. Attempting least-squares solution. "
            f"Original error: {e}",
            RuntimeWarning,
            stacklevel=2
        )
        lam, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        if rank < A.shape[0]:
            warnings.warn(
                f"Constraint system is rank-deficient (rank {rank}/{A.shape[0]}). "
                "Check for redundant constraints.",
                RuntimeWarning,
                stacklevel=2
            )

    # Recover constrained acceleration
    a = a0 + Minv @ (J.T @ lam)

    return a, lam


class HybridSolver:
    """
    Fixed-step solver combining semi-implicit Euler integration with KKT constraints.
    
    Uses symplectic (semi-implicit) Euler integration for stability and energy
    conservation. Constraints are enforced at the velocity level using KKT
    system with Baumgarte stabilization.
    
    Parameters
    ----------
    alpha : float
        Baumgarte position correction parameter [1/s].
        Recommended range: 1-10. Higher values provide stronger constraint
        enforcement but may cause instability. Use lower values for stiff systems.
    beta : float
        Baumgarte velocity correction parameter [-].
        Recommended range: 0.1-1.0. Typically β ≈ 0.2*α for good stability.
        
    Attributes
    ----------
    alpha : float
        Position stabilization parameter
    beta : float
        Velocity stabilization parameter
        
    Notes
    -----
    Semi-implicit Euler integration order:
        1. Solve KKT for accelerations: a = f(q_n, v_n, t_n)
        2. Update velocities: v_{n+1} = v_n + a * dt
        3. Update positions: q_{n+1} = q_n + v_{n+1} * dt
        
    This ordering (velocity before position) provides better energy conservation
    than explicit Euler and is symplectic for unconstrained systems.
    
    Examples
    --------
    >>> solver = HybridSolver(alpha=5.0, beta=1.0)
    >>> world = World(...)
    >>> for _ in range(num_steps):
    ...     world.step(solver, dt=0.01)
    """
    def __init__(self, alpha: float = 0.0, beta: float = 0.0) -> None:
        if alpha < 0:
            raise ValueError(f"Alpha must be non-negative, got {alpha}")
        if beta < 0:
            raise ValueError(f"Beta must be non-negative, got {beta}")
            
        self.alpha = float(alpha)
        self.beta = float(beta)

    def step(
        self, 
        bodies: List[RigidBody6DOF], 
        constraints: List[Constraint], 
        dt: float
    ) -> NDArray[np.float64]:
        """
        Advance simulation by one time step.
        
        Parameters
        ----------
        bodies : List[RigidBody6DOF]
            All bodies in the system
        constraints : List[Constraint]
            All constraints to enforce
        dt : float
            Time step size [s]
            
        Returns
        -------
        NDArray[np.float64]
            Computed accelerations (6N,) [m/s², rad/s²]
            
        Notes
        -----
        This method:
        1. Assembles the global KKT system
        2. Solves for constrained accelerations
        3. Integrates each body's state forward in time
        """
        if dt <= 0:
            raise ValueError(f"Time step must be positive, got {dt}")
            
        # Assemble and solve KKT system
        Minv, J, F, rhs, _ = assemble_system(bodies, constraints, self.alpha, self.beta)
        a, _ = solve_kkt(Minv, J, F, rhs)

        # Integrate each body
        for i, b in enumerate(bodies):
            a_lin = a[6*i:6*i+3]
            a_ang = a[6*i+3:6*i+6]
            b.integrate_semi_implicit(dt, a_lin, a_ang)
            
        return a


class HybridIVPSolver:
    """
    Variable-step stiff integrator using scipy.integrate.solve_ivp.
    
    Combines KKT constraint enforcement with adaptive-step implicit integration
    (Radau IIA or BDF methods). Suitable for stiff systems requiring high accuracy
    and automatic step size control.
    
    Parameters
    ----------
    method : str
        Integration method. Options:
        - 'Radau': Implicit Runge-Kutta (Radau IIA, order 5). Best for stiff systems.
        - 'BDF': Backward differentiation formulas (order 1-5). Good for very stiff ODEs.
        - 'RK45': Explicit Runge-Kutta (order 4-5). Use only for non-stiff systems.
    rtol : float
        Relative tolerance for adaptive stepping. Smaller = more accurate.
        Recommended: 1e-6 to 1e-8
    atol : float
        Absolute tolerance for adaptive stepping. Smaller = more accurate.
        Recommended: 1e-8 to 1e-10
    max_step : float | None
        Maximum allowed step size [s]. None for unlimited. Use to prevent
        overshooting fast dynamics.
    alpha : float
        Baumgarte position stabilization [1/s]
    beta : float
        Baumgarte velocity stabilization [-]
        
    Attributes
    ----------
    method : str
        Integration method name
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    max_step : float | None
        Maximum step size
    alpha : float
        Position stabilization parameter
    beta : float
        Velocity stabilization parameter
        
    Notes
    -----
    State vector per body: [p(3), q(4), v(3), ω(3)] = 13 DOF
    Total state size: 13 * num_bodies
    
    The solver uses quaternion integration with proper normalization to
    maintain unit quaternion constraint throughout adaptive stepping.
    
    Examples
    --------
    >>> solver = HybridIVPSolver(method='Radau', rtol=1e-6, atol=1e-8, 
    ...                          alpha=5.0, beta=2.0, max_step=1.0)
    >>> world = World(...)
    >>> sol = world.integrate_to(solver, t_end=100.0)
    """
    def __init__(
        self, 
        method: str = "Radau", 
        rtol: float = 1e-6, 
        atol: float = 1e-8,
        max_step: float | None = None, 
        alpha: float = 0.0, 
        beta: float = 0.0
    ) -> None:
        valid_methods = ["Radau", "BDF", "RK45", "DOP853", "LSODA"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")
        if rtol <= 0 or atol <= 0:
            raise ValueError("Tolerances must be positive")
        if alpha < 0 or beta < 0:
            raise ValueError("Baumgarte parameters must be non-negative")
            
        self.method = method
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.max_step = max_step
        self.alpha = float(alpha)
        self.beta = float(beta)

    def _pack(self, bodies: List[RigidBody6DOF]) -> NDArray[np.float64]:
        """Pack body states into flat array: [p, q, v, w] per body."""
        y = []
        for b in bodies:
            y.extend([*b.p, *b.q, *b.v, *b.w])
        return np.array(y, dtype=np.float64)

    def _unpack_to_world(self, y: NDArray[np.float64], bodies: List[RigidBody6DOF]) -> None:
        """Unpack flat state array into body objects."""
        for k, b in enumerate(bodies):
            off = 13 * k
            b.p[:] = y[off:off+3]
            b.q[:] = y[off+3:off+7]
            b.v[:] = y[off+7:off+10]
            b.w[:] = y[off+10:off+13]

    def integrate(self, world, t_end: float):
        """
        Integrate world state from current time to t_end using adaptive IVP solver.
        
        Parameters
        ----------
        world : World
            World object containing bodies, constraints, and forces
        t_end : float
            Final integration time [s]
            
        Returns
        -------
        sol : OdeResult
            scipy.integrate.OdeResult object containing:
            - t: array of time points
            - y: array of state vectors at each time point
            - success: whether integration succeeded
            - status: termination status (0=success, 1=event triggered)
            - t_events: times when terminal events occurred
            
        Notes
        -----
        - Updates world state to final integrated values
        - Logs state at each solver time point if logger is enabled
        - Terminal event stops integration when payload touches ground
        """
        try:
            from scipy.integrate import solve_ivp
        except ImportError as e:
            raise ImportError(
                "SciPy is required for HybridIVPSolver. "
                "Install with: pip install scipy>=1.8"
            ) from e

        bodies = world.bodies
        constraints = world.constraints

        def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            """Right-hand side function for ODE solver."""
            # Unpack state into bodies
            self._unpack_to_world(y, bodies)

            # Clear and apply forces at time t
            for b in bodies:
                b.clear_forces()
                for fb in b.per_body_forces:
                    fb.apply(b, t)
            for fg in world.global_forces:
                for b in bodies:
                    fg.apply(b, t)
            for fpair in world.interaction_forces:
                fpair.apply_pair(t)

            # Assemble and solve KKT system
            Minv, J, F, rhs_v, _ = assemble_system(bodies, constraints, self.alpha, self.beta)
            a, _ = solve_kkt(Minv, J, F, rhs_v)

            # Compose state derivative: [v, qdot, a_lin, a_ang]
            ydot = np.zeros_like(y)
            for i, b in enumerate(bodies):
                a_lin = a[6*i:6*i+3]
                a_ang = a[6*i+3:6*i+6]
                off = 13 * i
                
                # Position derivative
                ydot[off:off+3] = b.v
                
                # Quaternion derivative (ensure normalized first)
                b.q[:] = quat_normalize(b.q)
                ydot[off+3:off+7] = quat_derivative(b.q, b.w)
                
                # Velocity derivatives
                ydot[off+7:off+10] = a_lin
                ydot[off+10:off+13] = a_ang
                
            return ydot

        def touchdown_event(t: float, y: NDArray[np.float64]) -> float:
            """Terminal event: payload altitude crosses ground level."""
            z = y[13 * world.payload_index + 2]
            return float(z - world.ground_z)
        
        touchdown_event.terminal = True   # type: ignore[attr-defined]
        touchdown_event.direction = -1.0  # type: ignore[attr-defined]

        # Pack initial state
        y0 = self._pack(bodies)
        
        # Build solve_ivp kwargs conditionally (to handle max_step=None)
        solve_ivp_kwargs = {
            'method': self.method,
            'rtol': self.rtol,
            'atol': self.atol,
            'events': touchdown_event,
            'dense_output': False,
        }
        
        # Only add max_step if it's not None (scipy doesn't accept None in newer versions)
        if self.max_step is not None:
            solve_ivp_kwargs['max_step'] = self.max_step
        
        # Integrate
        sol = solve_ivp(
            rhs,
            t_span=(world.t, t_end),
            y0=y0,
            **solve_ivp_kwargs
        )

        # Update world with final state
        self._unpack_to_world(sol.y[:, -1], bodies)
        world.t = float(sol.t[-1])
        world.t_touchdown = float(sol.t_events[0][0]) if len(sol.t_events[0]) else None

        # Log integration trajectory if logger enabled
        if world.logger is not None and sol.t.size > 0:
            for k, tk in enumerate(sol.t):
                # Set state to this sample
                self._unpack_to_world(sol.y[:, k], bodies)
                world.t = float(tk)

                # Re-apply forces for consistent logging
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
