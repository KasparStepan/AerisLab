"""
World simulation orchestrator for constrained rigid body dynamics.

Manages bodies, forces, constraints, and time integration with optional
logging and automatic output organization.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

# Import System type for type hints (avoid circular import)
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.constraints import Constraint
from aerislab.dynamics.forces import Gravity, Spring
from aerislab.logger import CSVLogger
from aerislab.models.atmosphere.isa import FastISA, AtmosphereModel

if TYPE_CHECKING:
    from aerislab.components.system import System as SystemType

from .solver import HybridIVPSolver, HybridSolver

# Default output directory
DEFAULT_OUTPUT_DIR = Path("output")
EPSILON_GROUND = 1e-9  # Tolerance for ground detection


class World:
    """
    Container and orchestrator for multi-body dynamics simulation.

    Manages rigid bodies, forces, constraints, time evolution, and optional
    data logging with automatic output organization.

    Parameters
    ----------
    ground_z : float
        Ground plane altitude in world frame [m]. Simulation terminates when
        payload crosses this altitude (going downward).
    payload_index : int
        Index of the body to monitor for ground contact termination.
    simulation_name : str | None
        Name for this simulation. Used to organize output files. If None,
        logging is disabled by default. Use enable_logging() to activate.
    output_dir : Path | str | None
        Base directory for all simulation outputs. Defaults to "./output".
        Final structure: output_dir/simulation_name_timestamp/logs/ and /plots/
    auto_timestamp : bool
        If True, append timestamp to simulation folder name to prevent overwrites.
        Recommended for parameter studies and repeated runs.
    auto_save_plots : bool
        If True, automatically generate and save plots when simulation completes.
        Only works if logging is enabled.

    Attributes
    ----------
    bodies : List[RigidBody6DOF]
        All rigid bodies in the simulation
    global_forces : List
        Forces applied to all bodies (e.g., gravity)
    interaction_forces : List[Spring]
        Pairwise forces between bodies (e.g., springs, tethers)
    constraints : List[Constraint]
        Holonomic constraints (e.g., distance, point-weld)
    t : float
        Current simulation time [s]
    t_touchdown : float | None
        Time when payload touched ground [s], or None if not yet
    logger : CSVLogger | None
        Data logger instance, or None if logging disabled
    output_path : Path | None
        Path to simulation output directory

    Notes
    -----
    **Termination Logic:**
    By default, simulation stops when the payload body crosses ground_z going downward.
    Override with set_termination_callback() for custom termination conditions.

    **Output Organization:**
    When logging is enabled, creates:
        output_dir/
            simulation_name_20260109_101530/
                logs/
                    simulation.csv
                plots/
                    payload_traj.png
                    payload_vel_acc.png
                    ...

    Examples
    --------
    >>> # Explicit logging (recommended)
    >>> world = World(ground_z=0.0, payload_index=0)
    >>> world.enable_logging("parachute_drop")
    >>> # ... add bodies, forces, constraints ...
    >>> world.run(solver, duration=100.0, dt=0.01)
    >>> world.save_plots()  # Generate plots from logged data

    >>> # Convenience factory with auto-logging
    >>> world = World.with_logging(
    ...     name="my_sim",
    ...     ground_z=0.0,
    ...     auto_save_plots=True  # Plots generated automatically
    ... )
    """

    def __init__(
        self,
        ground_z: float = 0.0,
        payload_index: int = 0,
        simulation_name: str | None = None,
        output_dir: Path | str | None = None,
        auto_timestamp: bool = True,
        auto_save_plots: bool = False,
        atmosphere: AtmosphereModel = None,
    ) -> None:
        self.bodies: list[RigidBody6DOF] = []
        self.global_forces: list = []
        self.interaction_forces: list[Spring] = []
        self.constraints: list[Constraint] = []
        self._world_index: int | None = None  # lazy WORLD anchor, see .WORLD
        self.payload_index = int(payload_index)
        self.ground_z = float(ground_z)
        self.t = 0.0
        self.t_touchdown: float | None = None
        self.termination_callback: Callable[[World], bool] | None = None

        # Output configuration
        self._simulation_name = simulation_name
        self._output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        self._auto_timestamp = auto_timestamp
        self._auto_save_plots = auto_save_plots
        self.output_path: Path | None = None
        self.logger: CSVLogger | None = None

        # Dictionary to store separate force components for the current step
        self.force_breakdown: dict[str, NDArray] = {}

        # System-level management (component architecture)
        self.systems: list[SystemType] = []

        # Atmosphere model (optional, for aerodynamic forces)
        self.atmosphere = atmosphere if atmosphere is not None else FastISA()

        # Enable logging if name provided
        if simulation_name is not None:
            self.enable_logging(simulation_name)

    @classmethod
    def with_logging(
        cls,
        name: str,
        ground_z: float = 0.0,
        payload_index: int = 0,
        output_dir: Path | str | None = None,
        auto_save_plots: bool = True,
    ) -> World:
        """
        Convenience factory to create World with logging pre-enabled.

        This is the recommended way to create a World when you want automatic
        data logging and output organization.

        Parameters
        ----------
        name : str
            Simulation name (required)
        ground_z : float
            Ground plane altitude [m]
        payload_index : int
            Index of payload body for termination detection
        output_dir : Path | str | None
            Base output directory
        auto_save_plots : bool
            Automatically generate plots when simulation completes

        Returns
        -------
        World
            Configured World instance with logging enabled

        Examples
        --------
        >>> world = World.with_logging(
        ...     name="drop_test_01",
        ...     ground_z=0.0,
        ...     auto_save_plots=True
        ... )
        """
        return cls(
            ground_z=ground_z,
            payload_index=payload_index,
            simulation_name=name,
            output_dir=output_dir,
            auto_timestamp=True,
            auto_save_plots=auto_save_plots,
        )

    def enable_logging(self, name: str | None = None) -> Path:
        """
        Enable data logging with automatic output organization.

        Creates output directory structure and initializes CSV logger.
        Can be called at any time before or during simulation.

        Parameters
        ----------
        name : str | None
            Simulation name. If None, uses name from __init__ or raises error.

        Returns
        -------
        Path
            Path to the created output directory

        Raises
        ------
        ValueError
            If no simulation name available

        Notes
        -----
        Creates directory structure:
            output/simulation_name_timestamp/
                logs/
                plots/
        """
        if name is not None:
            self._simulation_name = name

        if self._simulation_name is None:
            raise ValueError(
                "Simulation name required for logging. "
                "Either pass name to __init__ or to enable_logging()."
            )

        # Create output path with optional timestamp
        if self._auto_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{self._simulation_name}_{timestamp}"
        else:
            folder_name = self._simulation_name

        self.output_path = self._output_dir / folder_name

        # Create directory structure
        logs_dir = self.output_path / "logs"
        plots_dir = self.output_path / "plots"
        logs_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logger
        csv_path = logs_dir / "simulation.csv"
        self.logger = CSVLogger(str(csv_path))

        print(f"[World] Logging enabled: {self.output_path}")
        print(f"        Logs: {logs_dir}")
        print(f"        Plots: {plots_dir}")

        return self.output_path

    def disable_logging(self) -> None:
        """
        Disable logging and close any open log files.

        Useful for performance testing or when you want to disable
        logging partway through a parameter study.
        """
        if self.logger is not None:
            self.logger.close()
            self.logger = None
            print("[World] Logging disabled")

    # --- Configuration Methods ---

    def set_logger(self, logger: CSVLogger) -> None:
        """
        Set custom logger instance.

        For advanced users who want custom logging behavior.
        Most users should use enable_logging() instead.
        """
        warnings.warn(
            "set_logger() is deprecated. Use enable_logging() for automatic "
            "output organization, or set world.logger directly for custom loggers.",
            DeprecationWarning,
            stacklevel=2
        )
        self.logger = logger

    @property
    def WORLD(self) -> int:  # noqa: N802 - constant-style accessor for the world index
        """
        Index of the implicit, immovable world-frame anchor body.

        Created lazily on first access: a ``fixed=True`` body at the origin is
        appended to ``self.bodies`` and its index cached. Use it to pin a DOF in
        absolute space without faking a heavy anchor, e.g.::

            DistanceConstraint(world.bodies, world.WORLD, bob, ...)

        Because it is appended (never inserted), all previously assigned body
        indices and ``payload_index`` remain valid. The body never moves and is
        excluded from logging/plots.
        """
        if self._world_index is None:
            anchor = RigidBody6DOF(
                name="__world__",
                mass=0.0,
                inertia_tensor_body=np.eye(3),
                position=np.zeros(3),
                orientation=np.array([0.0, 0.0, 0.0, 1.0]),
                fixed=True,
            )
            self._world_index = self.add_body(anchor)
        return self._world_index

    def add_body(self, body: RigidBody6DOF) -> int:
        """
        Add a rigid body to the simulation.

        Parameters
        ----------
        body : RigidBody6DOF
            The body to add

        Returns
        -------
        int
            Index of the added body (for constraint/joint definitions)
        """
        self.bodies.append(body)
        return len(self.bodies) - 1

    def add_global_force(self, f) -> None:
        """
        Add force that applies to all bodies (e.g., gravity).

        Parameters
        ----------
        f : Force
            Force object with apply(body, t) method
        """
        self.global_forces.append(f)

    def add_interaction_force(self, fpair: Spring) -> None:
        """
        Add pairwise interaction force (e.g., spring, tether).

        Parameters
        ----------
        fpair : Spring
            Spring-like force object with apply_pair(t) method
        """
        self.interaction_forces.append(fpair)

    def add_constraint(self, c: Constraint) -> None:
        """
        Add holonomic constraint (e.g., distance, weld joint).

        Parameters
        ----------
        c : Constraint
            Constraint object enforced via KKT solver
        """
        self.constraints.append(c)

    def set_termination_callback(self, fn: Callable[[World], bool]) -> None:
        """
        Set custom termination condition.

        Parameters
        ----------
        fn : Callable[[World], bool]
            Function that takes World and returns True to stop simulation.
            Called after each integration step.

        Examples
        --------
        >>> # Stop when payload reaches specific altitude
        >>> world.set_termination_callback(
        ...     lambda w: w.bodies[0].p[2] < 500.0
        ... )
        """
        self.termination_callback = fn

    def add_system(self, system: SystemType) -> None:
        """
        Add a multi-component system.

        Automatically registers all bodies and constraints from the system.
        Components manage their own forces via the System interface.

        Parameters
        ----------
        system : System
            Multi-component system with bodies and constraints

        Notes
        -----
        This is the preferred way to add complex assemblies like
        parachute-payload systems. The System handles component state
        updates and force application.

        Examples
        --------
        >>> from aerislab.components import System, Payload, Parachute
        >>> system = System("recovery_system")
        >>> system.add_component(Payload(...))
        >>> system.add_component(Parachute(...))
        >>> world.add_system(system)
        """

        self.systems.append(system)

        # Register all bodies from the system
        for component in system.components:
            self.add_body(component.body)

        # Register all constraints from the system
        for constraint in system.constraints:
            self.add_constraint(constraint)

        print(
            f"[World] Added system '{system.name}' with "
            f"{len(system.components)} components, "
            f"{len(system.constraints)} constraints"
        )

    # --- Fixed-Step Integration ---

    def step(self, solver: HybridSolver, dt: float) -> bool:
        """
        Advance simulation by one fixed time step.

        Parameters
        ----------
        solver : HybridSolver
            Fixed-step solver instance
        dt : float
            Time step [s]

        Returns
        -------
        bool
            True if termination condition met, False otherwise

        Notes
        -----
        Performs:
        1. Clear force accumulators
        2. Apply all forces
        3. Record pre-step payload altitude
        4. Solve KKT and integrate
        5. Log state (if enabled)
        6. Check termination
        7. Interpolate touchdown time if ground crossed
        """
        # 1) Clear per-body force accumulators
        for b in self.bodies:
            b.clear_forces()

        self.force_breakdown.clear()

        # 1.5) Update component states (deployment, actuation, etc.)
        for system in self.systems:
            system.update_all_states(self.t, dt)

        # 2) Apply forces
        # 2a) Apply system component forces
        for system in self.systems:
            system.apply_all_forces(self.t)

        # 2b) Apply per-body forces (non-component bodies)
        for b in self.bodies:
            for fb in b.per_body_forces:
                fb.apply(b, self.t)
                # Capture force if the object exposes it
                if hasattr(fb, "last_force"):
                    self.force_breakdown[f"{b.name}_{type(fb).__name__}"] = fb.last_force.copy()

        for fg in self.global_forces:
            for b in self.bodies:
                fg.apply(b, self.t)
                # Capture global force (requires apply() to update last_force per body)
                if hasattr(fg, "last_force"):
                    self.force_breakdown[f"{b.name}_{type(fg).__name__}"] = fg.last_force.copy()

        for fpair in self.interaction_forces:
            fpair.apply_pair(self.t)

        # 3) Record pre-step payload altitude
        payload = self.bodies[self.payload_index]
        z_pre = payload.p[2]

        # 4) Integrate
        solver.step(self.bodies, self.constraints, dt)
        self.t += dt

        # 5) Log state
        if self.logger is not None:
            self.logger.log(self)

        # 6) Check termination
        stop = False
        if self.termination_callback:
            stop = bool(self.termination_callback(self))
        else:
            # Default: ground contact
            z_post = payload.p[2]
            if (z_pre > self.ground_z + EPSILON_GROUND) and (z_post <= self.ground_z):
                # Linear interpolation for touchdown time
                frac = (z_pre - self.ground_z) / max((z_pre - z_post), EPSILON_GROUND)
                self.t_touchdown = float(self.t - dt + frac * dt)
                stop = True
            elif z_post <= self.ground_z:
                # Already below ground
                self.t_touchdown = self.t
                stop = True

        return stop

    def _constraint_label(self, c: Constraint) -> str:
        """Human-readable label for diagnostics, e.g. 'DistanceConstraint(pivot↔bob)'."""
        try:
            idx = c.index_map()
            names = []
            for k in idx:
                nm = self.bodies[k].name if 0 <= k < len(self.bodies) else f"#{k}"
                names.append("WORLD" if k == self._world_index else nm)
            return f"{type(c).__name__}({'↔'.join(names)})"
        except Exception:
            return type(c).__name__

    def validate_constraints(
        self, strict: bool = False, tol: float = 1e-9
    ) -> dict:
        """
        Static well-posedness check of the assembled constraint set.

        Runs once at the current state and reports authoring problems that would
        otherwise surface as silent-wrong results or a mid-run singular solve:

        - **Rank deficiency** of the Schur complement ``A = J·M⁻¹·Jᵀ`` (the
          matrix that must be invertible). This catches redundant/conflicting
          constraints *and* constraints acting only on fixed bodies — both make
          ``A`` singular even when ``J`` has full row rank.
        - **Initial-condition consistency**: ``‖C(q₀)‖`` and ``‖J·v₀‖`` per
          constraint, so a lock that disagrees with the starting state is caught.
        - **Per-body DOF accounting** (rows consumed vs. 6 DOF).

        Important: rank is configuration-dependent — passing here does *not*
        prove the system stays non-singular for all time (a pose can go singular
        mid-run). The runtime conditioning guard in ``solve_kkt`` remains the
        backstop for dynamic singularities. This check is for authoring errors.

        Parameters
        ----------
        strict : bool
            If True, raise ``RuntimeError`` when problems are found instead of
            only warning.
        tol : float
            Absolute tolerance for rank and IC-residual checks.

        Returns
        -------
        dict
            ``{ok, n_rows, rank, deficiency, cond, issues, per_body}``.
        """
        from .solver import assemble_system  # local import avoids cycle at module load

        issues: list[str] = []
        if not self.constraints:
            return {"ok": True, "n_rows": 0, "rank": 0, "deficiency": 0,
                    "cond": 0.0, "issues": [], "per_body": {}}

        Minv, J, _, _, v = assemble_system(self.bodies, self.constraints, 0.0, 0.0)
        m = J.shape[0]

        # --- Rank / conditioning of the Schur complement (the thing inverted) ---
        A = J @ (Minv @ J.T)
        rank = int(np.linalg.matrix_rank(A, tol=tol)) if m else 0
        deficiency = m - rank
        cond = float(np.linalg.cond(A)) if m and rank == m else float("inf")
        if deficiency > 0:
            issues.append(
                f"Schur complement A is rank-deficient: rank {rank}/{m} "
                f"(deficiency {deficiency}). Constraints are redundant, conflicting, "
                f"or act only on fixed bodies — multipliers (reaction forces) are "
                f"not unique."
            )
        elif cond > 1e10:
            issues.append(f"Constraint system is ill-conditioned (cond(A)={cond:.2e}).")

        # --- Per-constraint diagnostics: IC residuals + fixed-only detection ---
        JMinv = J @ Minv
        row = 0
        per_body: dict[str, int] = {}
        for c in self.constraints:
            r = c.rows()
            label = self._constraint_label(c)
            block = slice(row, row + r)

            C = np.atleast_1d(c.evaluate())
            if np.linalg.norm(C) > tol:
                issues.append(f"{label}: violated at start, ‖C‖={np.linalg.norm(C):.2e} "
                              f"(initial state off the constraint manifold).")
            Jv = J[block] @ v
            if np.linalg.norm(Jv) > tol:
                issues.append(f"{label}: velocity-inconsistent start, ‖J·v‖="
                              f"{np.linalg.norm(Jv):.2e}.")
            if np.allclose(JMinv[block], 0.0, atol=tol):
                issues.append(f"{label}: acts only on fixed/immovable DOFs — it "
                              f"adds {r} singular row(s) to A.")

            for k in c.index_map():
                b = self.bodies[k]
                if not getattr(b, "fixed", False):
                    per_body[b.name] = per_body.get(b.name, 0) + r
            row += r

        for name, used in per_body.items():
            if used > 6:
                issues.append(f"Body '{name}' is over-constrained: {used} rows on 6 DOF.")

        ok = not issues
        if not ok:
            msg = "Constraint validation found issues:\n  - " + "\n  - ".join(issues)
            if strict:
                raise RuntimeError(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        return {"ok": ok, "n_rows": m, "rank": rank, "deficiency": deficiency,
                "cond": cond, "issues": issues, "per_body": per_body}

    def run(
        self,
        solver: HybridSolver,
        duration: float,
        dt: float,
        log_interval: float = 1.0,
        validate: bool = True,
    ) -> None:
        """
        Run fixed-step simulation for specified duration.

        Parameters
        ----------
        solver : HybridSolver
            Fixed-step solver
        duration : float
            Simulation duration [s]
        dt : float
            Fixed time step [s]
        log_interval : float
            Interval [s] for printing progress to terminal. Set to <= 0 to disable.

        Notes
        -----
        - Logs initial state before integration
        - Stops early if termination condition met
        - Flushes logger and generates plots (if enabled) when complete
        """
        t_end = self.t + float(duration)
        last_log_time = self.t

        # Catch authoring errors (redundant/conflicting constraints, bad ICs)
        # before integrating; warns rather than raises so existing runs proceed.
        if validate and self.constraints:
            self.validate_constraints(strict=False)

        # Log initial state
        if self.logger is not None:
            self.logger.log(self)

        print(f"[World] Starting fixed-step simulation: {duration}s duration, dt={dt}s")

        try:
            while self.t < t_end:
                if self.step(solver, dt):
                    print(f"[World] Simulation terminated at t={self.t:.6f}s")
                    if self.t_touchdown is not None:
                        print(f"        Touchdown detected at t={self.t_touchdown:.6f}s")
                    break
                
                # Terminal Progress Log
                if log_interval > 0 and (self.t - last_log_time) >= log_interval:
                    # Basic status: Time and Altitude of payload
                    payload = self.bodies[self.payload_index]
                    z = payload.p[2]
                    vz = payload.v[2]
                    print(f"[World] t={self.t:6.2f}s | Payload z={z:8.2f}m, vz={vz:6.2f}m/s")
                    last_log_time = self.t

        finally:
            # Ensure data is written
            if self.logger:
                self.logger.flush()

            # Auto-generate plots if requested
            if self._auto_save_plots and self.logger is not None:
                print("[World] Auto-generating plots...")
                self.save_plots()

    # --- Variable-Step Integration ---

    def integrate_to(self, solver: HybridIVPSolver, t_end: float, log_interval: float = 1.0):
        """
        Integrate with adaptive-step IVP solver.

        Parameters
        ----------
        solver : HybridIVPSolver
            Variable-step solver
        t_end : float
            Final time [s]
        log_interval : float
            Interval [s] for printing progress. Set <= 0 to disable.

        Returns
        -------
        OdeResult
            scipy integration result object (merged if chunked)

        Notes
        -----
        - Automatically logs trajectory at solver time points
        - Auto-generates plots if enabled
        """
        if log_interval <= 0:
            # Single shot integration
            print(f"[World] Starting variable-step integration: {t_end - self.t:.2f}s duration")
            sol = solver.integrate(self, t_end)
        else:
            # Chunked integration for progress updates
            print(f"[World] Starting variable-step integration: {t_end - self.t:.2f}s duration")
            current_t = self.t
            
            # We need to accumulate results if we want to return a full solution object
            # But the primary output is the CSV log. The return value is rarely used by end users.
            # Let's just return the last solution object for now, or a dummy.
            
            final_sol = None
            
            while current_t < t_end:
                next_t = min(current_t + log_interval, t_end)
                
                # Run solver for this chunk
                sol = solver.integrate(self, next_t)
                final_sol = sol
                
                # Check for termination (ground contact)
                if sol.status != 0: # 1 means event occurred (touchdown)
                     if sol.t_events and len(sol.t_events[0]) > 0:
                         print(f"[World] Touchdown detected at t={sol.t_events[0][0]:.6f}s")
                     break
                
                # Update progress
                current_t = self.t # solver updates world.t
                
                # Terminal Log
                payload = self.bodies[self.payload_index]
                z = payload.p[2]
                vz = payload.v[2]
                print(f"[World] t={current_t:6.2f}s | Payload z={z:8.2f}m, vz={vz:6.2f}m/s")

        if self.logger is not None:
            self.logger.flush()

        # Auto-generate plots if requested
        if self._auto_save_plots and self.logger is not None:
            print("[World] Auto-generating plots...")
            self.save_plots()

        return final_sol

    # --- Plotting and Analysis ---

    def save_plots(
        self,
        bodies: list[str] | None = None,
        show: bool = False,
    ) -> None:
        """
        Generate and save standard analysis plots from logged data.

        Creates trajectory, velocity/acceleration, and force plots for
        specified bodies. Automatically uses correct output directory.

        Parameters
        ----------
        bodies : list[str] | None
            Body names to plot. If None, plots all bodies in simulation.
        show : bool
            If True, display plots interactively (blocks execution)

        Raises
        ------
        RuntimeError
            If logging is not enabled or no data logged yet

        Notes
        -----
        Generates for each body:
        - 3D trajectory plot
        - Velocity and acceleration time series
        - Force components time series

        Examples
        --------
        >>> world.run(solver, duration=100.0, dt=0.01)
        >>> world.save_plots(bodies=["payload", "canopy"])
        """
        if self.logger is None or self.output_path is None:
            raise RuntimeError(
                "Logging must be enabled to save plots. "
                "Call enable_logging() or use World.with_logging()."
            )

        from aerislab.visualization.plotting import (
            plot_forces,
            plot_trajectory_3d,
            plot_velocity_and_acceleration,
            plot_force_breakdown,
        )

        csv_path = self.output_path / "logs" / "simulation.csv"
        plots_dir = self.output_path / "plots"

        if not csv_path.exists():
            raise RuntimeError(
                f"No log file found at {csv_path}. "
                "Has the simulation been run yet?"
            )

        if bodies is None:
            bodies = [b.name for b in self.bodies if not getattr(b, "fixed", False)]

        print(f"[World] Generating plots for: {', '.join(bodies)}")
        for name in bodies:
            # 3D trajectory
            plot_trajectory_3d(
                str(csv_path), name,
                save_path=str(plots_dir / f"{name}_trajectory_3d.png"),
                show=show,
            )
            # Velocity and acceleration
            plot_velocity_and_acceleration(
                str(csv_path), name,
                save_path=str(plots_dir / f"{name}_velocity_acceleration.png"),
                show=show,
                magnitude=False
            )
            # Forces (Components)
            plot_forces(
                str(csv_path), name,
                save_path=str(plots_dir / f"{name}_forces.png"),
                show=show,
                magnitude=False
            )
            # Detailed Force Breakdown (New)
            plot_force_breakdown(
                str(csv_path), name,
                save_path=str(plots_dir / f"{name}_force_breakdown.png"),
                show=show
            )

        print(f"[World] Plots saved to: {plots_dir}")

    def get_energy(self) -> dict[str, float]:
        """
        Compute total system energy (diagnostic).

        Returns
        -------
        dict[str, float]
            Dictionary with keys:
            - 'kinetic': total kinetic energy [J]
            - 'potential': gravitational potential energy [J]
            - 'total': sum of kinetic and potential [J]

        Notes
        -----
        Only accounts for gravity in potential energy.
        Useful for validating energy conservation in unconstrained systems.

        Examples
        --------
        >>> energy = world.get_energy()
        >>> print(f"Total energy: {energy['total']:.2f} J")
        """
        KE = sum(b.kinetic_energy() for b in self.bodies)

        # Potential energy (gravity only)
        PE = 0.0
        for fg in self.global_forces:
            if isinstance(fg, Gravity):
                for b in self.bodies:
                    # PE = -m * g · r (relative to ground)
                    PE -= b.mass * np.dot(fg.g, b.p - np.array([0, 0, self.ground_z]))
                break  # Only first gravity force

        return {
            'kinetic': KE,
            'potential': PE,
            'total': KE + PE
        }
