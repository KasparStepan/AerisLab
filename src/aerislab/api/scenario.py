
"""
Scenario API: Fluent interface for defining and running simulations.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from aerislab.core.simulation import World
from aerislab.core.solver import HybridIVPSolver
from aerislab.components.system import System
from aerislab.components.base import Component
from aerislab.dynamics.forces import Gravity
from aerislab.dynamics.joints import RigidTetherJoint

SOLVER_PRESETS = {
    "default": {"method": "Radau", "rtol": 1e-6, "atol": 1e-8},
    "fast": {"method": "RK45", "rtol": 1e-3, "atol": 1e-6},
    "accurate": {"method": "Radau", "rtol": 1e-9, "atol": 1e-12},
    "stiff": {"method": "Radau", "rtol": 1e-6, "atol": 1e-8}, # Radau is good for stiff
}

class Scenario:
    def __init__(self, name: str, output_dir: str = "output"):
        self.name = name
        self.world = World.with_logging(
            name=name,
            output_dir=Path(output_dir),
            auto_save_plots=False # We handle this explicitly
        )
        self.systems: list[System] = []
        self.current_system: System | None = None
        
        # Default gravity
        self.world.add_global_force(Gravity(np.array([0, 0, -9.81])))

        self.world.add_global_force(Gravity(np.array([0, 0, -9.81])))
        
        # Solver defaults
        self._solver_params = SOLVER_PRESETS["default"]

    def set_initial_state(self, altitude: float | None = None, velocity: list[float] | None = None) -> 'Scenario':
        """
        Set the initial state of the primary system.
        
        Shifts the entire simulation vertically to strict altitude and sets velocity.
        Useful for running same system from different drop conditions.
        """
        if not self.world.bodies:
             # If called before adding systems, we can't do anything yet.
             # Ideally this should be called AFTER adding systems.
             # Or we store it and apply later? Storing is better UX but complex.
             # For now, let's warn if empty, or just return.
             print("[Scenario] Warning: No bodies to update. Call add_system() first.")
             return self

        # Primary body logic (same as before: finding payload index)
        # We shift ALL bodies to maintain relative positions
        primary_body = self.world.bodies[self.world.payload_index]
        
        if altitude is not None:
            current_z = primary_body.p[2]
            delta_z = altitude - current_z
            for body in self.world.bodies:
                body.p[2] += delta_z
                
        if velocity is not None:
            v_new = np.array(velocity, dtype=float)
            for body in self.world.bodies:
                # Setting velocity vector directly
                body.v = v_new.copy()
                
        return self

    def configure_solver(self, preset: str = "default", **kwargs) -> 'Scenario':
        """
        Configure the solver with a preset or custom overrides.
        
        Presets: 'default', 'fast', 'accurate', 'stiff'
        Kwargs: method, rtol, atol, alpha, beta
        """
        if preset in SOLVER_PRESETS:
            self._solver_params = SOLVER_PRESETS[preset].copy()
        
        self._solver_params.update(kwargs)
        return self

    def add_system(self, components: list[Component], name: str = "main_system") -> 'Scenario':
        """
        Create a new system with the given components.
        """
        sys = System(name=name)
        for comp in components:
            sys.add_component(comp)
        
        self.systems.append(sys)
        self.current_system = sys # Active system for connecting
        self.world.add_system(sys)
        return self

    def connect(self, comp1: Component, comp2: Component, type: str = "tether", length: float = 0.0) -> 'Scenario':
        """
        Connect two components in the current system.
        """
        if not self.current_system:
             raise RuntimeError("No active system. Call add_system() first.")

        # Find indices
        try:
             idx1 = self.current_system.components.index(comp1)
             idx2 = self.current_system.components.index(comp2)
        except ValueError:
             raise ValueError("Components must belong to the active system.")

        if type == "tether":
             # Auto-calculate attach points? Top of payload, bottom of parachute?
             # For improved UX, we might guess based on relative positions or just standard attach points
             # Radius helps: 
             r1 = comp1.body.radius
             r2 = comp2.body.radius
             
             # Default vertical stack assumption
             local_1 = np.array([0, 0, r1]) # Top
             local_2 = np.array([0, 0, -r2]) # Bottom
             
             joint = RigidTetherJoint(
                 body_i=idx1,
                 body_j=idx2,
                 attach_i_local=local_1,
                 attach_j_local=local_2,
                 length=length
             )
             
             # Attach to world bodies (System needs body list)
             constraint = joint.attach(self.current_system.get_bodies())
             self.current_system.add_constraint(constraint)
             
        return self

    def enable_plotting(self, show: bool = False) -> 'Scenario':
        """
        Enable plot generation at the end of simulation.
        
        Parameters
        ----------
        show : bool
            If True, display plots interactively (e.g. in Jupyter notebooks).
        """
        self._show_plots = show
        # If showing plots, we handle it manually to pass show=True.
        # If not showing, we can let World handle it automatically.
        if show:
            self.world._auto_save_plots = False
        else:
            self.world._auto_save_plots = True
        return self

    def run(self, duration: float = 60.0, log_interval: float = 1.0):
        print(f"Running Scenario: {self.name}")
        
        # Use configured solver params
        method = self._solver_params.get("method", "Radau")
        rtol = self._solver_params.get("rtol", 1e-6)
        atol = self._solver_params.get("atol", 1e-8)
        
        # Filter other potential keys for HybridIVPSolver
        # (HybridIVPSolver takes alpha, beta too)
        solver_kwargs = {k:v for k,v in self._solver_params.items() if k in ["method", "rtol", "atol", "alpha", "beta"]}
        
        print(f"[Scenario] Solver: {method} (rtol={rtol:.1e}, atol={atol:.1e})")
        solver = HybridIVPSolver(**solver_kwargs)
        
        self.world.integrate_to(solver, t_end=duration, log_interval=log_interval)
        
        # Manual plot generation if show=True
        # (If show=False, world handled it automatically above via _auto_save_plots)
        if getattr(self, '_show_plots', False):
             print("[Scenario] Generating plots...")
             self.world.save_plots(show=True)
             
        return self
