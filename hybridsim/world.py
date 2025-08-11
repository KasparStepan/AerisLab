from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from .solver import HybridSolver, HybridIVPSolver, SolverSettings, IVPSettings
from .logger import CSVLogger

@dataclass
class World:
    """Container for bodies, forces, constraints; orchestrates stepping."""
    dt: float
    solver: HybridSolver = field(default_factory=lambda: HybridSolver(SolverSettings()))
    contact_model: Optional["ContactModel"] = None

    bodies: list["RigidBody6DOF"] = field(default_factory=list)
    global_forces: list["Force"] = field(default_factory=list)
    interaction_forces: list["Spring"] = field(default_factory=list)  # legacy/optional
    constraints: list["Constraint"] = field(default_factory=list)
    logger: Optional[CSVLogger] = None

    time: float = 0.0

    # --- registration ---------------------------------------------------------

    def add_body(self, body: "RigidBody6DOF") -> None:
        self.bodies.append(body)

    def add_global_force(self, force: "Force") -> None:
        self.global_forces.append(force)

    def add_interaction_force(self, spring: "Spring") -> None:
        self.interaction_forces.append(spring)

    def add_constraint(self, c: "Constraint") -> None:
        self.constraints.append(c)

    # --- fixed-step stepping --------------------------------------------------

    def step(self) -> None:
        self.solver.step(self, self.dt)
        self.time += self.dt
        if self.logger is not None:
            self.logger.log(self.time, self.bodies)

    def run(self, duration: float) -> None:
        end = self.time + duration
        while self.time < end:
            self.step()

    # --- variable-step integration -------------------------------------------

    def integrate_to(self, t_end: float, ivp: Optional[HybridIVPSolver] = None) -> None:
        """
        Integrate from the current world.time to t_end using a variable-step stiff solver.
        """
        ivp_solver = ivp or HybridIVPSolver(settings=self.solver.settings)
        ivp_solver.integrate(self, t_end)
