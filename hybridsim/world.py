from __future__ import annotations
import numpy as np
from typing import Callable, Optional, List
from .body import RigidBody6DOF
from .forces import Force
from .solver import HybridSolver, HybridIVPSolver

class World:
    """Holds bodies, forces, constraints, time, and termination logic."""
    def __init__(self, ground_z: float = 0.0) -> None:
        self.bodies: List[RigidBody6DOF] = []
        self.global_forces: List[Force] = []
        self.interaction_forces: List = []  # springs etc.
        self.constraints: List = []
        self.time: float = 0.0
        self.ground_z: float = float(ground_z)
        self.payload_index: Optional[int] = None
        self.logger = None  # optional CSVLogger
        # default termination: stop when payload z <= ground_z
        self.termination_fn: Optional[Callable[['World'], bool]] = None

    # --- Registry ---
    def add_body(self, b: RigidBody6DOF) -> int:
        self.bodies.append(b)
        return len(self.bodies) - 1

    def set_payload(self, body_index: int) -> None:
        self.payload_index = int(body_index)

    def set_logger(self, logger) -> None:
        self.logger = logger

    def set_termination(self, fn: Callable[['World'], bool]) -> None:
        """Set user termination predicate."""
        self.termination_fn = fn

    # --- Fixed-step path ---
    def step(self, dt: float, solver: HybridSolver) -> None:
        solver.step(self, dt)

    def run(self, duration: float, dt: float, solver: Optional[HybridSolver] = None) -> None:
        solver = solver or HybridSolver()
        steps = int(np.ceil(duration / dt))
        for _ in range(steps):
            # check termination before stepping
            if self._should_terminate():
                break
            self.step(dt, solver)
            if self.logger is not None:
                self.logger.write(self.time, self)

            if self._should_terminate():
                break

    # --- Variable-step path ---
    def integrate_to(self, t_end: float, ivp: Optional[HybridIVPSolver] = None):
        ivp = ivp or HybridIVPSolver()
        # Let IVP event terminate; still log after completion
        sol = ivp.integrate_to(self, t_end)
        if self.logger is not None:
            self.logger.write(self.time, self)
        return sol

    # --- Helpers ---
    def _should_terminate(self) -> bool:
        if self.termination_fn is not None:
            return bool(self.termination_fn(self))
        # Default: stop when payload reaches ground
        if self.payload_index is None or self.payload_index < 0 or self.payload_index >= len(self.bodies):
            return False
        return bool(self.bodies[self.payload_index].p[2] <= self.ground_z)
