from __future__ import annotations
import numpy as np
from typing import Callable, List, Optional
from .body import RigidBody6DOF
from .forces import Gravity, Drag, Spring
from .constraints import Constraint
from .solver import HybridSolver, HybridIVPSolver

class World:
    """
    Container/orchestrator of bodies, forces, constraints, and time.

    Termination: stop when payload.z <= ground_z (no contact). For fixed-step,
    we detect crossing and linearly interpolate touchdown time t*.
    """
    def __init__(self, ground_z: float = 0.0, payload_index: int = 0) -> None:
        self.bodies: List[RigidBody6DOF] = []
        self.global_forces: List = []
        self.interaction_forces: List[Spring] = []
        self.constraints: List[Constraint] = []
        self.payload_index = int(payload_index)
        self.ground_z = float(ground_z)
        self.t = 0.0
        self.t_touchdown: Optional[float] = None
        self.logger = None
        self.termination_callback: Optional[Callable[['World'], bool]] = None

    # configuration
    def set_logger(self, logger) -> None:
        self.logger = logger

    def add_body(self, body: RigidBody6DOF) -> int:
        self.bodies.append(body)
        return len(self.bodies) - 1

    def add_global_force(self, f) -> None:
        self.global_forces.append(f)

    def add_interaction_force(self, fpair: Spring) -> None:
        self.interaction_forces.append(fpair)

    def add_constraint(self, c: Constraint) -> None:
        self.constraints.append(c)

    def set_termination_callback(self, fn: Callable[['World'], bool]) -> None:
        self.termination_callback = fn

    # --- Fixed-step API ---
    def step(self, solver: HybridSolver, dt: float) -> bool:
        # 1) clear per-body force accumulators
        for b in self.bodies:
            b.clear_forces()

        # 2) apply per-body & global & interaction forces
        for b in self.bodies:
            for fb in b.per_body_forces:
                fb.apply(b, self.t)
        for fg in self.global_forces:
            for b in self.bodies:
                fg.apply(b, self.t)
        for fpair in self.interaction_forces:
            fpair.apply_pair(self.t)

        # 3) record pre-step payload z
        payload = self.bodies[self.payload_index]
        z_pre = payload.p[2]

        # 4) KKT + integrate
        solver.step(self.bodies, self.constraints, dt)
        self.t += dt

        # 5) logging
        if self.logger is not None:
            self.logger.log(self)

        # 6) termination check (either custom or default ground)
        stop = False
        if self.termination_callback:
            stop = bool(self.termination_callback(self))
        else:
            z_post = payload.p[2]
            if (z_pre > self.ground_z) and (z_post <= self.ground_z):
                # linear interpolation for touchdown time
                frac = (z_pre - self.ground_z) / max((z_pre - z_post), 1e-12)
                self.t_touchdown = float(self.t - dt + frac * dt)
                stop = True
            elif z_post <= self.ground_z:
                # already below ground (edge)
                self.t_touchdown = self.t
                stop = True
        return stop

    def run(self, solver: HybridSolver, duration: float, dt: float) -> None:
        t_end = self.t + float(duration)
        while self.t < t_end:
            if self.step(solver, dt):
                break

    # --- Variable-step/API wrapper ---
    def integrate_to(self, solver: HybridIVPSolver, t_end: float):
        return solver.integrate(self, t_end)
