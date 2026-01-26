"""
Payload component - simple rigid body with aerodynamic drag.

Represents ballistic bodies, instrument bays, or any component
that needs basic drag modeling without complex state machines.
"""

from __future__ import annotations

import numpy as np

from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import Drag

from .base import Component


class Payload(Component):
    """
    Payload component with basic aerodynamic drag.

    Simple component for ballistic bodies, instruments, cargo, etc.
    Can be extended for sensor simulation, data acquisition, impact detection.

    Parameters
    ----------
    name : str
        Component name
    body : RigidBody6DOF
        Rigid body for payload dynamics
    Cd : float
        Drag coefficient [-]. Default 0.47 (sphere)
    area : float
        Reference area [m²]. Default 0.1
    rho : float
        Air density [kg/m³]. Default 1.225 (sea level)

    Examples
    --------
    >>> payload_body = RigidBody6DOF(
    ...     name="instrument_bay",
    ...     mass=10.0,
    ...     inertia_tensor_body=np.eye(3) * 0.1,
    ...     position=np.array([0, 0, 1000]),
    ...     orientation=np.array([0, 0, 0, 1])
    ... )
    >>> payload = Payload(
    ...     name="payload",
    ...     body=payload_body,
    ...     Cd=0.47,
    ...     area=np.pi * 0.2**2
    ... )

    Notes
    -----
    The Payload component is intentionally simple. For components requiring
    state machines (deployment, actuation), see Parachute or create a custom
    Component subclass.
    """

    def __init__(
        self,
        name: str,
        body: RigidBody6DOF,
        Cd: float = 0.47,
        area: float = 0.1,
        rho: float = 1.225,
    ):
        super().__init__(name, body)

        self.Cd = Cd
        self.area = area
        self.rho = rho

        # Add drag force
        drag = Drag(rho=rho, Cd=Cd, area=area, mode="quadratic")
        self.add_force(drag)

        # State tracking
        self._state = {
            "component_type": "payload",
            "Cd": Cd,
            "area": area,
        }

    def update_state(self, t: float, dt: float) -> None:
        """
        Update payload state.

        Payload has no active state changes - this is a passive component.
        Override in subclasses for active payload behaviors (e.g., sensor
        triggering, data recording, impact detection).

        Parameters
        ----------
        t : float
            Current simulation time [s]
        dt : float
            Time step [s]
        """
        pass  # Simple component - no state machine

    @property
    def altitude(self) -> float:
        """Current altitude (z-position) [m]."""
        return float(self.body.p[2])

    @property
    def speed(self) -> float:
        """Current speed magnitude [m/s]."""
        return float(np.linalg.norm(self.body.v))

    @property
    def vertical_velocity(self) -> float:
        """Current vertical velocity (z-component) [m/s]."""
        return float(self.body.v[2])
