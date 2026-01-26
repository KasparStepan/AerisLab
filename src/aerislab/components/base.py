"""
Base component abstraction for aerospace simulations.

Components wrap RigidBody6DOF with domain-specific behavior and force management.
Uses composition pattern - Component HAS-A RigidBody6DOF, not IS-A.

This design follows aerospace simulation best practices from NASA Trick,
JSBSim, and FlightGear frameworks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import Force


class Component(ABC):
    """
    Base class for aerospace simulation components.

    Components wrap RigidBody6DOF with domain-specific behavior
    and force management. Use composition pattern.

    Parameters
    ----------
    name : str
        Unique component identifier
    body : RigidBody6DOF
        Underlying rigid body dynamics

    Attributes
    ----------
    name : str
        Component identifier
    body : RigidBody6DOF
        Wrapped rigid body
    forces : list[Force]
        Forces attached to this component
    _state : dict
        Component-specific state variables for logging

    Examples
    --------
    >>> class CustomComponent(Component):
    ...     def update_state(self, t: float, dt: float) -> None:
    ...         pass  # Custom logic here

    Notes
    -----
    **Design Pattern: Composition over Inheritance**

    Instead of:
        class Parachute(RigidBody6DOF):  # BAD - tight coupling

    We use:
        class Parachute(Component):  # GOOD - loose coupling
            def __init__(self, body: RigidBody6DOF):
                self.body = body  # HAS-A relationship

    This allows:
    - Swapping body implementations (future: deformable bodies)
    - Clean separation of dynamics vs domain logic
    - Avoiding diamond inheritance problems
    """

    def __init__(self, name: str, body: RigidBody6DOF):
        self.name = name
        self.body = body
        self.forces: list[Force] = []
        self._state: dict = {}  # Component-specific state variables

    def add_force(self, force: Force) -> None:
        """
        Add force model to this component.

        Parameters
        ----------
        force : Force
            Force object with apply(body, t) method
        """
        self.forces.append(force)

    def apply_forces(self, t: float) -> None:
        """
        Apply all forces attached to this component.

        Parameters
        ----------
        t : float
            Current simulation time [s]
        """
        for force in self.forces:
            force.apply(self.body, t)

    @abstractmethod
    def update_state(self, t: float, dt: float) -> None:
        """
        Update component-specific state logic.

        Called before force application each timestep. Override for
        deployment logic, actuation, sensor simulation, etc.

        Parameters
        ----------
        t : float
            Current simulation time [s]
        dt : float
            Time step [s]
        """
        pass

    # -------------------------------------------------------------------------
    # Convenience properties - delegate to body
    # -------------------------------------------------------------------------

    @property
    def position(self) -> NDArray[np.float64]:
        """Component position in world frame [m]."""
        return self.body.p

    @property
    def velocity(self) -> NDArray[np.float64]:
        """Component velocity in world frame [m/s]."""
        return self.body.v

    @property
    def orientation(self) -> NDArray[np.float64]:
        """Component orientation quaternion [x,y,z,w]."""
        return self.body.q

    @property
    def angular_velocity(self) -> NDArray[np.float64]:
        """Component angular velocity [rad/s]."""
        return self.body.w

    @property
    def mass(self) -> float:
        """Component mass [kg]."""
        return self.body.mass

    # -------------------------------------------------------------------------
    # State serialization
    # -------------------------------------------------------------------------

    def get_state_dict(self) -> dict:
        """
        Get component state for logging/serialization.

        Returns
        -------
        dict
            Dictionary containing position, velocity, orientation,
            angular velocity, and component-specific state.
        """
        return {
            "name": self.name,
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "orientation": self.orientation.copy(),
            "angular_velocity": self.angular_velocity.copy(),
            **self._state,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', body='{self.body.name}')"
