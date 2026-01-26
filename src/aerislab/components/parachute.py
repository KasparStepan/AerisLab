"""
Parachute component with deployment state machine.

Manages:
- Deployment state transitions (STOWED → DEPLOYING → DEPLOYED)
- Altitude/velocity activation triggers
- Smooth drag force application via ParachuteDrag
- Future: reefing stages, opening shock modeling
"""

from __future__ import annotations

from enum import Enum, auto

import numpy as np

from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.forces import ParachuteDrag

from .base import Component


class DeploymentState(Enum):
    """
    Parachute deployment states.

    State Machine:
        STOWED → DEPLOYING → DEPLOYED
                    ↓
                 FAILED (future)
    """

    STOWED = auto()  # Packed, no drag
    DEPLOYING = auto()  # Inflation in progress
    DEPLOYED = auto()  # Fully inflated
    FAILED = auto()  # Future: deployment failure simulation


class Parachute(Component):
    """
    Parachute component with deployment logic and aerodynamics.

    Manages deployment state machine and wraps ParachuteDrag force.
    Automatically triggers deployment based on altitude or velocity thresholds.

    Parameters
    ----------
    name : str
        Component name (e.g., "main_chute", "drogue")
    body : RigidBody6DOF
        Rigid body for canopy dynamics
    Cd : float
        Deployed drag coefficient [-]. Typical: 1.0-2.0
    area : float
        Reference area when fully deployed [m²]
    activation_altitude : float | None
        Deploy when altitude drops below this value [m].
        Set to None to disable altitude trigger.
    activation_velocity : float
        Deploy when speed exceeds this value [m/s].
        Default 50.0
    rho : float
        Air density [kg/m³]. Default 1.225
    gate_sharpness : float
        Smooth deployment transition steepness. Higher = faster transition.
        Default 40.0

    Attributes
    ----------
    deployment_state : DeploymentState
        Current deployment state
    deployment_time : float | None
        Time when deployment was triggered [s]
    drag_force : ParachuteDrag
        Underlying drag force model

    Examples
    --------
    >>> canopy = RigidBody6DOF(
    ...     name="canopy",
    ...     mass=2.0,
    ...     inertia_tensor_body=0.1 * np.eye(3),
    ...     position=np.array([0, 0, 1000]),
    ...     orientation=np.array([0, 0, 0, 1])
    ... )
    >>> parachute = Parachute(
    ...     name="main_chute",
    ...     body=canopy,
    ...     Cd=1.5,
    ...     area=15.0,
    ...     activation_altitude=800.0,
    ...     activation_velocity=40.0
    ... )

    Notes
    -----
    **Deployment Logic**

    The parachute deploys when EITHER condition is met:
    - Altitude ≤ activation_altitude (if set)
    - Speed ≥ activation_velocity

    The ParachuteDrag force handles smooth area transition internally
    using a tanh gate function for numerical stability with stiff solvers.

    **Future Extensions**

    - Reefing stages (multi-stage deployment)
    - Opening shock modeling (transient force spike)
    - Deployment failure modes
    - Canopy oscillation damping
    """

    def __init__(
        self,
        name: str,
        body: RigidBody6DOF,
        Cd: float,
        area: float,
        activation_altitude: float | None = None,
        activation_velocity: float = 50.0,
        rho: float = 1.225,
        gate_sharpness: float = 40.0,
    ):
        super().__init__(name, body)

        # Deployment parameters
        self.activation_altitude = activation_altitude
        self.activation_velocity = activation_velocity
        self.deployment_state = DeploymentState.STOWED
        self.deployment_time: float | None = None

        # Aerodynamic parameters
        self.Cd = Cd
        self.area = area
        self.rho = rho
        self.gate_sharpness = gate_sharpness

        # Create parachute drag force
        self.drag_force = ParachuteDrag(
            rho=rho,
            Cd=Cd,
            area=area,
            activation_altitude=activation_altitude,
            activation_velocity=activation_velocity,
            gate_sharpness=gate_sharpness,
        )
        self.add_force(self.drag_force)

        # State tracking for logging
        self._update_state_dict()

    def _update_state_dict(self) -> None:
        """Update internal state dictionary for logging."""
        self._state = {
            "component_type": "parachute",
            "deployment_state": self.deployment_state.name,
            "deployment_time": self.deployment_time,
            "effective_area": self._compute_effective_area(),
            "Cd": self.Cd,
            "area": self.area,
        }

    def update_state(self, t: float, dt: float) -> None:
        """
        Update deployment state based on current conditions.

        Checks activation conditions and transitions state machine.
        Called by System before force application.

        Parameters
        ----------
        t : float
            Current simulation time [s]
        dt : float
            Time step [s]
        """
        if self.deployment_state == DeploymentState.STOWED:
            self._check_deployment_trigger(t)

        elif self.deployment_state == DeploymentState.DEPLOYING:
            self._check_deployment_complete(t)

        # Update state dict for logging
        self._update_state_dict()

    def _check_deployment_trigger(self, t: float) -> None:
        """Check if deployment conditions are met."""
        altitude = float(self.body.p[2])
        speed = float(np.linalg.norm(self.body.v))

        altitude_trigger = (
            self.activation_altitude is not None
            and altitude <= self.activation_altitude
        )
        velocity_trigger = speed >= abs(self.activation_velocity)

        if altitude_trigger or velocity_trigger:
            self.deploy(t)

    def _check_deployment_complete(self, t: float) -> None:
        """Check if deployment is complete (area fully open)."""
        if self.deployment_time is None:
            return

        effective_area = self._compute_effective_area()
        if effective_area >= 0.99 * self.area:
            self.deployment_state = DeploymentState.DEPLOYED

    def deploy(self, t: float) -> None:
        """
        Trigger parachute deployment.

        Parameters
        ----------
        t : float
            Current simulation time [s]
        """
        if self.deployment_state == DeploymentState.STOWED:
            self.deployment_state = DeploymentState.DEPLOYING
            self.deployment_time = t

            # Update drag force activation time
            self.drag_force.activation_time = t

            altitude = float(self.body.p[2])
            speed = float(np.linalg.norm(self.body.v))
            print(
                f"[{self.name}] Deployment initiated at t={t:.3f}s, "
                f"alt={altitude:.1f}m, vel={speed:.1f}m/s"
            )

    def _compute_effective_area(self) -> float:
        """
        Get current effective area (accounts for deployment dynamics).

        Returns
        -------
        float
            Current effective drag area [m²]
        """
        if self.deployment_state == DeploymentState.STOWED:
            return 0.0
        if self.deployment_time is None:
            return 0.0

        # Note: In a full implementation, this would compute the actual
        # area based on deployment dynamics. Currently simplified.
        return self.area  # Simplified - full deployment assumed after trigger

    @property
    def is_deployed(self) -> bool:
        """Check if parachute is fully deployed."""
        return self.deployment_state == DeploymentState.DEPLOYED

    @property
    def is_deploying(self) -> bool:
        """Check if parachute is currently deploying."""
        return self.deployment_state == DeploymentState.DEPLOYING

    @property
    def altitude(self) -> float:
        """Current altitude (z-position) [m]."""
        return float(self.body.p[2])

    @property
    def speed(self) -> float:
        """Current speed magnitude [m/s]."""
        return float(np.linalg.norm(self.body.v))
