"""
Force models for rigid body dynamics.

All force classes follow the Force protocol and can be applied to RigidBody6DOF instances.
Forces are applied at the center of mass unless otherwise specified.

Physical units:
- Forces: Newtons [N]
- Torques: Newton-meters [N·m]
- Velocities: meters per second [m/s]
- Areas: square meters [m²]
- Densities: kilograms per cubic meter [kg/m³]
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Protocol
from .body import RigidBody6DOF
import warnings

# Physical constants
EPSILON_VELOCITY = 1e-12  # Minimum velocity magnitude for drag calculations
EPSILON_DISTANCE = 1e-12  # Minimum distance for spring force calculations

# Parachute deployment constants
DEFAULT_GATE_SHARPNESS = 40.0  # Steepness of smooth activation transition [-]
DEFAULT_COLLAPSED_AREA = 1e-3  # Parachute area when collapsed [m²]


class Force(Protocol):
    """Protocol for force application to rigid bodies."""
    def apply(self, body: RigidBody6DOF, t: float | None = None) -> None:
        """
        Apply force to a rigid body.
        
        Parameters
        ----------
        body : RigidBody6DOF
            The body to apply force to
        t : float | None
            Current simulation time [s]. Optional for time-independent forces.
        """
        ...


class Gravity:
    """
    Uniform gravitational force.
    
    Applies constant gravitational acceleration to body's center of mass.
    Force: F = m * g
    
    Parameters
    ----------
    g : NDArray[np.float64]
        Gravitational acceleration vector in world frame [m/s²] (3,)
        Standard Earth gravity: [0, 0, -9.81]
        
    Examples
    --------
    >>> gravity = Gravity(np.array([0.0, 0.0, -9.81]))
    >>> gravity.apply(body)
    """
    def __init__(self, g: NDArray[np.float64]) -> None:
        self.g = np.asarray(g, dtype=np.float64)
        if self.g.shape != (3,):
            raise ValueError(f"Gravity vector must be (3,), got shape {self.g.shape}")

    def apply(self, body: RigidBody6DOF, t: float | None = None) -> None:
        """Apply gravitational force F = m * g to body center of mass."""
        body.apply_force(body.mass * self.g)


class Drag:
    """
    Aerodynamic drag force in world frame.
    
    Supports two drag models:
    - 'quadratic': F = -0.5 * ρ * Cd * A * |v| * v  (standard aerodynamic drag)
    - 'linear': F = -k * v  (Stokes drag for low Reynolds number)
    
    Parameters are runtime-modifiable to support time-varying configurations
    (e.g., parachute deployment, variable area).
    
    Parameters
    ----------
    rho : float
        Air density [kg/m³]. Standard sea level: 1.225 kg/m³
    Cd : float | Callable[[float, RigidBody6DOF], float]
        Drag coefficient [-]. Can be constant or time/state-dependent function.
        Typical values: sphere ≈ 0.47, parachute ≈ 1.3-1.8
    area : float | Callable[[float, RigidBody6DOF], float]
        Reference area [m²]. Can be constant or time/state-dependent function.
    mode : str
        Drag model: 'quadratic' (default) or 'linear'
    k_linear : float
        Linear drag coefficient for mode='linear' [N·s/m]
        
    Notes
    -----
    Force is applied at body center of mass (no induced torque).
    For asymmetric bodies, consider using apply_force with point_world.
    
    Examples
    --------
    >>> # Constant drag on a sphere
    >>> drag = Drag(rho=1.225, Cd=0.47, area=np.pi * 0.1**2)
    >>> 
    >>> # Time-varying parachute area
    >>> def area_func(t, body):
    ...     return 5.0 * min(1.0, t / 2.0)  # Deploy over 2 seconds
    >>> drag = Drag(rho=1.225, Cd=1.5, area=area_func)
    """
    def __init__(
        self,
        rho: float = 1.225,
        Cd: float | Callable[[float, RigidBody6DOF], float] = 1.0,
        area: float | Callable[[float, RigidBody6DOF], float] = 1.0,
        mode: str = "quadratic",
        k_linear: float = 0.0,
    ) -> None:
        if rho < 0:
            raise ValueError(f"Density must be non-negative, got {rho}")
        if mode not in ("quadratic", "linear"):
            raise ValueError(f"Mode must be 'quadratic' or 'linear', got '{mode}'")
        if mode == "linear" and k_linear < 0:
            raise ValueError(f"Linear drag coefficient must be non-negative, got {k_linear}")
            
        self.rho = float(rho)
        self.Cd = Cd
        self.area = area
        self.mode = mode
        self.k_linear = float(k_linear)

    def _value(self, val: float | Callable, t: float, body: RigidBody6DOF) -> float:
        """Evaluate parameter (constant or callable)."""
        return float(val(t, body)) if callable(val) else float(val)

    def apply(self, body: RigidBody6DOF, t: float | None = None) -> None:
        """
        Apply drag force to body center of mass.
        
        Parameters
        ----------
        body : RigidBody6DOF
            The body to apply drag to
        t : float | None
            Current simulation time [s]
        """
        tval = 0.0 if t is None else float(t)
        v = body.v
        speed = np.linalg.norm(v)
        
        if speed < EPSILON_VELOCITY:
            return  # No drag for stationary body
        
        if self.mode == "quadratic":
            Cd = self._value(self.Cd, tval, body)
            A = self._value(self.area, tval, body)
            
            # F = -0.5 * ρ * Cd * A * |v| * v
            f = -0.5 * self.rho * Cd * A * speed * v
            body.apply_force(f)
            
        elif self.mode == "linear":
            # F = -k * v
            body.apply_force(-self.k_linear * v)

        
class ParachuteDrag(Drag):
    """
    Specialized drag force for parachute systems with state-based deployment.
    
    Extends Drag with activation logic and smooth area transition for numerical stability
    with stiff IVP solvers. Parachute activates when velocity or altitude thresholds
    are exceeded, then smoothly transitions from collapsed to deployed state.
    
    Parameters
    ----------
    rho : float
        Air density [kg/m³]
    Cd : float | Callable
        Drag coefficient for deployed parachute [-]. Typical: 1.3-1.8
    area : float | Callable
        Reference area when fully deployed [m²]
    mode : str
        Drag model (typically 'quadratic')
    activation_time : float
        Initial activation time [s]. Updated when deployment conditions are met.
    activation_altitude : float | None
        Deploy when altitude drops below this value [m]. None to disable.
    activation_velocity : float
        Deploy when speed exceeds this value [m/s]. Can be negative to check descent rate.
    gate_sharpness : float
        Steepness of smooth activation transition [-]. Higher = steeper (but still smooth).
        Default: 40.0. Range: 10-100 for most applications.
    area_collapsed : float
        Parachute area when collapsed [m²]. Small non-zero value for numerical stability.
        Default: 1e-3 m²
        
    Notes
    -----
    The smooth transition uses a hyperbolic tangent gate function:
        A(t) = A_collapsed + 0.5 * (1 + tanh(k * (t - t_deploy))) * (A_deployed - A_collapsed)
    
    This ensures continuous derivatives for stiff IVP solvers while approximating
    a sharp deployment.
    
    Examples
    --------
    >>> # Deploy at 50 m/s descent velocity
    >>> para = ParachuteDrag(rho=1.225, Cd=1.5, area=10.0, activation_velocity=50.0)
    >>> 
    >>> # Deploy at 1000m altitude or 40 m/s, whichever comes first
    >>> para = ParachuteDrag(
    ...     rho=1.225, Cd=1.6, area=15.0,
    ...     activation_altitude=1000.0,
    ...     activation_velocity=40.0
    ... )
    """
    def __init__(
        self,
        rho: float = 1.225,
        Cd: float | Callable[[float, RigidBody6DOF], float] = 1.5,
        area: float | Callable[[float, RigidBody6DOF], float] = 1.0,
        mode: str = "quadratic",
        activation_time: float = 0.0,
        activation_altitude: float | None = None,
        activation_velocity: float = 50.0,
        gate_sharpness: float = DEFAULT_GATE_SHARPNESS,
        area_collapsed: float = DEFAULT_COLLAPSED_AREA,
    ) -> None:
        super().__init__(rho=rho, Cd=Cd, area=area, mode=mode)
        
        self.activation_time = float(activation_time)
        self.activation_altitude = (None if activation_altitude is None
                                   else float(activation_altitude))
        self.activation_velocity = float(activation_velocity)
        self.activation_status = False
        self.gate_sharpness = float(gate_sharpness)
        self.area_collapsed = float(area_collapsed)
        
        # Validation
        if self.gate_sharpness <= 0:
            raise ValueError(f"Gate sharpness must be positive, got {gate_sharpness}")
        if self.area_collapsed < 0:
            raise ValueError(f"Collapsed area must be non-negative, got {area_collapsed}")
        
    def apply(self, body: RigidBody6DOF, t: float | None = None) -> None:    
        """
        Apply parachute drag with activation logic.
        
        Checks deployment conditions and applies smooth drag force transition.
        
        Parameters
        ----------
        body : RigidBody6DOF
            The body with parachute attached
        t : float | None
            Current simulation time [s]
        """
        tval = 0.0 if t is None else float(t)
        v = body.v
        v_mag = np.linalg.norm(v)
        
        # Check activation conditions
        condition_vel = v_mag >= abs(self.activation_velocity)
        condition_alt = (self.activation_altitude is not None and 
                        body.p[2] <= self.activation_altitude)

        if (condition_vel or condition_alt) and not self.activation_status:
            self.activation_status = True
            self.activation_time = tval

        # Apply drag only if activated, using smooth area transition
        if self.activation_status:
            if v_mag < EPSILON_VELOCITY:
                return  # No drag at zero velocity
                
            Cd = self._value(self.Cd, tval, body)
            A = self._eval_smooth_area(tval, body)
            
            # Standard quadratic drag formula
            f = -0.5 * self.rho * Cd * A * v_mag * v
            body.apply_force(f)

    def _eval_smooth_area(self, t: float, body: RigidBody6DOF) -> float:
        """
        Evaluate parachute area with smooth activation transition.
        
        Uses hyperbolic tangent to create a smooth (but steep) transition
        from collapsed to deployed state. This ensures continuous derivatives
        for stiff IVP solvers.
        
        Parameters
        ----------
        t : float
            Current time [s]
        body : RigidBody6DOF
            The body (for callable area evaluation)
            
        Returns
        -------
        float
            Effective parachute area [m²]
            
        Notes
        -----
        Gate function: g(t) = 0.5 * (1 + tanh(k * (t - t_deploy)))
        - At t << t_deploy: g ≈ 0 (collapsed)
        - At t = t_deploy: g = 0.5 (50% deployed)
        - At t >> t_deploy: g ≈ 1 (fully deployed)
        """
        k = self.gate_sharpness
        A0 = self.area_collapsed
        
        # Smooth gate function: 0 before deployment, 1 after
        gate = 0.5 * (1.0 + np.tanh(k * (t - self.activation_time)))
        
        # Target deployed area (may be time-varying)
        target_area = self._value(self.area, t, body)
        
        # Linear interpolation weighted by smooth gate
        return A0 + gate * (target_area - A0)


class Spring:
    """
    Soft spring connection between two rigid bodies (Hooke + viscous damping).
    
    Models a deformable tether/spring between attachment points on two bodies.
    Not a rigid constraint - allows stretching with restoring force.
    
    Force law:
        F = -k * (|d| - L₀) * d_hat - c * v_rel_line
        
    where:
        d = separation vector from body B to body A attachment points
        d_hat = d / |d| (unit vector)
        v_rel_line = (v_A - v_B) · d_hat (relative velocity along line)
    
    Parameters
    ----------
    body_a : RigidBody6DOF
        First body
    body_b : RigidBody6DOF
        Second body
    attach_a_local : NDArray[np.float64]
        Attachment point on body A in body A's local frame [m] (3,)
    attach_b_local : NDArray[np.float64]
        Attachment point on body B in body B's local frame [m] (3,)
    k : float
        Spring stiffness [N/m]. Higher values → stiffer connection.
    c : float
        Damping coefficient [N·s/m]. Higher values → more damping.
    rest_length : float
        Natural (unstretched) length of spring [m]
        
    Notes
    -----
    - Force is applied at attachment points (generates torques if offset from CoM)
    - Equal and opposite forces applied to both bodies (Newton's 3rd law)
    - Handles zero-length springs (attachment points coincident when L₀=0)
    - Uses minimum distance threshold to avoid singularities
    
    Examples
    --------
    >>> # 5m tether between payload and parachute
    >>> spring = Spring(
    ...     payload, canopy,
    ...     attach_a_local=np.array([0, 0, 0]),
    ...     attach_b_local=np.array([0, 0, 0]),
    ...     k=1000.0, c=50.0, rest_length=5.0
    ... )
    """
    def __init__(
        self,
        body_a: RigidBody6DOF,
        body_b: RigidBody6DOF,
        attach_a_local: NDArray[np.float64],
        attach_b_local: NDArray[np.float64],
        k: float,
        c: float,
        rest_length: float,
    ) -> None:
        if k < 0:
            raise ValueError(f"Spring stiffness must be non-negative, got {k}")
        if c < 0:
            raise ValueError(f"Damping coefficient must be non-negative, got {c}")
        if rest_length < 0:
            raise ValueError(f"Rest length must be non-negative, got {rest_length}")
            
        self.a = body_a
        self.b = body_b
        self.ra_local = np.asarray(attach_a_local, dtype=np.float64)
        self.rb_local = np.asarray(attach_b_local, dtype=np.float64)
        self.k = float(k)
        self.c = float(c)
        self.L0 = float(rest_length)

    def apply_pair(self, t: float | None = None) -> None:
        """
        Apply spring forces to both bodies.
        
        Computes attachment point positions and velocities in world frame,
        then applies equal and opposite spring forces.
        
        Parameters
        ----------
        t : float | None
            Current simulation time [s]. Not used but included for interface consistency.
            
        Notes
        -----
        This method should be called once per timestep by the World orchestrator.
        """
        # Transform attachment points to world frame
        Ra = self.a.rotation_world()
        Rb = self.b.rotation_world()
        ra_w = Ra @ self.ra_local
        rb_w = Rb @ self.rb_local
        
        # World positions of attachment points
        pa = self.a.p + ra_w
        pb = self.b.p + rb_w
        
        # Separation vector (A to B)
        d = pa - pb
        dist = np.linalg.norm(d)
        
        if dist < EPSILON_DISTANCE:
            # Attachment points coincident - no force
            warnings.warn(
                f"Spring between '{self.a.name}' and '{self.b.name}' has near-zero length "
                f"({dist:.2e} m). Skipping force application.",
                RuntimeWarning,
                stacklevel=2
            )
            return
        
        d_hat = d / dist

        # Velocities of attachment points in world frame
        va = self.a.v + np.cross(self.a.w, ra_w)
        vb = self.b.v + np.cross(self.b.w, rb_w)
        
        # Relative velocity along spring line
        vrel = va - vb
        vrel_line = np.dot(vrel, d_hat)

        # Spring force: F = -k * (dist - L0) * d_hat - c * vrel_line * d_hat
        f_spring = -self.k * (dist - self.L0) * d_hat
        f_damping = -self.c * vrel_line * d_hat
        f = f_spring + f_damping
        
        # Apply equal and opposite forces at attachment points
        self.a.apply_force(+f, point_world=pa)
        self.b.apply_force(-f, point_world=pb)
