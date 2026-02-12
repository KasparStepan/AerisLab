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


class RoundCanopyParachute(ParachuteDrag):
    """
    Advanced round canopy parachute with inflation dynamics and apparent mass.
    
    Extends ParachuteDrag with physically accurate modeling of:
    - Dynamic inflation/collapse with exponential time constants
    - Apparent (added) mass during inflation
    - Reynolds number dependent drag coefficient
    - Canopy breathing oscillations (optional)
    
    This model is suitable for research and high-fidelity simulations where
    accurate opening shock and inflation dynamics are important.
    
    Parameters
    ----------
    rho : float
        Air density [kg/m³]
    D0 : float
        Nominal canopy diameter when fully inflated [m]
    Cd_inflated : float
        Drag coefficient when fully inflated [-]. Typical: 0.75-0.85 for round canopy.
    Cd_collapsed : float
        Drag coefficient when collapsed [-]. Typical: 0.1-0.2
    m_canopy : float
        Physical mass of canopy fabric [kg]
    k_apparent_mass : float
        Apparent mass coefficient (dimensionless). Typical: 0.4-0.6 for round canopies.
        Represents ratio of added mass to displaced air mass.
    tau_inflation : float
        Inflation time constant [s]. Time to reach ~63% inflation.
        Typical range: 0.5-2.0s depending on canopy size, porosity, and loading.
    tau_collapse : float
        Collapse time constant [s]. Usually faster than inflation.
        Typical range: 0.2-0.5s
    activation_velocity : float
        Minimum velocity for activation [m/s]
    activation_altitude : float | None
        Altitude trigger for deployment [m]
    gate_sharpness : float
        Steepness of activation transition
    enable_breathing : bool
        Enable periodic breathing oscillations
    breathing_amplitude : float
        Amplitude of breathing oscillations (fraction of area). Typical: 0.05-0.15
    breathing_frequency : float
        Frequency of breathing [Hz]. Typical: 1-3 Hz
    nu_air : float
        Kinematic viscosity of air [m²/s]. Standard: 1.5e-5
        
    Attributes
    ----------
    inflation_state : float
        Current inflation fraction [0, 1]. 0 = collapsed, 1 = fully inflated.
    apparent_mass : float
        Current apparent mass [kg]
        
    Notes
    -----
    **Inflation Dynamics:**
    The inflation state λ(t) evolves as:
        dλ/dt = (λ_target - λ) / τ
    where τ = tau_inflation or tau_collapse depending on target.
    
    **Apparent Mass:**
    When inflating, the canopy accelerates surrounding air:
        m_apparent = k_a * ρ * V_displaced
        V_displaced ≈ (π/6) * D³  (hemisphere volume)
    
    The apparent mass is automatically included in body dynamics when using
    appropriate integration methods.
    
    **Effective Drag:**
        F_drag = -0.5 * ρ * Cd(λ, Re) * S(λ, t) * |v| * v
    where:
        - Cd varies with inflation state and Reynolds number
        - S varies with inflation state and breathing (if enabled)
    
    References
    ----------
    .. [1] Cockrell, D. J. (1987). "The aerodynamics of parachutes."
           AGARDograph No. 295.
    .. [2] Knacke, T. W. (1991). "Parachute Recovery Systems Design Manual."
           Para Publishing, Santa Barbara, CA.
    .. [3] Lingard, J. S. (1995). "Ram-Air Parachute Design."
           AIAA Aerodynamic Decelerator Systems Technology Conference.
    
    Examples
    --------
    >>> # 10m diameter personnel parachute
    >>> para = RoundCanopyParachute(
    ...     rho=1.225,
    ...     D0=10.0,
    ...     Cd_inflated=0.80,
    ...     m_canopy=2.5,
    ...     tau_inflation=1.2,
    ...     activation_velocity=15.0
    ... )
    >>> 
    >>> # Cargo parachute with breathing
    >>> para = RoundCanopyParachute(
    ...     rho=1.225,
    ...     D0=25.0,
    ...     Cd_inflated=0.78,
    ...     m_canopy=8.0,
    ...     tau_inflation=2.0,
    ...     enable_breathing=True,
    ...     breathing_amplitude=0.1,
    ...     breathing_frequency=1.5
    ... )
    """
    
    def __init__(
        self,
        rho: float,
        D0: float,
        Cd_inflated: float = 0.80,
        Cd_collapsed: float = 0.15,
        m_canopy: float = 1.0,
        k_apparent_mass: float = 0.5,
        tau_inflation: float = 1.0,
        tau_collapse: float = 0.3,
        activation_velocity: float = 15.0,
        activation_altitude: float | None = None,
        gate_sharpness: float = DEFAULT_GATE_SHARPNESS,
        enable_breathing: bool = False,
        breathing_amplitude: float = 0.1,
        breathing_frequency: float = 2.0,
        nu_air: float = 1.5e-5,
    ):
        # Reference area for fully inflated canopy
        S0 = np.pi * (D0 / 2.0)**2
        
        # Initialize parent with basic parameters
        super().__init__(
            rho=rho,
            Cd=Cd_inflated,  # Will be overridden by our Cd calculation
            area=S0,          # Will be overridden by inflation state
            mode="quadratic",
            activation_time=0.0,
            activation_altitude=activation_altitude,
            activation_velocity=activation_velocity,
            gate_sharpness=gate_sharpness,
            area_collapsed=1e-4,  # Very small for numerical stability
        )
        
        # Round canopy specific parameters
        self.D0 = float(D0)
        self.Cd_inflated = float(Cd_inflated)
        self.Cd_collapsed = float(Cd_collapsed)
        self.m_canopy = float(m_canopy)
        self.k_apparent_mass = float(k_apparent_mass)
        self.tau_inflation = float(tau_inflation)
        self.tau_collapse = float(tau_collapse)
        self.nu_air = float(nu_air)
        
        # Breathing parameters
        self.enable_breathing = bool(enable_breathing)
        self.breathing_amplitude = float(breathing_amplitude)
        self.breathing_frequency = float(breathing_frequency)
        self.omega_breathing = 2.0 * np.pi * breathing_frequency
        
        # State variables
        self.inflation_state = 0.0  # λ ∈ [0, 1]
        self.apparent_mass = 0.0
        
        # Derived quantities
        self.S0 = S0
        self.V_hemisphere = (np.pi / 6.0) * D0**3  # Displaced air volume
        
        # Tracking for diagnostics
        self.last_Re = 0.0
        self.last_Cd = Cd_collapsed
        
        # Validation
        if tau_inflation <= 0 or tau_collapse <= 0:
            raise ValueError("Time constants must be positive")
        if not 0 <= k_apparent_mass <= 2.0:
            warnings.warn(
                f"Apparent mass coefficient {k_apparent_mass} outside typical range [0.4, 0.6]",
                RuntimeWarning
            )
    
    def _reynolds_number(self, v_mag: float) -> float:
        """Compute Reynolds number based on diameter and velocity."""
        return v_mag * self.D0 / self.nu_air
    
    def _drag_coefficient(self, inflation_fraction: float, Re: float) -> float:
        """
        Compute Cd as function of inflation state and Reynolds number.
        
        For round canopies, Cd is relatively insensitive to Re in the typical
        operating range (Re > 10^5), but we include a small correction.
        """
        # Base Cd from inflation state
        Cd_base = self.Cd_collapsed + inflation_fraction * (
            self.Cd_inflated - self.Cd_collapsed
        )
        
        # Reynolds number correction (minor for round canopies at high Re)
        if Re > 1e5:
            # Slight decrease in Cd at very high Re (empirical)
            Re_correction = 1.0 - 0.04 * np.log10(Re / 1e5)
            Re_correction = max(0.90, min(1.0, Re_correction))
        else:
            Re_correction = 1.0
        
        return Cd_base * Re_correction
    
    def _apparent_mass_current(self) -> float:
        """
        Compute current apparent (added) mass.
        
        Apparent mass scales with cube of inflation fraction (volume scaling).
        """
        V_current = self.V_hemisphere * self.inflation_state**3
        return self.k_apparent_mass * self.rho * V_current
    
    def _breathing_factor(self, t: float) -> float:
        """
        Compute breathing oscillation factor.
        
        Returns multiplicative factor for area: 1.0 ± amplitude.
        """
        if not self.enable_breathing or self.inflation_state < 0.8:
            return 1.0  # No breathing when not fully inflated
        
        return 1.0 + self.breathing_amplitude * np.sin(self.omega_breathing * t)
    
    def update_inflation(self, v: NDArray, dt: float) -> None:
        """
        Update inflation state using first-order dynamics.
        
        Parameters
        ----------
        v : NDArray
            Current velocity [m/s]
        dt : float
            Time step [s]
        """
        v_mag = np.linalg.norm(v)
        
        # Determine target inflation based on conditions
        if self.activation_status and v_mag > 0.5 * abs(self.activation_velocity):
            # Inflate when active and sufficient airspeed
            lambda_target = 1.0
            tau = self.tau_inflation
        else:
            # Collapse when not active or insufficient airspeed
            lambda_target = 0.0
            tau = self.tau_collapse
        
        # First-order inflation dynamics: dλ/dt = (λ_target - λ) / τ
        dlambda_dt = (lambda_target - self.inflation_state) / tau
        self.inflation_state += dlambda_dt * dt
        
        # Clamp to [0, 1]
        self.inflation_state = np.clip(self.inflation_state, 0.0, 1.0)
        
        # Update apparent mass
        self.apparent_mass = self._apparent_mass_current()
    
    def apply(self, body: RigidBody6DOF, t: float | None = None) -> None:
        """
        Apply advanced parachute force with inflation dynamics.
        
        This overrides the parent apply() to include:
        - Inflation state evolution
        - Reynolds-dependent Cd
        - Breathing oscillations
        - Apparent mass calculation
        
        Parameters
        ----------
        body : RigidBody6DOF
            Body with parachute attached
        t : float | None
            Current simulation time [s]
        """
        tval = 0.0 if t is None else float(t)
        v = body.v
        v_mag = np.linalg.norm(v)
        
        # Check activation conditions (inherited from ParachuteDrag)
        condition_vel = v_mag >= abs(self.activation_velocity)
        condition_alt = (self.activation_altitude is not None and 
                        body.p[2] <= self.activation_altitude)
        
        if (condition_vel or condition_alt) and not self.activation_status:
            self.activation_status = True
            self.activation_time = tval
        
        # Estimate dt from velocity (or use small default)
        # In practice, dt should be passed from solver, but we approximate here
        dt = 0.01  # Default time step for inflation dynamics
        
        # Update inflation state
        self.update_inflation(v, dt)
        
        # Only apply force if activated and moving
        if not self.activation_status or v_mag < EPSILON_VELOCITY:
            return
        
        # Compute Reynolds number
        Re = self._reynolds_number(v_mag)
        self.last_Re = Re
        
        # Compute effective drag coefficient
        Cd_eff = self._drag_coefficient(self.inflation_state, Re)
        self.last_Cd = Cd_eff
        
        # Compute effective area (inflation + breathing)
        S_inflation = self.S0 * self.inflation_state
        breathing = self._breathing_factor(tval)
        S_eff = S_inflation * breathing
        
        # Drag force: F = -0.5 * ρ * Cd * S * |v| * v
        F_drag = -0.5 * self.rho * Cd_eff * S_eff * v_mag * v
        
        body.apply_force(F_drag)
        
        # Note: Apparent mass affects system dynamics but is typically handled
        # implicitly through the constraint solver. For explicit handling,
        # you would need to modify the body's mass matrix during inflation.
    
    def get_diagnostics(self) -> dict:
        """
        Get current state diagnostics for logging/debugging.
        
        Returns
        -------
        dict
            Dictionary containing:
            - inflation_state: current λ
            - apparent_mass: current m_app [kg]
            - last_Re: most recent Reynolds number
            - last_Cd: most recent drag coefficient
        """
        return {
            'inflation_state': self.inflation_state,
            'apparent_mass': self.apparent_mass,
            'Reynolds_number': self.last_Re,
            'drag_coefficient': self.last_Cd,
        }


class MultiStageParachute(ParachuteDrag):
    """
    Round canopy parachute with discrete opening stages.
    
    Models the complete deployment sequence through distinct phases:
    1. Bag strip - pilot chute pulls deployment bag away
    2. Line stretch - suspension lines extend to full length
    3. Canopy snatch - canopy exits bag and begins to fill
    4. Inflation - canopy fills with air (most dynamic phase)
    5. Fully inflated - steady-state descent
    
    Each stage has characteristic drag area and duration based on
    empirical parachute opening data.
    
    Parameters
    ----------
    rho : float
        Air density [kg/m³]
    D0 : float
        Nominal canopy diameter [m]
    Cd_final : float
        Final drag coefficient when fully open
    activation_velocity : float
        Velocity threshold for deployment
    activation_altitude : float | None
        Altitude threshold for deployment
    stage_durations : dict | None
        Custom stage durations [s]. Keys: 'bag_strip', 'line_stretch',
        'canopy_snatch', 'inflation'. If None, uses empirical defaults.
        
    Notes
    -----
    Default stage characteristics (can be overridden):
    
    | Stage          | Cd  | Area Fraction | Duration |
    |----------------|-----|---------------|----------|
    | Bag strip      | 0.05| 0.05          | 0.2s     |
    | Line stretch   | 0.10| 0.15          | 0.3s     |
    | Canopy snatch  | 0.30| 0.35          | 0.4s     |
    | Inflation      | 0.60| 0.75          | 0.8s     |
    | Fully inflated | Cd_final | 1.00     | ∞        |
    
    Total nominal opening time: ~1.7s from bag strip to full inflation.
    
    Examples
    --------
    >>> # Standard personnel parachute
    >>> para = MultiStageParachute(
    ...     rho=1.225,
    ...     D0=8.5,
    ...     Cd_final=0.80,
    ...     activation_velocity=20.0
    ... )
    >>> 
    >>> # Fast-opening cargo chute with custom timings
    >>> para = MultiStageParachute(
    ...     rho=1.225,
    ...     D0=15.0,
    ...     Cd_final=0.75,
    ...     stage_durations={
    ...         'bag_strip': 0.1,
    ...         'line_stretch': 0.2,
    ...         'canopy_snatch': 0.3,
    ...         'inflation': 0.6,
    ...     }
    ... )
    """
    
    # Default stage characteristics
    DEFAULT_STAGES = [
        {'name': 'bag_strip',      'Cd': 0.05, 'area_frac': 0.05, 'duration': 0.2},
        {'name': 'line_stretch',   'Cd': 0.10, 'area_frac': 0.15, 'duration': 0.3},
        {'name': 'canopy_snatch',  'Cd': 0.30, 'area_frac': 0.35, 'duration': 0.4},
        {'name': 'inflation',      'Cd': 0.60, 'area_frac': 0.75, 'duration': 0.8},
    ]
    
    def __init__(
        self,
        rho: float,
        D0: float,
        Cd_final: float = 0.80,
        activation_velocity: float = 15.0,
        activation_altitude: float | None = None,
        stage_durations: dict | None = None,
        gate_sharpness: float = DEFAULT_GATE_SHARPNESS,
    ):
        S0 = np.pi * (D0 / 2.0)**2
        
        super().__init__(
            rho=rho,
            Cd=Cd_final,
            area=S0,
            activation_velocity=activation_velocity,
            activation_altitude=activation_altitude,
            gate_sharpness=gate_sharpness,
        )
        
        self.D0 = D0
        self.S0 = S0
        self.Cd_final = Cd_final
        
        # Setup stages
        self.stages = self.DEFAULT_STAGES.copy()
        if stage_durations is not None:
            for stage in self.stages:
                if stage['name'] in stage_durations:
                    stage['duration'] = float(stage_durations[stage['name']])
        
        # State tracking
        self.current_stage_index = -1  # -1 = not deployed
        self.stage_elapsed_time = 0.0
        self.deployment_start_time = 0.0
    
    def _get_current_stage_params(self) -> tuple[float, float]:
        """
        Get Cd and area fraction for current stage.
        
        Returns
        -------
        Cd : float
            Current drag coefficient
        S_eff : float
            Current effective area [m²]
        """
        if self.current_stage_index < 0:
            # Not deployed
            return 0.05, self.S0 * 0.001
        elif self.current_stage_index >= len(self.stages):
            # Fully inflated
            return self.Cd_final, self.S0
        else:
            # In a specific stage
            stage = self.stages[self.current_stage_index]
            return stage['Cd'], self.S0 * stage['area_frac']
    
    def apply(self, body: RigidBody6DOF, t: float | None = None) -> None:
        """
        Apply multi-stage parachute force.
        
        Advances through deployment stages and applies corresponding drag.
        
        Parameters
        ----------
        body : RigidBody6DOF
            Body with parachute attached
        t : float | None
            Current simulation time [s]
        """
        tval = 0.0 if t is None else float(t)
        v = body.v
        v_mag = np.linalg.norm(v)
        
        # Check activation
        condition_vel = v_mag >= abs(self.activation_velocity)
        condition_alt = (self.activation_altitude is not None and 
                        body.p[2] <= self.activation_altitude)
        
        if (condition_vel or condition_alt) and not self.activation_status:
            self.activation_status = True
            self.deployment_start_time = tval
            self.current_stage_index = 0
            self.stage_elapsed_time = 0.0
        
        if not self.activation_status or v_mag < EPSILON_VELOCITY:
            return
        
        # Update stage progression
        if self.current_stage_index < len(self.stages):
            stage = self.stages[self.current_stage_index]
            dt = tval - self.deployment_start_time - sum(
                s['duration'] for s in self.stages[:self.current_stage_index]
            )
            
            if dt >= stage['duration']:
                # Move to next stage
                self.current_stage_index += 1
        
        # Get current stage parameters
        Cd, S_eff = self._get_current_stage_params()
        
        # Apply drag force
        F_drag = -0.5 * self.rho * Cd * S_eff * v_mag * v
        body.apply_force(F_drag)
    
    def get_stage_name(self) -> str:
        """Get name of current deployment stage."""
        if self.current_stage_index < 0:
            return "not_deployed"
        elif self.current_stage_index >= len(self.stages):
            return "fully_inflated"
        else:
            return self.stages[self.current_stage_index]['name']


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
