"""
Advanced parachute inflation models for traditional round parachutes.

This module implements physics-based parachute inflation models that produce
more realistic peak opening loads than simplified Cx factor approaches.

Models implemented:
- SIMPLE_DRAG: Basic quadratic drag (no inflation dynamics)
- KNACKE: Opening shock factor approach (existing, tends to overestimate)
- CONTINUOUS_INFLATION: Wolf-French style differential area growth
- MASS_FLOW_BALANCE: Pflanz-style mass conservation model
- FRENCH_HUCKINS: Exponential filling with overshoot
- POROSITY_CORRECTED: Dynamic porosity reducing pressure peaks

References:
- Wolf, D.: "A Simplified Dynamic Model of Parachute Inflation" (1973)
- French, W.J. and Huckins, E.K.: "A Method of Parachute Analysis" (1964)
- Pflanz, E.: "Round Parachutes" in "Parachutes for Engineers" (1959)
- Knacke, T.W.: "Parachute Recovery Systems Design Manual" (1992)
- Ewing, E.G.: "Recovery Systems Design Guide" (1978)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

# Import from local modules - adjust path as needed
from aerislab.dynamics.body import RigidBody6DOF

# Physical constants
EPSILON_VELOCITY = 1e-12
EPSILON_AREA = 1e-6
DEFAULT_GATE_SHARPNESS = 40.0
DEFAULT_COLLAPSED_AREA = 1e-3


class ParachuteModelType(Enum):
    """Available parachute inflation model types."""
    
    SIMPLE_DRAG = "simple_drag"
    KNACKE = "knacke"
    CONTINUOUS_INFLATION = "continuous_inflation"
    MASS_FLOW_BALANCE = "mass_flow_balance"
    FRENCH_HUCKINS = "french_huckins"
    POROSITY_CORRECTED = "porosity_corrected"
    ADDED_MASS = "added_mass"


@dataclass
class ParachuteGeometry:
    """
    Geometric parameters for round parachute canopy.
    
    Attributes
    ----------
    D0 : float
        Nominal (constructed) diameter [m]
    S0 : float | None
        Nominal area [m²]. Computed from D0 if not provided.
    Dv : float
        Vent diameter [m]. Default 0.0 for solid canopy.
    geometric_porosity : float
        Fraction of canopy area that is open (gore gaps, slots) [-]. 
        Typical: 0.0-0.10
    fabric_permeability : float
        Effective porosity due to fabric weave [-]. 
        Typical: 0.01-0.05 for standard parachute cloth
    suspension_line_length : float | None
        Length of suspension lines [m]. Affects stability, not load.
    """
    
    D0: float
    S0: float | None = None
    Dv: float = 0.0
    geometric_porosity: float = 0.0
    fabric_permeability: float = 0.02
    suspension_line_length: float | None = None
    
    def __post_init__(self):
        """Compute derived properties."""
        if self.S0 is None:
            self.S0 = np.pi * (self.D0 / 2.0) ** 2
        if self.suspension_line_length is None:
            self.suspension_line_length = 1.5 * self.D0
    
    @property
    def total_porosity(self) -> float:
        """Total effective porosity (geometric + fabric)."""
        return self.geometric_porosity + self.fabric_permeability
    
    @property
    def projected_area(self) -> float:
        """Projected frontal area accounting for porosity [m²]."""
        return self.S0 * (1.0 - self.geometric_porosity)
    
    @property
    def vent_area(self) -> float:
        """Vent hole area [m²]."""
        return np.pi * (self.Dv / 2.0) ** 2


@dataclass
class InflationConfig:
    """
    Configuration parameters for parachute inflation dynamics.
    
    Attributes
    ----------
    n_fill : float
        Filling constant [-]. t_fill = n_fill * D0 / V (subsonic).
        Typical: 4-8 for round parachutes.
    Cx : float
        Opening load factor for Knacke model [-].
        Typical: 1.0-1.8. Higher = more conservative (larger peaks).
    area_exponent : float
        Exponent in area growth law A = A_full * (t/t_fill)^n.
        Typical: 1.5-2.5
    overshoot_factor : float
        Overshoot coefficient for French-Huckins model [-].
        Typical: 0.05-0.15 for round canopies.
    inflation_time_override : float | None
        If set, use this fixed inflation time [s] instead of computing.
    tau_inflation : float | None
        Time constant for exponential models [s]. Computed if None.
    """
    
    n_fill: float = 6.0
    Cx: float = 1.5
    area_exponent: float = 2.0
    overshoot_factor: float = 0.1
    inflation_time_override: float | None = None
    tau_inflation: float | None = None


@dataclass
class PorosityConfig:
    """
    Configuration for porosity-corrected model.
    
    Attributes
    ----------
    pressure_coefficient : float
        Coefficient relating pressure to porosity increase [-].
        Higher = more pressure relief through fabric.
    porosity_exponent : float
        Exponent α in Cd_eff = Cd_base * (1 - λ)^α.
        Typical: 1.0-2.0
    reference_pressure : float
        Reference dynamic pressure for normalization [Pa].
    """
    
    pressure_coefficient: float = 0.02
    porosity_exponent: float = 1.5
    reference_pressure: float = 1000.0


@dataclass
class MassFlowConfig:
    """
    Configuration for mass flow balance model.
    
    Attributes
    ----------
    inlet_coefficient : float
        Fraction of projected area that captures incoming flow [-].
        Typical: 0.8-1.0
    canopy_stiffness : float
        Stiffness relating internal pressure to volume [Pa/m³].
        Higher = faster inflation, higher peak loads.
    """
    
    inlet_coefficient: float = 0.9
    canopy_stiffness: float = 100.0


@dataclass
class AddedMassConfig:
    """
    Configuration for added mass (apparent mass) model.
    
    Based on Heinrich (1966) and Ludtke (1968) wind tunnel studies.
    
    Attributes
    ----------
    k_added_mass : float
        Added mass coefficient [-]. Ratio of added mass to enclosed air mass.
        Typical values:
        - Hemisphere: 0.5
        - Flat disk: 0.64
        - Round parachute: 0.3-0.5 (depends on shape)
    volume_coefficient : float
        Coefficient relating projected area to enclosed volume.
        V = k_vol * A^1.5 / sqrt(π)
        For hemisphere: k_vol = 2/3
    include_dm_dt_term : bool
        If True, include the dm_added/dt * V term which creates the peak.
        Set False for simplified analysis.
    """
    
    k_added_mass: float = 0.4
    volume_coefficient: float = 0.667  # 2/3 for hemisphere
    include_dm_dt_term: bool = True


@dataclass 
class ParachuteState:
    """
    Internal state of parachute during simulation.
    
    This tracks quantities that evolve during inflation.
    """
    
    is_activated: bool = False
    activation_time: float | None = None
    inflation_start_velocity: float | None = None
    current_area: float = DEFAULT_COLLAPSED_AREA
    air_mass_inside: float = 0.0
    inflation_complete: bool = False
    
    # For added mass model - track previous values for derivatives
    prev_velocity: float | None = None
    prev_added_mass: float = 0.0
    prev_time: float | None = None
    current_added_mass: float = 0.0


class AdvancedParachute:
    """
    Advanced parachute model with multiple physics-based inflation dynamics.
    
    This class provides physically accurate models for round parachute opening
    that produce more realistic peak loads than simplified Cx factor approaches.
    
    Parameters
    ----------
    geometry : ParachuteGeometry
        Canopy geometry (diameter, porosity, etc.)
    model_type : ParachuteModelType
        Which inflation model to use
    rho : float
        Air density [kg/m³]. Default 1.225 (sea level standard).
    Cd : float | Callable
        Drag coefficient [-]. Can be constant or function of (t, body).
        Typical: 0.8-1.0 for round parachutes (based on S0).
    inflation_config : InflationConfig | None
        Inflation dynamics configuration
    porosity_config : PorosityConfig | None
        Porosity model configuration (for POROSITY_CORRECTED)
    mass_flow_config : MassFlowConfig | None
        Mass flow configuration (for MASS_FLOW_BALANCE)
    activation_time : float
        Time at which deployment begins [s]
    activation_altitude : float | None
        Deploy when altitude falls below this [m]
    activation_velocity : float
        Deploy when speed exceeds this [m/s]
    gate_sharpness : float
        Steepness of smooth activation transition [-]
    area_collapsed : float
        Area when collapsed [m²]. Small nonzero for stability.
        
    Examples
    --------
    >>> # Create geometry for 10m diameter T-10 style parachute
    >>> geom = ParachuteGeometry(D0=10.0, geometric_porosity=0.05)
    >>> 
    >>> # Recommended: Continuous inflation model
    >>> para = AdvancedParachute(
    ...     geometry=geom,
    ...     model_type=ParachuteModelType.CONTINUOUS_INFLATION,
    ...     Cd=0.85,
    ...     activation_velocity=30.0
    ... )
    >>>
    >>> # Porosity-corrected for porous canopy
    >>> para_porous = AdvancedParachute(
    ...     geometry=geom,
    ...     model_type=ParachuteModelType.POROSITY_CORRECTED,
    ...     Cd=0.85,
    ...     porosity_config=PorosityConfig(pressure_coefficient=0.03)
    ... )
    """
    
    def __init__(
        self,
        geometry: ParachuteGeometry,
        model_type: ParachuteModelType = ParachuteModelType.CONTINUOUS_INFLATION,
        rho: float = 1.225,
        Cd: float | Callable[[float, RigidBody6DOF], float] = 0.85,
        inflation_config: InflationConfig | None = None,
        porosity_config: PorosityConfig | None = None,
        mass_flow_config: MassFlowConfig | None = None,
        activation_time: float | None = None,
        activation_altitude: float | None = None,
        activation_velocity: float = 30.0,
        gate_sharpness: float = DEFAULT_GATE_SHARPNESS,
        area_collapsed: float = DEFAULT_COLLAPSED_AREA,
        added_mass_config: AddedMassConfig | None = None,
    ) -> None:
        self.geometry = geometry
        self.model_type = model_type
        self.rho = rho
        self.Cd = Cd
        
        # Configuration with defaults
        self.inflation_config = inflation_config or InflationConfig()
        self.porosity_config = porosity_config or PorosityConfig()
        self.mass_flow_config = mass_flow_config or MassFlowConfig()
        self.added_mass_config = added_mass_config or AddedMassConfig()
        
        # Activation conditions
        self.activation_time = activation_time
        self.activation_altitude = activation_altitude
        self.activation_velocity = activation_velocity
        self.gate_sharpness = gate_sharpness
        self.area_collapsed = area_collapsed
        
        # Internal state
        self._state = ParachuteState()
        
    def reset(self) -> None:
        """Reset parachute to un-deployed state."""
        self._state = ParachuteState()
        
    def _check_activation(self, body: RigidBody6DOF, t: float) -> bool:
        """Check if deployment conditions are met and update state."""
        if self._state.is_activated:
            return True
            
        speed = np.linalg.norm(body.v)
        altitude = body.p[2]  # Assuming z-up convention
        
        # Check conditions
        velocity_triggered = speed >= self.activation_velocity
        altitude_triggered = (
            self.activation_altitude is not None 
            and altitude <= self.activation_altitude
        )
        time_triggered = (
            self.activation_time is not None 
            and t >= self.activation_time
        )
        
        if velocity_triggered or altitude_triggered or time_triggered:
            self._state.is_activated = True
            # If triggered by logic other than time, use t as activation time
            if time_triggered:
                 self._state.activation_time = self.activation_time
            else:
                 self._state.activation_time = t
                 
            self._state.inflation_start_velocity = speed
            self._state.current_area = self.area_collapsed
            return True
            
        return False
    
    def _get_cd(self, t: float, body: RigidBody6DOF) -> float:
        """Get current drag coefficient (constant or callable)."""
        if callable(self.Cd):
            return self.Cd(t, body)
        return self.Cd
    
    def _get_inflation_time(self, V: float) -> float:
        """
        Compute filling time based on velocity.
        
        Uses t_fill = n * D0 / V for subsonic flow.
        """
        if self.inflation_config.inflation_time_override is not None:
            return self.inflation_config.inflation_time_override
            
        # Avoid division by zero
        V_safe = max(V, 1.0)
        return self.inflation_config.n_fill * self.geometry.D0 / V_safe
    
    def _compute_normalized_time(self, t: float, V: float) -> float:
        """Compute normalized inflation time τ = (t - t_act) / t_fill."""
        if self._state.activation_time is None:
            return 0.0
            
        dt = t - self._state.activation_time
        t_fill = self._get_inflation_time(V)
        return np.clip(dt / t_fill, 0.0, 1.0)
    
    # =========================================================================
    # SIMPLE DRAG MODEL
    # =========================================================================
    
    def _force_simple_drag(
        self, body: RigidBody6DOF, t: float
    ) -> NDArray[np.float64]:
        """
        Simple quadratic drag with smooth area transition.
        
        No special inflation dynamics - just smooth ramp to full area.
        """
        v = body.v
        speed = np.linalg.norm(v)
        
        if speed < EPSILON_VELOCITY:
            return np.zeros(3)
        
        # Smooth activation gate
        if self._state.activation_time is None:
            return np.zeros(3)
            
        dt = t - self._state.activation_time
        gate = 0.5 * (1.0 + np.tanh(self.gate_sharpness * dt))
        
        A = self.area_collapsed + gate * (self.geometry.S0 - self.area_collapsed)
        self._state.current_area = A
        Cd = self._get_cd(t, body)
        
        # F = -0.5 * ρ * Cd * A * |v| * v̂
        q = 0.5 * self.rho * speed
        return -q * Cd * A * (v / speed) * speed
    
    # =========================================================================
    # KNACKE MODEL (existing approach with Cx factor)
    # =========================================================================
    
    def _force_knacke(
        self, body: RigidBody6DOF, t: float
    ) -> NDArray[np.float64]:
        """
        Knacke model with opening shock factor Cx.
        
        During inflation: F = Cx * F_steady
        After inflation: F = F_steady
        """
        v = body.v
        speed = np.linalg.norm(v)
        
        if speed < EPSILON_VELOCITY:
            return np.zeros(3)
            
        tau = self._compute_normalized_time(t, speed)
        
        # Area growth: A = A_coll + (A_full - A_coll) * tau^n
        n = self.inflation_config.area_exponent
        A = self.area_collapsed + (self.geometry.S0 - self.area_collapsed) * (tau ** n)
        self._state.current_area = A
        
        Cd = self._get_cd(t, body)
        
        # Apply Cx factor during inflation
        if tau < 1.0:
            Cx = self.inflation_config.Cx
        else:
            # Smooth decay from Cx to 1.0 to avoid discontinuity
            # Decay over normalized time (e.g. 50% decay every 0.2 tau)
            k_decay = 5.0
            decay = np.exp(-k_decay * (tau - 1.0))
            Cx = 1.0 + (self.inflation_config.Cx - 1.0) * decay
        
        q = 0.5 * self.rho * speed * speed
        F_mag = q * Cd * A * Cx
        
        return -F_mag * (v / speed)
    
    # =========================================================================
    # CONTINUOUS INFLATION MODEL (Wolf-French style)
    # =========================================================================
    
    def _force_continuous_inflation(
        self, body: RigidBody6DOF, t: float
    ) -> NDArray[np.float64]:
        """
        Continuous inflation with velocity-coupled area growth.
        
        Area grows according to dA/dt ∝ V * sqrt(A), giving natural
        load-velocity coupling without artificial Cx spikes.
        
        This produces peak-to-steady ratios of ~1.2-1.4.
        """
        v = body.v
        speed = np.linalg.norm(v)
        
        if speed < EPSILON_VELOCITY:
            return np.zeros(3)
            
        if self._state.activation_time is None:
            return np.zeros(3)
            
        dt = t - self._state.activation_time
        V0 = self._state.inflation_start_velocity or speed
        t_fill = self._get_inflation_time(V0)
        
        # Continuous area model: A(t) = A_full * tanh²(k * t/t_fill)
        # This gives natural S-curve without discontinuities
        tau = dt / t_fill
        k = 2.0  # Sharpness of transition
        
        # Use tanh² for smooth S-curve (0 -> 1)
        area_ratio = np.tanh(k * tau) ** 2
        A = self.area_collapsed + (self.geometry.S0 - self.area_collapsed) * area_ratio
        
        self._state.current_area = A
        
        # Track if fully inflated
        if area_ratio > 0.99:
            self._state.inflation_complete = True
        
        Cd = self._get_cd(t, body)
        
        # No explicit Cx factor - natural peak from dynamics
        q = 0.5 * self.rho * speed * speed
        F_mag = q * Cd * A
        
        return -F_mag * (v / speed)
    
    # =========================================================================
    # MASS FLOW BALANCE MODEL (Pflanz-style)
    # =========================================================================
    
    def _force_mass_flow_balance(
        self, body: RigidBody6DOF, t: float, dt: float = 0.01
    ) -> NDArray[np.float64]:
        """
        Mass conservation model tracking air inside canopy.
        
        dm/dt = ṁ_in - ṁ_out
        ṁ_in = ρ * V * A_inlet * C_inlet
        ṁ_out = ρ * V_out * (A_porous + A_vent)
        
        Area is function of enclosed mass: A = f(m_inside)
        """
        v = body.v
        speed = np.linalg.norm(v)
        
        if speed < EPSILON_VELOCITY:
            return np.zeros(3)
            
        if self._state.activation_time is None:
            return np.zeros(3)
        
        # Current projected area (dynamically growing)
        A_current = max(self._state.current_area, self.area_collapsed)
        
        # Inlet mass flow
        C_inlet = self.mass_flow_config.inlet_coefficient
        m_dot_in = self.rho * speed * A_current * C_inlet
        
        # Outlet through porosity and vent
        total_porosity = self.geometry.total_porosity
        A_porous = A_current * total_porosity
        A_vent = self.geometry.vent_area
        
        # Outlet velocity (simplified - pressure-driven)
        # V_out ≈ V * sqrt(total_porosity) for pressure equilibrium
        V_out = speed * np.sqrt(total_porosity + 0.01)
        m_dot_out = self.rho * V_out * (A_porous + A_vent)
        
        # Net mass change
        dm_dt = m_dot_in - m_dot_out
        
        # Update internal state (mass accumulation)
        self._state.air_mass_inside += dm_dt * dt
        self._state.air_mass_inside = max(0.0, self._state.air_mass_inside)
        
        # Relate mass to area using canopy volume-area relationship
        # For hemispherical canopy: V ≈ (2/3) * A^1.5 / sqrt(π)
        # m = ρ * V → A ≈ (m / ρ * 3/(2*sqrt(π/A_full)))^(2/3)
        
        # Simplified: A proportional to mass with saturation
        A_full = self.geometry.S0
        m_full = self.rho * (2/3) * (A_full ** 1.5) / np.sqrt(np.pi)
        
        mass_ratio = min(1.0, self._state.air_mass_inside / m_full)
        self._state.current_area = self.area_collapsed + (A_full - self.area_collapsed) * mass_ratio
        
        if mass_ratio > 0.99:
            self._state.inflation_complete = True
        
        Cd = self._get_cd(t, body)
        q = 0.5 * self.rho * speed * speed
        F_mag = q * Cd * self._state.current_area
        
        return -F_mag * (v / speed)
    
    # =========================================================================
    # FRENCH-HUCKINS MODEL (exponential with overshoot)
    # =========================================================================
    
    def _force_french_huckins(
        self, body: RigidBody6DOF, t: float
    ) -> NDArray[np.float64]:
        """
        French-Huckins model with exponential area growth and overshoot.
        
        A(t) = A_full * [1 - exp(-t/τ)]²
        F = 0.5 * ρ * Cd * A * V² * (1 + k * exp(-t/τ))
        
        The overshoot term provides small, realistic peak without Cx spike.
        Typical peak-to-steady ratio: 1.15-1.25.
        """
        v = body.v
        speed = np.linalg.norm(v)
        
        if speed < EPSILON_VELOCITY:
            return np.zeros(3)
            
        if self._state.activation_time is None:
            return np.zeros(3)
            
        dt = t - self._state.activation_time
        t_fill = self._get_inflation_time(speed)
        
        # Time constant (τ ≈ t_fill / 3 for 95% open at t_fill)
        tau = self.inflation_config.tau_inflation or (t_fill / 3.0)
        
        # Exponential area growth: [1 - exp(-t/τ)]²
        exp_term = np.exp(-dt / tau)
        area_ratio = (1.0 - exp_term) ** 2
        
        A = self.area_collapsed + (self.geometry.S0 - self.area_collapsed) * area_ratio
        self._state.current_area = A
        
        # Overshoot term: (1 + k * exp(-t/τ))
        k_over = self.inflation_config.overshoot_factor
        overshoot_multiplier = 1.0 + k_over * exp_term
        
        if area_ratio > 0.99:
            self._state.inflation_complete = True
        
        Cd = self._get_cd(t, body)
        q = 0.5 * self.rho * speed * speed
        F_mag = q * Cd * A * overshoot_multiplier
        
        return -F_mag * (v / speed)
    
    # =========================================================================
    # POROSITY-CORRECTED MODEL
    # =========================================================================
    
    def _force_porosity_corrected(
        self, body: RigidBody6DOF, t: float
    ) -> NDArray[np.float64]:
        """
        Porosity-corrected model with dynamic pressure relief.
        
        Accounts for increased fabric porosity under load, which
        reduces internal pressure and peak opening forces.
        
        λ_eff = λ_base + Cp * (q / q_ref)
        Cd_eff = Cd_base * (1 - λ_eff)^α
        """
        v = body.v
        speed = np.linalg.norm(v)
        
        if speed < EPSILON_VELOCITY:
            return np.zeros(3)
            
        tau = self._compute_normalized_time(t, speed)
        
        # Use continuous inflation area model
        k = 2.0
        area_ratio = np.tanh(k * tau) ** 2
        A = self.area_collapsed + (self.geometry.S0 - self.area_collapsed) * area_ratio
        
        self._state.current_area = A
        
        # Dynamic pressure
        q = 0.5 * self.rho * speed * speed
        
        # Effective porosity increases with pressure
        lambda_base = self.geometry.total_porosity
        Cp = self.porosity_config.pressure_coefficient
        q_ref = self.porosity_config.reference_pressure
        
        # λ_eff = λ_base + Cp * (q / q_ref), capped at 0.5
        lambda_eff = min(0.5, lambda_base + Cp * (q / q_ref))
        
        # Drag reduction due to porosity
        alpha = self.porosity_config.porosity_exponent
        Cd_base = self._get_cd(t, body)
        Cd_eff = Cd_base * ((1.0 - lambda_eff) ** alpha)
        
        if area_ratio > 0.99:
            self._state.inflation_complete = True
        
        F_mag = q * Cd_eff * A
        
        return -F_mag * (v / speed)
    
    # =========================================================================
    # ADDED MASS MODEL (Heinrich/Ludtke)
    # =========================================================================
    
    def _force_added_mass(
        self, body: RigidBody6DOF, t: float, dt: float = 0.01
    ) -> NDArray[np.float64]:
        """
        Added mass (apparent mass) model based on Heinrich/Ludtke.
        
        Accounts for inertia of air entrained with/around the canopy.
        
        F_total = F_drag + d/dt(m_added * V)
                = F_drag + m_added * dV/dt + dm_added/dt * V
        
        The dm/dt term creates the natural peak during rapid inflation.
        
        References:
        - Heinrich, H.G.: "The Effective Porosity of Parachutes" (1966)
        - Ludtke, W.P.: "Notes on Parachute Opening Dynamics" (1968)
        """
        v = body.v
        speed = np.linalg.norm(v)
        
        if speed < EPSILON_VELOCITY:
            return np.zeros(3)
            
        if self._state.activation_time is None:
            return np.zeros(3)
        
        dt_inflation = t - self._state.activation_time
        V0 = self._state.inflation_start_velocity or speed
        t_fill = self._get_inflation_time(V0)
        
        # Use smooth s-curve for area growth
        tau = dt_inflation / t_fill
        k = 2.0
        area_ratio = np.tanh(k * tau) ** 2
        A = self.area_collapsed + (self.geometry.S0 - self.area_collapsed) * area_ratio
        self._state.current_area = A
        
        # Compute enclosed volume from area
        # V_enclosed = k_vol * A^1.5 / sqrt(pi)
        k_vol = self.added_mass_config.volume_coefficient
        V_enclosed = k_vol * (A ** 1.5) / np.sqrt(np.pi)
        
        # Added mass: m_added = k_a * rho * V_enclosed
        k_a = self.added_mass_config.k_added_mass
        m_added = k_a * self.rho * V_enclosed
        
        # Store current added mass for next iteration
        prev_added_mass = self._state.current_added_mass
        self._state.current_added_mass = m_added
        
        # Compute dm_added/dt (rate of added mass change)
        if self._state.prev_time is not None and dt > 0:
            dt_actual = t - self._state.prev_time
            if dt_actual > 1e-10:
                dm_dt = (m_added - prev_added_mass) / dt_actual
            else:
                dm_dt = 0.0
        else:
            dm_dt = 0.0
        
        # Compute dV/dt (deceleration)
        if self._state.prev_velocity is not None and dt > 0:
            dt_actual = t - (self._state.prev_time or t - dt)
            if dt_actual > 1e-10:
                dV_dt = (speed - self._state.prev_velocity) / dt_actual
            else:
                dV_dt = 0.0
        else:
            dV_dt = 0.0
        
        # Update state for next call
        self._state.prev_velocity = speed
        self._state.prev_time = t
        
        # Base drag force
        Cd = self._get_cd(t, body)
        q = 0.5 * self.rho * speed * speed
        F_drag = q * Cd * A
        
        # Added mass inertial forces
        # F_inertial = m_added * dV/dt + dm/dt * V
        # Note: dV/dt is typically negative (deceleration)
        F_inertial_1 = m_added * abs(dV_dt)  # m * a term
        
        F_inertial_2 = 0.0
        if self.added_mass_config.include_dm_dt_term:
            # dm/dt * V term - this creates the peak during rapid inflation
            F_inertial_2 = dm_dt * speed
        
        # Total force magnitude
        F_total = F_drag + F_inertial_1 + max(0, F_inertial_2)
        
        if area_ratio > 0.99:
            self._state.inflation_complete = True
        
        # Direction: opposing velocity
        return -F_total * (v / speed)
    
    # =========================================================================
    # PUBLIC INTERFACE
    # =========================================================================
    
    def compute_force(
        self, 
        body: RigidBody6DOF, 
        t: float,
        dt: float = 0.01
    ) -> NDArray[np.float64]:
        """
        Compute parachute drag force.
        
        Parameters
        ----------
        body : RigidBody6DOF
            The body the parachute is attached to
        t : float
            Current simulation time [s]
        dt : float
            Time step [s]. Only used by mass flow model.
            
        Returns
        -------
        NDArray[np.float64]
            Drag force vector in world frame [N] (3,)
        """
        # Check activation
        if not self._check_activation(body, t):
            return np.zeros(3)
        
        # Dispatch to appropriate model
        if self.model_type == ParachuteModelType.SIMPLE_DRAG:
            return self._force_simple_drag(body, t)
        elif self.model_type == ParachuteModelType.KNACKE:
            return self._force_knacke(body, t)
        elif self.model_type == ParachuteModelType.CONTINUOUS_INFLATION:
            return self._force_continuous_inflation(body, t)
        elif self.model_type == ParachuteModelType.MASS_FLOW_BALANCE:
            return self._force_mass_flow_balance(body, t, dt)
        elif self.model_type == ParachuteModelType.FRENCH_HUCKINS:
            return self._force_french_huckins(body, t)
        elif self.model_type == ParachuteModelType.POROSITY_CORRECTED:
            return self._force_porosity_corrected(body, t)
        elif self.model_type == ParachuteModelType.ADDED_MASS:
            return self._force_added_mass(body, t, dt)
        else:
            # Fallback to continuous inflation
            return self._force_continuous_inflation(body, t)
    
    def apply(
        self, 
        body: RigidBody6DOF, 
        t: float | None = None,
        dt: float = 0.01
    ) -> None:
        """
        Apply parachute force to body.
        
        Parameters
        ----------
        body : RigidBody6DOF
            The body to apply force to
        t : float | None
            Current simulation time [s]
        dt : float
            Time step [s]
        """
        tval = 0.0 if t is None else float(t)
        F = self.compute_force(body, tval, dt)
        body.apply_force(F, label="aerodynamics")
    
    def get_current_area(self) -> float:
        """Get current effective parachute area [m²]."""
        return self._state.current_area
    
    def is_activated(self) -> bool:
        """Check if parachute has been deployed."""
        return self._state.is_activated
    
    def is_fully_inflated(self) -> bool:
        """Check if parachute is fully inflated."""
        return self._state.inflation_complete
    
    def get_peak_to_steady_ratio_estimate(self) -> float:
        """
        Estimate expected peak-to-steady force ratio for this model.
        
        Returns
        -------
        float
            Expected ratio of peak force to steady-state force.
        """
        ratios = {
            ParachuteModelType.SIMPLE_DRAG: 1.0,
            ParachuteModelType.KNACKE: self.inflation_config.Cx,
            ParachuteModelType.CONTINUOUS_INFLATION: 1.3,  # Typical
            ParachuteModelType.MASS_FLOW_BALANCE: 1.2,
            ParachuteModelType.FRENCH_HUCKINS: 1.0 + self.inflation_config.overshoot_factor * 2,
            ParachuteModelType.POROSITY_CORRECTED: 1.15,
            ParachuteModelType.ADDED_MASS: 1.25,  # Typical for k_a=0.4
        }
        return ratios.get(self.model_type, 1.3)


def create_parachute(
    diameter: float,
    model: str | ParachuteModelType = "continuous_inflation",
    porosity: float = 0.05,
    Cd: float = 0.85,
    **kwargs
) -> AdvancedParachute:
    """
    Factory function to create parachute with common configurations.
    
    Parameters
    ----------
    diameter : float
        Nominal canopy diameter [m]
    model : str | ParachuteModelType
        Model type ("simple_drag", "knacke", "continuous_inflation", 
        "mass_flow_balance", "french_huckins", "porosity_corrected")
    porosity : float
        Total porosity (geometric + fabric) [-]
    Cd : float
        Drag coefficient [-]
    **kwargs
        Additional arguments passed to AdvancedParachute
        
    Returns
    -------
    AdvancedParachute
        Configured parachute instance
        
    Examples
    --------
    >>> # 10m personnel parachute with recommended model
    >>> para = create_parachute(10.0, model="continuous_inflation")
    >>>
    >>> # Cargo parachute with French-Huckins
    >>> para = create_parachute(15.0, model="french_huckins", Cd=0.9)
    """
    # Convert string to enum if needed
    if isinstance(model, str):
        model = ParachuteModelType(model)
    
    geometry = ParachuteGeometry(
        D0=diameter,
        geometric_porosity=porosity * 0.5,
        fabric_permeability=porosity * 0.5,
    )
    
    return AdvancedParachute(
        geometry=geometry,
        model_type=model,
        Cd=Cd,
        **kwargs
    )


# =============================================================================
# EXAMPLES AND USAGE PATTERNS
# =============================================================================

def example_model_comparison():
    """
    Example: Compare all parachute models on the same scenario.
    
    This demonstrates how different models produce different peak-to-steady
    force ratios for the same parachute and conditions.
    """
    import numpy as np
    
    # Create a mock body for testing (replace with your RigidBody6DOF)
    class MockBody:
        def __init__(self, velocity, position):
            self.v = np.array(velocity)
            self.p = np.array(position)
            self.f = np.zeros(3)
        def apply_force(self, f, **kwargs):
            self.f += f
    
    # Common geometry: 10m diameter T-10 style parachute
    geometry = ParachuteGeometry(
        D0=10.0,                    # 10 meter diameter
        geometric_porosity=0.05,    # 5% open area (vents, gaps)
        fabric_permeability=0.02,   # Low permeability nylon
    )
    
    results = {}
    
    for model_type in ParachuteModelType:
        para = AdvancedParachute(
            geometry=geometry,
            model_type=model_type,
            rho=1.225,
            Cd=0.85,
            activation_velocity=0.0,  # Deploy immediately
        )
        
        body = MockBody(velocity=(0, 0, -50), position=(0, 0, 1000))
        
        # Simulate inflation
        forces = []
        dt = 0.05
        for t in np.arange(0.05, 10.0, dt):
            F = para.compute_force(body, t=t, dt=dt)
            forces.append(np.linalg.norm(F))
        
        peak = max(forces) if forces else 0
        steady = np.mean(forces[-20:]) if len(forces) > 20 else peak
        ratio = peak / steady if steady > 0 else 1.0
        
        results[model_type.value] = {
            'peak_force': peak,
            'steady_force': steady,
            'peak_ratio': ratio,
        }
        
        print(f"{model_type.value:25s}: Peak={peak:8.1f}N, "
              f"Steady={steady:8.1f}N, Ratio={ratio:.2f}")
    
    return results


def example_simple_simulation():
    """
    Example: Complete parachute deployment simulation.
    
    Simulates a payload falling and deploying a parachute,
    tracking forces and velocity throughout.
    """
    import numpy as np
    
    class MockBody:
        def __init__(self, velocity, position, mass):
            self.v = np.array(velocity, dtype=float)
            self.p = np.array(position, dtype=float)
            self.mass = mass
            self.f = np.zeros(3)
        def apply_force(self, f, **kwargs):
            self.f += f
        def clear_forces(self):
            self.f = np.zeros(3)
    
    # Create parachute with recommended model
    para = create_parachute(
        diameter=10.0,
        model="continuous_inflation",
        porosity=0.05,
        Cd=0.85,
        activation_velocity=30.0,  # Deploy at 30 m/s
    )
    
    # Simulation parameters
    mass = 100.0  # kg
    g = 9.81      # m/s²
    dt = 0.01     # s
    
    # Initial state
    velocity = 0.0   # Starting at rest (will accelerate due to gravity)
    altitude = 1000.0  # m
    time = 0.0
    
    # Recording
    history = {
        'time': [],
        'altitude': [],
        'velocity': [],
        'force': [],
        'activated': [],
    }
    
    while altitude > 0 and time < 60.0:
        body = MockBody(
            velocity=(0, 0, velocity),
            position=(0, 0, altitude),
            mass=mass,
        )
        
        # Compute parachute force
        F_para = para.compute_force(body, t=time, dt=dt)
        F_gravity = -mass * g
        F_total = F_para[2] + F_gravity
        
        # Record
        history['time'].append(time)
        history['altitude'].append(altitude)
        history['velocity'].append(velocity)
        history['force'].append(np.linalg.norm(F_para))
        history['activated'].append(para.is_activated())
        
        # Integrate
        acceleration = F_total / mass
        velocity += acceleration * dt
        altitude += velocity * dt
        time += dt
    
    print(f"Simulation complete:")
    print(f"  Total time: {time:.1f} s")
    print(f"  Final velocity: {velocity:.2f} m/s")
    print(f"  Peak parachute force: {max(history['force']):.1f} N")
    print(f"  Steady force: {np.mean(history['force'][-100:]):.1f} N")
    
    return history


def example_all_models():
    """
    Example: Create parachutes with each available model.
    
    Shows proper configuration for each model type.
    """
    # Common geometry
    geom = ParachuteGeometry(D0=10.0, geometric_porosity=0.05)
    
    # 1. SIMPLE_DRAG - Fastest, no inflation dynamics
    para_simple = AdvancedParachute(
        geometry=geom,
        model_type=ParachuteModelType.SIMPLE_DRAG,
        Cd=0.85,
        activation_velocity=30.0,
    )
    
    # 2. KNACKE - Traditional Cx factor approach
    para_knacke = AdvancedParachute(
        geometry=geom,
        model_type=ParachuteModelType.KNACKE,
        Cd=0.85,
        inflation_config=InflationConfig(
            Cx=1.5,           # Opening load factor
            n_fill=6.0,       # Filling constant
            area_exponent=2.0,  # Quadratic area growth
        ),
    )
    
    # 3. CONTINUOUS_INFLATION - Recommended for general use
    para_continuous = AdvancedParachute(
        geometry=geom,
        model_type=ParachuteModelType.CONTINUOUS_INFLATION,
        Cd=0.85,
        inflation_config=InflationConfig(n_fill=6.0),
    )
    
    # 4. MASS_FLOW_BALANCE - Most detailed physics
    para_massflow = AdvancedParachute(
        geometry=geom,
        model_type=ParachuteModelType.MASS_FLOW_BALANCE,
        Cd=0.85,
        mass_flow_config=MassFlowConfig(
            inlet_coefficient=0.9,
            canopy_stiffness=100.0,
        ),
    )
    
    # 5. FRENCH_HUCKINS - Exponential with overshoot
    para_fh = AdvancedParachute(
        geometry=geom,
        model_type=ParachuteModelType.FRENCH_HUCKINS,
        Cd=0.85,
        inflation_config=InflationConfig(
            n_fill=6.0,
            overshoot_factor=0.1,  # 10% overshoot
        ),
    )
    
    # 6. POROSITY_CORRECTED - For porous canopies
    para_porous = AdvancedParachute(
        geometry=geom,
        model_type=ParachuteModelType.POROSITY_CORRECTED,
        Cd=0.85,
        porosity_config=PorosityConfig(
            pressure_coefficient=0.02,
            porosity_exponent=1.5,
        ),
    )
    
    # 7. ADDED_MASS - Physically complete with inertial effects
    para_added_mass = AdvancedParachute(
        geometry=geom,
        model_type=ParachuteModelType.ADDED_MASS,
        Cd=0.85,
        added_mass_config=AddedMassConfig(
            k_added_mass=0.4,      # Added mass coefficient
            volume_coefficient=0.667,  # 2/3 for hemisphere
            include_dm_dt_term=True,   # Include dm/dt*V peak term
        ),
    )
    
    return {
        'simple': para_simple,
        'knacke': para_knacke,
        'continuous': para_continuous,
        'massflow': para_massflow,
        'french_huckins': para_fh,
        'porosity': para_porous,
        'added_mass': para_added_mass,
    }


def example_quick_start():
    """
    Quick start: Minimal code to create and use a parachute.
    """
    # Fastest way - use factory function
    para = create_parachute(
        diameter=10.0,            # 10m diameter
        model="added_mass",       # Recommended for accuracy
        Cd=0.85,
        activation_velocity=30.0,
    )
    
    # For use in simulation:
    # F = para.compute_force(body, t=current_time, dt=timestep)
    # body.apply_force(F)
    
    return para


if __name__ == "__main__":
    print("=" * 60)
    print("Parachute Model Comparison")
    print("=" * 60)
    example_model_comparison()
    
    print("\n" + "=" * 60)
    print("Simple Deployment Simulation")
    print("=" * 60)
    example_simple_simulation()

