"""
Tests for the FastISA atmosphere model.

These tests verify:
1. Computed properties match published ISA reference values (all layers)
2. The precomputed LUT agrees with the analytical formulas
3. The out-of-range path falls back to analytical computation correctly
4. Internal consistency: rho = p / (R * T) at every altitude
5. Physical monotonicity (p, rho decrease with altitude; troposphere lapse)
6. Edge cases: negative altitude, boundaries, non-unit resolution
"""

import numpy as np
import pytest

from aerislab.models.atmosphere.isa import FastISA

# ============================================================================
# Reference data
# ============================================================================

# Standard ISA reference points: altitude [m] -> (T [K], p [Pa], rho [kg/m^3]).
# Spans troposphere (<=11 km), lower stratosphere (11-20 km, isothermal),
# and upper stratosphere (>20 km, mild positive lapse).
ISA_REFERENCE = {
    0: (288.15, 101325.0, 1.225),
    2000: (275.15, 79495.0, 1.00649),
    5000: (255.65, 54019.9, 0.73612),
    8000: (236.15, 35599.8, 0.52517),
    11000: (216.65, 22631.7, 0.36392),
    15000: (216.65, 12044.5, 0.19367),
    20000: (216.65, 5474.81, 0.08803),
    25000: (221.65, 2511.0, 0.03947),
}

RTOL = 1e-3


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def model():
    """LUT covering the full reference range at 1 m resolution."""
    return FastISA(max_altitude=30000.0, resolution=1.0)


@pytest.fixture
def tropo_model():
    """Default-range LUT (10 km) so >10 km queries hit the analytical fallback."""
    return FastISA(max_altitude=10000.0, resolution=1.0)


# ============================================================================
# Construction
# ============================================================================

def test_constructs_without_error():
    """FastISA(...) should build its LUT in __init__ without raising."""
    m = FastISA()
    assert m.max_idx == int(10000.0 / 1.0) + 1
    assert m._rho.shape == (m.max_idx,)
    assert m._pressure.shape == (m.max_idx,)
    assert m._temperature.shape == (m.max_idx,)


def test_lut_is_populated(model):
    """Every LUT entry should be filled (no leftover zeros)."""
    assert np.all(model._rho > 0.0)
    assert np.all(model._pressure > 0.0)
    assert np.all(model._temperature > 0.0)


# ============================================================================
# Known-value accuracy
# ============================================================================

@pytest.mark.parametrize("altitude", sorted(ISA_REFERENCE))
def test_temperature_matches_reference(model, altitude):
    T_ref = ISA_REFERENCE[altitude][0]
    assert model.temperature(altitude) == pytest.approx(T_ref, rel=RTOL)


@pytest.mark.parametrize("altitude", sorted(ISA_REFERENCE))
def test_pressure_matches_reference(model, altitude):
    p_ref = ISA_REFERENCE[altitude][1]
    assert model.pressure(altitude) == pytest.approx(p_ref, rel=RTOL)


@pytest.mark.parametrize("altitude", sorted(ISA_REFERENCE))
def test_density_matches_reference(model, altitude):
    rho_ref = ISA_REFERENCE[altitude][2]
    assert model.density(altitude) == pytest.approx(rho_ref, rel=RTOL)


def test_sea_level_is_exact(model):
    """Sea-level values are definitional, so they should match closely."""
    assert model.temperature(0) == pytest.approx(288.15, rel=1e-6)
    assert model.pressure(0) == pytest.approx(101325.0, rel=1e-6)
    assert model.density(0) == pytest.approx(1.225, rel=1e-3)


def test_properties_not_swapped(model):
    """Guard against the classic pressure/temperature mix-up: their
    magnitudes differ by orders, so a swap would be obvious."""
    assert model.pressure(0) > 1e4   # Pa, ~1e5
    assert model.temperature(0) < 1e3  # K, ~3e2


# ============================================================================
# LUT vs. analytical agreement
# ============================================================================

@pytest.mark.parametrize("altitude", [0, 1000, 3500, 7000, 9999])
def test_lut_matches_analytical(model, altitude):
    """Within the table, getters must equal the analytical source of truth.
    The analytical tuple is ordered (rho, p, T)."""
    rho, p, T = model._calculate_analytical_properties(altitude)
    assert model.density(altitude) == pytest.approx(rho, rel=1e-9)
    assert model.pressure(altitude) == pytest.approx(p, rel=1e-9)
    assert model.temperature(altitude) == pytest.approx(T, rel=1e-9)


# ============================================================================
# Out-of-range fallback
# ============================================================================

@pytest.mark.parametrize("altitude", [15000, 20000, 25000])
def test_fallback_above_lut(tropo_model, altitude):
    """Above max_altitude (10 km here) the getters use the analytical path
    and must still match ISA reference values."""
    T_ref, p_ref, rho_ref = ISA_REFERENCE[altitude]
    assert tropo_model.temperature(altitude) == pytest.approx(T_ref, rel=RTOL)
    assert tropo_model.pressure(altitude) == pytest.approx(p_ref, rel=RTOL)
    assert tropo_model.density(altitude) == pytest.approx(rho_ref, rel=RTOL)


def test_fallback_continuous_at_boundary(tropo_model):
    """Just below and just above the LUT boundary should be nearly equal
    (no jump between table and analytical paths)."""
    top = (tropo_model.max_idx - 1) * tropo_model.resolution
    below = tropo_model.pressure(top)
    above = tropo_model.pressure(top + tropo_model.resolution)
    assert above == pytest.approx(below, rel=2e-3)


# ============================================================================
# Internal consistency
# ============================================================================

@pytest.mark.parametrize("altitude", [0, 2500, 6000, 11000, 18000, 24000])
def test_ideal_gas_consistency(model, altitude):
    """rho, p, T must satisfy the ideal gas law rho = p / (R * T)."""
    p = model.pressure(altitude)
    T = model.temperature(altitude)
    rho = model.density(altitude)
    assert rho == pytest.approx(p / (model.R * T), rel=1e-9)


# ============================================================================
# Monotonicity / physical trends
# ============================================================================

def test_pressure_strictly_decreases(model):
    sample = [model.pressure(h) for h in range(0, 25000, 500)]
    assert all(b < a for a, b in zip(sample, sample[1:]))


def test_density_strictly_decreases(model):
    sample = [model.density(h) for h in range(0, 25000, 500)]
    assert all(b < a for a, b in zip(sample, sample[1:]))


def test_troposphere_lapse_rate(model):
    """Temperature should drop at L = 6.5 K/km through the troposphere."""
    dT = model.temperature(1000) - model.temperature(0)
    assert dT == pytest.approx(-model.L * 1000, rel=RTOL)


def test_lower_stratosphere_isothermal(model):
    """11-20 km is constant temperature in the ISA model."""
    assert model.temperature(15000) == pytest.approx(model.temperature(12000), rel=1e-9)


# ============================================================================
# Edge cases
# ============================================================================

@pytest.mark.parametrize("altitude", [0.0, -10.0, -500.0])
def test_non_positive_altitude_clamps_to_sea_level(model, altitude):
    assert model.temperature(altitude) == pytest.approx(288.15, rel=1e-6)
    assert model.pressure(altitude) == pytest.approx(101325.0, rel=1e-6)
    assert model.density(altitude) == pytest.approx(model.density(0), rel=1e-9)


def test_resolution_other_than_one():
    """Coarser resolution should still index and interpolate-free lookup correctly."""
    m = FastISA(max_altitude=10000.0, resolution=100.0)
    # Grid point at 5000 m -> idx 50, must match analytical there.
    _, p, T = m._calculate_analytical_properties(5000.0)
    assert m.pressure(5000.0) == pytest.approx(p, rel=1e-9)
    assert m.temperature(5000.0) == pytest.approx(T, rel=1e-9)


def test_top_of_lut_index_in_bounds(model):
    """Querying exactly at max_altitude must not index out of bounds."""
    top = (model.max_idx - 1) * model.resolution
    # Should return a finite, positive pressure without IndexError.
    assert model.pressure(top) > 0.0
