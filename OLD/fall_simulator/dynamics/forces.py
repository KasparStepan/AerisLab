import numpy as np

def gravity_force(mass, gravity_vector=np.array([0.0, 0.0, -9.81])):
    """
    Computes gravitational force.

    Parameters:
    - mass: float, object's mass (kg)
    - gravity_vector: np.ndarray, gravitational acceleration vector (m/s²)

    Returns:
    - np.ndarray, gravitational force vector (N)
    """
    return mass * gravity_vector

def drag_force(velocity, air_density, drag_coefficient, area):
    """
    Computes aerodynamic drag force using standard drag equation.

    Parameters:
    - velocity: np.ndarray, velocity vector (m/s)
    - air_density: float, air density (kg/m³)
    - drag_coefficient: float, drag coefficient (dimensionless)
    - area: float, reference area (m²)

    Returns:
    - np.ndarray, drag force vector (N)
    """
    v_mag = np.linalg.norm(velocity)
    if v_mag == 0.0:
        return np.zeros(3)

    drag_direction = -velocity / v_mag  # Opposite to velocity
    drag_magnitude = 0.5 * air_density * v_mag**2 * drag_coefficient * area
    return drag_magnitude * drag_direction

def parachute_drag_force(velocity, parachute_deployed, air_density=1.225, drag_coefficient=1.5, parachute_area=10.0):
    """
    Computes parachute-induced drag force.

    Parameters:
    - velocity: np.ndarray, object's velocity vector (m/s)
    - parachute_deployed: bool, whether parachute is deployed
    - air_density: float, air density (kg/m³)
    - drag_coefficient: float, parachute drag coefficient
    - parachute_area: float, parachute cross-sectional area (m²)

    Returns:
    - np.ndarray, parachute drag force (N)
    """
    if parachute_deployed:
        return drag_force(velocity, air_density, drag_coefficient, parachute_area)
    else:
        return np.zeros(3)
