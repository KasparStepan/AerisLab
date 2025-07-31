import numpy as np

def rk4_step(state, derivative_func, dt):
    """
    Performs one Runge-Kutta 4th order (RK4) integration step.

    Parameters:
    - state: np.ndarray, the current state vector
    - derivative_func: callable, function f(state) returning derivative
    - dt: float, time step

    Returns:
    - np.ndarray, the updated state vector after dt
    """

    k1 = derivative_func(state)
    k2 = derivative_func(state + 0.5 * dt * k1)
    k3 = derivative_func(state + 0.5 * dt * k2)
    k4 = derivative_func(state + dt * k3)

    new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return new_state
