import numpy as np

def semi_implicit_step(state, derivative_func, dt):
    """
    Performs one time step using the semi-implicit Euler method.

    state: current state vector (pos, vel, quat, omega)
    derivative_func: function that computes derivatives from current state
    dt: time step
    """
    # Unpack state
    pos = state[0:3]
    vel = state[3:6]
    quat = state[6:10]
    omega = state[10:13]

    # Compute acceleration and angular acceleration
    dstate = derivative_func(state)
    accel = dstate[3:6]
    alpha = dstate[10:13]

    # Semi-implicit: update velocities first
    vel_new = vel + accel * dt
    omega_new = omega + alpha * dt

    # Then update position and orientation
    pos_new = pos + vel_new * dt

    # Simple quaternion update (you can replace with exact if needed)
    quat_new = quat + 0.5 * dt * quat_omega(omega_new) @ quat
    quat_new /= np.linalg.norm(quat_new)

    # Combine new state
    new_state = np.concatenate([pos_new, vel_new, quat_new, omega_new])
    return new_state

def quat_omega(omega):
    """Quaternion multiplication matrix for dq/dt = 0.5 * omega_matrix * q"""
    wx, wy, wz = omega
    return np.array([
        [0.0, -wx, -wy, -wz],
        [wx, 0.0, wz, -wy],
        [wy, -wz, 0.0, wx],
        [wz, wy, -wx, 0.0]
    ])
