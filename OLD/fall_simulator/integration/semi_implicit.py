import numpy as np

def semi_implicit_step(state, derivative_func, dt):
    """
    Semi-implicit Euler integration step for 6DOF rigid body.

    state: current [pos, vel, quat, omega]
    derivative_func: function(state) → dstate
    dt: timestep
    """
    pos = state[0:3]
    vel = state[3:6]
    quat = state[6:10]
    omega = state[10:13]

    dstate = derivative_func(state)
    accel = dstate[3:6]
    alpha = dstate[10:13]

    # Update velocities first (semi-implicit)
    vel_new = vel + accel * dt
    omega_new = omega + alpha * dt

    # Then update positions using new velocity
    pos_new = pos + vel_new * dt

    # Update quaternion (rotation)
    quat_new = quat + 0.5 * dt * quat_omega(omega_new) @ quat
    quat_new /= np.linalg.norm(quat_new)

    return np.concatenate([pos_new, vel_new, quat_new, omega_new])


def quat_omega(omega):
    wx, wy, wz = omega
    return np.array([
        [0.0, -wx, -wy, -wz],
        [wx, 0.0, wz, -wy],
        [wy, -wz, 0.0, wx],
        [wz, wy, -wx, 0.0]
    ])
