import numpy as np

def fully_implicit_step(state, derivative_func, dt, mass, tol=1e-6, max_iter=20):
    pos = state[0:3]
    vel = state[3:6]
    quat = state[6:10]
    omega = state[10:13]

    v_new = vel.copy()
    for _ in range(max_iter):
        pos_new = pos + dt * v_new
        test_state = np.concatenate([pos_new, v_new, quat, omega])
        a = derivative_func(test_state)[3:6]
        res = v_new - vel - dt * a

        if np.linalg.norm(res) < tol:
            break

        # Finite difference Jacobian
        J = np.zeros((3, 3))
        eps = 1e-6
        for i in range(3):
            dv = np.zeros(3)
            dv[i] = eps
            a_plus = derivative_func(np.concatenate([pos + dt*(v_new+dv), v_new+dv, quat, omega]))[3:6]
            a_minus = derivative_func(np.concatenate([pos + dt*(v_new-dv), v_new-dv, quat, omega]))[3:6]
            J[:, i] = (a_plus - a_minus) / (2 * eps)

        try:
            dv_corr = np.linalg.solve(np.eye(3) - dt * J, -res)
        except np.linalg.LinAlgError:
            break

        v_new += dv_corr

    pos_new = pos + dt * v_new
    quat_new = quat + 0.5 * dt * quat_omega(omega) @ quat
    quat_new /= np.linalg.norm(quat_new)
    omega_new = omega  # angular integration not implicit

    return np.concatenate([pos_new, v_new, quat_new, omega_new])

def quat_omega(omega):
    wx, wy, wz = omega
    return np.array([
        [0.0, -wx, -wy, -wz],
        [wx, 0.0, wz, -wy],
        [wy, -wz, 0.0, wx],
        [wz, wy, -wx, 0.0]
    ])
