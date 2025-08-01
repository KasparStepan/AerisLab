import numpy as np
from math_utils import quaternion_multiply, normalize_quaternion

def integrate_rigid_body(body, dt, scheme='semi'):
    print(f"[DEBUG] {body.name}: pos={body.position}, vel={body.linear_velocity}, ω={body.angular_velocity}")
    if scheme in ('explicit', 'semi'):
        a = body.force * body.inv_mass
        body.linear_velocity += a * dt
        body.position += body.linear_velocity * dt
        Iw = body.get_inertia_world()
        omega = body.angular_velocity
        alpha = np.linalg.inv(Iw) @ (body.torque - np.cross(omega, Iw @ omega))
        body.angular_velocity += alpha * dt
        dq = quaternion_multiply(body.orientation, np.concatenate(([0.], body.angular_velocity))) * 0.5
        body.orientation = normalize_quaternion(body.orientation + dq * dt)
    elif scheme == 'rk4':
        a0 = body.force * body.inv_mass
        v0 = body.linear_velocity
        p_mid = body.position + v0 * (dt / 2)
        v_mid = body.linear_velocity + a0 * (dt / 2)
        body.position += v_mid * dt
        body.linear_velocity += a0 * dt
        Iw = body.get_inertia_world()
        alpha = np.linalg.inv(Iw) @ (body.torque - np.cross(body.angular_velocity, Iw @ body.angular_velocity))
        body.angular_velocity += alpha * dt
        dq = quaternion_multiply(body.orientation, np.concatenate(([0.], body.angular_velocity))) * 0.5
        body.orientation = normalize_quaternion(body.orientation + dq * dt)
    else:
        raise ValueError(f"Unknown integration scheme: {scheme}")
    body.clear_forces()
    print(f"[DEBUG] {body.name}: new pos={body.position}, vel={body.linear_velocity}, ω={body.angular_velocity}")