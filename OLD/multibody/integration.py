import numpy as np

def integrate(body, dt, method='semi_euler'):
    if method == 'semi_euler':
        body._semi_implicit_euler(dt)
    elif method == 'explicit_euler':
        explicit_euler(body, dt)
    elif method == 'rk4':
        rk4_integrate(body, dt)
    else:
        raise ValueError(f"Unknown integration method: {method}")

def explicit_euler(body, dt):
    y = body.get_state_vector()
    dy = body.compute_state_derivative()
    body.set_state_vector(y + dt * dy)
    body.reset_forces()

def rk4_integrate(body, dt):
    y0 = body.get_state_vector()

    def f(state):
        body.set_state_vector(state)
        return body.compute_state_derivative()

    k1 = f(y0)
    k2 = f(y0 + 0.5 * dt * k1)
    k3 = f(y0 + 0.5 * dt * k2)
    k4 = f(y0 + dt * k3)

    y_next = y0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    body.set_state_vector(y_next)
    body.reset_forces()
