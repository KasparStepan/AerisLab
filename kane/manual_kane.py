import numpy as np

def compute_partial_velocities(bodies, generalized_speeds):
    # Placeholder: returns manually defined partial velocity vectors
    # User must implement analytic derivatives of each body velocity w.r.t each speed
    return partial_velocities

def compute_generalized_forces(bodies, partial_velocities):
    Fg = np.zeros(len(generalized_speeds))
    for (b_idx,u_idx,pv) in partial_velocities:
        F = bodies[b_idx].force
        Fg[u_idx] += np.dot(F, pv)
    return Fg

def compute_generalized_inertia(bodies, partial_velocities):
    Fi = np.zeros(len(generalized_speeds))
    for (b_idx,u_idx,pv) in partial_velocities:
        a = bodies[b_idx].acceleration  # must be computed
        Fi[u_idx] += - bodies[b_idx].mass * np.dot(a, pv)
    return Fi
