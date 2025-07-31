import numpy as np

def sphere_sphere_collision(body_i, body_j, restitution=0.5, friction=0.1):
    Δ = body_j.position - body_i.position
    dist = np.linalg.norm(Δ)
    rsum = body_i.radius + body_j.radius
    if 1e-6 < dist < rsum:
        n = Δ/dist
        v_rel = body_j.velocity - body_i.velocity
        v_norm = np.dot(v_rel, n)
        if v_norm < 0:
            J = -(1+restitution)*v_norm / (1/body_i.mass + 1/body_j.mass)
            impulse = J * n
            body_i.velocity += impulse/body_i.mass
            body_j.velocity -= impulse/body_j.mass
            v_tan = v_rel - v_norm*n
            if np.linalg.norm(v_tan)>1e-6:
                t = v_tan/np.linalg.norm(v_tan)
                Jf = friction * abs(J)
                body_i.velocity += Jf*t / body_i.mass
                body_j.velocity -= Jf*t / body_j.mass
            print(f"[COLLISION] {body_i.name}-{body_j.name}: impulse={impulse}")
