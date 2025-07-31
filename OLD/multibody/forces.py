import numpy as np

def gravity_force(mass, gravity=np.array([0, 0, -9.81])):
    return mass * gravity

def drag_force(velocity, air_density=1.225, drag_coefficient=0.5, area=1.0):
    speed = np.linalg.norm(velocity)
    if speed == 0:
        return np.zeros(3)
    drag = -0.5 * air_density * speed * velocity * drag_coefficient * area
    return drag
