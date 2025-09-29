import numpy as np

a = np.array([1, 2, 3])
magnitude = np.linalg.norm(a)
magnitude_old_way = np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
print(f"Magnitude of {a} is {magnitude}")
print(f"Magnitude of {a} (old way) is {magnitude_old_way}")

