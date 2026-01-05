"""
Just my junkyard to test some synthax. This is just a space for me to fix my stupidity :D

Please dont judge me for this.
"""

import numpy as np

a = np.array([1, 2, 3])
magnitude = np.linalg.norm(a)
magnitude_old_way = np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
print(f"Magnitude of {a} is {magnitude}")
print(f"Magnitude of {a} (old way) is {magnitude_old_way}")

position_1 = np.array([1.0, 2.0, 3.0])
position_2 = np.array([4.0, 5.0, 6.0])
direction = position_2 - position_1
distance = np.linalg.norm(direction)
print(f"Direction from {position_1} to {position_2} is {direction}")
print(f"Distance between {position_1} and {position_2} is {distance}")


print(np.zeros(3))