#Imports from AerisLab specific modules
from aerislab.dynamics.body import RigidBody6DOF

#Imports from standart libraries
from enum import Enum


class ParachuteType(Enum):
    """
    This calss stores enumarations for different parachute types.
    """

    SIMPLE_DRAG = "simple_drag"
    KNACKE = "knacke"
    LUDTKE_APPARENT_MASS = "ludtke_apparent_mass"
    WOLF_DYNAMIC = "wolf_dynamic"
    FILLING_TIME_POLYNOMIAL = "filling_time_polynomial"
    ELLIPSOID_APARENT_MASS = "ellipsoid_apparent_mass"


class ParachuteV2(RigidBody6DOF):