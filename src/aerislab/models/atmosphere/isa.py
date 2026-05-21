import numpy as np
from aerislab.models.atmosphere.base import AtmosphereModel
import math


class FastISA(AtmosphereModel):
    def __init__(self, max_altitude: float = 10000.0, resolution: float = 1.0):
        self.resolution = resolution
        self.max_idx = int(max_altitude / resolution) + 1

        # Air properties at sea level
        self.T0 = 288.15  # K
        self.P0 = 101325.0  # Pa
        self.rho0 = 1.225  # kg/m^3
        self.L = 0.0065  # K/m (Teplotní gradient)
        self.R = 287.05  # J/(kg*K) (Plynová konstanta)
        self.g = 9.80665  # m/s^2 (Gravitační zrychlení)

        # Alocation of memory
        self._rho = np.zeros(self.max_idx, dtype=np.float64)
        self._pressure = np.zeros(self.max_idx, dtype=np.float64)
        self._temperature = np.zeros(self.max_idx, dtype=np.float64)

        # Precalculate table of properties
        self._precalculate_LUT()

    def _precalculate_LUT(self):
        """
        Go through altitudes and calculate properties using ISA formulas.
        """

        for i in range(self.max_idx):
            # Calculate altitude
            alt = i * self.resolution
            
            # Calculate properties at each step of altitude
            self._rho[i], self._pressure[i], self._temperature[i] = self._calculate_analytical_properties(alt)

    def _calculate_analytical_properties(self, altitude: float) -> tuple[np.float64, np.float64, np.float64]: 
        """
        Calculate properties at a given altitude using ISA formulas.
        """
        if altitude <= 11000.0:
            # Troposféra
            T = self.T0 - self.L * altitude
            p = self.P0 * math.pow(T / self.T0, self.g / (self.L * self.R))

        elif altitude <= 20000.0:
            # Spodní stratosféra (Konstantní teplota)
            T_11 = 216.65
            p_11 = 22632.1
            T = T_11
            p = p_11 * math.exp(-self.g * (altitude - 11000.0) / (self.R * T_11))
        
        else:
            # Svrchní stratosféra (Mírný nárůst teploty)
            T_20 = 216.65
            p_20 = 5474.89
            L_20 = 0.001
            T = T_20 + L_20 * (altitude - 20000.0)
            p = p_20 * math.pow(T / T_20, -self.g / (L_20 * self.R))
            
        rho = p / (self.R * T)
        return rho, p, T

    def pressure(self, altitude):
        if altitude <= 0.0: return self._pressure[0]

        idx = int(altitude/self.resolution)
        if idx < self.max_idx:
            return self._pressure[idx]
        else:
            return self._calculate_analytical_properties(altitude)[1]
    
    def temperature(self, altitude):
        if altitude <= 0.0: return self._temperature[0]

        idx = int(altitude/self.resolution)
        if idx < self.max_idx:
            return self._temperature[idx]
        else:
            return self._calculate_analytical_properties(altitude)[2]
    
    def density(self, altitude):
        if altitude <= 0.0: return self._rho[0]

        idx = int(altitude/self.resolution)
        if idx < self.max_idx:
            return self._rho[idx]
        else:
            return self._calculate_analytical_properties(altitude)[0]
    
