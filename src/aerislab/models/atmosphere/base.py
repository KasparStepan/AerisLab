from typing import Protocol

class AtmosphereModel(Protocol):
    def density(self, altitude: float) -> float:
        """Returns the atmospheric density at a given altitude."""
        ...

    def pressure(self, altitude: float) -> float:
        """Returns the atmospheric pressure at a given altitude."""
        ...

    def temperature(self, altitude: float) -> float:
        """Returns the atmospheric temperature at a given altitude."""
        ...