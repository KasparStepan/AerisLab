"""
Hybrid simulation protocols for state management, auxiliary dynamics, and inertial properties.
"""


from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class StateProvider(Protocol):
    def num_states(self) -> int:
        ...

    def pack_state(self,out: NDArray[np.float64]) -> None:
        ...

    def unpack_state(self,y: NDArray[np.float64]) -> None:
        ...

@runtime_checkable
class AuxDynamics(Protocol):
    def compute_derivatives(self, t: float) -> NDArray[np.float64]:
        ...

@runtime_checkable
class InertialProvider(Protocol):
    def mass_matrix_world(self) -> NDArray[np.float64]:
        ...

    def inv_mass_matrix_world(self) -> NDArray[np.float64]:
        ...



