"""
Multi-component system assembly and management.

A System groups related Components together (e.g., parachute + payload)
and manages inter-component constraints like tethers and joints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aerislab.dynamics.body import RigidBody6DOF
from aerislab.dynamics.constraints import Constraint

from .base import Component

if TYPE_CHECKING:
    pass


class System:
    """
    Multi-component aerospace system.

    Manages components, inter-component constraints (tethers, joints),
    and provides unified interface for simulation.

    Parameters
    ----------
    name : str
        System identifier

    Attributes
    ----------
    name : str
        System identifier
    components : list[Component]
        All components in the system
    constraints : list[Constraint]
        Inter-component constraints (tethers, joints, etc.)

    Examples
    --------
    >>> system = System(name="parachute_system")
    >>> payload_idx = system.add_component(payload)
    >>> canopy_idx = system.add_component(parachute)
    >>> system.add_constraint(tether)  # Connect components

    Notes
    -----
    **System vs World**

    - System: Domain-level grouping of related components
    - World: Simulation orchestrator that runs physics

    A World can contain multiple Systems (e.g., multiple parachute-payload
    pairs for cluster analysis).

    **Workflow**

    1. Create System
    2. Add Components (each wraps a RigidBody6DOF)
    3. Add Constraints between components
    4. Add System to World via world.add_system(system)
    5. Run simulation - World handles physics, System manages components
    """

    def __init__(self, name: str):
        self.name = name
        self.components: list[Component] = []
        self.constraints: list[Constraint] = []

    def add_component(self, component: Component) -> int:
        """
        Add component to system.

        Parameters
        ----------
        component : Component
            Component to add

        Returns
        -------
        int
            Index of added component (for constraint/joint definitions)
        """
        self.components.append(component)
        return len(self.components) - 1

    def add_constraint(self, constraint: Constraint) -> None:
        """
        Add inter-component constraint.

        Parameters
        ----------
        constraint : Constraint
            Constraint connecting two components (e.g., DistanceConstraint
            for tether, PointWeldConstraint for rigid connection)
        """
        self.constraints.append(constraint)

    def get_bodies(self) -> list[RigidBody6DOF]:
        """
        Get all rigid bodies for solver.

        Returns
        -------
        list[RigidBody6DOF]
            List of rigid bodies from all components
        """
        return [comp.body for comp in self.components]

    def update_all_states(self, t: float, dt: float) -> None:
        """
        Update state for all components.

        Called by World before force application.

        Parameters
        ----------
        t : float
            Current simulation time [s]
        dt : float
            Time step [s]
        """
        for comp in self.components:
            comp.update_state(t, dt)

    def apply_all_forces(self, t: float) -> None:
        """
        Apply forces for all components.

        Parameters
        ----------
        t : float
            Current simulation time [s]
        """
        for comp in self.components:
            comp.apply_forces(t)

    def get_component(self, name: str) -> Component | None:
        """
        Get component by name.

        Parameters
        ----------
        name : str
            Component name to find

        Returns
        -------
        Component | None
            Component if found, None otherwise
        """
        for comp in self.components:
            if comp.name == name:
                return comp
        return None

    def get_component_by_index(self, index: int) -> Component:
        """
        Get component by index.

        Parameters
        ----------
        index : int
            Component index

        Returns
        -------
        Component
            Component at given index

        Raises
        ------
        IndexError
            If index is out of range
        """
        return self.components[index]

    def __len__(self) -> int:
        """Number of components in system."""
        return len(self.components)

    def __repr__(self) -> str:
        return (
            f"System(name='{self.name}', "
            f"components={len(self.components)}, "
            f"constraints={len(self.constraints)})"
        )

    def summary(self) -> str:
        """
        Get human-readable system summary.

        Returns
        -------
        str
            Multi-line summary of system contents
        """
        lines = [f"System: {self.name}"]
        lines.append(f"  Components ({len(self.components)}):")
        for i, comp in enumerate(self.components):
            lines.append(f"    [{i}] {comp}")
        lines.append(f"  Constraints ({len(self.constraints)}):")
        for i, con in enumerate(self.constraints):
            lines.append(f"    [{i}] {con.__class__.__name__}")
        return "\n".join(lines)
