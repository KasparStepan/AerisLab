"""
AerisLab Physics Models.

This module contains physics models that can be used by components:
- Aerodynamics (drag coefficients, added mass, FSI interface)
- Deployment (reefing, inflation dynamics)
- Atmosphere (ISA, custom profiles)

These models are separate from force implementations in dynamics/forces.py.
Forces apply models to bodies; models encapsulate the physics/mathematics.

Future Modules
--------------
- aerodynamics.py: Drag coefficient tables, Mach effects, added mass
- deployment.py: Inflation dynamics, reefing schedules, opening shock
- atmosphere.py: ISA, custom atmospheric profiles, wind models
- materials.py: Fabric properties, line stretch, porosity effects
"""

# Placeholder for future development
# from .aerodynamics import DragModel, AddedMassModel
# from .deployment import InflationModel, ReefingSchedule
# from .atmosphere import StandardAtmosphere, WindField

__all__: list[str] = []
