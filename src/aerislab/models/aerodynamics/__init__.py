"""
Aerodynamics models module.

Contains physical models for:
- Parachute inflation and drag (ParachuteModels)
"""

from .parachute_models import (
    AdvancedParachute,
    ParachuteGeometry,
    ParachuteModelType,
    InflationConfig,
    PorosityConfig,
    MassFlowConfig,
    AddedMassConfig,
    create_parachute,
)
