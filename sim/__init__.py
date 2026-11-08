"""Safar simulation core.

The package is import-safe and dependency-free (standard library only), so it
can be tested, scripted or embedded without pulling in FastAPI or a database.
"""

from sim.agent import TrainingConfig, TrainingResult, train
from sim.controller import Action, PurePursuitController, TabularPolicy
from sim.engine import SimulationEngine, SimulationResult, run_scenario
from sim.metrics import MetricsCollector
from sim.scenario import NPCConfig, ObstacleConfig, Scenario, builtin_scenarios
from sim.track import Track
from sim.vehicle import Vehicle, VehicleParams, VehicleState

__all__ = [
    "Action",
    "MetricsCollector",
    "NPCConfig",
    "ObstacleConfig",
    "PurePursuitController",
    "Scenario",
    "SimulationEngine",
    "SimulationResult",
    "TabularPolicy",
    "Track",
    "TrainingConfig",
    "TrainingResult",
    "Vehicle",
    "VehicleParams",
    "VehicleState",
    "builtin_scenarios",
    "run_scenario",
    "train",
]

__version__ = "1.0.0"
