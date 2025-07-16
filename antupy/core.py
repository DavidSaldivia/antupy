"""
module with the core classes for AntuPy
"""

from collections.abc import Iterable
from typing import TypedDict

class Simulation():
    pass

class Output(TypedDict):
    pass

class Analyser():
    def get_simulation_instance(self, cases: Iterable) -> Simulation:
        return Simulation()
    def run_simulation(self) -> Output:
        return Output()

