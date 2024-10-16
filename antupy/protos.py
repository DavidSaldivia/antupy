from __future__ import annotations

from collections.abc import Iterable
import pandas as pd
from typing import Protocol, Self, TypedDict, TypeAlias


# base classes
class Analyser(Protocol):
    def input(self) -> Input:
        ...
    def get_simulation_instance() -> Simulator:
        ...
    def output(self) -> dict[str,float|dict]:
        ...


class Simulator(Protocol):
    def layout() -> Layout:
        ...
    def run_simulation() -> Output:
        ...


class Model(Protocol):
    solver: Solver
    @classmethod
    def set_model(cls, by:tuple[str|None,str|None] = (None,None)) -> Self:
        ...
    def simulate(self, ts:pd.DataFrame) -> pd.DataFrame:
        ...


class TimeSeriesGenerator(Protocol):
    @classmethod
    def settings(cls, dict) -> Self:
        ...
    def get_data(self, cols:list[str]) -> pd.DataFrame:
        ...


class Solver(Protocol):
    def run_simulation(self, model: Model, ts: pd.DataFrame) -> pd.DataFrame:
        ...


Input: TypeAlias = list[Model|TimeSeriesGenerator]
Layout: TypeAlias = dict[tuple[str,str], tuple[str,str]]
Output: TypeAlias = dict[str,float|Iterable]
