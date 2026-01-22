from __future__ import annotations

from collections.abc import Iterable
import pandas as pd
from typing import Protocol, Self, TypedDict, TypeAlias


class Analyser(Protocol):
    def inputs(self) -> Input:
        ...
    def get_simulation_instance(self) -> Simulation:
        ...
    def output(self) -> dict[str,float|dict]:
        ...

class Simulation(Protocol):
    components: dict[str,Model|TimeSeriesGenerator|None]
    layout: Layout | None
    output: Output | None
    def run(self) -> None:
        ...

class Model(Protocol):
    @classmethod
    def set_model(cls, by:tuple[str|None,str|None]) -> Self:
        ...
    def simulate(self, ts:pd.DataFrame) -> pd.DataFrame:
        ...

class TimeSeriesGenerator(Protocol):
    @classmethod
    def settings(cls, dict) -> Self:
        ...
    def get_data(self, cols:list[str]) -> pd.DataFrame:
        ...



Input: TypeAlias = list[Model|TimeSeriesGenerator]
Layout: TypeAlias = dict[tuple[str,str], tuple[str,str]]
Output: TypeAlias = dict[str,float|Iterable]
