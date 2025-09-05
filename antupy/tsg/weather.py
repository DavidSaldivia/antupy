"""module for weather forecast. It should include functions for the following tasks:
- TMY loading and generation.
- MERRA2 wrapper and other open-source weather data.
- weather data generator
"""
import pandas as pd

from antupy.constants import DIRECTORY
from antupy.units import Var

class Weather():
    def __init__(self):
        pass
    def create_ts(self) -> pd.DataFrame:
        ts = pd.DataFrame()
        return ts