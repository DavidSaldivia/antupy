from dataclasses import dataclass

import pandas as pd
import polars as pl
import numpy as np

from antupy import Var

#------------------------------------
@dataclass
class TimeParams():
    """Time parameters for simulations. Useful for annual simulations, stochastic simulations, representative days simulations.

    Parameters:
        START (Variable): initial time of the simulation (in 'hr') (first hour of the year for annual simulation)
        STOP (Variable): final time of the year (in 'hr').
        STEP (Variable): timestep for the simulation (in 'min')
        YEAR (Variable): Year of the simulation (useful only for annual simulations)

    """

    START: Var = Var(0, "hr")
    STOP: Var = Var(8760, "hr")
    STEP: Var = Var(60, "min")
    YEAR: Var = Var(1800, "-")

    @property
    def DAYS(self) -> Var:
        """Based on START and STOP, returns the days of simulation.

        Returns:
            Variable: Days of simulation
        """
        START = self.START.gv("hr")
        STOP = self.STOP.gv("hr")
        return Var( int((STOP-START)/24), "d")
    
    @property
    def PERIODS(self) -> Var:
        """
        Based on START, STOP and STEP, it returns the number of periods.

        Returns:
            Variable: Periods of simulation.
        """
        START = self.START.gv("hr")
        STOP = self.STOP.gv("hr")
        STEP_h = self.STEP.gv("hr")
        return Var( int(np.ceil((STOP - START)/STEP_h)), "-")

    @property
    def idx(self) -> pl.Series:
        """It is the datetime series for the simulation (Polars-based)

        Returns:
            pl.Series: datetime series used for all the timeseries and simulation results
        """
        START = int(self.START.gv("hr"))
        STOP = int(self.STOP.gv("hr"))
        STEP = int(self.STEP.gv("min"))
        YEAR = int(self.YEAR.gv("-"))
        PERIODS = int(self.PERIODS.gv("-"))
        # Create datetime range
        return pl.datetime_range(
            start=pl.datetime(YEAR, 1, 1, 0, 0, 0) + pl.duration(hours=START),
            end=pl.datetime(YEAR, 1, 1, 0, 0, 0) + pl.duration(hours=STOP),
            interval=f"{STEP}m",  # minutes
            time_unit="ms",
            eager=True,
        ).slice(0, PERIODS)

    @property
    def idx_pd(self) -> pd.DatetimeIndex:
        """Pandas DatetimeIndex conversion of the simulation datetime series

        Returns:
            pd.DatetimeIndex: pandas index used for compatibility with pandas-based code
        """
        # Recreate the pandas DatetimeIndex using the same parameters as the original implementation
        START = int(self.START.gv("hr"))
        STEP = int(self.STEP.gv("min"))
        YEAR = int(self.YEAR.gv("-"))
        PERIODS = int(self.PERIODS.gv("-"))
        start_time = pd.to_datetime(f"{YEAR}-01-01 00:00:00") + pd.DateOffset(hours=START)
        return pd.date_range(start=start_time, periods=PERIODS, freq=f"{STEP}min")
    

if __name__ == "__main__":
    tp = TimeParams()
    print(tp.idx)
    print(tp.idx_pd)

    tp = TimeParams(
        START = Var(0, "hr"),
        STOP = Var(48, "hr"),
        STEP = Var(30, "min"),
    )
    print(tp.idx)