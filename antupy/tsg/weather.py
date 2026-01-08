"""module for weather forecast. It should include functions for the following tasks:
- TMY loading and generation.
- MERRA2 wrapper and other open-source weather data.
- weather data generator
"""

from dataclasses import dataclass, field

import pandas as pd

from antupy.ddd_au import DIRECTORY
from antupy import Var
from antupy.tsg.settings import TimeParams
from antupy.utils.loc import Location

import os
import pandas as pd
import numpy as np
import xarray as xr

from typing import Optional, Union, Any, Literal, Protocol, runtime_checkable

from antupy.ddd_au import (
    DIRECTORY,
    DEFINITIONS,
    SIMULATIONS_IO
)
from antupy.utils.loc.loc_au import (
    LocationAU,
    _from_postcode
)
from antupy.utils.loc.loc_cl import LocationCL

DIR_DATA = DIRECTORY.DIR_DATA
DEFINITION_SEASON = DEFINITIONS.SEASON
LOCATIONS_METEONORM = DEFINITIONS.LOCATIONS_METEONORM
LOCATIONS_STATE = DEFINITIONS.LOCATIONS_STATE
LOCATIONS_COORDINATES = DEFINITIONS.LOCATIONS_COORDINATES
TS_WEATHER = SIMULATIONS_IO.TS_TYPES["weather"]

#--------------
DIR_METEONORM = os.path.join(DIR_DATA["weather"], "meteonorm_processed")
DIR_MERRA2 = os.path.join(DIR_DATA["weather"], "merra2_processed")
DIR_NCI = os.path.join(DIR_DATA["weather"], "nci_processed")

FILES_WEATHER = {
    "METEONORM_TEMPLATE" : os.path.join(DIR_METEONORM, "meteonorm_{:}.csv"),  #expected LOCATIONS_METEONORM
    "MERRA2" : os.path.join(DIR_MERRA2, "merra2_processed_all.nc"),
    "NCI": "",
}

_VARIABLE_RANGES = {
    "GHI" : (1000.,1000.),
    "temp_amb" : (10.0,40.0),
    "temp_mains" : (10.0,30.0),
}

# Type alias for simulation types
WeatherSimulationType = Literal["tmy", "mc", "historical", "constant_day"]
WeatherDatasetType = Literal["meteonorm", "merra2", "local", ""]
WeatherSubsetType = Literal["all", "annual", "season", "month", "date", None]



@runtime_checkable
class Weather(Protocol):
    """
    Weather generator protocol. Defines the interface for weather data generation for thermal and PV simulations.
    
    Required attributes:
        dataset: Source of weather data (e.g., "meteonorm", "merra2")
        location: Location where the simulation is performed (str or Location object)
        time_params: Time parameters defining the simulation period and timesteps
    """
    
    dataset: str
    location: str | Location
    time_params: TimeParams
    
    def load_data(self) -> pd.DataFrame:
        """Load weather data based on the instance's time_params.
        
        Returns:
            A dataframe with the weather timeseries using time_params.idx_pd as index.
        """
        ...


@dataclass
class TMY(Weather):
    """
    TMY (Typical Meteorological Year) weather generator.
    One year of data, usually with TMY files.
    
    Parameters:
        dataset: Source of weather data. Options: "meteonorm", "merra2".
        location: City where the simulation is performed.
        time_params: Time parameters defining the simulation period.
    """
    
    dataset: str = "meteonorm"
    location: str | Location = field(default_factory=lambda: LocationAU("Sydney"))
    time_params: TimeParams = field(default_factory=TimeParams)
    
    def load_data(self) -> pd.DataFrame:
        """Load TMY data based on the instance's time_params."""
        ts_index = self.time_params.idx_pd
        ts_df = pd.DataFrame(index=ts_index, columns=TS_WEATHER)
        return _load_tmy(ts_df, dataset=self.dataset, location=self.location, columns=TS_WEATHER)


@dataclass
class WeatherMC(Weather):
    """
    Monte Carlo weather generator.
    Random sample of temporal unit (e.g. days) from set (month, week, day).
    
    Parameters:
        dataset: Source of weather data. Options: "meteonorm", "merra2", "nci".
        location: City where the simulation is performed.
        time_params: Time parameters defining the simulation period.
        subset: The subset to generate data. Options: "annual", "season", "month", "date".
        random: Whether generates data randomly or periodically.
        value: The value used on subset (season name, month number, or date).
    """
    
    dataset: str = "meteonorm"
    location: str | Location = field(default_factory=lambda: LocationAU("Sydney"))
    time_params: TimeParams = field(default_factory=TimeParams)
    subset: str | None = None
    random: bool = False
    value: str | int | None = None
    
    def load_data(self) -> pd.DataFrame:
        """Load Monte Carlo weather data based on the instance's time_params."""
        ts_index = self.time_params.idx_pd
        ts_df = pd.DataFrame(index=ts_index, columns=TS_WEATHER)
        return _load_montecarlo(ts_df, dataset=self.dataset, location=self.location, 
                                subset=self.subset, value=self.value, columns=TS_WEATHER)


@dataclass
class WeatherHist(Weather):
    """
    Historical weather generator.
    Specific dates for a specific location from historical datasets.
    
    Parameters:
        dataset: Source of weather data. Options: "merra2", "nci", "local".
        location: City where the simulation is performed.
        time_params: Time parameters defining the simulation period.
        file_path: Path to the weather file location.
        list_dates: Set of dates to load.
    """
    
    dataset: str = "merra2"
    location: str | Location = field(default_factory=lambda: LocationAU("Sydney"))
    time_params: TimeParams = field(default_factory=TimeParams)
    file_path: str | None = None
    list_dates: pd.DatetimeIndex | pd.Timestamp | None = None
    
    def load_data(self) -> pd.DataFrame:
        """Load historical weather data based on the instance's time_params."""
        ts_index = self.time_params.idx_pd
        ts_df = pd.DataFrame(index=ts_index, columns=TS_WEATHER)
        return _load_historical(ts_df, file_path=self.file_path, columns=TS_WEATHER)


@dataclass
class WeatherConstantDay(Weather):
    """
    Constant day weather generator.
    Environmental variables kept constant throughout the simulation.
    
    Parameters:
        dataset: Source of weather data (usually empty for constant values).
        location: City where the simulation is performed.
        time_params: Time parameters defining the simulation period.
        random: Whether to generate random values within ranges.
        value: Specific constant values to use.
        subset: Additional subset parameter.
    """
    
    dataset: str = ""
    location: str | Location = field(default_factory=lambda: LocationAU("Sydney"))
    time_params: TimeParams = field(default_factory=TimeParams)
    random: bool = False
    value: str | int | None = None
    subset: str | None = None
    
    def load_data(self) -> pd.DataFrame:
        """Load constant day weather data based on the instance's time_params."""
        ts_index = self.time_params.idx_pd
        ts_df = pd.DataFrame(index=ts_index, columns=TS_WEATHER)
        return _load_day_constant_random(ts_df)
    


#----------
def _load_day_constant_random(
    timeseries: pd.DataFrame,
    ranges: dict[str,tuple] = _VARIABLE_RANGES,
    seed_id: Optional[int] = None,
    columns: list[str] = TS_WEATHER,
) -> pd.DataFrame:
    
    if seed_id is None:
        seed = np.random.SeedSequence().entropy
    else:
        seed = seed_id
    rng = np.random.default_rng(seed)
    
    idx = pd.to_datetime(timeseries.index)
    dates = np.unique(idx.date)
    DAYS = len(dates)

    df_weather_days = pd.DataFrame( index=dates, columns=columns)
    df_weather_days.index = pd.to_datetime(df_weather_days.index)
    for lbl in ranges.keys():
        df_weather_days[lbl] = rng.uniform(
            ranges[lbl][0],
            ranges[lbl][1],
            size=DAYS,
        )
    df_weather = df_weather_days.loc[idx.date]
    df_weather.index = idx
    timeseries[columns] = df_weather[columns]
    return timeseries


#---------------------------------
def _random_days_from_dataframe(
    timeseries: pd.DataFrame,
    df_sample: pd.DataFrame,
    seed_id: Optional[int] = None,
    columns: Optional[list[str]] = TS_WEATHER,
) -> pd.DataFrame :
    if seed_id is None:
        seed = np.random.SeedSequence().entropy
    else:
        seed = seed_id
    rng = np.random.default_rng(seed)

    df_sample_new = df_sample.copy()
    df_sample_idx = pd.to_datetime(df_sample_new.index)
    ts_index = pd.to_datetime(timeseries.index)

    list_dates = np.unique(df_sample_idx.date)
    DAYS = len(np.unique(ts_index.date))
    list_picked_dates = rng.choice( list_dates, size=DAYS )
    df_sample_new["date"] = df_sample_idx.date
    set_picked_days = [
        df_sample_new[df_sample_new["date"]==date] for date in list_picked_dates
    ]
    df_final = pd.concat(set_picked_days)
    df_final.index = ts_index
    timeseries[columns] = df_final[columns]
    
    return timeseries

#---------------------------------
def from_tmy(
        timeseries: pd.DataFrame,
        TMY: pd.DataFrame,
        columns: Optional[list[str]] = TS_WEATHER,
    ) -> pd.DataFrame :
    
    rows_timeseries = len(timeseries)
    rows_tmy = len(TMY)
    
    if rows_tmy <= rows_timeseries:
        N = int( np.ceil( rows_timeseries/rows_tmy ) )
        TMY_extended = pd.concat([TMY]*N, ignore_index=True)
        TMY_final = TMY_extended.iloc[:rows_timeseries]
    else:
        TMY_final = TMY.iloc[:rows_timeseries]

    TMY_final.index = timeseries.index
    timeseries[columns] = TMY_final[columns]
    return timeseries

# -------------
def _load_tmy(
    ts: pd.DataFrame,
    params: dict | None = None,
    *,
    dataset: str | None = None,
    location: str | Location | None = None,
    columns: list[str] | None = TS_WEATHER,
) -> pd.DataFrame:
    
    # Handle both dict params and keyword arguments
    if params is not None:
        # Legacy dict-based interface
        dataset = params["dataset"]
        location = params["location"]
    elif dataset is None or location is None:
        raise ValueError("Either params dict or dataset+location keywords must be provided")
    
    YEAR = pd.to_datetime(ts.index).year[0]
    
    # At this point, dataset and location are guaranteed to be not None
    assert dataset is not None and location is not None
    
    # Convert Location objects to string for processing
    location_str = str(location) if not isinstance(location, str) else location
    
    if dataset == "meteonorm":
        df_dataset = _load_dataset_meteonorm(location_str, YEAR)
    elif dataset == "merra2":
        # For MERRA2, convert LocationCL to LocationAU if needed, or pass as is if compatible
        if isinstance(location, LocationCL):
            # Convert to string representation for MERRA2
            location_for_merra2 = str(location)
        else:
            location_for_merra2 = location
        df_dataset = _load_dataset_merra2(ts, location_for_merra2, YEAR)  # type: ignore
    else:
        raise ValueError(f"dataset: {dataset} is not available.")
    return from_tmy( ts, df_dataset, columns=columns )


def _load_dataset_meteonorm(
        location: str,
        YEAR: int = 2022,
        START: int = 0,
        STEP: int = 3,
) -> pd.DataFrame:

    if location not in DEFINITIONS.LOCATIONS_METEONORM:
        raise ValueError(f"location {location} not in available METEONORM files")
    
    df_dataset = pd.read_csv(
        os.path.join(
            DIR_METEONORM,
            FILES_WEATHER["METEONORM_TEMPLATE"].format(location),
        ),
        index_col=0
    )
    PERIODS = len(df_dataset)

    start_time = pd.to_datetime(f"{YEAR}-01-01 00:00:00") + pd.DateOffset(hours=START)
    df_dataset.index = pd.date_range( start=start_time, periods=PERIODS, freq=f"{STEP}min")
    df_dataset["date"] = df_dataset.index
    df_dataset["date"] = df_dataset["date"].apply(lambda x: x.replace(year=YEAR))
    df_dataset = df_dataset.set_index(pd.to_datetime(df_dataset["date"]))
    return df_dataset


def _load_dataset_merra2(
        ts: pd.DataFrame,
        location: LocationAU | str | tuple | int,
        YEAR: int,
        STEP:int = 5,
        file_dataset:str = FILES_WEATHER["MERRA2"],
        ) -> pd.DataFrame:

    if isinstance(location, int):   #postcode
        (lon,lat) = _from_postcode(location, get="coords")
    elif isinstance(location,str):   #city
        loc = LocationAU(location)
        (lon,lat) = (loc.lon, loc.lat)
    elif isinstance(location, tuple): #(longitude, latitude) tuple
        (lon,lat) = (location)
    elif isinstance(location, LocationAU):
        (lon,lat) = (location.lon, location.lat)
    else:
        raise ValueError(f"location {location} not in available format.")

    data_weather = xr.open_dataset(file_dataset)
    lons = np.array(data_weather.lon)
    lats = np.array(data_weather.lat)
    lon_a = lons[(abs(lons-lon)).argmin()]
    lat_a = lats[(abs(lats-lat)).argmin()]
    df_w = data_weather.sel(lon=lon_a,lat=lat_a).to_dataframe()

    df_w.index = pd.to_datetime(df_w.index).tz_localize('UTC')
    tz = 'Australia/Brisbane'
    df_w.index = df_w.index.tz_convert(tz)
    df_w.index = df_w.index.tz_localize(None)
    df_w.rename(columns={'SWGDN':'GHI','T2M':'Temp_Amb'},inplace=True)
    df_w = df_w[['GHI','Temp_Amb']].copy()
    df_w = df_w.resample(f"{STEP}T").interpolate()       #Getting the data in half hours
    
    ts["GHI"] = df_w["GHI"]
    ts["Temp_Amb"] = df_w["Temp_Amb"] - 273.15
    
    #########################################
    #Replace later for the closest city
    df_aux = _load_dataset_meteonorm("Sydney", YEAR)
    df_aux = df_aux.resample(f"{STEP}T").interpolate()       #Getting the data in half hours
    ts["Temp_Mains"] = df_aux["Temp_Mains"]
    #########################################

    return ts

#----------
def _load_montecarlo(
    ts: pd.DataFrame,
    params: dict | None = None,
    *,
    dataset: str | None = None,
    location: str | Location | None = None,
    subset: str | None = None,
    value: str | int | None = None,
    columns: Optional[list[str]] = TS_WEATHER,
) -> pd.DataFrame:
    
    # Handle both dict params and keyword arguments
    if params is not None:
        # Legacy dict-based interface
        dataset = params["dataset"]
        location = params["location"]
        subset = params["subset"]
        value = params["value"]
    elif any(x is None for x in [dataset, location, subset]):
        raise ValueError("Either params dict or dataset+location+subset keywords must be provided")
    
    # Convert Location objects to string for processing
    location_str = str(location) if not isinstance(location, str) else location
    
    ts_index = pd.to_datetime(ts.index)

    # At this point, required parameters are guaranteed to be not None
    assert dataset is not None and location is not None and subset is not None

    if dataset == "meteonorm":
        df_dataset = _load_dataset_meteonorm(location_str)
    elif dataset == "merra2":
        # For MERRA2, ensure location is in correct format
        if isinstance(location, LocationAU):
            location_for_merra2 = location
        elif isinstance(location, LocationCL) or hasattr(location, 'value'):
            # Convert non-AU Location objects to string representation for MERRA2
            location_for_merra2 = str(location)
        else:
            location_for_merra2 = location
        # Type assertion to help type checker since we've converted to compatible types
        df_dataset = _load_dataset_merra2(ts, location_for_merra2, ts_index.year[0])  # type: ignore
    else:
        raise ValueError(f"dataset: {dataset} is not available.")
    
    df_dataset.index = pd.to_datetime(df_dataset.index)
    if subset == 'annual':
        df_sample = df_dataset[
            df_dataset.index.year==value
            ]
    elif subset == 'season':
        # value should be a string for season
        if not isinstance(value, str):
            raise ValueError(f"For season subset, value must be a string, got {type(value)}")
        df_sample = df_dataset[
            df_dataset.index.isin(DEFINITION_SEASON[value])
            ]
    elif subset == 'month':
        # value should be an int for month
        if not isinstance(value, int):
            raise ValueError(f"For month subset, value must be an int, got {type(value)}")
        df_sample = df_dataset[
            df_dataset.index.month==value
            ]  
    elif subset == 'date':
        # value should have a date() method (datetime/Timestamp)
        if not hasattr(value, 'date'):
            raise ValueError(f"For date subset, value must be a datetime object, got {type(value)}")
        df_sample = df_dataset[
            df_dataset.index.date==value.date()  # type: ignore
            ]
    else:
        raise ValueError(f"subset: {subset} not in available options.")
    df_weather = _random_days_from_dataframe( ts, df_sample, columns=columns )
    return df_weather

#----------------
def _load_historical(
    ts: pd.DataFrame,
    params: dict | None = None,
    *,
    file_path: str | None = None,
    columns: Optional[list[str]] = TS_WEATHER,
) -> pd.DataFrame:
    
    # Handle both dict params and keyword arguments
    if params is not None:
        # Legacy dict-based interface
        file_path = params["file_path"]
    elif file_path is None:
        raise ValueError("Either params dict or file_path keyword must be provided")
    
    # At this point, file_path is guaranteed to be not None
    assert file_path is not None
    
    ts_ = pd.read_csv(file_path, index_col=0)
    ts_.index = pd.to_datetime(ts.index)
    return ts_

def main():
    from antupy.tsg.settings import TimeParams

    tp = TimeParams(YEAR=Var(2020,"-"), STEP=Var(30,"min"))

    #----------------
    # TMY with Meteonorm
    tmy_weather = TMY(dataset="meteonorm", location="Sydney", time_params=tp)
    ts_tmy = tmy_weather.load_data()
    print("TMY Meteonorm:", ts_tmy[TS_WEATHER])

    #----------------
    # TMY with MERRA2
    location = LocationAU(2035)
    tp2 = TimeParams(YEAR=Var(2020,"-"), STEP=Var(30,"min"))
    tmy_weather_merra = TMY(dataset="merra2", location=str(location), time_params=tp2)
    ts_tmy_merra = tmy_weather_merra.load_data()
    print("TMY MERRA2:", ts_tmy_merra[TS_WEATHER])

    #----------------
    # Monte Carlo
    mc_weather = WeatherMC(
        dataset="meteonorm",
        location=str(LocationAU(2035)),
        time_params=tp,
        subset="month",
        value=5
    )
    ts_mc = mc_weather.load_data()
    print("Monte Carlo:", ts_mc[TS_WEATHER])

    #----------------
    # Constant day
    constant_weather = WeatherConstantDay(time_params=tp)
    ts_constant = constant_weather.load_data()
    print("Constant Day:", ts_constant[TS_WEATHER])

    return


if __name__ == "__main__":
    main()
    pass
