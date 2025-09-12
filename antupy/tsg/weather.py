"""module for weather forecast. It should include functions for the following tasks:
- TMY loading and generation.
- MERRA2 wrapper and other open-source weather data.
- weather data generator
"""

from dataclasses import dataclass

import pandas as pd

from antupy.constants import DIRECTORY
from antupy.units import Var


import os
import pandas as pd
import numpy as np
import xarray as xr

from typing import Optional, Union, Any

from antupy.constants import (
    DIRECTORY,
    DEFINITIONS,
    DEFAULT,
    SIMULATIONS_IO
)
from antupy.location.au import (
    LocationAU,
    _from_postcode
)

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
_VARIABLES_NAMES = {
    "GHI": "Irradiance",
    "temp_amb": "Ambient Temperature",
    "temp_mains": "Mains Temperature",
}
_VARIABLE_DEFAULTS = {
    "GHI" : 1000.,
    "temp_amb" : 25.,
    "temp_mains" : 20.,
}
_VARIABLE_RANGES = {
    "GHI" : (1000.,1000.),
    "temp_amb" : (10.0,40.0),
    "temp_mains" : (10.0,30.0),
}
_TYPES_SIMULATION = [
    "tmy",                      # One year of data. Usually with tmy files.
    "mc",                       # Random sample of temporal unit (e.g. days) from set (month, week, day).
    "historical",               # Specific dates for a specific location (SolA, EE, EWS, etc).
    "constant_day",             # Constant values each day
    ]
_SIMS_PARAMS: dict[str, dict[str,Any]] = {
    "tmy": {
        "dataset": ['METEONORM',],
        "location": LOCATIONS_METEONORM,
    },
    "mc": {
        "dataset": ['METEONORM', 'MERRA2', 'NCI'],
        "location": DEFAULT.LOCATION,
        "subset": ['all', 'annual', 'season', 'month', 'date'],
        "random": [True, False],
        "value": None,
    },
    "historical": {
        "dataset": ['MERRA2', 'NCI', "local"],
        "location": str,
        "file_path": str,
        "list_dates": [pd.DatetimeIndex, pd.Timestamp],
    },
    "constant_day": {
        "dataset": [],
        "random": [True, False],
        "values": _VARIABLE_DEFAULTS,
        "ranges": _VARIABLE_RANGES,
    }
}
list_aux = list()
for d in _SIMS_PARAMS.keys():
    list_aux.append(_SIMS_PARAMS[d]["dataset"])
DATASET_ALL = list(dict.fromkeys( [ x for xs in list_aux for x in xs ] )) #flatten and delete dupl.



@dataclass
class Weather():
    """
    Weather generator. It generates weather data for thermal and PV simulations using one of four options depending on the type of simulation. Depending on this options it requires one or more Parameters.
    Check the module timeseries.weather for details.

    Parameters: 
        type_sim: Type of simulation. Options are: "tmy" (annual simulation), "mc" (montecarlo), "historical" (historical data files), "constant_day" (environmental variables kept constant)
        dataset: Source of weather data. Options: "meteonorm", "merra2".
        location: City where the simulation is performed.
        subset: For mc simulations, the subset to generate data. Options: "annual", "season", "month", "date". Depending on the choice, the "value" parameter is used.
        random: Whether generates data randomly or periodically. If True, picks randomly days from subset. If False, it repeats subset until the required number of days are met.
        value: The value used on subset. If "season", options: "summer", "autumn", "winter", spring". If "month", the month as integer. If "date", the specific date.
        file_path: A string with the weather file location (If "historical" is chosen)
        list_dates: Used only for "historical" simulations. A set of dates to load.

    """

    type_sim: str = "tmy"
    dataset: str = "meteonorm"
    location: str = "Sydney"
    subset: str | None = None
    random: bool = False
    value: str | int | None = None

    file_path: str | None = None
    list_dates: pd.DatetimeIndex | pd.Timestamp | None = None


    def params(self) -> dict[str, str | int | bool | pd.DatetimeIndex | None]:
        if self.type_sim == "tmy":
            params = {
                "dataset": self.dataset,
                "location": self.location,
            }
        elif self.type_sim == "mc":
            params = {
                "dataset": self.dataset,
                "location": self.location,
                "subset": self.subset,
                "random": self.random,
                "value": self.value,
            }
        elif self.type_sim == "historical":
            params = {
                "dataset": self.dataset,
                "location": self.location,
                "file_path": self.file_path,
                "list_dates": self.list_dates,
            },
        elif self.type_sim == "constant_day":
            params = {
                "dataset": self.dataset,
                "random": self.random,
                "value": self.value,
                "subset": self.subset,
        }
        else:
            raise ValueError(f"Type of simulation {self.type_sim} is not valid. Options are: 'tmy', 'mc', 'historical', 'constant_day'.")
        return params
    

    def load_data(self, ts_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Load data defined by self.params

        Args:
            ts_index (pd.DatetimeIndex): The dataframe's index defined by the simulation.

        Returns:
            pd.DataFrame: A dataframe with the weather timeseries.
        """
        params = self.params()
        ts_wea = _load_weather_data(
                    ts_index, type_sim = self.type_sim, params = params
                )
        return ts_wea
    


#----------
def load_day_constant_random(
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
def random_days_from_dataframe(
    timeseries: pd.DataFrame,
    df_sample: pd.DataFrame,
    seed_id: Optional[int] = None,
    columns: Optional[list[str]] = TS_WEATHER,
) -> pd.DataFrame :
    """
    This function randomly assign the weather variables of a set of days
    to the timeseries DataFrame. It returns timeseries updated
        
    Parameters
    ----------
    timeseries : pd.DataFrame
        DESCRIPTION.
    set_days : pd.DataFrame
        DESCRIPTION.
    columns : Optional[list[str]], optional
        DESCRIPTION. The default is TS_WEATHER.
    : TYPE
        DESCRIPTION.

    Returns
    -------
    timeseries.

    """
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

#---------------------------------
def from_file(
    timeseries: pd.DataFrame,
    file_path: str = "",
    columns: list[str] = TS_WEATHER,
    subset_random: str | None = None,
    subset_value: str | int | pd.Timestamp | None = None,
) -> pd.DataFrame :
    """
    It returns the dataframe timeseries with the weather loaded from a file.
    It admits optional parameters subset_random and subset_value to select a subset
    from the source and select randomly days from that subset.
    If subset_random is None, load the file as TMY. If the simulation period is longer
    the file is repeated to match it.

    Parameters
    ----------
    timeseries : pd.DataFrame
        The DataFrame defined by profile_new.
    file_path : str
        Path to the file. It is assumed the file is in the correct format.
    columns : Optional[list[str]], optional
        DESCRIPTION. The default is TS_WEATHER.
    subset_random : Optional[str], optional
                    'all': pick from all the dataset,
                    'annual': the year is defined as subset value.
                    'season': the season is defined by subset_value
                                ('summer', 'autumn', 'winter', 'spring')
                    'month': the month is defined by the integer subset_value (1-12),
                    'date': the specific date is defined by a pd.datetime,
                    None: There is not randomization. subset_value is ignored.
                    The default is None.
    subset_value : Optional[str,int], optional. Check previous definition.
                    The default is None.

    Returns
    -------
    timeseries : TYPE
        Returns timeseries with the environmental variables included.

    """
    
    set_days = pd.read_csv(file_path, index_col=0)
    set_days.index = pd.to_datetime(set_days.index)
    if subset_random is None:
        pass
    elif subset_random == 'annual':
        set_days = set_days[
            set_days.index.year==subset_value
            ]
    elif subset_random == 'season':
        set_days = set_days[
            set_days.index.isin(DEFINITION_SEASON[subset_value])
            ]
    elif subset_random == 'month':
        set_days = set_days[
            set_days.index.month==subset_value
            ]  
    elif subset_random == 'date':
        set_days = set_days[
            set_days.index.date==pd.to_datetime(subset_value).date()
            ]  
    
    if subset_random is None:
        timeseries = from_tmy(
            timeseries, set_days, columns=columns
            )
    else:
        timeseries = random_days_from_dataframe(
            timeseries, set_days, columns=columns
            )   
    return timeseries

# -------------
def load_tmy(
    ts: pd.DataFrame,
    params: dict,
    columns: list[str] | None = TS_WEATHER,
) -> pd.DataFrame:
    
    YEAR = pd.to_datetime(ts.index).year[0]
    if type(params["location"]) == str:
        location = params["location"]
    else:
        location = params["location"]
    dataset = params["dataset"]
    if dataset == "meteonorm":
        df_dataset = load_dataset_meteonorm(location, YEAR)
    elif dataset == "merra2":
        df_dataset = load_dataset_merra2(ts, location, YEAR)
    else:
        raise ValueError(f"dataset: {dataset} is not available.")
    return from_tmy( ts, df_dataset, columns=columns )


def load_dataset_meteonorm(
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

    temp_mains = df_dataset["temp_mains"].to_numpy()
    df_dataset["temp_mains"] = np.concatenate((temp_mains[PERIODS//2:], temp_mains[:PERIODS//2]))

    start_time = pd.to_datetime(f"{YEAR}-01-01 00:00:00") + pd.DateOffset(hours=START)
    df_dataset.index = pd.date_range( start=start_time, periods=PERIODS, freq=f"{STEP}min")
    df_dataset["date"] = df_dataset.index
    df_dataset["date"] = df_dataset["date"].apply(lambda x: x.replace(year=YEAR))
    df_dataset.index = pd.to_datetime(df_dataset["date"])
    return df_dataset


def load_dataset_merra2(
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
    df_aux = load_dataset_meteonorm("Sydney", YEAR)
    df_aux = df_aux.resample(f"{STEP}T").interpolate()       #Getting the data in half hours
    ts["Temp_Mains"] = df_aux["Temp_Mains"]
    #########################################

    return ts

#----------
def load_montecarlo(
    ts: pd.DataFrame,
    params: dict,
    columns: Optional[list[str]] = TS_WEATHER,
) -> pd.DataFrame:
    
    dataset = params["dataset"]
    location = params["location"]
    subset = params["subset"]
    value = params["value"]
    ts_index = pd.to_datetime(ts.index)

    if dataset == "meteonorm":
        df_dataset = load_dataset_meteonorm(location)
    elif dataset == "merra2":
        df_dataset = load_dataset_merra2(ts, location, ts_index.year[0])
    else:
        raise ValueError(f"dataset: {dataset} is not available.")
    
    df_dataset.index = pd.to_datetime(df_dataset.index)
    if subset == 'annual':
        df_sample = df_dataset[
            df_dataset.index.year==value
            ]
    elif subset == 'season':
        df_sample = df_dataset[
            df_dataset.index.isin(DEFINITION_SEASON[value])
            ]
    elif subset == 'month':
        df_sample = df_dataset[
            df_dataset.index.month==value
            ]  
    elif subset == 'date':
        df_sample = df_dataset[
            df_dataset.index.date==value.date()
            ]
    else:
        raise ValueError(f"subset: {subset} not in available options.")
    df_weather = random_days_from_dataframe( ts, df_sample, columns=columns )
    return df_weather

#----------------
def load_historical(
    ts: pd.DataFrame,
    params: dict,
    columns: Optional[list[str]] = TS_WEATHER,
) -> pd.DataFrame:
    file_path = params["file_path"]
    ts_ = pd.read_csv(file_path, index_col=0)
    ts_.index = pd.to_datetime(ts.index)
    return ts_

#----------
def _load_weather_data(
        ts: pd.DataFrame | pd.DatetimeIndex,
        type_sim: str,
        params: dict = {},
        columns: Optional[list[str]] = TS_WEATHER,
) -> pd.DataFrame:
    
    if isinstance(ts, pd.DatetimeIndex):
        ts_ = pd.DataFrame(index = ts, columns = columns)
    else:
        ts_ = ts.copy()

    if type_sim == "tmy":
        df_weather = load_tmy(ts_, params, columns)
    elif type_sim == "mc":
        df_weather = load_montecarlo(ts_, params, columns)
    elif type_sim == "historical":
        df_weather = load_historical(ts_, params, columns)
    elif type_sim == "constant_day":
        df_weather = load_day_constant_random(ts_)
    else:
        raise ValueError(f"{type_sim} not in {_TYPES_SIMULATION}")
    
    return df_weather

def main():
    from antupy.tsg import TimeParams


    tp = TimeParams(YEAR=Var(2020,"-"), STEP=Var(30,"min"))
    ts = tp.idx_pd

    #----------------
    type_sim = "tmy"
    params = {
        "dataset": "meteonorm",
        "location": "Sydney"
    }
    ts = _load_weather_data(ts, type_sim, params)
    print(ts[TS_WEATHER])

    #----------------
    type_sim = "tmy"
    YEAR = Var(2020,"-")
    location = LocationAU(2035)
    ts = TimeParams(YEAR=YEAR, STEP=Var(30,"min")).idx_pd
    params = {
        "dataset": "merra2",
        "location": LocationAU(2035)
    }
    ts = _load_weather_data(ts, type_sim, params)
    print(ts[TS_WEATHER])

    #----------------
    type_sim = "mc"
    params = {
        "dataset": "meteonorm",
        "location": LocationAU(2035),
        "subset": "month",
        "value": 5
    }
    ts = _load_weather_data(ts, type_sim, params)
    print(ts[TS_WEATHER])

    #----------------
    type_sim = "constant_day"
    ts = _load_weather_data(ts, type_sim)
    print(ts[TS_WEATHER])

    return


if __name__ == "__main__":
    main()
    pass
