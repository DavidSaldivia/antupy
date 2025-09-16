"""
Test fixtures for weather module testing.

This module provides reusable sample weather data and helper functions
for comprehensive testing of the weather.py module.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Base directory for fixtures
FIXTURES_DIR = Path(__file__).parent

def create_sample_tmy_data(location: str = "Sydney", year: int = 2023) -> pd.DataFrame:
    """
    Create realistic TMY (Typical Meteorological Year) sample data.
    
    Parameters:
        location: Australian city name
        year: Year for the data
        
    Returns:
        DataFrame with typical weather patterns for Australian conditions
    """
    # 8760 hours in a year
    hours = np.arange(8760)
    start_date = pd.Timestamp(f'{year}-01-01')
    index = pd.date_range(start=start_date, periods=8760, freq='H')
    
    # Create realistic Australian weather patterns
    day_of_year = np.array([dt.timetuple().tm_yday for dt in index])
    hour_of_day = index.hour.values
    
    # Solar irradiance patterns
    # Annual cycle: peak in summer (Dec-Feb), minimum in winter (Jun-Aug)
    annual_solar_factor = 0.7 + 0.3 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    # Daily cycle: solar peak at noon, zero at night
    daily_solar_factor = np.maximum(0, np.cos(np.pi * (hour_of_day - 12) / 12))
    # Random weather variability
    np.random.seed(42)  # For reproducible fixtures
    weather_noise = 0.8 + 0.4 * np.random.random(8760)
    
    ghi = 1200 * annual_solar_factor * daily_solar_factor * weather_noise
    ghi = np.maximum(0, ghi)  # No negative irradiance
    
    # Temperature patterns for Australia
    if location.lower() in ['darwin', 'cairns']:
        # Tropical: warm year-round, wet/dry seasons
        base_temp = 28 + 3 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
        daily_temp_range = 8
    elif location.lower() in ['perth', 'adelaide']:
        # Mediterranean: mild winters, hot dry summers
        base_temp = 20 + 8 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
        daily_temp_range = 12
    elif location.lower() in ['sydney', 'melbourne', 'brisbane']:
        # Temperate: distinct seasons
        base_temp = 18 + 7 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
        daily_temp_range = 10
    else:
        # Default Australian pattern
        base_temp = 20 + 6 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
        daily_temp_range = 10
    
    # Daily temperature cycle: peak mid-afternoon, minimum pre-dawn
    daily_temp_cycle = daily_temp_range * 0.5 * np.cos(2 * np.pi * (hour_of_day - 15) / 24)
    temp_noise = 2 * (np.random.random(8760) - 0.5)
    temp_amb = base_temp + daily_temp_cycle + temp_noise
    
    # Mains water temperature (ground temperature with lag)
    temp_mains = 18 + 4 * np.cos(2 * np.pi * (day_of_year - 105) / 365)  # 3-month lag
    
    # Relative humidity (higher in winter, varies with temperature)
    base_humidity = 60 + 20 * np.cos(2 * np.pi * (day_of_year - 195) / 365)  # Winter peak
    humidity_temp_factor = -0.5 * (temp_amb - 20)  # Lower RH with higher temp
    humidity_noise = 10 * (np.random.random(8760) - 0.5)
    relative_humidity = np.clip(base_humidity + humidity_temp_factor + humidity_noise, 20, 95)
    
    # Wind speed (typically higher in winter, lower at night)
    seasonal_wind = 8 + 4 * np.cos(2 * np.pi * (day_of_year - 195) / 365)
    daily_wind_factor = 0.7 + 0.3 * np.cos(2 * np.pi * (hour_of_day - 15) / 24)
    wind_noise = 3 * np.random.exponential(1, 8760)  # Exponential for wind gusts
    wind_speed = np.maximum(0, seasonal_wind * daily_wind_factor + wind_noise)
    
    # Wind direction (prevailing westerlies with some variation)
    base_wind_dir = 270  # Westerly
    wind_dir_variation = 60 * (np.random.random(8760) - 0.5)
    wind_direction = (base_wind_dir + wind_dir_variation) % 360
    
    # Atmospheric pressure
    pressure_base = 1013.25
    pressure_variation = 15 * np.sin(2 * np.pi * hours / (24 * 7))  # Weekly cycle
    pressure_noise = 5 * (np.random.random(8760) - 0.5)
    pressure = pressure_base + pressure_variation + pressure_noise
    
    return pd.DataFrame({
        'GHI': ghi,
        'DNI': 0.8 * ghi + 50 * np.random.random(8760),  # Direct normal from global
        'DHI': 0.3 * ghi + 30 * np.random.random(8760),  # Diffuse from global
        'temp_amb': temp_amb,
        'Temp_Amb': temp_amb,  # Alternative naming
        'temp_mains': temp_mains,
        'Temp_Mains': temp_mains,  # Alternative naming
        'RH': relative_humidity,
        'WS': wind_speed,
        'WD': wind_direction,
        'P': pressure,
    }, index=index)


def create_sample_meteonorm_data(location: str = "Sydney") -> pd.DataFrame:
    """Create sample data in METEONORM format (3-minute intervals)."""
    # METEONORM typically uses 3-minute intervals for a full year
    periods = 8760 * 20  # 3-minute intervals
    index = pd.date_range('2023-01-01', periods=periods, freq='3min')
    
    # Use the TMY function and downsample
    hourly_data = create_sample_tmy_data(location, 2023)
    
    # Interpolate to 3-minute intervals
    resampled = hourly_data.resample('3min').interpolate(method='linear')
    
    # Add some noise to make it more realistic
    np.random.seed(42)
    for col in ['GHI', 'temp_amb', 'temp_mains']:
        if col in resampled.columns:
            noise_factor = 0.02  # 2% noise
            noise = 1 + noise_factor * (np.random.random(len(resampled)) - 0.5)
            resampled[col] *= noise
    
    return resampled.iloc[:periods]  # Ensure exact length


def create_sample_merra2_data(location: str = "Sydney", year: int = 2023) -> dict:
    """
    Create sample data in MERRA2 NetCDF-like format.
    Returns a dictionary that can be used to mock xarray datasets.
    """
    # MERRA2 typically has hourly data
    hours = 8760
    index = pd.date_range(f'{year}-01-01', periods=hours, freq='H')
    
    # Get base data
    base_data = create_sample_tmy_data(location, year)
    
    # MERRA2 uses different variable names and units
    return {
        'time': index,
        'SWGDN': np.array(base_data['GHI']),  # Surface Incoming Shortwave Flux
        'T2M': np.array(base_data['temp_amb']) + 273.15,  # Temperature at 2m (Kelvin)
        'U2M': np.array(base_data['WS']) * 0.7,  # U-component of wind
        'V2M': np.array(base_data['WS']) * 0.3,  # V-component of wind
        'PS': np.array(base_data['P']) * 100,  # Surface pressure (Pa)
        'QV2M': np.array(base_data['RH']) / 100 * 0.02,  # Specific humidity
    }


def save_fixture_files():
    """Create and save all fixture files."""
    fixtures_weather_dir = FIXTURES_DIR / "weather"
    fixtures_weather_dir.mkdir(exist_ok=True)
    
    # Australian cities to create fixtures for
    cities = ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Darwin"]
    
    for city in cities:
        # TMY data (CSV format)
        tmy_data = create_sample_tmy_data(city, 2023)
        tmy_file = fixtures_weather_dir / f"tmy_{city.lower()}_2023.csv"
        tmy_data.to_csv(tmy_file)
        
        # METEONORM format
        meteo_data = create_sample_meteonorm_data(city)
        meteo_file = fixtures_weather_dir / f"meteonorm_{city.lower()}.csv"
        meteo_data.to_csv(meteo_file)
        
        # MERRA2 metadata (JSON for testing)
        merra2_data = create_sample_merra2_data(city, 2023)
        merra2_meta = {
            'location': city,
            'year': 2023,
            'variables': list(merra2_data.keys()),
            'records': len(merra2_data['time']),
            'description': f'Sample MERRA2-like data for {city}, Australia'
        }
        merra2_file = fixtures_weather_dir / f"merra2_{city.lower()}_meta.json"
        with open(merra2_file, 'w') as f:
            json.dump(merra2_meta, f, indent=2)
    
    # Create a small sample for testing edge cases
    small_sample = create_sample_tmy_data("Sydney", 2023).iloc[:168]  # One week
    small_file = fixtures_weather_dir / "sample_week_sydney.csv"
    small_sample.to_csv(small_file)
    
    # Create malformed data for error testing
    malformed_data = pd.DataFrame({
        'invalid_column': [1, 2, 3],
        'GHI': [-100, 'invalid', None],  # Invalid values
    }, index=pd.date_range('2023-01-01', periods=3, freq='H'))
    malformed_file = fixtures_weather_dir / "malformed_data.csv"
    malformed_data.to_csv(malformed_file)
    
    print(f"Created weather fixtures in {fixtures_weather_dir}")


if __name__ == "__main__":
    save_fixture_files()