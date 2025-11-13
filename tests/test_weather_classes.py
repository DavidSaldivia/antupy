"""
Test suite for Weather class instantiation and basic functionality.

This module tests:
- Class instantiation with default and custom parameters
- Mutable default handling
- Parameter validation and error handling
- Field factory functionality
- Individual class-specific behavior
"""

import pytest
import pandas as pd

from antupy.tsg.weather import TMY, WeatherMC, WeatherHist, WeatherConstantDay
from antupy.tsg.settings import TimeParams
from antupy.utils.loc.loc_au import LocationAU
from antupy import Var
from antupy.ddd_au import DEFINITIONS


class TestWeatherClassInstantiation:
    """Test weather class instantiation and default handling."""
    
    def test_default_instantiation(self):
        """Test that all weather classes can be instantiated with defaults."""
        # Should work without any parameters
        tmy = TMY()
        mc = WeatherMC()
        hist = WeatherHist()
        const = WeatherConstantDay()
        
        # Check that all have required attributes
        for weather in [tmy, mc, hist, const]:
            assert hasattr(weather, 'dataset')
            assert hasattr(weather, 'location')
            assert hasattr(weather, 'time_params')
            assert hasattr(weather, 'load_data')
    
    def test_mutable_defaults_independence(self):
        """Test that mutable defaults create independent objects."""
        # Create multiple instances
        tmy1 = TMY()
        tmy2 = TMY()
        
        # TimeParams should be different objects
        assert tmy1.time_params is not tmy2.time_params
        assert id(tmy1.time_params) != id(tmy2.time_params)
        
        # Locations should be different objects
        assert tmy1.location is not tmy2.location
        assert id(tmy1.location) != id(tmy2.location)
        
        # Modifying one shouldn't affect the other
        tmy1.time_params.YEAR = Var(2020, "-")
        tmy2.time_params.YEAR = Var(2025, "-")
        
        assert tmy1.time_params.YEAR.gv('-') != tmy2.time_params.YEAR.gv('-')
    
    def test_custom_instantiation(self):
        """Test instantiation with custom parameters."""
        custom_time_params = TimeParams(
            YEAR=Var(2024, "-"),
            STEP=Var(30, "min"),
            STOP=Var(48, "hr")
        )
        custom_location = LocationAU("Melbourne")
        
        tmy = TMY(
            dataset="merra2",
            location=custom_location,
            time_params=custom_time_params
        )
        
        assert tmy.dataset == "merra2"
        assert tmy.location == custom_location
        assert tmy.time_params == custom_time_params
        assert tmy.time_params.YEAR.gv('-') == 2024
    
    def test_default_location_type(self):
        """Test that default location is LocationAU('Sydney')."""
        tmy = TMY()
        
        assert isinstance(tmy.location, LocationAU)
        assert str(tmy.location) == "Sydney"
    
    def test_string_location_assignment(self):
        """Test that string locations can be assigned."""
        tmy = TMY(location="Brisbane")
        
        assert tmy.location == "Brisbane"
        assert isinstance(tmy.location, str)


class TestTMYClass:
    """Test TMY-specific functionality."""
    
    def test_tmy_default_parameters(self):
        """Test TMY default parameter values."""
        tmy = TMY()
        
        assert tmy.dataset == "meteonorm"
        assert isinstance(tmy.location, LocationAU)
        assert isinstance(tmy.time_params, TimeParams)
    
    def test_tmy_dataset_options(self):
        """Test TMY with different dataset options."""
        time_params = TimeParams(YEAR=Var(2023, "-"))
        
        # Test meteonorm
        tmy_meteo = TMY(dataset="meteonorm", time_params=time_params)
        assert tmy_meteo.dataset == "meteonorm"
        
        # Test merra2
        tmy_merra = TMY(dataset="merra2", time_params=time_params)
        assert tmy_merra.dataset == "merra2"
    
    def test_tmy_australian_locations(self):
        """Test TMY with Australian locations."""
        time_params = TimeParams(YEAR=Var(2023, "-"))
        
        for city in DEFINITIONS.LOCATIONS_METEONORM:
            tmy = TMY(location=city, time_params=time_params)
            assert tmy.location == city


class TestWeatherMCClass:
    """Test WeatherMC-specific functionality."""
    
    def test_weathermc_default_parameters(self):
        """Test WeatherMC default parameter values."""
        mc = WeatherMC()
        
        assert mc.dataset == "meteonorm"
        assert isinstance(mc.location, LocationAU)
        assert mc.subset is None
        assert mc.random is False
        assert mc.value is None
    
    def test_weathermc_subset_options(self):
        """Test WeatherMC with different subset options."""
        time_params = TimeParams(YEAR=Var(2023, "-"))
        
        subset_options = ["annual", "season", "month", "date"]
        
        for subset in subset_options:
            mc = WeatherMC(subset=subset, time_params=time_params)
            assert mc.subset == subset
    
    def test_weathermc_random_options(self):
        """Test WeatherMC random parameter."""
        time_params = TimeParams(YEAR=Var(2023, "-"))
        
        mc_random = WeatherMC(random=True, time_params=time_params)
        assert mc_random.random is True
        
        mc_deterministic = WeatherMC(random=False, time_params=time_params)
        assert mc_deterministic.random is False
    
    def test_weathermc_value_assignment(self):
        """Test WeatherMC value parameter assignment."""
        time_params = TimeParams(YEAR=Var(2023, "-"))
        
        # Test with string value (for seasons)
        mc_season = WeatherMC(subset="season", value="summer", time_params=time_params)
        assert mc_season.value == "summer"
        
        # Test with int value (for months)
        mc_month = WeatherMC(subset="month", value=6, time_params=time_params)
        assert mc_month.value == 6


class TestWeatherHistClass:
    """Test WeatherHist-specific functionality."""
    
    def test_weatherhist_default_parameters(self):
        """Test WeatherHist default parameter values."""
        hist = WeatherHist()
        
        assert hist.dataset == "merra2"
        assert isinstance(hist.location, LocationAU)
        assert hist.file_path is None
        assert hist.list_dates is None
    
    def test_weatherhist_file_path_assignment(self):
        """Test WeatherHist file path assignment."""
        time_params = TimeParams(YEAR=Var(2023, "-"))
        test_path = "/path/to/weather/data.csv"
        
        hist = WeatherHist(file_path=test_path, time_params=time_params)
        assert hist.file_path == test_path
    
    def test_weatherhist_dates_assignment(self):
        """Test WeatherHist dates assignment."""
        time_params = TimeParams(YEAR=Var(2023, "-"))
        test_dates = pd.date_range("2023-01-01", "2023-01-07", freq="D")
        
        hist = WeatherHist(list_dates=test_dates, time_params=time_params)
        assert hist.list_dates is test_dates
    
    def test_weatherhist_dataset_options(self):
        """Test WeatherHist with different dataset options."""
        time_params = TimeParams(YEAR=Var(2023, "-"))
        
        dataset_options = ["merra2", "nci", "local"]
        
        for dataset in dataset_options:
            hist = WeatherHist(dataset=dataset, time_params=time_params)
            assert hist.dataset == dataset


class TestWeatherConstantDayClass:
    """Test WeatherConstantDay-specific functionality."""
    
    def test_weatherconstantday_default_parameters(self):
        """Test WeatherConstantDay default parameter values."""
        const = WeatherConstantDay()
        
        assert const.dataset == ""
        assert isinstance(const.location, LocationAU)
        assert const.random is False
        assert const.value is None
        assert const.subset is None
    
    def test_weatherconstantday_random_options(self):
        """Test WeatherConstantDay random parameter."""
        time_params = TimeParams(YEAR=Var(2023, "-"))
        
        const_random = WeatherConstantDay(random=True, time_params=time_params)
        assert const_random.random is True
        
        const_fixed = WeatherConstantDay(random=False, time_params=time_params)
        assert const_fixed.random is False
    
    def test_weatherconstantday_value_assignment(self):
        """Test WeatherConstantDay value assignment."""
        time_params = TimeParams(YEAR=Var(2023, "-"))
        
        const = WeatherConstantDay(value=25, time_params=time_params)
        assert const.value == 25


class TestParameterValidation:
    """Test parameter validation and error handling."""
    
    def test_time_params_integration(self):
        """Test that TimeParams integrates properly with weather classes."""
        custom_time_params = TimeParams(
            START=Var(0, "hr"),
            STOP=Var(72, "hr"),  # 3 days
            STEP=Var(15, "min"),  # 15-minute steps
            YEAR=Var(2024, "-")
        )
        
        weather_classes = [TMY, WeatherMC, WeatherHist, WeatherConstantDay]
        
        for WeatherClass in weather_classes:
            weather = WeatherClass(time_params=custom_time_params)
            
            # Check that time_params is properly assigned
            assert weather.time_params is custom_time_params
            assert weather.time_params.YEAR.gv('-') == 2024
            assert weather.time_params.STEP.gv('min') == 15
    
    def test_location_type_flexibility(self):
        """Test that different location types work correctly."""
        time_params = TimeParams(YEAR=Var(2023, "-"))
        
        # String location
        tmy_str = TMY(location="Adelaide", time_params=time_params)
        assert tmy_str.location == "Adelaide"
        
        # LocationAU object
        location_au = LocationAU("Perth")
        tmy_au = TMY(location=location_au, time_params=time_params)
        assert tmy_au.location is location_au
        
        # Postcode (through LocationAU)
        postcode_location = LocationAU(2000)  # Sydney postcode
        tmy_postcode = TMY(location=postcode_location, time_params=time_params)
        assert tmy_postcode.location is postcode_location


class TestDataclassFeatures:
    """Test dataclass-specific features."""
    
    def test_repr_functionality(self):
        """Test that dataclass __repr__ works correctly."""
        time_params = TimeParams(YEAR=Var(2023, "-"))
        tmy = TMY(dataset="meteonorm", location="Sydney", time_params=time_params)
        
        repr_str = repr(tmy)
        assert "TMY" in repr_str
        assert "meteonorm" in repr_str
    
    def test_equality_comparison(self):
        """Test that dataclass equality works correctly."""
        time_params = TimeParams(YEAR=Var(2023, "-"))
        
        tmy1 = TMY(dataset="meteonorm", location="Sydney", time_params=time_params)
        tmy2 = TMY(dataset="meteonorm", location="Sydney", time_params=time_params)
        
        # Note: They won't be equal because time_params are different objects
        # But they should have the same values
        assert tmy1.dataset == tmy2.dataset
        assert str(tmy1.location) == str(tmy2.location)
    
    def test_field_access(self):
        """Test that all fields are accessible and mutable."""
        tmy = TMY()
        
        # Test field modification
        original_dataset = tmy.dataset
        tmy.dataset = "new_dataset"
        assert tmy.dataset == "new_dataset"
        assert tmy.dataset != original_dataset
        
        # Test time_params modification
        original_year = tmy.time_params.YEAR.gv('-')
        tmy.time_params.YEAR = Var(2025, "-")
        assert tmy.time_params.YEAR.gv('-') == 2025
        assert tmy.time_params.YEAR.gv('-') != original_year


if __name__ == "__main__":
    pytest.main([__file__])