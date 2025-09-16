"""
Test suite for Weather Protocol compliance and runtime checking.

This module tests:
- Protocol compliance for all weather classes
- Runtime checkable functionality
- Attribute existence and types
- Method signature compliance
"""

import pytest
from typing import get_type_hints
import pandas as pd

from antupy.tsg.weather import Weather, TMY, WeatherMC, WeatherHist, WeatherConstantDay
from antupy.tsg.settings import TimeParams
from antupy.loc import Location
from antupy.loc.loc_au import LocationAU
from antupy import Var


class TestWeatherProtocolCompliance:
    """Test that all weather classes implement the Weather Protocol correctly."""
    
    def test_weather_protocol_is_runtime_checkable(self):
        """Test that Weather Protocol is runtime checkable."""
        # The Protocol should be decorated with @runtime_checkable
        # Test by trying isinstance with a valid object
        time_params = TimeParams(YEAR=Var(2023, "-"))
        tmy = TMY(time_params=time_params)
        
        # This should work without errors if Protocol is runtime checkable
        assert isinstance(tmy, Weather)
    
    @pytest.mark.parametrize("weather_class", [TMY, WeatherMC, WeatherHist, WeatherConstantDay])
    def test_weather_classes_implement_protocol(self, weather_class):
        """Test that all weather classes implement the Weather Protocol."""
        # Create instance with minimal setup
        time_params = TimeParams(YEAR=Var(2023, "-"))
        instance = weather_class(time_params=time_params)
        
        # Runtime protocol checking
        assert isinstance(instance, Weather), f"{weather_class.__name__} should implement Weather Protocol"
    
    @pytest.mark.parametrize("weather_class", [TMY, WeatherMC, WeatherHist, WeatherConstantDay])
    def test_required_attributes_exist(self, weather_class):
        """Test that all weather classes have required Protocol attributes."""
        time_params = TimeParams(YEAR=Var(2023, "-"))
        instance = weather_class(time_params=time_params)
        
        # Check required attributes
        assert hasattr(instance, 'dataset'), f"{weather_class.__name__} missing 'dataset' attribute"
        assert hasattr(instance, 'location'), f"{weather_class.__name__} missing 'location' attribute"
        assert hasattr(instance, 'time_params'), f"{weather_class.__name__} missing 'time_params' attribute"
        assert hasattr(instance, 'load_data'), f"{weather_class.__name__} missing 'load_data' method"
    
    @pytest.mark.parametrize("weather_class", [TMY, WeatherMC, WeatherHist, WeatherConstantDay])
    def test_attribute_types(self, weather_class):
        """Test that attributes have correct types."""
        time_params = TimeParams(YEAR=Var(2023, "-"))
        instance = weather_class(time_params=time_params)
        
        # Check attribute types
        assert isinstance(instance.dataset, str), f"{weather_class.__name__}.dataset should be str"
        assert isinstance(instance.location, (str, LocationAU)), f"{weather_class.__name__}.location should be str or Location"
        assert isinstance(instance.time_params, TimeParams), f"{weather_class.__name__}.time_params should be TimeParams"
        assert callable(instance.load_data), f"{weather_class.__name__}.load_data should be callable"
    
    @pytest.mark.parametrize("weather_class", [TMY, WeatherMC, WeatherHist, WeatherConstantDay])
    def test_load_data_method_signature(self, weather_class):
        """Test that load_data method has correct signature."""
        time_params = TimeParams(YEAR=Var(2023, "-"), STOP=Var(24, "hr"))
        
        # Create instances with proper parameters for each class
        if weather_class == WeatherMC:
            instance = weather_class(
                time_params=time_params,
                subset="month",
                value=6  # June
            )
        elif weather_class == WeatherHist:
            instance = weather_class(
                time_params=time_params,
                file_path="/mock/path/to/weather.csv"  # Mock path
            )
        else:
            instance = weather_class(time_params=time_params)
        
        # Verify that load_data method exists and is callable
        assert hasattr(instance, 'load_data')
        assert callable(getattr(instance, 'load_data'))
        
        # For this test, we don't actually call load_data since it would require
        # actual data files or mocking. We just verify the method signature exists.
    
    def test_non_weather_objects_fail_protocol_check(self):
        """Test that non-weather objects fail Protocol compliance."""
        non_weather_objects = [
            "string",
            42,
            {"dataset": "fake", "location": "nowhere"},
            None,
            []
        ]
        
        for obj in non_weather_objects:
            assert not isinstance(obj, Weather), f"{obj} should NOT implement Weather Protocol"


class TestRuntimeProtocolChecking:
    """Test runtime protocol checking functionality."""
    
    def test_isinstance_with_valid_weather_objects(self):
        """Test isinstance() works correctly with valid weather objects."""
        time_params = TimeParams(YEAR=Var(2023, "-"))
        
        weather_objects = [
            TMY(time_params=time_params),
            WeatherMC(time_params=time_params),
            WeatherHist(time_params=time_params),
            WeatherConstantDay(time_params=time_params)
        ]
        
        for weather_obj in weather_objects:
            assert isinstance(weather_obj, Weather)
            # Also test that they're instances of their specific class
            assert isinstance(weather_obj, type(weather_obj))
    
    def test_protocol_based_function_dispatch(self):
        """Test using Protocol for function dispatch."""
        def process_weather(weather) -> str:  # Remove type hint to avoid static type error
            """Function that accepts any Weather implementation."""
            if not isinstance(weather, Weather):
                return "Not a weather object"
            return f"Processing {weather.dataset} data for {weather.location}"
        
        time_params = TimeParams(YEAR=Var(2023, "-"))
        tmy = TMY(dataset="meteonorm", location="Sydney", time_params=time_params)
        
        result = process_weather(tmy)
        assert "meteonorm" in result
        assert "Sydney" in result or "LocationAU" in result
        
        # Test with invalid object
        result = process_weather("not weather")
        assert result == "Not a weather object"
    
    def test_protocol_type_checking_in_lists(self):
        """Test Protocol checking with collections of weather objects."""
        time_params = TimeParams(YEAR=Var(2023, "-"))
        
        mixed_objects = [
            TMY(time_params=time_params),
            "not weather",
            WeatherMC(time_params=time_params),
            42,
            WeatherConstantDay(time_params=time_params)
        ]
        
        weather_objects = [obj for obj in mixed_objects if isinstance(obj, Weather)]
        
        assert len(weather_objects) == 3
        for obj in weather_objects:
            assert hasattr(obj, 'load_data')
            assert hasattr(obj, 'dataset')


class TestProtocolInheritanceAndCompatibility:
    """Test Protocol inheritance and compatibility scenarios."""
    
    def test_location_protocol_compatibility(self):
        """Test that Location Protocol works with weather objects."""
        time_params = TimeParams(YEAR=Var(2023, "-"))
        
        # Test with LocationAU object
        location_au = LocationAU("Melbourne")
        weather = TMY(location=location_au, time_params=time_params)
        
        assert isinstance(weather.location, Location)
        assert isinstance(weather, Weather)
    
    def test_protocol_with_inheritance_chain(self):
        """Test Protocol checking works with class inheritance."""
        # All weather classes should be instances of their parent classes and Protocol
        time_params = TimeParams(YEAR=Var(2023, "-"))
        
        tmy = TMY(time_params=time_params)
        assert isinstance(tmy, TMY)  # Specific class
        assert isinstance(tmy, Weather)  # Protocol
    
    def test_protocol_attribute_access_safety(self):
        """Test safe attribute access after Protocol validation."""
        def safe_weather_processor(obj):
            """Safely process any object that might be a weather object."""
            if isinstance(obj, Weather):
                # Safe to access Protocol attributes
                return {
                    'dataset': obj.dataset,
                    'location': str(obj.location),
                    'periods': obj.time_params.PERIODS.gv('-')
                }
            return None
        
        time_params = TimeParams(YEAR=Var(2023, "-"), STOP=Var(48, "hr"))
        weather = TMY(time_params=time_params)
        
        result = safe_weather_processor(weather)
        assert result is not None
        assert 'dataset' in result
        assert 'location' in result
        assert 'periods' in result
        
        # Test with non-weather object
        result = safe_weather_processor("not weather")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__])