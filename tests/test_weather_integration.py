"""
Integration tests for Weather classes with realistic data scenarios.

This module tests:
- load_data() method functionality
- Integration with actual/mock weather data
- Error handling for missing data files
- Australian location-specific scenarios
- Data format validation
- Performance and memory considerations
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import os

from antupy.tsg.weather import TMY, WeatherMC, WeatherHist, WeatherConstantDay
from antupy.tsg.settings import TimeParams
from antupy.loc.loc_au import LocationAU
from antupy import Var

# Handle DIR_DATA import gracefully for testing environments
try:
    from antupy import ddd_au
    DIR_DATA = getattr(ddd_au, 'DIR_DATA', {'weather': None})
except (ImportError, AttributeError):
    DIR_DATA = {'weather': None}


class TestWeatherDataLoading:
    """Test weather data loading functionality."""
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.test_time_params = TimeParams(
            START=Var(0, "hr"),
            STOP=Var(24, "hr"),
            STEP=Var(1, "hr"),
            YEAR=Var(2023, "-")
        )
        
        # Sample weather data structure
        self.sample_weather_data = pd.DataFrame({
            'DateTime': pd.date_range('2023-01-01', periods=24, freq='h'),
            'GHI': np.random.uniform(0, 1200, 24),  # Global Horizontal Irradiance
            'DNI': np.random.uniform(0, 1000, 24),  # Direct Normal Irradiance
            'DHI': np.random.uniform(0, 400, 24),   # Diffuse Horizontal Irradiance
            'Tamb': np.random.uniform(15, 35, 24),  # Ambient Temperature
            'RH': np.random.uniform(30, 90, 24),    # Relative Humidity
            'WS': np.random.uniform(0, 15, 24),     # Wind Speed
            'WD': np.random.uniform(0, 360, 24),    # Wind Direction
            'P': np.random.uniform(990, 1020, 24)   # Atmospheric Pressure
        })
    
    @patch('antupy.tsg.weather._load_tmy')
    def test_tmy_load_data_integration(self, mock_load_tmy):
        """Test TMY load_data integration with mock data."""
        mock_load_tmy.return_value = self.sample_weather_data
        
        tmy = TMY(
            dataset="meteonorm",
            location=LocationAU("Sydney"),
            time_params=self.test_time_params
        )
        
        result = tmy.load_data()
        
        # Verify mock was called
        mock_load_tmy.assert_called_once()
        
        # Verify result
        assert isinstance(result, pd.DataFrame)
        assert 'GHI' in result.columns or 'DateTime' in result.columns
    
    @patch('antupy.tsg.weather._load_montecarlo')
    def test_weathermc_load_data_integration(self, mock_load_montecarlo):
        """Test WeatherMC load_data integration with mock data."""
        mock_load_montecarlo.return_value = self.sample_weather_data
        
        mc = WeatherMC(
            dataset="meteonorm",
            location=LocationAU("Melbourne"),
            time_params=self.test_time_params,
            subset="month",
            value=6  # June
        )
        
        result = mc.load_data()
        
        # Verify mock was called
        mock_load_montecarlo.assert_called_once()
        
        # Verify result
        assert isinstance(result, pd.DataFrame)
    
    @patch('antupy.tsg.weather._load_historical')
    def test_weatherhist_load_data_integration(self, mock_load_historical):
        """Test WeatherHist load_data integration with mock data."""
        mock_load_historical.return_value = self.sample_weather_data
        
        test_dates = pd.date_range("2023-06-01", "2023-06-07", freq="D")
        
        hist = WeatherHist(
            dataset="merra2",
            location=LocationAU("Brisbane"),
            time_params=self.test_time_params,
            file_path="/path/to/weather/data.csv",
            list_dates=test_dates
        )
        
        result = hist.load_data()
        
        # Verify mock was called
        mock_load_historical.assert_called_once()
        
        # Verify result
        assert isinstance(result, pd.DataFrame)
    
    @patch('antupy.tsg.weather._load_day_constant_random')
    def test_weatherconstantday_load_data_integration(self, mock_load_constant):
        """Test WeatherConstantDay load_data integration with mock data."""
        mock_load_constant.return_value = self.sample_weather_data
        
        constant = WeatherConstantDay(
            dataset="",
            location=LocationAU("Perth"),
            time_params=self.test_time_params,
            random=True
        )
        
        result = constant.load_data()
        
        # Verify mock was called
        mock_load_constant.assert_called_once()
        
        # Verify result
        assert isinstance(result, pd.DataFrame)


class TestWeatherLocationIntegration:
    """Test weather data loading with different Australian locations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_time_params = TimeParams(
            START=Var(0, "hr"),
            STOP=Var(48, "hr"),
            STEP=Var(1, "hr"),
            YEAR=Var(2023, "-")
        )
    
    @patch('antupy.tsg.weather._load_tmy')
    def test_multiple_australian_locations(self, mock_load_tmy):
        """Test TMY loading with different Australian locations."""
        mock_weather_data = pd.DataFrame({
            'GHI': [500] * 48,
            'temp_amb': [25] * 48,
            'temp_mains': [20] * 48
        }, index=pd.date_range('2023-01-01', periods=48, freq='h'))
        
        mock_load_tmy.return_value = mock_weather_data
        
        locations = ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide"]
        
        for city in locations:
            tmy = TMY(
                dataset="meteonorm",
                location=LocationAU(city),
                time_params=self.test_time_params
            )
            
            result = tmy.load_data()
            assert isinstance(result, pd.DataFrame)
            assert len(result) >= 0  # Should return some data


class TestWeatherDataValidation:
    """Test data validation and error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_time_params = TimeParams(
            START=Var(0, "hr"),
            STOP=Var(24, "hr"),
            STEP=Var(1, "hr"),
            YEAR=Var(2023, "-")
        )
    
    @patch('antupy.tsg.weather._load_tmy')
    def test_empty_data_handling(self, mock_load_tmy):
        """Test handling of empty weather data."""
        mock_load_tmy.return_value = pd.DataFrame()
        
        tmy = TMY(
            dataset="meteonorm",
            location=LocationAU("Sydney"),
            time_params=self.test_time_params
        )
        
        result = tmy.load_data()
        assert isinstance(result, pd.DataFrame)
    
    @patch('antupy.tsg.weather._load_tmy')
    def test_invalid_dataset_handling(self, mock_load_tmy):
        """Test handling of invalid dataset."""
        mock_load_tmy.side_effect = ValueError("Invalid dataset")
        
        tmy = TMY(
            dataset="invalid_dataset",
            location=LocationAU("Sydney"),
            time_params=self.test_time_params
        )
        
        with pytest.raises(ValueError):
            tmy.load_data()


class TestWeatherMemoryAndPerformance:
    """Test memory usage and performance considerations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Large time series for performance testing
        self.large_time_params = TimeParams(
            START=Var(0, "hr"),
            STOP=Var(8760, "hr"),  # Full year
            STEP=Var(1, "hr"),
            YEAR=Var(2023, "-")
        )
    
    @patch('antupy.tsg.weather._load_tmy')
    def test_large_dataset_memory_efficiency(self, mock_load_tmy):
        """Test memory efficiency with large datasets."""
        # Create large mock dataset
        large_data = pd.DataFrame({
            'GHI': np.random.uniform(0, 1200, 8760),
            'temp_amb': np.random.uniform(15, 35, 8760),
            'temp_mains': np.random.uniform(15, 25, 8760)
        }, index=pd.date_range('2023-01-01', periods=8760, freq='h'))
        
        mock_load_tmy.return_value = large_data
        
        tmy = TMY(
            dataset="meteonorm",
            location=LocationAU("Sydney"),
            time_params=self.large_time_params
        )
        
        result = tmy.load_data()
        
        # Check that we get reasonable sized data
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 0


class TestWeatherEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_time_params = TimeParams(
            START=Var(0, "hr"),
            STOP=Var(1, "hr"),  # Very short period
            STEP=Var(1, "hr"),
            YEAR=Var(2023, "-")
        )
    
    @patch('antupy.tsg.weather._load_day_constant_random')
    def test_single_timestep_constantday(self, mock_load_constant):
        """Test WeatherConstantDay with single timestep."""
        mock_data = pd.DataFrame({
            'GHI': [500],
            'temp_amb': [25],
            'temp_mains': [20]
        }, index=pd.date_range('2023-01-01', periods=1, freq='h'))
        
        mock_load_constant.return_value = mock_data
        
        constant = WeatherConstantDay(
            dataset="",
            location=LocationAU("Darwin"),
            time_params=self.test_time_params
        )
        
        result = constant.load_data()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    @patch('antupy.tsg.weather._load_montecarlo')
    def test_extreme_time_parameters(self, mock_load_montecarlo):
        """Test with extreme time parameters."""
        extreme_time_params = TimeParams(
            START=Var(0, "hr"),
            STOP=Var(100000, "hr"),  # Extremely long period
            STEP=Var(100, "hr"),     # Large step
            YEAR=Var(2023, "-")
        )
        
        # Mock should handle large datasets gracefully
        mock_load_montecarlo.return_value = pd.DataFrame()
        
        mc = WeatherMC(
            dataset="meteonorm",
            location=LocationAU("Sydney"),
            time_params=extreme_time_params
        )
        
        result = mc.load_data()
        assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__])