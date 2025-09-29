"""
Test suite for Weather utility functions using the sample_week_sydney.csv fixture.

This module tests:
- load_day_constant_random function
- random_days_from_dataframe function  
- from_tmy function
- from_file function
- Basic functionality with real data
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from antupy.tsg.weather import (
    _load_day_constant_random,
    _random_days_from_dataframe,
    from_tmy,
    _load_dataset_meteonorm,
    TS_WEATHER,
    _VARIABLE_RANGES
)
from antupy.tsg.settings import TimeParams
from antupy.loc.loc_au import LocationAU
from antupy import Var


class TestLoadDayConstantRandom:
    """Test load_day_constant_random utility function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.timeseries = pd.DataFrame(
            index=pd.date_range('2023-01-01', periods=48, freq='h'),
            columns=TS_WEATHER
        )
    
    def test_basic_functionality(self):
        """Test basic constant day generation."""
        result = _load_day_constant_random(
            self.timeseries.copy(),
            seed_id=42,
            columns=["GHI", "temp_amb"]
        )
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == self.timeseries.shape
        assert "GHI" in result.columns
        assert "temp_amb" in result.columns
        assert not result["GHI"].isna().any()
        assert not result["temp_amb"].isna().any()
    
    def test_default_parameters(self):
        """Test with default parameters."""
        result = _load_day_constant_random(self.timeseries.copy())
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == self.timeseries.shape
        # Should have all TS_WEATHER columns
        for col in TS_WEATHER:
            assert col in result.columns
    
    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        result1 = _load_day_constant_random(
            self.timeseries.copy(),
            seed_id=42,
            columns=["GHI"]
        )
        result2 = _load_day_constant_random(
            self.timeseries.copy(),
            seed_id=42,
            columns=["GHI"]
        )
        
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        result1 = _load_day_constant_random(
            self.timeseries.copy(),
            seed_id=42,
            columns=["temp_amb"]  # Use temp_amb which has a range, not GHI
        )
        result2 = _load_day_constant_random(
            self.timeseries.copy(),
            seed_id=123,
            columns=["temp_amb"]
        )
        
        # Results should be different
        assert not result1["temp_amb"].equals(result2["temp_amb"])


class TestRandomDaysFromDataframe:
    """Test random_days_from_dataframe utility function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.timeseries = pd.DataFrame(
            index=pd.date_range('2023-01-01', periods=120, freq='h'),  # 5 days
            columns=TS_WEATHER
        )
        
        # Path to our real fixture file
        self.fixture_path = Path(__file__).parent / "fixtures" / "weather" / "weather" / "sample_week_sydney.csv"
        self.fixture_data = pd.read_csv(self.fixture_path, index_col=0, parse_dates=True)
    
    def test_basic_functionality(self):
        """Test basic random day selection."""
        result = _random_days_from_dataframe(
            self.timeseries.copy(),
            self.fixture_data,
            seed_id=42,
            columns=["GHI", "temp_amb"]
        )
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == self.timeseries.shape
        assert not result["GHI"].isna().all()
        assert not result["temp_amb"].isna().all()
    
    def test_reproducibility_with_seed(self):
        """Test reproducible results with same seed."""
        result1 = _random_days_from_dataframe(
            self.timeseries.copy(),
            self.fixture_data,
            seed_id=42,
            columns=["GHI"]
        )
        result2 = _random_days_from_dataframe(
            self.timeseries.copy(),
            self.fixture_data,
            seed_id=42,
            columns=["GHI"]
        )
        
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        result1 = _random_days_from_dataframe(
            self.timeseries.copy(),
            self.fixture_data,
            seed_id=42,
            columns=["GHI"]
        )
        result2 = _random_days_from_dataframe(
            self.timeseries.copy(),
            self.fixture_data,
            seed_id=123,
            columns=["GHI"]
        )
        
        # Results should be different
        assert not result1["GHI"].equals(result2["GHI"])


class TestFromTmy:
    """Test from_tmy utility function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.timeseries = pd.DataFrame(
            index=pd.date_range('2023-01-01', periods=48, freq='h'),
            columns=TS_WEATHER
        )
        
        # Use our fixture data as TMY
        self.fixture_path = Path(__file__).parent / "fixtures" / "weather" / "weather" / "sample_week_sydney.csv"
        self.tmy_data = pd.read_csv(self.fixture_path, index_col=0, parse_dates=True)
    
    def test_basic_functionality(self):
        """Test basic TMY functionality."""
        result = from_tmy(
            self.timeseries.copy(),
            self.tmy_data,
            columns=["GHI", "temp_amb"]
        )
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == self.timeseries.shape
        assert "GHI" in result.columns
        assert "temp_amb" in result.columns
    
    def test_time_alignment(self):
        """Test that time alignment works correctly."""
        # Create a shorter timeseries
        short_ts = pd.DataFrame(
            index=pd.date_range('2023-01-01', periods=24, freq='h'),  # 1 day
            columns=["GHI"]
        )
        
        result = from_tmy(short_ts, self.tmy_data, columns=["GHI"])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 24
        assert result.index.equals(short_ts.index)
    
    def test_default_columns(self):
        """Test with default columns."""
        result = from_tmy(self.timeseries.copy(), self.tmy_data)
        
        assert isinstance(result, pd.DataFrame)
        # Should use TS_WEATHER columns that exist in TMY data
        common_cols = set(TS_WEATHER).intersection(set(self.tmy_data.columns))
        for col in common_cols:
            assert col in result.columns


class TestPrivateHelperFunctions:
    """Test private helper functions using simplified mocking."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_ts = pd.DataFrame(
            index=pd.date_range('2023-01-01', periods=24, freq='h'),
            columns=TS_WEATHER
        )
        
        # Path to our real fixture file
        self.fixture_path = Path(__file__).parent / "fixtures" / "weather" / "weather" / "sample_week_sydney.csv"
    
    @patch('os.path.join')
    @patch('pandas.read_csv')
    def test_load_dataset_meteonorm(self, mock_read_csv, mock_join):
        """Test _load_dataset_meteonorm with simplified mocking."""
        # Create a mock dataframe with the expected structure
        mock_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=24, freq='h'),
            'GHI': [100] * 24,
            'temp_amb': [20] * 24,
            'temp_mains': [15] * 24
        })
        mock_read_csv.return_value = mock_data
        mock_join.return_value = '/mock/meteonorm/path/Sydney.csv'
        
        result = _load_dataset_meteonorm("Sydney", YEAR=2023)
        
        mock_read_csv.assert_called_once()
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_load_dataset_meteonorm_invalid_location(self):
        """Test _load_dataset_meteonorm with invalid location."""
        with pytest.raises(ValueError):
            _load_dataset_meteonorm("InvalidCity", YEAR=2023)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling for utility functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Path to our real fixture file
        self.fixture_path = Path(__file__).parent / "fixtures" / "weather" / "weather" / "sample_week_sydney.csv"
    
    def test_load_day_constant_random_with_valid_ranges(self):
        """Test load_day_constant_random with valid range values."""
        timeseries = pd.DataFrame(
            index=pd.date_range('2023-01-01', periods=24, freq='h'),
            columns=["GHI"]
        )
        
        # Test with valid range
        valid_ranges = {"GHI": (100.0, 1000.0)}
        
        result = _load_day_constant_random(
            timeseries,
            ranges=valid_ranges,
            seed_id=42,
            columns=["GHI"]
        )
        
        assert isinstance(result, pd.DataFrame)
        # Values should be within the specified range
        assert result["GHI"].min() >= 100.0
        assert result["GHI"].max() <= 1000.0
    
    def test_random_days_from_dataframe_with_matching_columns(self):
        """Test random_days_from_dataframe with matching columns."""
        timeseries = pd.DataFrame(
            index=pd.date_range('2023-01-01', periods=24, freq='h'),
            columns=["GHI", "temp_amb"]
        )
        
        # Load real fixture data
        df_sample = pd.read_csv(self.fixture_path, index_col=0, parse_dates=True)
        
        # Test with columns that exist in both
        result = _random_days_from_dataframe(
            timeseries,
            df_sample,
            seed_id=42,
            columns=["GHI", "temp_amb"]
        )
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == timeseries.shape
        assert not result["GHI"].isna().all()  # Should have some data
        assert not result["temp_amb"].isna().all()  # Should have some data
    
    def test_from_tmy_with_valid_tmy_data(self):
        """Test from_tmy with valid TMY data."""
        timeseries = pd.DataFrame(
            index=pd.date_range('2023-01-01', periods=24, freq='h'),
            columns=["GHI"]
        )
        
        # Create sample TMY data
        tmy_data = pd.DataFrame(
            index=pd.date_range('2022-01-01', periods=48, freq='h'),
            columns=["GHI"]
        )
        tmy_data["GHI"] = np.random.uniform(0, 1000, 48)
        
        result = from_tmy(timeseries, tmy_data, columns=["GHI"])
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == timeseries.shape
        assert not result["GHI"].isna().all()
    


if __name__ == "__main__":
    pytest.main([__file__])