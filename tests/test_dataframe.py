"""
Comprehensive tests for antupy.DataFrame with units support.

Based on test_dataframe_demo.py and antupy_dataframe_demo.py functionality.
"""

import pytest
import warnings
import pandas as pd
import numpy as np
import antupy as ap

from antupy.dataframe import dataframe_with_units


class TestDataFrameBasics:
    """Test basic DataFrame creation and functionality."""
    
    def test_dataframe_creation_with_list_units(self):
        """Test creating DataFrame with list units."""
        df = ap.DataFrame({
            'temperature': [20, 25, 30],
            'pressure': [1013, 1015, 1010]
        }, units=['°C', 'hPa'])
        
        assert isinstance(df, ap.DataFrame)
        assert df.units == ['°C', 'hPa']
        assert len(df.units) == len(df.columns)
    
    def test_dataframe_creation_with_dict_units(self):
        """Test creating DataFrame with dict units."""
        df = ap.DataFrame({
            'velocity': [10, 15, 20],
            'acceleration': [2.5, 3.0, 1.8]
        }, units={'velocity': 'm/s', 'acceleration': 'm/s²'})
        
        assert df.units == ['m/s', 'm/s²']
        assert df.unit('velocity') == {'velocity': 'm/s'}
        assert df.unit('acceleration') == {'acceleration': 'm/s²'}
    
    def test_dataframe_creation_without_units(self):
        """Test creating DataFrame without units (should default to empty strings)."""
        df = ap.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        assert df.units == ['', '']
        assert all(unit == '' for unit in df.units)
    
    def test_dataframe_creation_from_pandas(self):
        """Test creating antupy DataFrame from pandas DataFrame."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df = ap.DataFrame(pd_df, units=['m', 's'])
        
        assert isinstance(df, ap.DataFrame)
        assert df.units == ['m', 's']
        assert df.values.tolist() == [[1, 3], [2, 4]]


class TestUnitsProperty:
    """Test the units property and its setter."""
    
    def test_units_getter(self):
        """Test that units property returns a copy."""
        df = ap.DataFrame({'A': [1], 'B': [2]}, units=['m', 's'])
        units = df.units
        units[0] = 'km'  # Modify the returned list
        
        # Original should be unchanged
        assert df.units == ['m', 's']
    
    def test_units_setter_valid(self):
        """Test setting units with valid list."""
        df = ap.DataFrame({'A': [1], 'B': [2]})
        df.units = ['m', 's']
        
        assert df.units == ['m', 's']
    
    def test_units_setter_invalid_length(self):
        """Test setting units with wrong length raises error."""
        df = ap.DataFrame({'A': [1], 'B': [2]})
        
        with pytest.raises(ValueError, match="Length of units"):
            df.units = ['m']  # Too short
        
        with pytest.raises(ValueError, match="Length of units"):
            df.units = ['m', 's', 'kg']  # Too long


class TestUnitMethod:
    """Test the unit() method for querying units."""
    
    def test_unit_method_all_units(self):
        """Test unit() with no arguments returns all units."""
        df = ap.DataFrame({
            'temp': [20, 25],
            'pressure': [1013, 1015],
            'humidity': [60, 65]
        }, units=['°C', 'hPa', '%'])
        
        result = df.unit()
        expected = {'temp': '°C', 'pressure': 'hPa', 'humidity': '%'}
        assert result == expected
    
    def test_unit_method_single_column_string(self):
        """Test unit() with single column name as string."""
        df = ap.DataFrame({'temp': [20], 'pressure': [1013]}, units=['°C', 'hPa'])
        
        assert df.unit('temp') == {'temp': '°C'}
        assert df.unit('pressure') == {'pressure': 'hPa'}
    
    def test_unit_method_single_column_list(self):
        """Test unit() with single column name as list."""
        df = ap.DataFrame({'temp': [20], 'pressure': [1013]}, units=['°C', 'hPa'])
        
        assert df.unit(['temp']) == {'temp': '°C'}
    
    def test_unit_method_multiple_columns(self):
        """Test unit() with multiple column names."""
        df = ap.DataFrame({
            'temp': [20], 'pressure': [1013], 'humidity': [60]
        }, units=['°C', 'hPa', '%'])
        
        result = df.unit(['temp', 'humidity'])
        expected = {'temp': '°C', 'humidity': '%'}
        assert result == expected
    
    def test_unit_method_invalid_column(self):
        """Test unit() with non-existent column raises error."""
        df = ap.DataFrame({'A': [1]}, units=['m'])
        
        with pytest.raises(KeyError, match="Column 'B' not found"):
            df.unit('B')
        
        with pytest.raises(KeyError, match="Column 'C' not found"):
            df.unit(['A', 'C'])
    
    def test_unit_method_invalid_type(self):
        """Test unit() with invalid argument type raises error."""
        df = ap.DataFrame({'A': [1]}, units=['m'])
        
        with pytest.raises(TypeError, match="cols must be a string, list of strings, or None"):
            df.unit(123)


class TestSetUnitsMethod:
    """Test the set_units() method."""
    
    def test_set_units_with_dict(self):
        """Test set_units() with dictionary."""
        df = ap.DataFrame({'A': [1], 'B': [2], 'C': [3]})
        
        df.set_units({'A': 'm', 'C': 'kg'})
        assert df.units == ['m', '', 'kg']
    
    def test_set_units_with_dict_partial(self):
        """Test set_units() with dict updates only specified columns."""
        df = ap.DataFrame({'A': [1], 'B': [2]}, units=['old_A', 'old_B'])
        
        df.set_units({'A': 'new_A'})
        assert df.units == ['new_A', 'old_B']
    
    def test_set_units_with_list(self):
        """Test set_units() with list."""
        df = ap.DataFrame({'A': [1], 'B': [2]})
        
        df.set_units(['m', 's'])
        assert df.units == ['m', 's']
    
    def test_set_units_list_wrong_length(self):
        """Test set_units() with wrong length list raises error."""
        df = ap.DataFrame({'A': [1], 'B': [2]})
        
        with pytest.raises(ValueError, match="Length of units"):
            df.set_units(['m'])
    
    def test_set_units_invalid_type(self):
        """Test set_units() with invalid type raises error."""
        df = ap.DataFrame({'A': [1]})
        
        with pytest.raises(TypeError, match="units must be a dict or list"):
            df.set_units('invalid')


class TestDataFrameInitialization:
    """Test DataFrame initialization edge cases."""
    
    def test_init_units_wrong_length(self):
        """Test initialization with wrong units length raises error."""
        with pytest.raises(ValueError, match="Length of units"):
            ap.DataFrame({'A': [1], 'B': [2]}, units=['m'])
    
    def test_init_units_invalid_type(self):
        """Test initialization with invalid units type raises error."""
        with pytest.raises(TypeError, match="units must be a list, dict, or None"):
            ap.DataFrame({'A': [1]}, units='invalid')
    
    def test_init_empty_dataframe(self):
        """Test initialization of empty DataFrame."""
        df = ap.DataFrame()
        assert df.units == []
        
        df_with_units = ap.DataFrame(units=[])
        assert df_with_units.units == []


class TestPandasCompatibility:
    """Test compatibility with pandas operations."""
    
    def test_no_warnings_basic_operations(self):
        """Test that basic operations don't generate pandas warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            df = ap.DataFrame({
                'temperature': [20, 25, 30],
                'pressure': [1013, 1015, 1010]
            }, units=['°C', 'hPa'])
            
            # Perform basic operations
            _ = df.units
            _ = df.unit()
            _ = df.unit('temperature')
            df.set_units(['K', 'Pa'])
            
            # Check for pandas warnings
            pandas_warnings = [warning for warning in w if 'Pandas' in str(warning.message)]
            assert len(pandas_warnings) == 0
    
    def test_column_selection_preserves_type(self):
        """Test that column selection preserves DataFrame type."""
        df = ap.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        }, units=['m', 's', 'kg'])
        
        subset = df[['A', 'C']]
        assert isinstance(subset, ap.DataFrame)
        assert hasattr(subset, 'units')
    
    def test_mathematical_operations_preserve_type(self):
        """Test that mathematical operations preserve DataFrame type."""
        df = ap.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        }, units=['m', 's'])
        
        df2 = df * 2
        assert isinstance(df2, ap.DataFrame)
        assert hasattr(df2, 'units')
    
    def test_indexing_operations(self):
        """Test various indexing operations."""
        df = ap.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        }, units=['m', 's'])
        
        # Row indexing
        row = df.iloc[0]
        assert isinstance(row, pd.Series)
        
        # Boolean indexing
        filtered = df[df['A'] > 1]
        assert isinstance(filtered, ap.DataFrame)
        assert hasattr(filtered, 'units')


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    def test_environmental_data_example(self):
        """Test with environmental monitoring data like in the demo."""
        np.random.seed(42)
        n_points = 5
        
        env_data = ap.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_points, freq='h'),
            'air_temp': 20 + np.random.normal(0, 2, n_points),
            'rel_humidity': 60 + np.random.normal(0, 10, n_points),
            'atm_pressure': 1013 + np.random.normal(0, 5, n_points),
            'wind_speed': np.abs(np.random.normal(3, 1, n_points)),
            'solar_irrad': np.maximum(0, 800 + np.random.normal(0, 200, n_points))
        }, units=['', '°C', '%', 'hPa', 'm/s', 'W/m²'])
        
        assert len(env_data) == n_points
        assert env_data.unit('air_temp') == {'air_temp': '°C'}
        assert env_data.unit(['air_temp', 'rel_humidity']) == {'air_temp': '°C', 'rel_humidity': '%'}
        
        # Test that we can get weather-related units
        weather_units = env_data.unit(['air_temp', 'rel_humidity', 'atm_pressure'])
        expected_weather = {'air_temp': '°C', 'rel_humidity': '%', 'atm_pressure': 'hPa'}
        assert weather_units == expected_weather
    
    def test_unit_conversion_scenario(self):
        """Test unit conversion scenario from demo."""
        measurement_data = ap.DataFrame({
            'temp_celsius': [0, 10, 20, 30, 40],
            'temp_kelvin': [273.15, 283.15, 293.15, 303.15, 313.15],
            'pressure_atm': [1.0, 1.1, 0.9, 1.2, 0.8]
        }, units=['°C', 'K', 'atm'])
        
        # Add new column with conversion
        measurement_data['pressure_pa'] = measurement_data['pressure_atm'] * 101325
        
        # Update units to include new column
        current_units = measurement_data.units.copy()
        current_units.append('Pa')
        measurement_data.units = current_units
        
        assert len(measurement_data.units) == 4
        assert measurement_data.units[-1] == 'Pa'
        assert measurement_data.unit('pressure_pa') == {'pressure_pa': 'Pa'}
    
    def test_dynamic_column_addition(self):
        """Test adding columns and updating units dynamically."""
        df = ap.DataFrame({
            'length': [1, 2, 3],
            'width': [2, 3, 4]
        }, units=['m', 'm'])
        
        # Add calculated column
        df['area'] = df['length'] * df['width']
        
        # Update units to include new column
        df.units = df.units + ['m²']
        
        assert len(df.units) == 3
        assert df.units == ['m', 'm', 'm²']
        assert df.unit('area') == {'area': 'm²'}


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_column_names(self):
        """Test with empty or unusual column names."""
        df = ap.DataFrame({
            '': [1, 2],
            ' ': [3, 4]
        }, units=['unit1', 'unit2'])
        
        assert df.units == ['unit1', 'unit2']
        assert df.unit('') == {'': 'unit1'}
        assert df.unit(' ') == {' ': 'unit2'}
    
    def test_unicode_units(self):
        """Test with unicode characters in units."""
        df = ap.DataFrame({'temp': [20]}, units=['°C'])
        assert df.units == ['°C']
        assert df.unit('temp') == {'temp': '°C'}
    
    def test_long_unit_strings(self):
        """Test with long unit strings."""
        long_unit = 'kg⋅m²⋅s⁻²⋅A⁻¹'
        df = ap.DataFrame({'energy': [100]}, units=[long_unit])
        assert df.units == [long_unit]
        assert df.unit('energy') == {'energy': long_unit}


def test_convenience_function():
    """Test the convenience function dataframe_with_units."""
    df = dataframe_with_units({
        'A': [1, 2],
        'B': [3, 4]
    }, units=['m', 's'])
    
    assert isinstance(df, ap.DataFrame)
    assert df.units == ['m', 's']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])