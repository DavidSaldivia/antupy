"""
Test the custom DataFrame implementation.
"""

import antupy as ap
import pandas as pd


def test_basic_dataframe_creation():
    """Test basic DataFrame creation with units."""
    # Test with list units
    df = ap.DataFrame({
        'temperature': [20, 25, 30],
        'pressure': [1013, 1015, 1010]
    }, units=['°C', 'hPa'])
    
    assert hasattr(df, 'units')
    assert df.units == ['°C', 'hPa']
    assert hasattr(df, 'unit')
    assert callable(df.unit)
    
    print("✅ Basic DataFrame creation works")


def test_unit_method():
    """Test the unit() method functionality."""
    df = ap.DataFrame({
        'temp': [20, 25], 
        'pressure': [1013, 1015],
        'humidity': [60, 65]
    }, units=['°C', 'hPa', '%'])
    
    # Test getting all units
    all_units = df.unit()
    expected = {'temp': '°C', 'pressure': 'hPa', 'humidity': '%'}
    assert all_units == expected
    
    # Test getting single column unit
    temp_unit = df.unit('temp')
    assert temp_unit == {'temp': '°C'}
    
    # Test getting multiple columns
    multi_units = df.unit(['temp', 'pressure'])
    expected_multi = {'temp': '°C', 'pressure': 'hPa'}
    assert multi_units == expected_multi
    
    print("✅ Unit method works correctly")


def test_dict_units():
    """Test DataFrame creation with dict units."""
    df = ap.DataFrame({
        'temp': [20, 25],
        'pressure': [1013, 1015]
    }, units={'temp': '°C', 'pressure': 'hPa'})
    
    assert df.units == ['°C', 'hPa']
    assert df.unit() == {'temp': '°C', 'pressure': 'hPa'}
    
    print("✅ Dict units work correctly")


def test_set_units():
    """Test the set_units method."""
    df = ap.DataFrame({
        'temp': [20, 25],
        'pressure': [1013, 1015]
    })
    
    # Initially no units
    assert df.units == ['', '']
    
    # Set units using dict
    df.set_units({'temp': '°C', 'pressure': 'hPa'})
    assert df.units == ['°C', 'hPa']
    
    # Set units using list
    df.set_units(['K', 'Pa'])
    assert df.units == ['K', 'Pa']
    
    print("✅ Set units method works correctly")


def test_pandas_operations():
    """Test that the DataFrame still works as a pandas DataFrame."""
    df = ap.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    }, units=['m', 's'])
    
    # Test basic pandas operations
    assert len(df) == 3
    assert list(df.columns) == ['A', 'B']
    assert df.loc[0, 'A'] == 1
    
    # Test that units are preserved
    assert df.units == ['m', 's']
    
    print("✅ Pandas operations work correctly")


if __name__ == '__main__':
    test_basic_dataframe_creation()
    test_unit_method()
    test_dict_units()
    test_set_units()
    test_pandas_operations()
    
    print("\n🎉 All tests passed! The antupy.DataFrame works correctly.")
    
    # Demo usage
    print("\n📊 Demo usage:")
    df = ap.DataFrame({
        'temperature': [20.5, 25.0, 30.2],
        'pressure': [1013.25, 1015.0, 1010.5],
        'humidity': [60, 65, 70]
    }, units=['°C', 'hPa', '%'])
    
    print(f"DataFrame:\n{df}")
    print(f"\nUnits: {df.units}")
    print(f"All units: {df.unit()}")
    print(f"Temperature unit: {df.unit('temperature')}")
    print(f"Temperature & Pressure units: {df.unit(['temperature', 'pressure'])}")