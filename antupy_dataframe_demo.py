"""
Comprehensive example of antupy.DataFrame usage.

This script demonstrates all the features of the custom DataFrame with units support.
"""

import antupy as ap
import pandas as pd
import numpy as np


def main():
    print("🔬 antupy.DataFrame - Custom DataFrame with Units Support")
    print("=" * 60)
    
    # 1. Basic creation with list units
    print("\n1. Creating DataFrame with list units:")
    df1 = ap.DataFrame({
        'temperature': [20.5, 25.0, 30.2, 18.7, 22.3],
        'pressure': [1013.25, 1015.0, 1010.5, 1020.2, 1012.8],
        'humidity': [60, 65, 70, 55, 58]
    }, units=['°C', 'hPa', '%'])
    
    print(df1)
    print(f"Units: {df1.units}")
    print(f"Units as dict: {df1.unit()}")
    
    # 2. Creating with dict units
    print("\n2. Creating DataFrame with dict units:")
    df2 = ap.DataFrame({
        'velocity': [10, 15, 20],
        'acceleration': [2.5, 3.0, 1.8]
    }, units={'velocity': 'm/s', 'acceleration': 'm/s²'})
    
    print(df2)
    print(f"Velocity unit: {df2.unit('velocity')}")
    print(f"Multiple units: {df2.unit(['velocity', 'acceleration'])}")
    
    # 3. Setting units after creation
    print("\n3. Setting units after DataFrame creation:")
    df3 = ap.DataFrame({
        'length': [1.5, 2.0, 3.2],
        'width': [0.8, 1.2, 1.5],
        'area': [1.2, 2.4, 4.8]
    })
    
    print("Before setting units:")
    print(f"Units: {df3.units}")
    
    # Set units using dict
    df3.set_units({'length': 'm', 'width': 'm', 'area': 'm²'})
    print("\nAfter setting units with dict:")
    print(f"Units: {df3.units}")
    
    # Set units using list
    df3.set_units(['cm', 'cm', 'cm²'])
    print("\nAfter changing to cm units:")
    print(f"Units: {df3.units}")
    
    # 4. Working with pandas operations
    print("\n4. Pandas operations (units are preserved where possible):")
    df4 = ap.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [10, 20, 30, 40],
        'C': [0.1, 0.2, 0.3, 0.4]
    }, units=['kg', 'N', 'm/s'])
    
    print("Original DataFrame:")
    print(df4)
    print(f"Units: {df4.units}")
    
    # Select columns
    subset = df4[['A', 'C']]
    print(f"\nSubset columns A,C:")
    print(subset)
    print(f"Subset units: {subset.units if hasattr(subset, 'units') else 'Not preserved'}")
    
    # Mathematical operations
    df4['D'] = df4['A'] * 2  # This creates a new column without units
    print(f"\nAfter adding column D (A*2):")
    print(df4)
    print(f"Units (new column has no unit): {df4.units}")
    
    # 5. Practical example: Scientific data
    print("\n5. Practical example - Environmental monitoring data:")
    
    # Simulate some environmental data
    np.random.seed(42)
    n_points = 10
    
    env_data = ap.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_points, freq='h'),
        'air_temp': 20 + np.random.normal(0, 2, n_points),
        'rel_humidity': 60 + np.random.normal(0, 10, n_points),
        'atm_pressure': 1013 + np.random.normal(0, 5, n_points),
        'wind_speed': np.abs(np.random.normal(3, 1, n_points)),
        'solar_irrad': np.maximum(0, 800 + np.random.normal(0, 200, n_points))
    }, units=['', '°C', '%', 'hPa', 'm/s', 'W/m²'])
    
    print(env_data)
    print(f"\nData types and units:")
    for col, unit in zip(env_data.columns, env_data.units):
        print(f"  {col}: {env_data[col].dtype} [{unit}]")
    
    # Get specific measurements with their units
    print(f"\nTemperature data: {env_data.unit('air_temp')}")
    print(f"Weather units: {env_data.unit(['air_temp', 'rel_humidity', 'atm_pressure'])}")
    
    # 6. Advanced usage
    print("\n6. Advanced usage - Unit conversion example:")
    
    # Create data in different units
    measurement_data = ap.DataFrame({
        'temp_celsius': [0, 10, 20, 30, 40],
        'temp_kelvin': [273.15, 283.15, 293.15, 303.15, 313.15],
        'pressure_atm': [1.0, 1.1, 0.9, 1.2, 0.8]
    }, units=['°C', 'K', 'atm'])
    
    print(measurement_data)
    print(f"Original units: {measurement_data.unit()}")
    
    # Convert to different units (manual conversion for demonstration)
    measurement_data['pressure_pa'] = measurement_data['pressure_atm'] * 101325
    # Update units to include the new column
    current_units = measurement_data.units.copy()
    current_units.append('Pa')
    measurement_data.units = current_units
    
    print(f"\nAfter adding pressure in Pascals:")
    print(measurement_data)
    print(f"Updated units: {measurement_data.unit()}")
    
    print("\n✅ antupy.DataFrame demonstration complete!")
    print("\nKey features:")
    print("- ✓ Inherits all pandas DataFrame functionality")
    print("- ✓ Adds .units attribute (list of units for each column)")
    print("- ✓ Provides .unit() method to query units")
    print("- ✓ Provides .set_units() method to update units")
    print("- ✓ Supports both list and dict formats for units")
    print("- ✓ Perfect for scientific computing with physical quantities")


if __name__ == "__main__":
    main()