# -*- coding: utf-8 -*-
"""
Property-based tests for parametric analysis using Hypothesis.

These tests automatically generate edge cases and verify invariants.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from unittest.mock import Mock
from typing import Any

from hypothesis import given, strategies as st, assume, settings, example
from hypothesis.extra.pandas import data_frames, column, range_indexes

from antupy.core import Var, Array, Simulation
from antupy.analyser.par import Parametric, ParametricSettings, ParameterValue


class MockSimulation(Simulation):
    """Mock simulation for hypothesis testing."""
    
    def __init__(self):
        super().__init__()
        self.out = {
            "efficiency": 0.85,
            "power": 1000.0,
            "cost": 50.0,
            "temperature": 25.0
        }
        # Dynamic attributes for parameter updating
        
    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
    
    def run_simulation(self, verbose=False):
        """Mock simulation that produces realistic outputs."""
        # Create some realistic relationships between inputs and outputs
        if hasattr(self, 'input_temp') and getattr(self, 'input_temp', None) is not None:
            input_temp_attr = getattr(self, 'input_temp')
            temp_val = input_temp_attr.v if hasattr(input_temp_attr, 'v') else input_temp_attr
            # Efficiency decreases with very high or very low temperatures
            if isinstance(temp_val, (int, float)):
                self.out["efficiency"] = max(0.1, min(0.95, 0.8 - abs(temp_val - 25) * 0.01))
                self.out["temperature"] = temp_val
        
        if hasattr(self, 'power_rating') and getattr(self, 'power_rating', None) is not None:
            power_rating_attr = getattr(self, 'power_rating')
            power_val = power_rating_attr.v if hasattr(power_rating_attr, 'v') else power_rating_attr
            if isinstance(power_val, (int, float)) and power_val > 0:
                self.out["power"] = power_val * self.out["efficiency"]
                self.out["cost"] = power_val * 0.05  # $0.05 per watt


# Hypothesis strategies for generating test data
@st.composite
def parameter_values(draw):
    """Strategy for generating parameter values (lists or Arrays)."""
    param_type = draw(st.sampled_from(['list', 'array']))
    
    if param_type == 'list':
        # Generate lists of various types
        value_type = draw(st.sampled_from(['float', 'int', 'str']))
        if value_type == 'float':
            return draw(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), 
                                min_size=2, max_size=10))  # Ensure min_size > 1 to avoid empty after filtering
        elif value_type == 'int':
            return draw(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=2, max_size=10))
        else:  # str
            return draw(st.lists(st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=['Lu', 'Ll'])), 
                                min_size=2, max_size=5))
    else:  # array
        # Generate Array objects
        values = draw(st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False), 
                              min_size=2, max_size=10))  # Ensure min_size > 1
        unit = draw(st.sampled_from(['°C', 'K', 'W', 'm/s', 'm2/s', 'm']))  # Use valid units only
        if unit:
            return Array(values, unit)
        else:
            return Array(values)  # No unit parameter when None


@st.composite
def parameter_dict(draw):
    """Strategy for generating parameter dictionaries."""
    # Generate parameter names - ensure we get at least 1
    param_names = draw(st.lists(
        st.text(min_size=3, max_size=20, alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'),
        min_size=2, max_size=5, unique=True  # Start with min_size=2 to ensure we have parameters
    ))
    
    # Generate values for each parameter
    params = {}
    for name in param_names:
        params[name] = draw(parameter_values())
    
    return params


@st.composite 
def parametric_settings_strategy(draw):
    """Strategy for generating ParametricSettings."""
    params_out = draw(st.lists(
        st.text(min_size=3, max_size=15, alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'),
        min_size=1, max_size=8, unique=True
    ))
    
    return ParametricSettings(
        params_out=params_out,
        save_results_detailed=draw(st.booleans()),
        dir_output=None,  # Keep simple for testing
        path_results=None,
        verbose=draw(st.booleans())
    )


class TestParametricHypothesis:
    """Property-based tests for Parametric class."""
    
    @given(parameter_dict())
    @settings(max_examples=50, deadline=5000)  # Reasonable limits for CI
    def test_setup_cases_always_produces_valid_dataframe(self, params):
        """Property: setup_cases should always produce a valid DataFrame."""
        # No need for assume() statements since we built constraints into the strategy
        
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim)
        
        try:
            cases, units = parametric.setup_cases(params)  # type: ignore
            
            # Properties that should always hold
            assert isinstance(cases, pd.DataFrame)
            assert len(cases) > 0
            assert len(cases.columns) == len(params)
            assert list(cases.columns) == list(params.keys())
            assert isinstance(units, dict)
            assert set(units.keys()) == set(params.keys())
            
            # Check that cartesian product is correct
            expected_rows = 1
            for values in params.values():
                if hasattr(values, '__len__'):
                    expected_rows *= len(values)
            assert len(cases) == expected_rows
            
        except Exception as e:
            # If we get an exception, it should be a reasonable one
            assert isinstance(e, (ValueError, TypeError, AttributeError)), f"Unexpected exception: {e}"
    
    @given(st.lists(st.floats(min_value=0, max_value=100, allow_nan=False), min_size=1, max_size=20))
    @example([20.0, 25.0, 30.0])  # Ensure we test a known good case
    def test_temperature_parameter_processing(self, temperatures):
        """Property: Temperature parameters should be processed correctly."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim)
        
        params: dict[str, ParameterValue] = {'input_temp': Array(temperatures, '°C')}
        
        cases, units = parametric.setup_cases(params)
        
        # Properties
        assert len(cases) == len(temperatures)
        assert units['input_temp'] == '°C'
        assert list(cases['input_temp']) == temperatures
    
    @given(parametric_settings_strategy())
    def test_parametric_initialization_with_random_settings(self, settings):
        """Property: Parametric should initialize correctly with any valid settings."""
        mock_sim = MockSimulation()
        
        parametric = Parametric(mock_sim, settings)
        
        assert parametric.base_case is mock_sim
        assert parametric.settings is settings
        assert parametric.cases is None
        assert parametric.units is None
        assert parametric.results is None
    
    @given(st.lists(st.floats(min_value=1, max_value=1000, allow_nan=False), min_size=1, max_size=10))
    @settings(max_examples=20)  # Limit for performance
    def test_analysis_output_properties(self, power_values):
        """Property: Analysis should produce results with expected properties."""
        mock_sim = MockSimulation()
        settings = ParametricSettings(params_out=['efficiency', 'power', 'cost'])
        parametric = Parametric(mock_sim, settings)
        
        params: dict[str, ParameterValue] = {'power_rating': Array(power_values, 'W')}
        
        cases, units = parametric.setup_cases(params)
        results = parametric.run_analysis()
        
        # Properties that should always hold
        assert len(results) == len(power_values)
        assert 'power_rating' in results.columns
        assert 'efficiency' in results.columns
        assert 'power' in results.columns
        assert 'cost' in results.columns
        
        # Physics-based properties
        for _, row in results.iterrows():
            # Efficiency should be between 0 and 1
            assert 0 <= row['efficiency'] <= 1
            # Power output should be less than or equal to power rating
            assert row['power'] <= row['power_rating']
            # Cost should be positive
            assert row['cost'] >= 0
    
    @given(st.integers(min_value=1, max_value=100))
    def test_run_analysis_handles_various_case_counts(self, n_cases):
        """Property: run_analysis should handle any reasonable number of cases."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim)
        
        # Create n_cases by using range
        params: dict[str, ParameterValue] = {'case_id': list(range(n_cases))}
        
        cases, units = parametric.setup_cases(params)
        results = parametric.run_analysis()
        
        assert len(results) == n_cases
        assert len(results.columns) >= 1  # At least the input parameter
    
@st.composite
def dot_notation_names(draw):
    """Generate parameter names with dot notation."""
    # Generate 2-4 parts separated by dots
    n_parts = draw(st.integers(min_value=2, max_value=4))
    parts = []
    for _ in range(n_parts):
        part = draw(st.text(
            min_size=1, max_size=10,
            alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
        ))
        parts.append(part)
    return '.'.join(parts)

    @given(
        dot_notation_names(),
        st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False), min_size=1, max_size=10)
    )
    def test_parameter_names_with_dots(self, param_name, values):
        """Property: Parameter names with dots should be handled correctly."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim)
        
        params: dict[str, ParameterValue] = {param_name: values}
        
        try:
            cases, units = parametric.setup_cases(params)
            assert param_name in cases.columns
            assert param_name in units
        except (ValueError, AttributeError):
            # Some parameter names might not be valid - that's OK
            pass


class TestEdgeCaseDiscovery:
    """Tests specifically designed to discover edge cases."""
    
    @given(data_frames(
        columns=[
            column('param1', elements=st.floats(min_value=-1e10, max_value=1e10, allow_nan=False)),
            column('param2', elements=st.integers(min_value=-1000, max_value=1000)),
        ],
        index=range_indexes(min_size=1, max_size=100)
    ))
    @settings(max_examples=20)
    def test_analysis_with_extreme_dataframes(self, df):
        """Test analysis with DataFrames that might have extreme values."""
        mock_sim = MockSimulation()
        settings = ParametricSettings(params_out=['efficiency'])
        parametric = Parametric(mock_sim, settings)
        
        units: dict[str, str | None] = {'param1': None, 'param2': None}
        
        try:
            results = parametric.run_analysis(df, units)
            
            # Basic sanity checks
            assert len(results) == len(df)
            assert 'param1' in results.columns
            assert 'param2' in results.columns
            assert 'efficiency' in results.columns
            
        except (OverflowError, ValueError) as e:
            # Some extreme values might cause legitimate errors
            # This is actually useful information about the limits of our system
            print(f"Found limitation with extreme values: {e}")
    
    @given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=50))
    def test_performance_with_varying_sizes(self, values):
        """Test that performance doesn't degrade catastrophically with size."""
        
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim)
        
        params: dict[str, ParameterValue] = {'values': values}
        
        import time
        start_time = time.time()
        
        cases, units = parametric.setup_cases(params)
        setup_time = time.time() - start_time
        
        # Setup time should be reasonable (less than 1 second for reasonable inputs)
        if len(values) < 100:
            assert setup_time < 1.0, f"Setup took {setup_time:.2f}s for {len(values)} values"
    
    @given(st.lists(st.text(min_size=1, max_size=100, alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '), min_size=2, max_size=10))
    def test_string_parameter_edge_cases(self, string_values):
        """Test handling of various string parameter edge cases."""
        # No need to filter - generate only valid strings from the start
        
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim)
        
        params: dict[str, ParameterValue] = {'text_param': string_values}
        
        cases, units = parametric.setup_cases(params)
        results = parametric.run_analysis()
        
        assert len(results) == len(string_values)
        assert units['text_param'] is None  # String parameters shouldn't have units


def test_hypothesis_finds_real_bug():
    """
    Example of how Hypothesis can find real bugs by testing properties.
    This test demonstrates the power of property-based testing.
    """
    mock_sim = MockSimulation()
    parametric = Parametric(mock_sim)
    
    # This will try many different parameter combinations automatically
    @given(parameter_dict())
    @settings(max_examples=100)
    def property_test(params):
        # No need for assume() statements since we built constraints into the strategy
        
        cases, units = parametric.setup_cases(params)  # type: ignore
        
        # Property: All columns in cases should have the expected names
        assert set(cases.columns) == set(params.keys())
        
        # Property: Units dict should have same keys as params
        assert set(units.keys()) == set(params.keys())
        
        # Property: No NaN values in input columns
        for col in cases.columns:
            assert not cases[col].isna().any(), f"Found NaN in column {col}"
    
    # Run the property test
    property_test()


if __name__ == "__main__":
    # Run a quick demonstration
    print("Running Hypothesis property-based tests...")
    pytest.main([__file__, "-v", "--tb=short"])
