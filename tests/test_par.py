# -*- coding: utf-8 -*-
"""
Unit tests for the parametric analysis module with simplified API.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import pickle
from unittest.mock import Mock, patch, MagicMock
from typing import Any

from antupy.core import Var, Array
from antupy.analyser.par import Parametric, analysis, ParameterValue


class MockSimulation:
    """Mock simulation class for testing that inherits from expected base."""
    
    def __init__(self):
        self.out = {
            "eta_something": 0.85,
            "annual_something": 1000.0,
            "specific_something": 50.0,
            "average_something": 25.0
        }
        # Nested object for testing dot notation
        self.subsystem = Mock()
        self.subsystem.temperature = None
        self.subsystem.flow_rate = None
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Allow dynamic attribute setting for testing."""
        super().__setattr__(name, value)
        
    def run_simulation(self, verbose=False):
        """Mock simulation run that modifies outputs based on parameters."""
        # Simple mock behavior: efficiency depends on temperature
        if hasattr(self.subsystem, 'temperature') and self.subsystem.temperature:
            temp_val = self.subsystem.temperature.v if hasattr(self.subsystem.temperature, 'v') else self.subsystem.temperature
            self.out["eta_something"] = 0.8 + (temp_val - 20) * 0.01  # Higher temp = higher efficiency
            
        if hasattr(self.subsystem, 'flow_rate') and self.subsystem.flow_rate:
            flow_val = self.subsystem.flow_rate.v if hasattr(self.subsystem.flow_rate, 'v') else self.subsystem.flow_rate
            self.out["annual_something"] = 1000.0 * flow_val  # Annual proportional to flow rate


class TestParametric:
    """Tests for the simplified Parametric class interface."""
    
    def test_parametric_creation(self):
        """Test basic Parametric instance creation."""
        mock_sim = MockSimulation()
        params_out = ["eta_something", "annual_something"]
        
        parametric = Parametric(mock_sim, params_out)
        
        assert parametric.base_case is mock_sim
        assert parametric.params_out == params_out
        assert parametric.verbose == True  # default
        assert parametric.save_results_detailed == False  # default
        assert parametric.dir_output is None  # default
        assert parametric.path_results is None  # default
        assert parametric.cases is None
        assert parametric.units is None
        assert parametric.results is None
    
    def test_parametric_with_custom_settings(self):
        """Test Parametric creation with custom settings."""
        mock_sim = MockSimulation()
        params_out = ["efficiency"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_output = Path(tmpdir)
            path_results = Path(tmpdir) / "results.csv"
            
            parametric = Parametric(
                mock_sim, 
                params_out,
                save_results_detailed=True,
                dir_output=dir_output,
                path_results=path_results,
                verbose=False
            )
            
            assert parametric.params_out == params_out
            assert parametric.save_results_detailed == True
            assert parametric.dir_output == dir_output
            assert parametric.path_results == path_results
            assert parametric.verbose == False

    def test_setup_cases(self):
        """Test case setup from parameters."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim, ["eta_something"])
        
        params = {
            'temperature': Array([20.0, 25.0, 30.0], '°C'),
            'size': ['small', 'medium', 'large']
        }
        
        cases, units = parametric.setup_cases(params)
        
        assert len(cases) == 9  # 3 x 3 combinations
        assert list(cases.columns) == ['temperature', 'size']
        assert units['temperature'] == '°C'
        assert units['size'] is None
        
        # Check that cases are stored in the instance
        assert parametric.cases is not None
        assert parametric.units is not None

    def test_run_analysis(self):
        """Test running complete parametric analysis."""
        mock_sim = MockSimulation()
        parametric = Parametric(
            mock_sim, 
            ["eta_something", "annual_something"],
            verbose=False
        )
        
        params = {
            'subsystem.temperature': Array([20.0, 30.0], '°C'),
            'subsystem.flow_rate': Array([0.5, 1.0], 'm3/s')
        }
        
        results = parametric.run_analysis(params)
        
        # Check structure
        assert len(results) == 4  # 2 x 2 combinations
        assert 'subsystem.temperature' in results.columns
        assert 'subsystem.flow_rate' in results.columns  
        assert 'eta_something' in results.columns
        assert 'annual_something' in results.columns
        
        # Check that simulation parameters affected outputs
        # Higher temperature should give higher efficiency
        temp_20_rows = results[results['subsystem.temperature'] == 20.0]
        temp_30_rows = results[results['subsystem.temperature'] == 30.0]
        
        assert temp_30_rows['eta_something'].iloc[0] > temp_20_rows['eta_something'].iloc[0]

    def test_run_analysis_with_file_saving(self):
        """Test analysis with file output."""
        mock_sim = MockSimulation()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            parametric = Parametric(
                mock_sim,
                params_in,
                ["eta_something"],
                dir_output=tmpdir,
                path_results=Path(tmpdir) / "results.csv",
                save_results_detailed=True,
                verbose=False
            )
            
            params = {'temperature': [20.0, 25.0]}
            results = parametric.run_analysis()
            
            # Check results CSV was created
            assert Path(tmpdir, "results.csv").exists()
            
            # Check detailed simulation files were saved
            assert Path(tmpdir, "sim_0.plk").exists()
            assert Path(tmpdir, "sim_1.plk").exists()

    def test_save_results(self):
        """Test manual results saving."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim, ["eta_something"], verbose=False)
        
        params = {'temperature': [20.0, 25.0]}
        results = parametric.run_analysis(params)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            try:
                parametric.save_results(tmp.name)
                
                # Check file was created and has content
                saved_df = pd.read_csv(tmp.name, index_col=0)
                pd.testing.assert_frame_equal(results, saved_df)
            finally:
                Path(tmp.name).unlink(missing_ok=True)

    def test_get_summary(self):
        """Test summary statistics generation."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim, ["eta_something", "annual_something"], verbose=False)
        
        # Test empty summary
        summary = parametric.get_summary()
        assert summary["status"] == "No analysis completed"
        
        # Test with results
        params = {'temperature': [20.0, 25.0, 30.0]}
        parametric.run_analysis(params)
        
        summary = parametric.get_summary()
        assert summary["total_cases"] == 3
        assert summary["input_parameters"] == ["temperature"]
        assert summary["output_parameters"] == ["eta_something", "annual_something"]
        assert summary["completed"] == True
        assert "eta_something_stats" in summary
        assert "mean" in summary["eta_something_stats"]

    def test_parameter_update_direct_attribute(self):
        """Test updating direct simulation attributes."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim, ["eta_something"])
        
        # Test direct attribute update
        row = pd.Series({'direct_param': 42.0})
        units = {'direct_param': None}
        
        parametric._update_parameters(mock_sim, row, units)
        assert mock_sim.direct_param == 42.0

    def test_parameter_update_nested_attribute(self):
        """Test updating nested simulation attributes with dot notation."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim, ["eta_something"])
        
        # Test nested attribute update with units
        row = pd.Series({'subsystem.temperature': 25.0})
        units = {'subsystem.temperature': '°C'}
        
        parametric._update_parameters(mock_sim, row, units)
        
        assert isinstance(mock_sim.subsystem.temperature, Var)
        assert mock_sim.subsystem.temperature.v == 25.0
        assert mock_sim.subsystem.temperature.u == '°C'

    def test_parameter_update_nested_no_units(self):
        """Test updating nested attributes without units."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim, ["eta_something"])
        
        row = pd.Series({'subsystem.flow_rate': 1.5})
        units = {'subsystem.flow_rate': None}
        
        parametric._update_parameters(mock_sim, row, units)
        assert mock_sim.subsystem.flow_rate == 1.5

    def test_array_parameter_handling(self):
        """Test handling of Array parameters."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim, ["eta_something"], verbose=False)
        
        # Array with units
        temp_array = Array([20.0, 25.0], '°C')
        params = {'temperature': temp_array}
        
        cases, units = parametric.setup_cases(params)
        
        assert len(cases) == 2
        assert units['temperature'] == '°C'
        assert list(cases['temperature']) == [20.0, 25.0]

    def test_mixed_parameter_types(self):
        """Test handling mixed parameter types."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim, ["eta_something"], verbose=False)
        
        params = {
            'numeric_list': [1, 2, 3],
            'string_list': ['a', 'b', 'c'],
            'array_param': Array([10.0, 20.0], 'W'),
            'single_string': ['test']
        }
        
        cases, units = parametric.setup_cases(params)
        
        assert len(cases) == 18  # 3 * 3 * 2 * 1
        assert units['array_param'] == 'W'
        assert units['numeric_list'] is None


class TestLegacyFunction:
    """Tests for the legacy analysis function."""
    
    def test_analysis_function_basic(self):
        """Test basic usage of legacy analysis function."""
        mock_sim = MockSimulation()
        
        # Create test data
        cases_df = pd.DataFrame({
            'temp': [20.0, 25.0],
            'flow': [0.5, 1.0]
        })
        units_dict = {'temp': '°C', 'flow': 'm3/s'}
        
        results = analysis(
            cases_in=cases_df,
            units_in=units_dict,
            params_out=["eta_something"],
            base_case=mock_sim,
            verbose=False
        )
        
        assert len(results) == 4
        assert 'temp' in results.columns
        assert 'flow' in results.columns
        assert 'eta_something' in results.columns

    def test_analysis_function_missing_base_case(self):
        """Test that analysis function requires base_case."""
        cases_df = pd.DataFrame({'temp': [20.0]})
        units_dict = {'temp': '°C'}
        
        with pytest.raises(TypeError, match="base_case must be a Simulation or Plant instance"):
            analysis(
                cases_in=cases_df,
                units_in=units_dict,
                base_case=None
            )

    def test_analysis_function_default_params(self):
        """Test analysis function with default output parameters."""
        mock_sim = MockSimulation()
        
        cases_df = pd.DataFrame({'temp': [20.0]})
        units_dict = {'temp': '°C'}
        
        results = analysis(
            cases_in=cases_df,
            units_in=units_dict,
            base_case=mock_sim,
            verbose=False
        )
        
        # Should have default output parameters
        expected_params = ["eta_something", "annual_something", "specific_something", "average_something"]
        for param in expected_params:
            assert param in results.columns


class TestErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_simulation_error_handling(self):
        """Test handling of simulation errors during analysis."""
        mock_sim = MockSimulation()
        
        # Mock simulation that raises an error
        def failing_run(verbose=False):
            raise RuntimeError("Simulation failed!")
        
        mock_sim.run_simulation = failing_run
        
        parametric = Parametric(mock_sim, ["eta_something"], verbose=False)
        
        params = {'temperature': [20.0]}
        
        # Should not crash but handle the error gracefully
        with patch('builtins.print') as mock_print:
            results = parametric.run_analysis(params)
            
            # Should have printed error message
            mock_print.assert_called()
            error_calls = [call for call in mock_print.call_args_list 
                          if "Error occurred" in str(call)]
            assert len(error_calls) > 0

    def test_output_extraction_error(self):
        """Test handling of output parameter extraction errors."""
        mock_sim = MockSimulation()
        # Remove expected output to cause KeyError
        mock_sim.out = {}
        
        parametric = Parametric(mock_sim, ["missing_param"], verbose=False)
        
        params = {'temperature': [20.0]}
        
        with patch('builtins.print') as mock_print:
            results = parametric.run_analysis(params)
            
            # Should have handled KeyError gracefully
            assert 'missing_param' in results.columns
            # Value should be NaN due to failed extraction
            assert pd.isna(results['missing_param'].iloc[0])

    def test_empty_parameter_dict(self):
        """Test handling of empty parameter dictionary."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim, ["eta_something"])
        
        params = {}
        
        # Should handle empty params gracefully
        cases, units = parametric.setup_cases(params)
        assert len(cases) == 0
        assert len(units) == 0

    def test_invalid_parameter_values(self):
        """Test handling of invalid parameter values."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim, ["eta_something"])
        
        # Test with None values (should be handled by Array class)
        params = {'test_param': [1, 2, None, 4]}
        
        # Should not crash during setup
        cases, units = parametric.setup_cases(params)
        assert len(cases) == 4
        assert pd.isna(cases['test_param'].iloc[2])


if __name__ == "__main__":
    pytest.main([__file__])
