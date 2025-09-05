# -*- coding: utf-8 -*-
"""
Unit tests for the parametric analysis module.
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
from antupy.analyser.par import Parametric, ParametricSettings, settings, analysis, ParameterValue


class MockSimulation:
    """Mock simulation class for testing."""
    
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


class TestParametricSettings:
    """Test cases for ParametricSettings dataclass."""
    
    def test_default_settings(self):
        """Test default settings initialization."""
        settings = ParametricSettings()
        
        assert len(settings.params_out) == 4
        assert "eta_something" in settings.params_out
        assert settings.save_results_detailed is False
        assert settings.dir_output is None
        assert settings.path_results is None
        assert settings.verbose is True
    
    def test_custom_settings(self):
        """Test custom settings initialization."""
        custom_params = ["efficiency", "cost"]
        settings = ParametricSettings(
            params_out=custom_params,
            save_results_detailed=True,
            dir_output="test_output",
            verbose=False
        )
        
        assert settings.params_out == custom_params
        assert settings.save_results_detailed is True
        assert isinstance(settings.dir_output, Path)
        assert settings.dir_output.name == "test_output"
        assert settings.verbose is False
    
    def test_path_conversion(self):
        """Test automatic path conversion in __post_init__."""
        settings = ParametricSettings(
            dir_output="/tmp/test",
            path_results="/tmp/results.csv"
        )
        
        assert isinstance(settings.dir_output, Path)
        assert isinstance(settings.path_results, Path)
        assert str(settings.dir_output).endswith("test")


class TestParametric:
    """Test cases for Parametric class."""
    
    @pytest.fixture
    def mock_simulation(self):
        """Create a mock simulation for testing."""
        return MockSimulation()
    
    @pytest.fixture
    def parametric_instance(self, mock_simulation):
        """Create a Parametric instance for testing."""
        return Parametric(mock_simulation)
    
    @pytest.fixture
    def sample_parameters(self):
        """Sample parameters for testing."""
        return {
            'subsystem.temperature': Array([20.0, 25.0, 30.0], '°C'),
            'subsystem.flow_rate': Array([0.1, 0.2], 'm3/s'),
            'system_type': ['A', 'B']
        }
    
    def test_initialization(self, mock_simulation):
        """Test Parametric class initialization."""
        parametric = Parametric(mock_simulation)
        
        assert parametric.base_case is mock_simulation
        assert isinstance(parametric.settings, ParametricSettings)
        assert parametric.cases is None
        assert parametric.units is None
        assert parametric.results is None
    
    def test_initialization_with_custom_settings(self, mock_simulation):
        """Test initialization with custom settings."""
        custom_settings = ParametricSettings(params_out=["efficiency"])
        parametric = Parametric(mock_simulation, custom_settings)
        
        assert parametric.settings is custom_settings
        assert parametric.settings.params_out == ["efficiency"]
    
    def test_setup_cases(self, parametric_instance, sample_parameters):
        """Test case setup with various parameter types."""
        cases, units = parametric_instance.setup_cases(sample_parameters)
        
        # Check dimensions (3 temps × 2 flows × 2 types = 12 cases)
        assert len(cases) == 12
        assert len(cases.columns) == 3
        
        # Check columns
        expected_cols = ['subsystem.temperature', 'subsystem.flow_rate', 'system_type']
        assert list(cases.columns) == expected_cols
        
        # Check units
        assert units['subsystem.temperature'] == '°C'
        assert units['subsystem.flow_rate'] == 'm3/s'
        assert units['system_type'] is None
        
        # Check that cases are stored in instance
        assert parametric_instance.cases is not None
        assert parametric_instance.units is not None
        pd.testing.assert_frame_equal(parametric_instance.cases, cases)
    
    def test_setup_cases_with_non_array_values(self, parametric_instance):
        """Test setup with non-Array values."""
        params = {
            'param1': [1, 2, 3],
            'param2': ['x', 'y']
        }
        
        cases, units = parametric_instance.setup_cases(params)
        
        assert len(cases) == 6  # 3 × 2 = 6 cases
        assert units['param1'] is None
        assert units['param2'] is None
    
    def test_run_analysis_without_setup(self, parametric_instance):
        """Test running analysis without setting up cases first."""
        with pytest.raises(ValueError, match="No cases defined"):
            parametric_instance.run_analysis()
    
    def test_run_analysis_basic(self, parametric_instance, sample_parameters):
        """Test basic analysis run."""
        # Setup cases
        parametric_instance.setup_cases(sample_parameters)
        
        # Run analysis
        results = parametric_instance.run_analysis()
        
        # Check results structure
        assert len(results) == 12
        assert len(results.columns) == 7  # 3 input + 4 output columns
        
        # Check that output columns exist and are not all NaN
        for param in parametric_instance.settings.params_out:
            assert param in results.columns
            assert not results[param].isna().all()
        
        # Check that results are stored
        assert parametric_instance.results is not None
        pd.testing.assert_frame_equal(parametric_instance.results, results)
    
    def test_run_analysis_with_provided_cases(self, parametric_instance):
        """Test analysis with externally provided cases."""
        cases = pd.DataFrame({
            'subsystem.temperature': [20.0, 30.0],
            'param2': [1, 2]
        })
        units = {'subsystem.temperature': '°C', 'param2': None}
        
        results = parametric_instance.run_analysis(cases, units)
        
        assert len(results) == 2
        assert 'subsystem.temperature' in results.columns
        assert 'param2' in results.columns
    
    def test_update_parameters_simple(self, parametric_instance):
        """Test parameter updating for simple attributes."""
        sim = MockSimulation()
        row = pd.Series({'simple_param': 42})
        units = {'simple_param': None}
        
        parametric_instance._update_parameters(sim, row, units)
        
        # Check that the attribute was set using getattr
        assert hasattr(sim, 'simple_param')
        assert getattr(sim, 'simple_param') == 42
    
    def test_update_parameters_nested(self, parametric_instance):
        """Test parameter updating for nested attributes."""
        sim = MockSimulation()
        row = pd.Series({
            'subsystem.temperature': 25.0,
            'subsystem.flow_rate': 0.15
        })
        units = {
            'subsystem.temperature': '°C',
            'subsystem.flow_rate': 'm3/s'
        }
        
        parametric_instance._update_parameters(sim, row, units)
        
        # Check that nested attributes were set correctly
        assert isinstance(sim.subsystem.temperature, Var)
        assert sim.subsystem.temperature.v == 25.0
        assert sim.subsystem.temperature.u == '°C'
        
        assert isinstance(sim.subsystem.flow_rate, Var)
        assert sim.subsystem.flow_rate.v == 0.15
        assert sim.subsystem.flow_rate.u == 'm3/s'
    
    def test_update_parameters_nested_without_units(self, parametric_instance):
        """Test nested parameter updating without units."""
        sim = MockSimulation()
        row = pd.Series({'subsystem.param': 100})
        units = {'subsystem.param': None}
        
        parametric_instance._update_parameters(sim, row, units)
        
        assert sim.subsystem.param == 100
        assert not isinstance(sim.subsystem.param, Var)
    
    def test_save_results_without_analysis(self, parametric_instance):
        """Test saving results before running analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_results.csv"
            
            with pytest.raises(ValueError, match="No results to save"):
                parametric_instance.save_results(filepath)
    
    def test_save_results(self, parametric_instance, sample_parameters):
        """Test saving results to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_results.csv"
            
            # Run analysis first
            parametric_instance.setup_cases(sample_parameters)
            parametric_instance.run_analysis()
            
            # Save results
            parametric_instance.save_results(filepath)
            
            # Verify file was created and contains expected data
            assert filepath.exists()
            
            saved_data = pd.read_csv(filepath)
            assert len(saved_data) == 12
            assert 'subsystem.temperature' in saved_data.columns
    
    def test_get_summary_no_analysis(self, parametric_instance):
        """Test summary generation without analysis."""
        summary = parametric_instance.get_summary()
        
        assert summary["status"] == "No analysis completed"
    
    def test_get_summary_with_analysis(self, parametric_instance, sample_parameters):
        """Test summary generation after analysis."""
        # Run analysis
        parametric_instance.setup_cases(sample_parameters)
        parametric_instance.run_analysis()
        
        summary = parametric_instance.get_summary()
        
        # Check basic information
        assert summary["total_cases"] == 12
        assert summary["completed"] is True
        assert len(summary["input_parameters"]) == 3
        assert len(summary["output_parameters"]) == 4
        
        # Check that statistics are included for output parameters
        assert "eta_something_stats" in summary
        stats = summary["eta_something_stats"]
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
    
    @patch('pickle.dump')
    def test_detailed_results_saving(self, mock_pickle_dump, parametric_instance, sample_parameters):
        """Test saving detailed simulation results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = ParametricSettings(
                save_results_detailed=True,
                dir_output=tmpdir
            )
            parametric = Parametric(parametric_instance.base_case, settings)
            
            # Run analysis with limited cases for faster testing
            limited_params: dict[str, ParameterValue] = {
                'subsystem.temperature': Array([20.0, 25.0], '°C')
            }
            parametric.setup_cases(limited_params)
            parametric.run_analysis()
            
            # Verify pickle.dump was called for each case
            assert mock_pickle_dump.call_count == 2


class TestLegacyFunctions:
    """Test legacy function interfaces."""
    
    def test_settings_function(self):
        """Test legacy settings function."""
        params = {
            'temp': Array([20, 30], '°C'),
            'size': ['small', 'large']
        }
        
        cases, units = settings(params)
        
        assert len(cases) == 4  # 2 × 2 = 4 cases
        assert units['temp'] == '°C'
        assert units['size'] is None
    
    def test_analysis_function(self):
        """Test legacy analysis function."""
        mock_sim = MockSimulation()
        
        cases = pd.DataFrame({
            'subsystem.temperature': [20.0, 30.0]
        })
        units: dict[str, str | None] = {'subsystem.temperature': '°C'}
        
        # Use type ignore to bypass strict typing for test mock
        results = analysis(
            cases_in=cases,
            units_in=units,
            base_case=mock_sim,  # type: ignore
            verbose=False
        )
        
        assert len(results) == 2
        assert 'eta_something' in results.columns
    
    def test_analysis_function_with_none_base_case(self):
        """Test legacy analysis function with None base case."""
        cases = pd.DataFrame({'param': [1, 2]})
        units: dict[str, str | None] = {'param': None}
        
        with pytest.raises(TypeError, match="base_case must be a Simulation or Plant"):
            analysis(cases_in=cases, units_in=units, base_case=None)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_workflow(self):
        """Test complete parametric study workflow."""
        # Create mock simulation
        mock_sim = MockSimulation()
        
        # Define parameters
        params: dict[str, ParameterValue] = {
            'subsystem.temperature': Array([20.0, 25.0, 30.0], '°C'),
            'subsystem.flow_rate': Array([0.1, 0.2], 'm3/s')
        }
        
        # Create parametric study (use type ignore for test mock)
        study = Parametric(mock_sim)  # type: ignore
        
        # Setup and run
        cases, units = study.setup_cases(params)
        results = study.run_analysis()
        
        # Verify results make sense
        assert len(results) == 6  # 3 temps × 2 flows
        
        # Check that efficiency varies with temperature (our mock behavior)
        temp_20_results = results[results['subsystem.temperature'] == 20.0]
        temp_30_results = results[results['subsystem.temperature'] == 30.0]
        
        assert temp_30_results['eta_something'].iloc[0] > temp_20_results['eta_something'].iloc[0]
        
        # Check that annual output varies with flow rate
        flow_01_results = results[results['subsystem.flow_rate'] == 0.1]
        flow_02_results = results[results['subsystem.flow_rate'] == 0.2]
        
        assert flow_02_results['annual_something'].iloc[0] > flow_01_results['annual_something'].iloc[0]
    
    def test_error_handling_in_simulation(self):
        """Test error handling when simulation fails."""
        class FailingSimulation(MockSimulation):
            def run_simulation(self, verbose=False):
                raise RuntimeError("Simulation failed")
        
        failing_sim = FailingSimulation()
        study = Parametric(failing_sim)  # type: ignore
        
        params: dict[str, ParameterValue] = {'param': [1, 2]}
        study.setup_cases(params)
        
        # The analysis should raise the simulation error
        with pytest.raises(RuntimeError, match="Simulation failed"):
            study.run_analysis()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
