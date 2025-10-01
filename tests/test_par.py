# -*- coding: utf-8 -*-
"""
Comprehensive unit tests for the enhanced Parametric class with ap.Frame integration.
"""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock
import numpy as np

from antupy import Var, Array, Frame
from antupy.core import Simulation
from antupy.analyser.par import Parametric, ParameterValue


class MockSimulation(Simulation):
    """Enhanced mock simulation class for testing with Var outputs."""
    
    def __init__(self):
        # Initialize with Var objects that have units
        self.out = {
            "efficiency": Var(0.85, "-"),
            "power_output": Var(1000.0, "W"),
            "temperature_out": Var(50.0, "°C"),
            "flow_rate_out": Var(0.15, "m3/s"),
            "eta_something": 0.85,
            "annual_something": 1000.0,
            "specific_something": 50.0,
            "average_something": 25.0
        }
        
        # Nested object for testing dot notation (use simple object instead of Mock)
        class SimpleObj:
            def __init__(self):
                self.temperature = None
                self.pressure = None
                self.flow_rate = None
        
        self.subsystem = SimpleObj()
        
        # Direct attributes
        self.flow_rate = None
        self.operating_mode = None
    
    def __setattr__(self, name: str, value) -> None:
        """Allow dynamic attribute setting for testing."""
        super().__setattr__(name, value)
    
    def run_simulation(self, verbose=False):
        """
        Mock simulation run that modifies outputs based on parameters.
        
        This simulates realistic behavior where simulation outputs
        depend on input parameters and return Var objects with units.
        """
        # Update efficiency based on subsystem temperature
        if hasattr(self.subsystem, 'temperature') and self.subsystem.temperature is not None:
            temp_val = self.subsystem.temperature.v if hasattr(self.subsystem.temperature, 'v') else self.subsystem.temperature
            # Higher temperature = higher efficiency (simple model)
            # Convert from Celsius to efficiency factor
            efficiency_val = 0.6 + temp_val * 0.01  # Efficiency improves with temp (Celsius)
            self.out["efficiency"] = Var(max(0.1, min(0.95, efficiency_val)), "-")
            self.out["eta_something"] = max(0.1, min(0.95, efficiency_val))
            
            # Temperature out depends on input temperature
            temp_out = temp_val + 10  # Output is 10K higher than input
            self.out["temperature_out"] = Var(temp_out, "K")
        
        # Update power output based on flow rate
        if hasattr(self, 'flow_rate') and self.flow_rate:
            flow_val = self.flow_rate.v if hasattr(self.flow_rate, 'v') else self.flow_rate
            # Power proportional to flow rate
            power_val = 800.0 + flow_val * 2000.0  # Base 800W + 2000W per m3/s
            self.out["power_output"] = Var(power_val, "W")
            self.out["annual_something"] = power_val
            
            # Flow rate out is 90% of input (some losses)
            flow_out = flow_val * 0.9
            self.out["flow_rate_out"] = Var(flow_out, "m3/s")
        
        # Update based on operating mode
        if hasattr(self, 'operating_mode') and self.operating_mode:
            if self.operating_mode == "high_performance":
                # Boost all outputs by 10%
                current_eff = self.out["efficiency"].v if hasattr(self.out["efficiency"], 'v') else self.out["efficiency"]
                current_power = self.out["power_output"].v if hasattr(self.out["power_output"], 'v') else self.out["power_output"]
                self.out["efficiency"] = Var(current_eff * 1.1, "-")
                self.out["power_output"] = Var(current_power * 1.1, "W")
                self.out["eta_something"] = current_eff * 1.1
                self.out["annual_something"] = current_power * 1.1
            elif self.operating_mode == "eco_mode":
                # Reduce outputs by 5%
                current_eff = self.out["efficiency"].v if hasattr(self.out["efficiency"], 'v') else self.out["efficiency"]
                current_power = self.out["power_output"].v if hasattr(self.out["power_output"], 'v') else self.out["power_output"]
                self.out["efficiency"] = Var(current_eff * 0.95, "-")
                self.out["power_output"] = Var(current_power * 0.95, "W")
                self.out["eta_something"] = current_eff * 0.95
                self.out["annual_something"] = current_power * 0.95


class TestParametricCreation:
    """Test Parametric class creation and initialization."""
    
    def test_parametric_creation_new_api(self):
        """Test basic Parametric instance creation with new API."""
        mock_sim = MockSimulation()
        params_in = {
            'temperature': Array([300, 350], 'K'),
            'flow_rate': Array([0.1, 0.2], 'm3/s')
        }
        params_out = ["efficiency", "power_output"]
        
        parametric = Parametric(
            base_case=mock_sim,
            params_in=params_in,
            params_out=params_out
        )
        
        assert parametric.base_case is mock_sim
        assert parametric.params_in == params_in
        assert parametric.params_out == params_out
        assert parametric.verbose == True  # default
        assert parametric.save_results_detailed == False  # default
        assert parametric.dir_output is None  # default
        assert parametric.path_results is None  # default
        assert parametric.cases is None
        assert parametric.results is None

    def test_parametric_with_custom_settings(self):
        """Test Parametric creation with custom settings."""
        mock_sim = MockSimulation()
        params_in = {'temperature': Array([20.0, 25.0], '°C')}
        params_out = ["efficiency"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            parametric = Parametric(
                base_case=mock_sim,
                params_in=params_in,
                params_out=params_out,
                save_results_detailed=True,
                dir_output=tmpdir,
                path_results=Path(tmpdir) / "results.csv",
                verbose=False
            )
            
            assert parametric.save_results_detailed == True
            assert parametric.dir_output == Path(tmpdir)
            assert parametric.path_results == Path(tmpdir) / "results.csv"
            assert parametric.verbose == False


class TestSetupCases:
    """Test case setup functionality."""
    
    def test_setup_cases_with_units(self):
        """Test case setup preserves units in ap.Frame."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim, {}, ["efficiency"])
        
        params = {
            'temperature': Array([300.0, 320.0, 340.0], 'K'),
            'flow_rate': Array([0.1, 0.15, 0.2], 'm3/s'),
            'mode': ['normal', 'eco', 'high_performance']
        }
        
        cases = parametric.setup_cases(params)
        
        # Check that it's a Frame with correct structure
        assert isinstance(cases, Frame)
        assert len(cases) == 27  # 3 x 3 x 3 combinations
        assert list(cases.columns) == ['temperature', 'flow_rate', 'mode']
        
        # Check units are preserved
        assert cases.unit('temperature')['temperature'] == 'K'
        assert cases.unit('flow_rate')['flow_rate'] == 'm3/s'
        assert cases.unit('mode')['mode'] == ''  # No unit for strings
        
        # Check units property
        expected_units = {'temperature': 'K', 'flow_rate': 'm3/s', 'mode': ''}
        assert cases.units == expected_units
        
        # Check that cases are stored in the instance
        assert parametric.cases is not None
        assert isinstance(parametric.cases, Frame)

    def test_setup_cases_mixed_types(self):
        """Test case setup with mixed parameter types."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim, {}, ["efficiency"])
        
        params = {
            'temperature': Array([20.0, 25.0, 30.0], '°C'),
            'size': ['small', 'medium', 'large'],
            'count': [1, 2, 3]
        }
        
        cases = parametric.setup_cases(params)
        
        assert len(cases) == 27  # 3 x 3 x 3 combinations
        assert list(cases.columns) == ['temperature', 'size', 'count']
        assert cases.units['temperature'] == '°C'
        assert cases.units['size'] == ''  # No unit for strings
        assert cases.units['count'] == ''  # No unit for plain lists

    def test_empty_params_in(self):
        """Test handling of empty input parameters."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim, {}, ["efficiency"])
        
        cases = parametric.setup_cases({})
        
        assert isinstance(cases, Frame)
        assert len(cases) == 0
        assert len(cases.columns) == 0


class TestOutputExtraction:
    """Test output extraction functionality."""
    
    def test_extract_outputs_with_units(self):
        """Test output extraction preserves units from Var objects."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim, {}, ["efficiency", "power_output"])
        
        # Set up mock simulation with known Var outputs
        mock_sim.out = {
            "efficiency": Var(0.87, "-"),
            "power_output": Var(1250.0, "W"),
            "temperature_out": Var(75.0, "°C"),
        }
        
        params_out = ["efficiency", "power_output", "temperature_out"]
        values, units = parametric._extract_outputs(mock_sim, params_out)
        
        # Check values
        assert values == [0.87, 1250.0, 75.0]
        
        # Check units
        assert units == ["-", "W", "°C"]

    def test_extract_outputs_mixed_types(self):
        """Test output extraction handles mixed output types."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim, {}, ["efficiency"])
        
        # Mix of Var objects, plain numbers, and Arrays
        mock_sim.out = {
            "efficiency": Var(0.87, "-"),
            "plain_number": 42.5,
            "integer_val": 100,
            "array_single": Array([1.5], "m"),
            "array_multi": Array([1.0, 2.0, 3.0], "kg"),
        }
        
        params_out = ["efficiency", "plain_number", "integer_val", "array_single", "array_multi"]
        values, units = parametric._extract_outputs(mock_sim, params_out)
        
        # Check values
        assert values[0] == 0.87  # Var
        assert values[1] == 42.5  # Plain float
        assert values[2] == 100.0  # Integer converted to float
        assert values[3] == 1.5  # Single-element Array
        assert values[4] == 2.0  # Multi-element Array (mean)
        
        # Check units
        assert units == ["-", "", "", "m", "kg"]


class TestParameterUpdate:
    """Test parameter updating functionality."""
    
    def test_update_parameters_with_units(self):
        """Test parameter updating creates Var objects with correct units."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim, {}, ["efficiency"])
        
        case_params = {
            'flow_rate': 0.15,
            'subsystem.temperature': 350.0,
            'operating_mode': 'eco_mode'
        }
        
        input_units = {
            'flow_rate': 'm3/s',
            'subsystem.temperature': 'K',
            'operating_mode': ''  # No unit for string
        }
        
        parametric._update_parameters(mock_sim, case_params, input_units)
        
        # Check direct attribute with unit
        assert isinstance(mock_sim.flow_rate, Var)
        assert mock_sim.flow_rate.v == 0.15
        assert mock_sim.flow_rate.u == 'm3/s'
        
        # Check nested attribute with unit
        assert isinstance(mock_sim.subsystem.temperature, Var)
        assert mock_sim.subsystem.temperature.v == 350.0
        assert mock_sim.subsystem.temperature.u == 'K'
        
        # Check string attribute without unit
        assert mock_sim.operating_mode == 'eco_mode'
        assert not isinstance(mock_sim.operating_mode, Var)

    def test_parameter_update_direct_attribute(self):
        """Test updating direct simulation attributes."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim, {}, ["efficiency"])
        
        case_params = {'direct_param': 42.0}
        input_units = {'direct_param': ''}
        
        parametric._update_parameters(mock_sim, case_params, input_units)
        assert mock_sim.direct_param == 42.0

    def test_parameter_update_nested_attribute(self):
        """Test updating nested simulation attributes with dot notation."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim, {}, ["efficiency"])
        
        case_params = {'subsystem.temperature': 25.0}
        input_units = {'subsystem.temperature': '°C'}
        
        parametric._update_parameters(mock_sim, case_params, input_units)
        
        assert isinstance(mock_sim.subsystem.temperature, Var)
        assert mock_sim.subsystem.temperature.v == 25.0
        assert mock_sim.subsystem.temperature.u == '°C'


class TestRunAnalysis:
    """Test complete analysis workflow."""
    
    def test_run_analysis_complete_workflow(self):
        """Test complete analysis workflow with units preserved."""
        mock_sim = MockSimulation()
        
        params_in = {
            'subsystem.temperature': Array([300, 320], 'K'),
            'flow_rate': Array([0.1, 0.2], 'm3/s')
        }
        params_out = ["efficiency", "power_output", "temperature_out", "flow_rate_out"]
        
        parametric = Parametric(
            base_case=mock_sim,
            params_in=params_in,
            params_out=params_out,
            verbose=False
        )
        
        results = parametric.run_analysis()
        
        # Check that results is a Frame
        assert isinstance(results, Frame)
        
        # Check structure
        assert len(results) == 4  # 2 x 2 combinations
        expected_columns = ['subsystem.temperature', 'flow_rate', 'efficiency', 'power_output', 'temperature_out', 'flow_rate_out']
        assert list(results.columns) == expected_columns
        
        # Check input units preserved
        assert results.unit('subsystem.temperature')['subsystem.temperature'] == 'K'
        assert results.unit('flow_rate')['flow_rate'] == 'm3/s'
        
        # Check output units preserved
        assert results.unit('efficiency')['efficiency'] == '-'
        assert results.unit('power_output')['power_output'] == 'W'
        assert results.unit('temperature_out')['temperature_out'] == 'K'
        assert results.unit('flow_rate_out')['flow_rate_out'] == 'm3/s'
        
        # Check that values are reasonable (mock simulation logic)
        efficiency_values = results['efficiency'].values
        assert all(0.1 <= eff <= 0.95 for eff in efficiency_values if not np.isnan(eff))
        
        power_values = results['power_output'].values
        assert all(800 <= power <= 1200 for power in power_values if not np.isnan(power))

    def test_run_analysis_with_legacy_outputs(self):
        """Test analysis with legacy output names (backward compatibility)."""
        mock_sim = MockSimulation()
        
        params_in = {
            'subsystem.temperature': Array([20.0, 30.0], '°C'),
            'subsystem.flow_rate': Array([0.5, 1.0], 'm3/s')
        }
        params_out = ["eta_something", "annual_something"]
        
        parametric = Parametric(
            base_case=mock_sim,
            params_in=params_in,
            params_out=params_out,
            verbose=False
        )
        
        results = parametric.run_analysis()
        
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


class TestOutputArrays:
    """Test get_output_arrays functionality."""
    
    def test_get_output_arrays(self):
        """Test getting results as Array objects with units."""
        mock_sim = MockSimulation()
        
        params_in = {
            'flow_rate': Array([0.1, 0.15], 'm3/s')
        }
        params_out = ["efficiency", "power_output"]
        
        parametric = Parametric(
            base_case=mock_sim,
            params_in=params_in,
            params_out=params_out,
            verbose=False
        )
        
        results = parametric.run_analysis()
        
        # Test single column
        efficiency_array = parametric.get_output_arrays('efficiency')
        assert isinstance(efficiency_array, Array)
        assert efficiency_array.u == '-'
        assert len(efficiency_array.value) == 2
        
        # Test multiple columns
        multi_arrays = parametric.get_output_arrays(['efficiency', 'power_output'])
        assert isinstance(multi_arrays, dict)
        assert 'efficiency' in multi_arrays
        assert 'power_output' in multi_arrays
        assert isinstance(multi_arrays['efficiency'], Array)
        assert isinstance(multi_arrays['power_output'], Array)
        assert multi_arrays['efficiency'].u == '-'
        assert multi_arrays['power_output'].u == 'W'
        
        # Test all columns
        all_arrays = parametric.get_output_arrays()
        assert isinstance(all_arrays, dict)
        assert len(all_arrays) == 3  # flow_rate (input) + efficiency + power_output
        assert all(isinstance(arr, Array) for arr in all_arrays.values())


class TestSummary:
    """Test summary functionality."""
    
    def test_get_summary_with_units(self):
        """Test summary includes comprehensive unit information."""
        mock_sim = MockSimulation()
        
        params_in = {
            'subsystem.temperature': Array([300, 320], 'K'),
            'operating_mode': ['normal', 'eco_mode']
        }
        params_out = ["efficiency", "power_output"]
        
        parametric = Parametric(
            base_case=mock_sim,
            params_in=params_in,
            params_out=params_out,
            verbose=False
        )
        
        results = parametric.run_analysis()
        summary = parametric.get_summary()
        
        # Check basic info
        assert summary['total_cases'] == 4
        assert summary['input_parameters'] == ['subsystem.temperature', 'operating_mode']
        assert summary['output_parameters'] == ['efficiency', 'power_output']
        assert summary['completed'] == True
        
        # Check unit information
        assert 'input_units' in summary
        assert 'output_units' in summary
        assert summary['input_units']['subsystem.temperature'] == 'K'
        assert summary['input_units']['operating_mode'] == ''
        assert summary['output_units']['efficiency'] == '-'
        assert summary['output_units']['power_output'] == 'W'
        
        # Check statistics with units
        assert 'efficiency_stats' in summary
        assert 'power_output_stats' in summary
        
        eff_stats = summary['efficiency_stats']
        assert 'mean' in eff_stats
        assert 'std' in eff_stats
        assert 'min' in eff_stats
        assert 'max' in eff_stats
        assert 'unit' in eff_stats
        assert eff_stats['unit'] == '-'

    def test_get_summary_empty(self):
        """Test summary before running analysis."""
        mock_sim = MockSimulation()
        parametric = Parametric(mock_sim, {}, ["efficiency"])
        
        summary = parametric.get_summary()
        assert summary["status"] == "No analysis completed"


class TestFileSaving:
    """Test file saving functionality."""
    
    def test_file_saving_functionality(self):
        """Test detailed file saving functionality."""
        mock_sim = MockSimulation()
        
        params_in = {'flow_rate': Array([0.1, 0.15], 'm3/s')}
        params_out = ["efficiency"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            parametric = Parametric(
                base_case=mock_sim,
                params_in=params_in,
                params_out=params_out,
                save_results_detailed=False,  # Disable pickle saving for test
                dir_output=tmpdir,
                path_results=Path(tmpdir) / "results.csv",
                verbose=False
            )
            
            results = parametric.run_analysis()
            
            # Check CSV results were saved
            assert Path(tmpdir, "results.csv").exists()
            
            # Check CSV content can be read back
            import pandas as pd
            saved_df = pd.read_csv(Path(tmpdir) / "results.csv")
            assert len(saved_df) == 2  # Two simulation cases
            assert 'flow_rate' in saved_df.columns
            assert 'efficiency' in saved_df.columns


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        mock_sim = MockSimulation()
        
        # Test get_output_arrays before running analysis
        parametric = Parametric(mock_sim, {}, ["efficiency"])
        
        with pytest.raises(ValueError, match="No results available"):
            parametric.get_output_arrays('efficiency')
        
        # Test missing output parameter
        params_in = {'flow_rate': Array([0.1], 'm3/s')}
        parametric = Parametric(
            base_case=mock_sim,
            params_in=params_in,
            params_out=["nonexistent_output"],
            verbose=False
        )
        
        with pytest.raises(KeyError, match="not in sim.out"):
            parametric.run_analysis()


class TestUnitsConsistency:
    """Test unit consistency throughout workflow."""
    
    def test_units_consistency_through_workflow(self):
        """Test that units remain consistent throughout the entire workflow."""
        mock_sim = MockSimulation()
        
        # Define parameters with specific units
        params_in = {
            'subsystem.temperature': Array([300, 350], 'K'),
            'flow_rate': Array([0.1, 0.2], 'm3/s'),
            'operating_mode': ['normal', 'high_performance']
        }
        params_out = ["efficiency", "power_output", "temperature_out", "flow_rate_out"]
        
        parametric = Parametric(
            base_case=mock_sim,
            params_in=params_in,
            params_out=params_out,
            verbose=False
        )
        
        # Run complete analysis
        results = parametric.run_analysis()
        
        # Verify input units consistency
        for param, expected_unit in [('subsystem.temperature', 'K'), ('flow_rate', 'm3/s'), ('operating_mode', '')]:
            actual_unit = results.unit(param)[param]
            assert actual_unit == expected_unit, f"Unit mismatch for {param}: expected {expected_unit}, got {actual_unit}"
        
        # Verify output units consistency
        for param, expected_unit in [('efficiency', '-'), ('power_output', 'W'), ('temperature_out', 'K'), ('flow_rate_out', 'm3/s')]:
            actual_unit = results.unit(param)[param]
            assert actual_unit == expected_unit, f"Unit mismatch for {param}: expected {expected_unit}, got {actual_unit}"
        
        # Get arrays and verify units are preserved
        efficiency_array = parametric.get_output_arrays('efficiency')
        assert efficiency_array.u == '-'
        
        power_array = parametric.get_output_arrays('power_output')
        assert power_array.u == 'W'
        
        # Get all arrays and verify all units
        all_arrays = parametric.get_output_arrays()
        expected_units = {
            'subsystem.temperature': 'K',
            'flow_rate': 'm3/s',
            'operating_mode': '',
            'efficiency': '-',
            'power_output': 'W',
            'temperature_out': 'K',
            'flow_rate_out': 'm3/s'
        }
        
        for param, expected_unit in expected_units.items():
            if param in all_arrays:
                actual_unit = all_arrays[param].u
                assert actual_unit == expected_unit, f"Array unit mismatch for {param}: expected {expected_unit}, got {actual_unit}"


if __name__ == "__main__":
    pytest.main([__file__, '-v'])