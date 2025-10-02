# -*- coding: utf-8 -*-
"""
Enhanced Parametric analysis module with ap.Frame integration.

This module provides the Parametric class for setting up and running
parametric studies with full unit tracking using ap.Frame instead of pd.DataFrame.
"""

import os
import copy
import itertools
import pickle
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, overload

import numpy as np

from antupy.core import Simulation, Plant
from antupy.var import Var
from antupy.array import Array
from antupy.frame import Frame


# Type aliases for better readability
ParameterValue = Array | Iterable[str | int | float]
SimulationType = Simulation | Plant


class Parametric:
    """
    Enhanced parametric analysis manager with full unit tracking.
    
    This class handles the setup, execution, and management of parametric 
    studies where multiple input parameters are varied systematically 
    to explore their effects on simulation outputs. Results are stored
    in ap.Frame with full unit preservation.
    
    Parameters
    ----------
    base_case : Simulation or Plant
        Base simulation or plant object to use as template for all parametric runs.
        This object will be deep-copied for each simulation case.
    params_in : dict[str, Array or Iterable]
        Dictionary mapping parameter names to their value ranges.
        Keys are parameter names (support dot notation for nested attributes).
        Values can be Array objects with units or iterables of values.
    params_out : list[str], optional
        List of output parameter names to extract from simulation results.
        If empty list (default), all available outputs will be extracted.
    save_results_detailed : bool, optional
        Whether to save detailed simulation objects as pickle files.
        Default is False to save disk space.
    dir_output : Path, str, or None, optional
        Directory path to save detailed simulation results.
        Created automatically if it doesn't exist. Default is None.
    path_results : Path, str, or None, optional
        File path for saving summary results CSV after each simulation.
        Enables incremental saving for long-running studies. Default is None.
    verbose : bool, optional
        Whether to print progress information during analysis.
        Default is True.
    
    Attributes
    ----------
    base_case : Simulation or Plant
        The base simulation template object.
    params_in : dict[str, Array or Iterable]
        Input parameters dictionary as provided during initialization.
    params_out : list[str]
        Output parameters to extract from simulation results.
    save_results_detailed : bool
        Flag indicating whether to save detailed simulation objects.
    dir_output : Path or None
        Directory for saving detailed simulation results.
    path_results : Path or None
        File path for saving incremental CSV results.
    verbose : bool
        Flag for progress information printing.
    cases : Frame or None
        Frame containing all input parameter combinations to analyze.
        Created by setup_cases() method with units preserved.
    results : Frame or None
        Complete results Frame with input parameters and output metrics.
        Available after successful run_analysis() execution with units.
        
    Examples
    --------
    Basic parametric study with units preserved:
    
    >>> from antupy.analyser.par import Parametric
    >>> from antupy.array import Array
    >>> 
    >>> # Define parameter ranges with units
    >>> params_in = {
    ...     'temperature': Array([20, 25, 30], '°C'),
    ...     'flow_rate': Array([0.1, 0.2, 0.3], 'm3/s')
    ... }
    >>> 
    >>> # Create and run analysis
    >>> study = Parametric(
    ...     base_case=my_simulation,
    ...     params_in=params_in,
    ...     params_out=['efficiency', 'cost']
    ... )
    >>> results = study.run_analysis()  # Returns ap.Frame with units
    >>> 
    >>> # Access results with units
    >>> efficiency_array = results.get_values('efficiency')  # Array with units
    >>> all_outputs = results.get_values(['efficiency', 'cost'])  # Dict of Arrays
    """
    
    def __init__(
        self,
        base_case: SimulationType,
        params_in: Mapping[str, ParameterValue],
        params_out: list[str] = [],
        save_results_detailed: bool = False,
        dir_output: Path | str | None = None,
        path_results: Path | str | None = None,
        verbose: bool = True
    ):
        self.base_case = base_case
        self.params_in = params_in
        self.params_out = params_out
        self.save_results_detailed = save_results_detailed
        self.verbose = verbose
        
        # Convert paths to Path objects
        self.dir_output = Path(dir_output) if dir_output is not None else None
        self.path_results = Path(path_results) if path_results is not None else None
        
        # Internal state - using Frame instead of DataFrame
        self.cases: Frame | None = None
        self.results: Frame | None = None

    def setup_cases(
        self, 
        params_in: Mapping[str, ParameterValue]
    ) -> Frame:
        """
        Create parametric run matrix from input parameters as ap.Frame.
        
        Generates all combinations of input parameters using Cartesian product.
        Order of parameters follows "first=outer" convention where the first
        parameter varies slowest and the last parameter varies fastest.
        Units are preserved in the returned Frame.
        
        Parameters
        ----------
        params_in : dict[str, Array or Iterable]
            Dictionary mapping parameter names to their value ranges.
            Keys are parameter names (support dot notation for nested attributes).
            Values can be Array objects with units or iterables of strings/numbers.
            
        Returns
        -------
        cases : Frame
            Frame with all parameter combinations as rows and units preserved.
            Columns correspond to parameter names from params_in keys.
            Number of rows equals the product of all parameter range lengths.
            
        Examples
        --------
        Basic parameter setup with units:
        
        >>> params = {
        ...     'temp': Array([20, 30], 'K'),
        ...     'size': ['small', 'large']
        ... }
        >>> cases = study.setup_cases(params)
        >>> print(cases.units)
        {'temp': 'K', 'size': ''}
        """
        cols_in = list(params_in.keys())
        
        # Handle empty parameters case
        if not cols_in:
            self.cases = Frame()
            return self.cases
        
        params_values = []
        params_units = []
        
        for lbl, values in params_in.items():
            if isinstance(values, Array):
                # Extract values and unit from Array
                params_values.append(values.value.tolist())
                params_units.append(values.u)
            else:
                # Handle iterable (list, tuple, etc.)
                params_values.append(list(values))
                params_units.append("")  # No unit for plain iterables

        # Create all combinations
        combinations = list(itertools.product(*params_values))
        
        # Create Frame with units
        cases = Frame(
            combinations,
            columns=cols_in,
            units=params_units
        )
        
        self.cases = cases
        return cases

    def _extract_outputs(
        self, 
        sim: SimulationType, 
        params_out: list[str]
    ) -> tuple[list[float], list[str]]:
        """
        Extract output values and units from simulation object.
        
        Supports extraction from Var, Array, and plain numeric types.
        For Array objects with multiple values, returns the mean.
        
        Parameters
        ----------
        sim : Simulation or Plant
            Simulation object with .out dictionary containing results.
        params_out : list[str]
            List of parameter names to extract from sim.out.
            
        Returns
        -------
        tuple[list[float], list[str]]
            Values as list of floats and corresponding units as list of strings.
            
        Examples
        --------
        Extract from simulation with mixed output types:
        
        >>> sim.out = {
        ...     'efficiency': Var(0.85, '-'),
        ...     'power': Array([1000, 1100], 'W'),
        ...     'count': 42
        ... }
        >>> values, units = self._extract_outputs(sim, ['efficiency', 'power', 'count'])
        >>> # Returns: ([0.85, 1050.0, 42.0], ['-', 'W', ''])
        """
        values = []
        units = []
        
        for param in params_out:
            if param not in sim.out:
                raise KeyError(f"Parameter '{param}' not found in sim.out")
            
            output = sim.out[param]
            
            if isinstance(output, Var):
                # Extract value and unit from Var
                values.append(float(output.v))
                units.append(output.u)
            elif isinstance(output, Array):
                # Extract value and unit from Array (use mean for multiple values)
                if len(output.value) == 1:
                    values.append(float(output.value[0]))
                else:
                    values.append(float(np.mean(output.value)))
                units.append(output.u)
            else:
                # Handle plain numbers (int, float)
                values.append(float(output))
                units.append("")  # No unit for plain numbers
        
        return values, units

    def run_analysis(self) -> Frame:
        """
        Execute parametric analysis with configured input parameters.

        Creates all parameter combinations using setup_cases() and runs
        simulations for each, collecting specified output metrics.
        Results are returned as ap.Frame with full unit tracking.

        Returns
        -------
        Frame
            Complete results Frame with input parameters and output metrics.
            Input columns contain parameter values with units from params_in.
            Output columns contain extracted metrics with units from sim.out.
            Each row represents one completed simulation case.

        Raises
        ------
        KeyError
            If any parameter in params_out is not found in simulation outputs.
        Exception
            If simulation execution fails (errors are printed but not re-raised).      

        Examples
        --------
        Run analysis and access results with units:

        >>> study = Parametric(
        ...     base_case=simulation,
        ...     params_in={'temp': Array([20, 30], 'K')},
        ...     params_out=['efficiency', 'power_output']
        ... )
        >>> results = study.run_analysis()
        >>>
        >>> # Get values with units preserved
        >>> efficiency_array = results.get_values('efficiency')
        >>> all_results = results.get_values()
        >>>
        >>> # Access units
        >>> print(results.units)
        >>> print(results.unit('efficiency'))
        """
        # Setup cases from input parameters
        cases_in = self.setup_cases(self.params_in)

        # Initialize results with input cases and their units
        results_data = cases_in.values.copy()  # Copy input data
        results_columns = list(cases_in.columns)
        input_units = list(cases_in.units.values())

        # Will track output units as we go
        output_units = {}
        
        # For auto-detection case, we'll determine the structure dynamically
        results = None
        detected_params_out = None

        # If we have explicit params_out, initialize results Frame now
        if self.params_out:
            num_output_cols = len(self.params_out)
            if num_output_cols > 0:
                output_data = np.full((len(cases_in), num_output_cols), np.nan)
                results_data = np.column_stack([results_data, output_data])
                results_columns.extend(self.params_out)
                all_units_list = input_units + [""] * num_output_cols
            else:
                all_units_list = input_units
                
            results = Frame(
                results_data,
                columns=results_columns,
                units=all_units_list
            )

        # Create output directory if needed
        if self.dir_output is not None:
            self.dir_output.mkdir(parents=True, exist_ok=True)

        # Run simulations
        for index in range(len(cases_in)):
            if self.verbose:
                print(f'RUNNING SIMULATION {index + 1}/{len(cases_in)}')

            # Get input parameters for this case
            case_params = {}
            for col in cases_in.columns:
                case_params[col] = cases_in.iloc[index][col]

            # Create simulation copy and update parameters
            sim = copy.deepcopy(self.base_case)
            self._update_parameters(sim, case_params, cases_in.units)

            # Run simulation
            sim.run_simulation(verbose=self.verbose)

            # Determine output parameters (for auto-detection on first run)
            if not self.params_out:
                params_out = list(sim.out.keys())
                if detected_params_out is None:
                    detected_params_out = params_out
                    # Now we know the structure, create the results Frame
                    num_output_cols = len(params_out)
                    if num_output_cols > 0:
                        output_data = np.full((len(cases_in), num_output_cols), np.nan)
                        results_data = np.column_stack([results_data, output_data])
                        results_columns.extend(params_out)
                        all_units_list = input_units + [""] * num_output_cols
                    else:
                        all_units_list = input_units
                    
                    results = Frame(
                        results_data,
                        columns=results_columns,
                        units=all_units_list
                    )
            else:
                params_out = self.params_out
                # results Frame already initialized for explicit params_out case
                
                missing = [k for k in params_out if k not in sim.out]
                if missing:
                    raise KeyError(f"The following params_out are not in sim.out: {missing}")

            # Extract outputs with units
            try:
                values_out, units_out = self._extract_outputs(sim, params_out)

                # Store values in results
                for param, value in zip(params_out, values_out):
                    results.loc[index, param] = value

                # Update output units tracking (first time we see each output)
                for param, unit in zip(params_out, units_out):
                    if param not in output_units:
                        output_units[param] = unit

            except Exception as e:
                print(f"Error extracting outputs from simulation {index + 1}: {e}")
                continue

            # Save detailed results if requested
            if self.save_results_detailed and self.dir_output:
                pickle_path = self.dir_output / f'sim_{index}.plk'
                with open(pickle_path, "wb") as file:
                    pickle.dump(sim, file, protocol=pickle.HIGHEST_PROTOCOL)

            # Save intermediate results to CSV if requested
            if self.path_results is not None:
                results.to_csv(self.path_results)

            if self.verbose:
                print(f"Case {index + 1} completed: {dict(zip(params_out, values_out))}")
        
        # Handle case where no results were created (empty input)
        if results is None:
            results = Frame(
                results_data,
                columns=results_columns,
                units=input_units
            )

        # Update Frame with complete unit information
        all_units = {}
        for col in results.columns:
            if col in cases_in.columns:
                all_units[col] = cases_in.unit(col)[col]
            elif col in output_units:
                all_units[col] = output_units[col]
            else:
                all_units[col] = ""
        
        # Create final Frame with correct units
        final_results = Frame(
            results.values,
            columns=results.columns,
            units=[all_units[col] for col in results.columns]
        )
        
        self.results = final_results
        return final_results

    def _update_parameters(
        self,
        simulation: SimulationType,
        case_params: dict[str, Any],
        input_units: dict[str, str]
    ) -> None:
        """
        Update simulation object parameters with values from a case.
        
        Supports both direct attributes and nested attributes using dot notation.
        Creates Var objects for parameters with units.
        
        Parameters
        ----------
        simulation : Simulation or Plant
            Simulation object to update.
        case_params : dict[str, Any]
            Dictionary mapping parameter names to values for this case.
        input_units : dict[str, str]
            Dictionary mapping parameter names to their units.
            
        Examples
        --------
        Update simulation parameters:
        
        >>> case_params = {'temperature': 25.0, 'subsystem.pressure': 2.0}
        >>> input_units = {'temperature': '°C', 'subsystem.pressure': 'bar'}
        >>> self._update_parameters(sim, case_params, input_units)
        >>> # sim.temperature becomes Var(25.0, '°C')
        >>> # sim.subsystem.pressure becomes Var(2.0, 'bar')
        """
        for param_name, value in case_params.items():
            unit = input_units.get(param_name, "")
            
            if '.' in param_name:
                # Handle nested attribute with dot notation
                parts = param_name.split('.')
                obj = simulation
                
                # Navigate to the parent object
                for part in parts[:-1]:
                    if not hasattr(obj, part):
                        # Create intermediate object if it doesn't exist
                        setattr(obj, part, type('', (), {})())
                    obj = getattr(obj, part)
                
                # Set the final attribute
                final_attr = parts[-1]
                if unit:  # Create Var object if unit is specified
                    setattr(obj, final_attr, Var(value, unit))
                else:  # Set plain value if no unit
                    setattr(obj, final_attr, value)
            else:
                # Handle direct attribute
                if unit:  # Create Var object if unit is specified
                    setattr(simulation, param_name, Var(value, unit))
                else:  # Set plain value if no unit
                    setattr(simulation, param_name, value)

    @overload
    def get_output_arrays(self, cols: str) -> Array: ...
    
    @overload
    def get_output_arrays(self, cols: list[str]) -> dict[str, Array]: ...
    
    @overload
    def get_output_arrays(self, cols: None = None) -> dict[str, Array]: ...
    
    def get_output_arrays(self, cols: str | list[str] | None = None) -> dict[str, Array] | Array:
        """
        Get analysis results as Array objects with units preserved.
        
        Parameters
        ----------
        cols : str, list[str], or None
            Column name(s) to return as Arrays. If None, returns all columns.
            
        Returns
        -------
        dict[str, Array] or Array
            If cols is str: single Array object.
            If cols is list or None: dict mapping column names to Array objects.
            
        Raises
        ------
        ValueError
            If no results are available (run_analysis() not called yet).
        """
        if self.results is None:
            raise ValueError("No results available. Run run_analysis() first.")
        
        return self.results.get_values(cols)

    def get_summary(self) -> dict[str, Any]:
        """
        Generate comprehensive summary of parametric analysis results.
        
        Returns
        -------
        dict[str, Any]
            Summary statistics including case counts, parameter lists,
            completion status, units information, and statistical summaries
            for each output parameter.
            
        Examples
        --------
        Get analysis summary:
        
        >>> summary = study.get_summary()
        >>> print(f"Completed {summary['total_cases']} cases")
        >>> print(f"Input units: {summary['input_units']}")
        >>> print(f"Efficiency stats: {summary['efficiency_stats']}")
        """
        if self.results is None:
            return {"status": "No analysis completed"}
        
        # Basic information
        summary = {
            "total_cases": len(self.results),
            "input_parameters": [col for col in self.results.columns if col in self.cases.columns],
            "output_parameters": [col for col in self.results.columns if col not in self.cases.columns],
            "completed": True
        }
        
        # Units information
        summary["input_units"] = {}
        summary["output_units"] = {}
        
        for col in summary["input_parameters"]:
            summary["input_units"][col] = self.results.unit(col)[col]
        
        for col in summary["output_parameters"]:
            summary["output_units"][col] = self.results.unit(col)[col]
        
        # Statistical summaries for output parameters
        for col in summary["output_parameters"]:
            data = self.results[col].dropna()  # Remove NaN values
            if len(data) > 0:
                unit = self.results.unit(col)[col]
                summary[f"{col}_stats"] = {
                    "mean": float(data.mean()),
                    "std": float(data.std()),
                    "min": float(data.min()),
                    "max": float(data.max()),
                    "unit": unit
                }
        
        return summary


def main():
    pass

if __name__ == "__main__":
    main()