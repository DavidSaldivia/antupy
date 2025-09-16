# -*- coding: utf-8 -*-
"""
Parametric analysis module for running simulation campaigns.

This module provides the Parametric class for setting up and running
parametric studies with multiple input variables and output metrics.
"""

import os
import copy
import itertools
import pickle
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from antupy.core import Var, Array, Simulation, Plant


# Type aliases for better readability
ParameterValue = Array | Iterable[str | int | float]
SimulationType = Simulation | Plant


class Parametric:
    """
    Parametric analysis manager for simulation campaigns.
    
    This class handles the setup, execution, and management of parametric 
    studies where multiple input parameters are varied systematically 
    to explore their effects on simulation outputs.
    
    Parameters
    ----------
    base_case : SimulationType
        Base simulation or plant object to use as template.
    params_out : list[str]
        List of output parameter names to extract from simulations.
    save_results_detailed : bool, optional
        Whether to save detailed simulation objects. Default is False.
    dir_output : Path or str, optional
        Directory to save results. Default is None.
    path_results : Path or str, optional
        Path for saving summary results CSV. Default is None.
    verbose : bool, optional
        Whether to print progress information. Default is True.
    
    Attributes
    ----------
    base_case : SimulationType
        The base simulation template.
    params_out : list[str]
        Output parameters to extract from simulations.
    save_results_detailed : bool
        Whether to save detailed simulation objects.
    dir_output : Path or None
        Directory for saving results.
    path_results : Path or None
        Path for saving summary results CSV.
    verbose : bool
        Whether to print progress information.
    cases : pd.DataFrame or None
        DataFrame containing all parameter combinations to analyze.
    units : dict[str, str] or None
        Dictionary mapping parameter names to their units.
    results : pd.DataFrame or None
        Results DataFrame with inputs and outputs.
        
    Examples
    --------
    Basic parametric study:
    
    >>> from antupy.analyser.par import Parametric
    >>> from antupy.core import Array
    >>> 
    >>> # Define parameter ranges
    >>> params_in = {
    ...     'temperature': Array([20, 25, 30], '°C'),
    ...     'flow_rate': Array([0.1, 0.2, 0.3], 'm3/s')
    ... }
    >>> 
    >>> # Create and run analysis
    >>> study = Parametric(base_simulation, params_out=['efficiency', 'cost'])
    >>> results = study.run_analysis(params_in)
    
    With file saving:
    
    >>> study = Parametric(
    ...     base_simulation, 
    ...     params_out=['efficiency', 'cost', 'emissions'],
    ...     dir_output='results/',
    ...     path_results='summary.csv',
    ...     verbose=True
    ... )
    >>> results = study.run_analysis(params_in)
    """
    
    def __init__(
        self, 
        base_case: SimulationType,
        params_in: dict[str, ParameterValue],
        params_out: list[str],
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
        
        # Internal state
        self.cases: pd.DataFrame | None = None
        self.units: dict[str, Optional[str]] | None = None
        self.results: pd.DataFrame | None = None

    def setup_cases(
        self, 
        params_in: dict[str, ParameterValue]
    ) -> tuple[pd.DataFrame, dict[str, str | None]]:
        """
        Create parametric run matrix from input parameters.
        
        Generates all combinations of input parameters using Cartesian product.
        Order of parameters follows "first=outer" convention.
        
        Parameters
        ----------
        params_in : dict[str, ParameterValue]
            Dictionary mapping parameter names to their value ranges.
            Values can be Array objects or iterables of strings/numbers.
            
        Returns
        -------
        tuple[pd.DataFrame, dict[str, Optional[str]]]
            DataFrame with all parameter combinations and units dictionary.
            
        Examples
        --------
        >>> params = {
        ...     'temp': Array([20, 30], '°C'),
        ...     'size': ['small', 'large']
        ... }
        >>> cases, units = study.setup_cases(params)
        """

        cols_in = list(params_in.keys())
        
        # Handle empty parameters case
        if not cols_in:
            empty_df = pd.DataFrame()
            self.cases = empty_df
            self.units = {}
            return empty_df, {}
        
        params_values = []
        params_units = {}
        
        for lbl, values in params_in.items():
            if isinstance(values, Array):
                params_units[lbl] = values.u
                values = values.gv(values.u)
            elif isinstance(values, Iterable):
                values = list(values)
                params_units[lbl] = None
            else:
                params_units[lbl] = None
            params_values.append(values)

        cases = pd.DataFrame(
            list(itertools.product(*params_values)), 
            columns=cols_in,
        )
        
        # Store for later use
        self.cases = cases
        self.units = params_units
        
        return cases, params_units
    
    def run_analysis(
        self, 
    ) -> pd.DataFrame:
        """
        Execute parametric analysis with given input parameters.
        
        Creates all parameter combinations and runs simulations for each,
        collecting specified output metrics.
        
        Parameters
        ----------
        params_in : dict[str, ParameterValue]
            Dictionary mapping parameter names to their value ranges.
            Values can be Array objects or iterables of strings/numbers.
            
        Returns
        -------
        pd.DataFrame
            Results DataFrame with input parameters and output metrics.
            
        Examples
        --------
        >>> params = {
        ...     'temp': Array([20, 30], '°C'),
        ...     'size': ['small', 'large']
        ... }
        >>> results = study.run_analysis(params)
        """
        # Setup cases from input parameters
        cases_in, units_in = self.setup_cases(self.params_in)
        
        results = cases_in.copy()
        
        # Initialize output columns
        for col in self.params_out:
            results[col] = np.nan
        
        # Create output directory if needed
        if self.dir_output is not None:
            self.dir_output.mkdir(parents=True, exist_ok=True)

        # Run simulations
        for index, row in results.iterrows():
            idx = int(index)  #type: ignore
            
            if self.verbose:
                print(f'RUNNING SIMULATION {idx + 1}/{len(results)}')
                
            # Create simulation copy and update parameters
            sim = copy.deepcopy(self.base_case)
            self._update_parameters(sim, row[cases_in.columns], units_in)
            
            # Run simulation
            try:
                sim.run_simulation(verbose=self.verbose)
            except Exception as e:
                print(f"Error occurred during simulation {idx + 1}: {e}")
                continue
            
            # Extract outputs
            try:
                params_out = self.params_out
                values_out = [sim.out[lbl] for lbl in params_out]
                results.loc[idx, params_out] = values_out
            except Exception as e:
                print(f"Error occurred while extracting outputs: {e}")

            # Save detailed results if requested
            if self.save_results_detailed and self.dir_output:
                pickle_path = self.dir_output / f'sim_{idx}.plk'
                with open(pickle_path, "wb") as file:
                    pickle.dump(sim, file, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save intermediate results
            if self.path_results is not None:
                results.to_csv(self.path_results)
            
            if self.verbose:
                print(results.loc[idx])
        
        self.results = results
        return results
    
    def _update_parameters(
        self,
        simulation: SimulationType,
        row_in: pd.Series,
        units_in: dict[str, str | None]
    ) -> None:
        """
        Update simulation parameters for specific run.
        
        Updates simulation object attributes based on parameter values.
        Handles nested attributes using dot notation (e.g., 'object.param').
        
        Parameters
        ----------
        simulation : SimulationType
            Simulation instance to update.
        row_in : pd.Series
            Parameter values for this run.
        units_in : dict[str, Optional[str]]
            Parameter units mapping.
        """
        for key, value in row_in.items():
            key_str = str(key)
            
            if '.' in key_str:
                # Handle nested attributes
                obj_name, param_name = key_str.split('.', 1)
                obj = getattr(simulation, obj_name)
                
                # Create Var with units if specified
                unit = units_in.get(key_str)
                param_value = Var(value, unit) if unit is not None else value
                
                setattr(obj, param_name, param_value)
                if hasattr(obj, "__post_init__") and not isinstance(obj, (Var, Array)):
                    obj.__post_init__()

                setattr(simulation, obj_name, obj)

            else:
                # Direct attribute
                unit = units_in.get(key_str)
                param_value = Var(value, unit) if unit is not None else value
                setattr(simulation, key_str, param_value)

            if hasattr(simulation, "__post_init__"):
                simulation.__post_init__()

    def save_results(self, filepath: Path | str) -> None:
        """
        Save analysis results to CSV file.
        
        Parameters
        ----------
        filepath : Path or str
            Output file path.
            
        Raises
        ------
        ValueError
            If no results are available to save.
        """
        if self.results is None:
            raise ValueError("No results to save. Run analysis first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(filepath, index=False)
    
    def get_summary(self) -> dict[str, Any]:
        """
        Get summary statistics of the parametric study.
        
        Returns
        -------
        dict[str, Any]
            Summary information including case count, parameter ranges, etc.
        """
        if self.cases is None or self.results is None:
            return {"status": "No analysis completed"}
        
        summary = {
            "total_cases": len(self.cases),
            "input_parameters": list(self.cases.columns),
            "output_parameters": self.params_out,
            "completed": self.results is not None,
        }
        
        if self.results is not None:
            # Add output statistics
            for param in self.params_out:
                if param in self.results.columns:
                    values = self.results[param].dropna()
                    summary[f"{param}_stats"] = {
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                    }
        
        return summary


# Legacy function interfaces for backward compatibility
def settings(params_in: dict[str, ParameterValue] = {}) -> tuple[pd.DataFrame, dict]:
    """Legacy function for creating parametric cases. Use Parametric class instead."""
    parametric = Parametric(base_case=None)  # type: ignore
    return parametric.setup_cases(params_in)


def analysis(
    cases_in: pd.DataFrame,
    units_in: dict[str, str | None],
    params_in: dict[str, ParameterValue] = {},
    params_out: list[str] | None = None,
    base_case: SimulationType | None = None,
    save_results_detailed: bool = False,
    dir_output: str | None = None,
    path_results: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Legacy function for running parametric analysis. 
    
    Note: This function is deprecated. Use the Parametric class directly:
    
    >>> parametric = Parametric(base_case, params_out)
    >>> results = parametric.run_analysis(params_in)
    """
    if params_out is None:
        params_out = ["eta_something", "annual_something", "specific_something", "average_something"]
    
    if base_case is None:
        raise TypeError("base_case must be a Simulation or Plant instance.")
    
    # Create Parametric instance with new API
    parametric = Parametric(
        base_case=base_case,
        params_in=params_in,
        params_out=params_out,
        save_results_detailed=save_results_detailed,
        dir_output=dir_output,
        path_results=path_results,
        verbose=verbose
    )
    
    # Convert DataFrame back to parameter dict format
    # This is a bit hacky but maintains backwards compatibility
    params_dict = {}
    for col in cases_in.columns:
        unit = units_in.get(col)
        if unit:
            params_dict[col] = Array(cases_in[col].values.tolist(), unit)
        else:
            params_dict[col] = cases_in[col].values.tolist()
    
    return parametric.run_analysis()


def main():
    pass

if __name__ == "__main__":
    main()