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
SimulationObject = Simulation | Plant


@dataclass
class ParametricSettings:
    """
    Configuration settings for parametric analysis.
    
    Parameters
    ----------
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
    """
    params_out: list[str] = field(default_factory=lambda: [
        "eta_something", "annual_something", "specific_something", "average_something"
    ])
    save_results_detailed: bool = False
    dir_output: Optional[Path | str] = None
    path_results: Optional[Path | str] = None
    verbose: bool = True

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if self.dir_output is not None:
            self.dir_output = Path(self.dir_output)
        if self.path_results is not None:
            self.path_results = Path(self.path_results)


class Parametric:
    """
    Parametric analysis manager for simulation campaigns.
    
    This class handles the setup, execution, and management of parametric 
    studies where multiple input parameters are varied systematically 
    to explore their effects on simulation outputs.
    
    Parameters
    ----------
    base_case : SimulationObject
        Base simulation or plant object to use as template.
    settings : ParametricSettings, optional
        Configuration settings for the analysis. Default creates new instance.
    
    Attributes
    ----------
    base_case : SimulationObject
        The base simulation template.
    settings : ParametricSettings
        Analysis configuration settings.
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
    >>> params = {
    ...     'temperature': Array([20, 25, 30], '°C'),
    ...     'flow_rate': Array([0.1, 0.2, 0.3], 'm3/s')
    ... }
    >>> 
    >>> # Create and run analysis
    >>> study = Parametric(base_simulation)
    >>> study.setup_cases(params)
    >>> results = study.run_analysis()
    
    Custom output parameters:
    
    >>> settings = ParametricSettings(
    ...     params_out=['efficiency', 'cost', 'emissions'],
    ...     verbose=True
    ... )
    >>> study = Parametric(base_simulation, settings)
    """
    
    def __init__(
        self, 
        base_case: SimulationObject, 
        settings: Optional[ParametricSettings] = None
    ):
        self.base_case = base_case
        self.settings = settings or ParametricSettings()
        self.cases: Optional[pd.DataFrame] = None
        self.units: Optional[dict[str, Optional[str]]] = None
        self.results: Optional[pd.DataFrame] = None
    
    def setup_cases(
        self, 
        params_in: dict[str, ParameterValue]
    ) -> tuple[pd.DataFrame, dict[str, Optional[str]]]:
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
        params_values = []
        params_units = {}
        
        for lbl, values in params_in.items():
            if isinstance(values, Array):
                params_units[lbl] = values.u
                values = values.gv(values.u)
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
        cases_in: Optional[pd.DataFrame] = None,
        units_in: Optional[dict[str, Optional[str]]] = None
    ) -> pd.DataFrame:
        """
        Execute parametric analysis across all cases.
        
        Runs simulations for each parameter combination and collects
        specified output metrics.
        
        Parameters
        ----------
        cases_in : pd.DataFrame, optional
            DataFrame with parameter combinations. Uses self.cases if None.
        units_in : dict[str, Optional[str]], optional
            Parameter units dictionary. Uses self.units if None.
            
        Returns
        -------
        pd.DataFrame
            Results DataFrame with input parameters and output metrics.
            
        Raises
        ------
        ValueError
            If no cases have been set up and none provided.
        """
        # Use provided or stored cases
        if cases_in is None:
            if self.cases is None:
                raise ValueError("No cases defined. Call setup_cases() first or provide cases_in.")
            cases_in = self.cases
            
        if units_in is None:
            units_in = self.units or {}
            
        params_in = cases_in.columns
        results = cases_in.copy()
        
        # Initialize output columns
        for col in self.settings.params_out:
            results[col] = np.nan
        
        # Create output directory if needed
        if self.settings.dir_output is not None:
            if isinstance(self.settings.dir_output, Path):
                self.settings.dir_output.mkdir(parents=True, exist_ok=True)
            elif isinstance(self.settings.dir_output, str):
                self.settings.dir_output = Path(self.settings.dir_output)
                self.settings.dir_output.mkdir(parents=True, exist_ok=True)
            else:
                raise TypeError("dir_output must be a Path, str, or None.")

        # Run simulations
        for index, row in results.iterrows():
            idx = int(index) if isinstance(index, int) else int(index) #type: ignore
            
            if self.settings.verbose:
                print(f'RUNNING SIMULATION {idx + 1}/{len(results)}')
                
            # Create simulation copy and update parameters
            sim = copy.deepcopy(self.base_case)
            self._update_parameters(sim, row[params_in], units_in)
            
            # Run simulation
            sim.run_simulation(verbose=self.settings.verbose)
            
            # Extract outputs
            try:
                params_out = self.settings.params_out
                values_out = [sim.out[lbl] for lbl in params_out]
                results.loc[idx, params_out] = values_out
            except Exception as e:
                print(f"Error occurred while extracting outputs: {e}")

            # Save detailed results if requested
            if self.settings.save_results_detailed and self.settings.dir_output:
                pickle_path = self.settings.dir_output / f'sim_{idx}.plk'
                with open(pickle_path, "wb") as file:
                    pickle.dump(sim, file, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save intermediate results
            if self.settings.path_results is not None:
                results.to_csv(self.settings.path_results)
            
            if self.settings.verbose:
                print(results.loc[idx])
        
        self.results = results
        return results
    
    def _update_parameters(
        self,
        simulation: SimulationObject,
        row_in: pd.Series,
        units_in: dict[str, Optional[str]]
    ) -> None:
        """
        Update simulation parameters for specific run.
        
        Updates simulation object attributes based on parameter values.
        Handles nested attributes using dot notation (e.g., 'object.param').
        
        Parameters
        ----------
        simulation : SimulationObject
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
                setattr(simulation, obj_name, obj)
            else:
                # Direct attribute
                setattr(simulation, key_str, value)
    
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
            "output_parameters": self.settings.params_out,
            "completed": self.results is not None,
        }
        
        if self.results is not None:
            # Add output statistics
            for param in self.settings.params_out:
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
    params_out: list[str] | None = None,
    base_case: SimulationObject | None = None,
    save_results_detailed: bool = False,
    dir_output: str | None = None,
    path_results: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Legacy function for running analysis. Use Parametric class instead."""
    if params_out is None:
        params_out = ["eta_something", "annual_something", "specific_something", "average_something"]
    
    settings_obj = ParametricSettings(
        params_out=params_out,
        save_results_detailed=save_results_detailed,
        dir_output=dir_output,
        path_results=path_results,
        verbose=verbose
    )
    if base_case is not None:
        parametric = Parametric(base_case, settings_obj)  # type: ignore
    else:
        raise TypeError("base_case must be a Simulation or Plant instance.")
    
    return parametric.run_analysis(cases_in, units_in)


if __name__ == "__main__":
    pass