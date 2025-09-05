# -*- coding: utf-8 -*-

import os
import copy
import itertools
import pickle
from typing import Iterable

import numpy as np
import pandas as pd
from antupy import Var, Array, Simulation, Plant


_PARAMS_OUT = ["eta_something", "annual_something", "specific_something", "average_something"]  # Example output parameters

#-------------
def settings(
        params_in : dict[str, Array | Iterable[str|int]] = {},
        ) -> tuple[pd.DataFrame, dict]:
    """ 
    This function creates a parametric run. A pandas dataframe with all the runs required. In practice, it is a cross product between the variables values.
    The order of running for params_in is "first=outer".

    Args:
        params_in (dict[str, Array], optional): dictionary with (parameter : values) parameters.

    Returns:
        pd.DataFrame: set with simulation runs to be performed in the parametric analysis
        dict: dictionary with units for each parameter

    """
    cols_in = list(params_in.keys())
    params_values = []
    params_units = {}
    for lbl in params_in:
        values = params_in[lbl]
        if type(values)==Array:
            params_units[lbl] = values.u
            values = values.gv(values.u)
        else:
            params_units[lbl] = None
        params_values.append(values)

    runs = pd.DataFrame(
        list(itertools.product(*params_values)), 
        columns=cols_in,
        )
    return (runs, params_units)

#-----------------------------
def analysis(
    cases_in: pd.DataFrame,
    units_in: dict[str,str],
    params_out: list[str] = _PARAMS_OUT,
    base_case = Simulation(),
    save_results_detailed: bool = False,
    dir_output: str | None = None,
    path_results: str | None = None,
    verbose: bool = True,
    ) -> pd.DataFrame:
    """Run the parametric analysis
    Parametric.analysis performs a set of simulations changing a set of parameters (params_in).
    The combination of all possible values is performed.

    Args:
        cases_in (pd.DataFrame): a dataframe with all the inputss.
        units_in (dict): list of units with params_units[key] = unit
        params_out (List): list of labels of expected output. Defaults to PARAMS_OUT.
        sim_base (_type_, optional): Simulation instance used as base case. Defaults to general.Simulation().
        save_results_detailed (bool, optional): Defaults to False.
        dir_output (bool, optional): Defaults to None.
        save_results_general (bool, optional): Defaults to False.
        path_results (str, optional): Defaults to None.
        verbose (bool, optional): Defaults to True.

    Returns:
        pd.DataFrame: An updated version of the runs_in dataframe, with the output values.
    """

    params_in = cases_in.columns
    runs_out = cases_in.copy()
    for col in params_out:
        runs_out[col] = np.nan
      
    for (index, row) in runs_out.iterrows():
        if isinstance(index, int):
            idx = index
        else:
            idx = int(index) # type: ignore

        if verbose:
            print(f'RUNNING SIMULATION {int(idx)}/{len(runs_out)}')
        sim = copy.copy(base_case)
        updating_parameters( simulation=sim, row_in = row[params_in], units_in = units_in )
        sim.run_simulation(verbose=verbose)
        values_out = [sim.out[lbl] for lbl in params_out]
        runs_out.loc[idx, params_out] = values_out
        
        #----------------
        #General results?
        if dir_output is not None:
            if not os.path.exists(dir_output):
                os.mkdir(dir_output)
            runs_out.to_csv(path_results)
        
        if save_results_detailed and dir_output is not None:
            pickle_path = os.path.join(dir_output, f'sim_{int(idx)}.plk')
            with open(pickle_path, "wb") as file:
                pickle.dump(sim, file, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(runs_out.loc[int(idx)])
        
    return runs_out

#-------------
def updating_parameters(
        simulation: Simulation,
        row_in: pd.Series,
        units_in: dict = {},
) -> None:
    """Updating parameters for those of the specific run.
    This function update the simulation object. It takes the string and converts it into GS attributes.

    Args:
        simulation (general.Simulation): Simulation instance.
        row_in (pd.Series): values of the specific run (it contains all input and output of the parametric study)
        units_in (dict): labels of the parameters (input)
    """

    for (k, value) in row_in.items():
        key = str(k)
        if '.' in key:
            (obj_name, param_name) = key.split('.')

            #Retrieving first level attribute (i.e.: DEWH, household, sim, etc.)
            object = getattr(simulation, obj_name)
            
            unit = units_in[key]
            if unit is not None:
                param_value = Var(value, unit)
            else:
                param_value = value
            setattr(object, param_name, param_value)

            # Reassigning the first level attribute to GS
            setattr(simulation, obj_name, object)

        else:
            setattr(simulation, key, value)
    return None


#------
if __name__ == "__main__":
    pass