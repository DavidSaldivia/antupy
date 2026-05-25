"""
Custom DataFrame with unit tracking functionality.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List, Any, overload, Literal, Self, Sequence

from antupy.core.units import Unit
from antupy.core.var import Var
from antupy.core.array import Array


class Frame(pd.DataFrame):
    """
    Enhanced DataFrame with unit tracking capabilities.
    
    Extends pandas DataFrame to include:
    - .units property for tracking column units (returns dict)
    - .unit() method for querying units
    - .set_units() method for updating units with conversion
    - .get_values() method for returning Arrays with units
    - .df property for getting plain pandas DataFrame
    """
    
    # Ensure pandas recognizes our custom attributes
    _metadata = ['_units']
    
    def __new__(
        cls, 
        data=None, 
        index=None, 
        columns=None, 
        dtype=None, 
        copy=None, 
        units: list[str] | dict[str, str] | None = None,
        **kwargs
    ):
        """
        Create new Frame instance.
        
        This method handles the actual object creation and ensures
        compatibility with pandas DataFrame creation patterns.
        """
        
        # Create the pandas DataFrame using the parent's __new__
        # Pass only the arguments that pandas DataFrame.__new__ expects
        obj = super(Frame, cls).__new__(cls)
        
        # Initialize as a DataFrame manually
        pd.DataFrame.__init__(
            obj,
            data=data,          # type: ignore
            index=index,        # type: ignore
            columns=columns,    # type: ignore
            dtype=dtype,        # type: ignore
            copy=copy,          # type: ignore
            **kwargs
        )
        
        # Initialize units attribute
        obj._init_units(units)
        
        return obj
    
    def __init__(
        self, 
        data=None, 
        index=None, 
        columns=None, 
        dtype=None, 
        copy=None, 
        units: list[str] | dict[str, str] | None = None,
        **kwargs
    ):
        """
        Initialize Frame instance.
        
        Note: The actual DataFrame initialization is done in __new__,
        this is just for compatibility.
        """
        # Don't call super().__init__ here as it's already done in __new__
        pass
    
    def _init_units(self, units: list[str] | dict[str, str] | None):
        """Initialize units attribute."""
        if units is None:
            self._units = [""] * len(self.columns)
        elif isinstance(units, dict):
            self._units = [units.get(col, "") for col in self.columns]
        elif isinstance(units, list):
            if len(units) != len(self.columns):
                raise ValueError(f"Length of units ({len(units)}) must match number of columns ({len(self.columns)})")
            self._units = units.copy()
        else:
            raise TypeError("units must be a list, dict, or None")
    
    @property
    def units(self) -> dict[str, str]:
        """Get the units for all columns as a dictionary."""
        return dict(zip(self.columns, self._units))
    
    @units.setter
    def units(self, value: list[str] | dict[str, str]):
        """Set the units for all columns."""
        if isinstance(value, dict):
            self._units = [value.get(col, "") for col in self.columns]
        elif isinstance(value, list):
            if len(value) != len(self.columns):
                raise ValueError(f"Length of units ({len(value)}) must match number of columns ({len(self.columns)})")
            self._units = value.copy()
        else:
            raise TypeError("units must be a list or dict")
    
    @property
    def df(self) -> pd.DataFrame:
        """Return plain pandas DataFrame without unit metadata."""
        return pd.DataFrame(self)
    
    def unit(self, cols: str | list[str] | None = None) -> dict[str, str]:
        """
        Get units for specified columns.
        
        Parameters
        ----------
        cols : str, list[str], or None
            Column name(s) to get units for. If None, returns all units.
            
        Returns
        -------
        dict[str, str]
            Mapping of column names to their units.
        """
        return self.get_units(cols)
    
    def get_units(self, cols: str | list[str] | None = None) -> dict[str, str]:
        """
        Get units for specified columns.
        
        Parameters
        ----------
        cols : str, list[str], or None
            Column name(s) to get units for. If None, returns all units.
            
        Returns
        -------
        dict[str, str]
            Mapping of column names to their units.
        """
        if cols is None:
            return dict(zip(self.columns, self._units))
        elif isinstance(cols, str):
            if cols not in self.columns:
                raise KeyError(f"Column '{cols}' not found in DataFrame")
            idx = list(self.columns).index(cols)
            return {cols: self._units[idx]}
        elif isinstance(cols, list):
            result = {}
            for col in cols:
                if col not in self.columns:
                    raise KeyError(f"Column '{col}' not found in DataFrame")
                idx = list(self.columns).index(col)
                result[col] = self._units[idx]
            return result
        else:
            raise TypeError("cols must be a string, list of strings, or None")
    
    @overload
    def get_values(self, cols: None = None) -> dict[str, Array]: ...
    
    @overload
    def get_values(self, cols: list[str]) -> dict[str, Array]: ...
    
    @overload
    def get_values(self, cols: str) -> Array: ...
    
    def get_values(self, cols: str | list[str] | None = None) -> dict[str, Array] | Array:
        """
        Get column data as Array objects with units.
        
        Parameters
        ----------
        cols : str, list[str], or None
            Column name(s) to get values for. If None, returns all columns.
            If str, returns single Array. If list, returns dict of Arrays.
            
        Returns
        -------
        dict[str, Array] or Array
            If cols is None or list: dict mapping column names to Array objects.
            If cols is str: single Array object for that column.
        """
        if cols is None:
            # Return all columns as dict
            result = {}
            for i, col in enumerate(self.columns):
                unit = self._units[i]
                # Access underlying numpy array directly
                values = self.values[:, i]
                result[col] = Array(np.array(values), unit)
            return result
        elif isinstance(cols, str):
            # Return single Array
            if cols not in self.columns:
                raise KeyError(f"Column '{cols}' not found in DataFrame")
            idx = list(self.columns).index(cols)
            unit = self._units[idx]
            # Access underlying numpy array directly
            values = self.values[:, idx]
            return Array(np.array(values), unit)
        elif isinstance(cols, list):
            # Return dict of specified columns
            result = {}
            for col in cols:
                if col not in self.columns:
                    raise KeyError(f"Column '{col}' not found in DataFrame")
                idx = list(self.columns).index(col)
                unit = self._units[idx]
                # Access underlying numpy array directly
                values = self.values[:, idx]
                result[col] = Array(np.array(values), unit)
            return result
        else:
            raise TypeError("cols must be a string, list of strings, or None")
    
    @overload
    def gv(self, cols: None = None) -> dict[str, Array]: ...
    
    @overload
    def gv(self, cols: list[str]) -> dict[str, Array]: ...
    
    @overload
    def gv(self, cols: str) -> Array: ...
    
    def gv(self, cols: str | list[str] | None = None) -> dict[str, Array] | Array:
        """Alias for get_values()."""
        return self.get_values(cols)
    
    def set_units(self, units: dict[str, str] | list[str]) -> None:
        """
        Set units for columns with automatic unit conversion.
        
        Uses Array.su() method to perform unit conversion on each column.
        
        Parameters
        ----------
        units : dict[str, str] or list[str]
            If dict, maps column names to new units (with conversion).
            If list, must match column order for unit conversion.
        """
        if isinstance(units, dict):
            for col, new_unit in units.items():
                if col in self.columns:
                    # Get current data and unit directly from numpy
                    idx = list(self.columns).index(col)
                    current_data = self.values[:, idx]
                    current_unit = self._units[idx]
                    
                    # Convert using Array.su() method
                    if current_unit:  # Only convert if current unit is not empty
                        array_obj = Array(np.array(current_data), current_unit)
                        converted_array = array_obj.su(new_unit)
                        # Set values directly in underlying array
                        self.values[:, idx] = converted_array.v
                    
                    # Update the unit
                    self._units[idx] = new_unit
                else:
                    raise KeyError(f"Column '{col}' not found in DataFrame")
                    
        elif isinstance(units, list):
            if len(units) != len(self.columns):
                raise ValueError(f"Length of units ({len(units)}) must match number of columns ({len(self.columns)})")
            
            # Convert each column
            for i, (col, new_unit) in enumerate(zip(self.columns, units)):
                current_unit = self._units[i]
                current_data = self.values[:, i]
                
                # Convert using Array.su() method
                if current_unit:  # Only convert if current unit is not empty
                    array_obj = Array(np.array(current_data), current_unit)
                    converted_array = array_obj.su(new_unit)
                    # Set values directly in underlying array
                    self.values[:, i] = converted_array.v
                
                # Update the unit
                self._units[i] = new_unit
        else:
            raise TypeError("units must be a dict or list")
    
    def su(self, units: dict[str, str] | list[str]) -> None:
        """Alias for set_units()."""
        self.set_units(units)
    
    @property
    def _constructor(self):
        """Return the constructor for this class (required for pandas subclassing)."""
        return Frame
    
    @property
    def _constructor_sliced(self):
        """Return pandas Series constructor for compatibility."""
        return pd.Series

    # Use pandas' built-in metadata system instead of overriding __finalize__
    _metadata = ['_units']


class Framepl():

    def __init__(
            self,
            _data: pl.DataFrame | Framepl | pd.DataFrame | dict[str, Any] | np.ndarray | Sequence |None = None,
            _units: dict[str, str] | None = None
    ):
        if isinstance(_data, Framepl):
            self.data = _data.df("pl")
            self.units = _data.units
        elif isinstance(_data, (pl.DataFrame, pd.DataFrame, dict, np.ndarray)):
            self.data = pl.DataFrame(_data)
            if _units is not None:
                self.units = {k:Unit(v) for k, v in _units.items()}
            else:
                self.units = {k: Unit("-") for k in self.data.columns}
        else:
            raise TypeError("data must be a Framepl, pl.DataFrame, pd.DataFrame, dict, or None")

    def __add__(self, other: Framepl) -> Framepl:
        if not isinstance(other, Framepl):
            raise TypeError("Can only add Framepl to Framepl")

        for col in self.data.columns:
            unit1 = self.units.get(col, Unit("-"))
            unit2 = other.units.get(col, Unit("-"))
            if unit1 and unit2 and unit1 != unit2:
                raise ValueError(f"Units in col {col} are not compatible: '{unit1}' and '{unit2}'")
        return Framepl(self.data + other.gv(self.u), self.u)
    
    def __mul__(
            self, other: Var | Array | Framepl
    ) -> Framepl:
        
        if isinstance(other, (Var, Array)):
            return Framepl(self.data * other.v, )
        elif isinstance(other, Framepl):
            pass
        else:
            raise TypeError("Can only multiply Framepl by Var, Array, or Framepl")

    def __len__(self) -> int:
        return len(self.data)
    
    @overload
    def __getitem__(self, key: str) -> Array: ...

    @overload
    def __getitem__(self, key: list[str]) -> Framepl: ...

    def __getitem__(
            self,
            key: str | list[str] | None = None
        ) -> Array | Framepl:
        if isinstance(key, str):
            if key not in self.data.columns:
                raise KeyError(f"Column '{key}' not found in Frame")
            unit = self.units.get(key, "")
            values = self.data[key].to_numpy()
            return Array(values, unit)
        elif isinstance(key, list):
            return Framepl(self.data.select(key), {k: self.u[k] for k in key})
        elif key is None:
            return self
        else:
            raise TypeError("key must be a string, list of strings, or None")

    @property
    def v(self) -> pl.DataFrame:
        """Return the underlying DataFrame."""
        return self.data

    @property
    def u(self) -> dict[str, str]:
        """Return the units dictionary."""
        return {k: v.label_unit for k, v in self.units.items()}
    
    @overload
    def df(self, _type: Literal["pl"]) -> pl.DataFrame: ...

    @overload
    def df(self, _type: Literal["pd"]) -> pd.DataFrame: ...

    def df(self, _type: Literal["pl", "pd"] = "pl") -> pl.DataFrame | pd.DataFrame:
        """Return the underlying DataFrame."""
        if _type == "pl":
            return self.data
        elif _type == "pd":
            return self.data.to_pandas()
        else:
            raise ValueError("Invalid type specified. Use 'pl' or 'pd'. Default is pl.")


    def get_value(
        self,
        cols: str | list[str] | dict[str, str] | None = None
    ) -> pl.DataFrame | pl.Series:
        if cols is None:
            return self.v
        elif isinstance(cols, str):
            return self.v[cols]
        elif isinstance(cols, list):
            return self.v.select(cols)
        elif isinstance(cols, dict):
            return self.v.select(cols.keys())
        else:
            raise TypeError("cols must be a string, list of strings, dict, or None")


    # just placeholder for now
    def set_units(
            self,
            units: dict[str, str] | list[str]
    ) -> Self:
        pass
        return self

    def gv(
            self,
            cols: str | list[str] | dict[str, str] | None = None
    ) -> pl.DataFrame | pl.Series:
        return self.get_value(cols)

    def su(
            self,
            units: dict[str, str] | list[str]
    ) -> Self:
        return self.set_units(units)

    @property
    def columns(self) -> list[str]:
        """Return the list of column names."""
        return self.data.columns
     

def main():

    pass

if __name__ == "__main__":
    main()