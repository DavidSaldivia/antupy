"""
Custom DataFrame implementation with units support.

This module provides the antupy.DataFrame function that creates pandas DataFrames
with unit tracking capabilities for scientific computing applications.
"""

import pandas as pd
from typing import Union, List, Dict, Optional


def DataFrame(data=None, index=None, columns=None, dtype=None, copy_data=None, units=None):
    """
    Create a pandas DataFrame with unit tracking capabilities.
    
    This function creates a regular pandas DataFrame and adds:
    - A .units attribute that tracks units for each column
    - A .unit() method to get/set units for specific columns
    
    Parameters
    ----------
    data : array-like, Iterable, dict, or DataFrame, optional
        Dict can contain Series, arrays, constants, dataclass or list-like objects.
    index : Index or array-like, optional
        Index to use for resulting frame.
    columns : Index or array-like, optional
        Column labels to use for resulting frame.
    dtype : dtype, optional
        Data type to force. Only a single dtype is allowed.
    copy_data : bool, optional
        Copy data from inputs.
    units : list of str or dict, optional
        Units for each column. Can be:
        - List of strings (same order as columns)
        - Dict mapping column names to unit strings
        - None (defaults to empty units)
    
    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame with added .units attribute and .unit() method.
    
    Examples
    --------
    >>> import antupy as ap
    >>> df = ap.DataFrame({
    ...     'temperature': [20, 25, 30],
    ...     'pressure': [1013, 1015, 1010]
    ... }, units=['°C', 'hPa'])
    >>> df.units
    ['°C', 'hPa']
    >>> df.unit(['temperature'])
    {'temperature': '°C'}
    """
    
    # Create the base DataFrame
    if copy_data is None:
        df = pd.DataFrame(data, index=index, columns=columns, dtype=dtype)
    else:
        df = pd.DataFrame(data, index=index, columns=columns, dtype=dtype, copy=copy_data)
    
    # Initialize units
    if units is None:
        df.units = [""] * len(df.columns)
    elif isinstance(units, dict):
        # Convert dict to list in column order
        df.units = [units.get(col, "") for col in df.columns]
    elif isinstance(units, list):
        if len(units) != len(df.columns):
            raise ValueError(f"Length of units ({len(units)}) must match number of columns ({len(df.columns)})")
        df.units = units.copy()
    else:
        raise TypeError("units must be a list, dict, or None")
    
    def unit_method(cols: Union[List[str], str, None] = None) -> Dict[str, str]:
        """
        Get units for specified columns.
        
        Parameters
        ----------
        cols : list of str, str, or None
            Column name(s) to get units for.
            If None, returns units for all columns.
            If str, returns units for single column.
            If list, returns units for specified columns.
        
        Returns
        -------
        dict
            Dictionary mapping column names to their units.
        """
        if cols is None:
            # Return all column units
            return {col: unit for col, unit in zip(df.columns, df.units)}
        elif isinstance(cols, str):
            # Single column
            if cols not in df.columns:
                raise KeyError(f"Column '{cols}' not found in DataFrame")
            col_idx = list(df.columns).index(cols)
            return {cols: df.units[col_idx]}
        elif isinstance(cols, list):
            # Multiple columns
            result = {}
            for col in cols:
                if col not in df.columns:
                    raise KeyError(f"Column '{col}' not found in DataFrame")
                col_idx = list(df.columns).index(col)
                result[col] = df.units[col_idx]
            return result
        else:
            raise TypeError("cols must be a string, list of strings, or None")
    
    def set_units_method(units_arg: Union[Dict[str, str], List[str]]):
        """
        Set units for columns.
        
        Parameters
        ----------
        units_arg : dict or list
            Units to set. Can be:
            - Dict mapping column names to unit strings
            - List of unit strings (same order as columns)
        
        Returns
        -------
        DataFrame
            Returns self for method chaining.
        """
        if isinstance(units_arg, dict):
            # Update units based on dict
            for col, unit in units_arg.items():
                if col in df.columns:
                    col_idx = list(df.columns).index(col)
                    df.units[col_idx] = unit
        elif isinstance(units_arg, list):
            if len(units_arg) != len(df.columns):
                raise ValueError(f"Length of units ({len(units_arg)}) must match number of columns ({len(df.columns)})")
            df.units = units_arg.copy()
        else:
            raise TypeError("units must be a dict or list")
        
        return df
    
    # Add methods to the DataFrame instance
    df.unit = unit_method
    df.set_units = set_units_method
    
    return df
    
    @property
    def _constructor(self):
        """Return the constructor for this class."""
        return DataFrame
    
    def _constructor_sliced(self, *args, **kwargs):
        """Return constructor for Series-like operations."""
        # For single column selection, return regular pandas Series
        return pd.Series
    
    def unit(self, cols: Union[List[str], str, None] = None) -> Dict[str, str]:
        """
        Get units for specified columns.
        
        Parameters
        ----------
        cols : list of str, str, or None
            Column name(s) to get units for.
            If None, returns units for all columns.
            If str, returns units for single column.
            If list, returns units for specified columns.
        
        Returns
        -------
        dict
            Dictionary mapping column names to their units.
        
        Examples
        --------
        >>> df = ap.DataFrame({'temp': [20, 25]}, units=['°C'])
        >>> df.unit()
        {'temp': '°C'}
        >>> df.unit('temp')
        {'temp': '°C'}
        >>> df.unit(['temp'])
        {'temp': '°C'}
        """
        if cols is None:
            # Return all column units
            return {col: unit for col, unit in zip(self.columns, self.units)}
        elif isinstance(cols, str):
            # Single column
            if cols not in self.columns:
                raise KeyError(f"Column '{cols}' not found in DataFrame")
            col_idx = list(self.columns).index(cols)
            return {cols: self.units[col_idx]}
        elif isinstance(cols, list):
            # Multiple columns
            result = {}
            for col in cols:
                if col not in self.columns:
                    raise KeyError(f"Column '{col}' not found in DataFrame")
                col_idx = list(self.columns).index(col)
                result[col] = self.units[col_idx]
            return result
        else:
            raise TypeError("cols must be a string, list of strings, or None")
    
    def set_units(self, units: Union[Dict[str, str], List[str]]) -> 'DataFrame':
        """
        Set units for columns.
        
        Parameters
        ----------
        units : dict or list
            Units to set. Can be:
            - Dict mapping column names to unit strings
            - List of unit strings (same order as columns)
        
        Returns
        -------
        DataFrame
            Returns self for method chaining.
        
        Examples
        --------
        >>> df = ap.DataFrame({'temp': [20, 25], 'pressure': [1013, 1015]})
        >>> df.set_units({'temp': '°C', 'pressure': 'hPa'})
        >>> df.units
        ['°C', 'hPa']
        """
        if isinstance(units, dict):
            # Update units based on dict
            for col, unit in units.items():
                if col in self.columns:
                    col_idx = list(self.columns).index(col)
                    self.units[col_idx] = unit
        elif isinstance(units, list):
            if len(units) != len(self.columns):
                raise ValueError(f"Length of units ({len(units)}) must match number of columns ({len(self.columns)})")
            self.units = units.copy()
        else:
            raise TypeError("units must be a dict or list")
        
        return self
    
    def __finalize__(self, other, method=None, **kwargs):
        """
        Propagate metadata from other to result.
        
        This is called by pandas when operations create new DataFrames.
        """
        result = super().__finalize__(other, method, **kwargs)
        
        # If other is our custom DataFrame, copy units
        if hasattr(other, 'units'):
            if hasattr(result, 'columns') and len(result.columns) > 0:
                # For operations that preserve columns, try to preserve units
                if len(other.columns) == len(result.columns) and all(other.columns == result.columns):
                    # Same columns, copy units directly
                    result.units = other.units.copy() if hasattr(other, 'units') else [""] * len(result.columns)
                else:
                    # Different columns, initialize empty units
                    result.units = [""] * len(result.columns)
            else:
                result.units = []
        else:
            # Initialize empty units for new DataFrame
            if hasattr(result, 'columns'):
                result.units = [""] * len(result.columns)
            else:
                result.units = []
        
        return result
    
    def __getitem__(self, key):
        """Override getitem to handle units properly."""
        result = super().__getitem__(key)
        
        # If result is a DataFrame (multiple columns selected)
        if isinstance(result, pd.DataFrame) and not isinstance(result, DataFrame):
            # Convert back to our custom DataFrame with appropriate units
            if isinstance(key, list):
                # Multiple columns selected
                selected_units = []
                for col in key:
                    if col in self.columns:
                        col_idx = list(self.columns).index(col)
                        selected_units.append(self.units[col_idx])
                    else:
                        selected_units.append("")
                result = DataFrame(result, units=selected_units)
            else:
                # Single column that somehow returned DataFrame
                result = DataFrame(result, units=[""] * len(result.columns))
        
        return result
    
    def copy(self, deep: bool = True) -> 'DataFrame':
        """
        Copy the DataFrame including units.
        
        Parameters
        ----------
        deep : bool, default True
            Make a deep copy, including a copy of the data and the indices.
        
        Returns
        -------
        DataFrame
            Copy of the DataFrame with units preserved.
        """
        result = super().copy(deep=deep)
        result = DataFrame(result, units=self.units.copy())
        return result
    
    def __repr__(self) -> str:
        """String representation including units."""
        base_repr = super().__repr__()
        if any(self.units):  # Only show units if any are set
            units_str = f"Units: {dict(zip(self.columns, self.units))}"
            return f"{base_repr}\n{units_str}"
        return base_repr
    
    def info(self, verbose: Optional[bool] = None, buf=None, max_cols: Optional[int] = None, 
             memory_usage: Optional[Union[bool, str]] = None, show_counts: Optional[bool] = None) -> None:
        """
        Print a concise summary including units information.
        """
        super().info(verbose=verbose, buf=buf, max_cols=max_cols, 
                    memory_usage=memory_usage, show_counts=show_counts)
        
        if any(self.units):  # Only show units if any are set
            print(f"\nColumn Units:")
            for col, unit in zip(self.columns, self.units):
                if unit:
                    print(f"  {col}: {unit}")


