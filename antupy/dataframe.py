"""
Custom DataFrame with unit tracking functionality.
"""

import pandas as pd
from typing import Dict, List, Union, Optional, Any


class DataFrame(pd.DataFrame):
    """
    Enhanced DataFrame with unit tracking capabilities.
    
    Extends pandas DataFrame to include:
    - .units attribute for tracking column units
    - .unit() method for querying units
    - .set_units() method for updating units
    """
    
    # Ensure pandas recognizes our custom attributes
    _metadata = ['_units']
    
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=None, units=None):
        """
        Initialize DataFrame with optional units.
        
        Parameters:
        -----------
        data : array-like, dict, or DataFrame
            Data for the DataFrame
        index : Index or array-like
            Row labels
        columns : Index or array-like  
            Column labels
        dtype : dtype
            Data type to force
        copy : bool
            Copy data from inputs
        units : list, dict, or None
            Units for columns. If list, must match column order.
            If dict, maps column names to units.
        """
        # Initialize the parent DataFrame
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        
        # Initialize units
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
    def units(self) -> List[str]:
        """Get the units for all columns."""
        return self._units.copy()
    
    @units.setter
    def units(self, value: List[str]):
        """Set the units for all columns."""
        if len(value) != len(self.columns):
            raise ValueError(f"Length of units ({len(value)}) must match number of columns ({len(self.columns)})")
        self._units = value.copy()
    
    def unit(self, cols: Optional[Union[str, List[str]]] = None) -> Dict[str, str]:
        """
        Get units for specified columns.
        
        Parameters:
        -----------
        cols : str, list of str, or None
            Column name(s) to get units for. If None, returns all units.
            
        Returns:
        --------
        dict
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
    
    def set_units(self, units: Union[Dict[str, str], List[str]]) -> None:
        """
        Set units for columns.
        
        Parameters:
        -----------
        units : dict or list
            If dict, maps column names to units.
            If list, must match column order.
        """
        if isinstance(units, dict):
            for col, unit in units.items():
                if col in self.columns:
                    idx = list(self.columns).index(col)
                    self._units[idx] = unit
        elif isinstance(units, list):
            if len(units) != len(self.columns):
                raise ValueError(f"Length of units ({len(units)}) must match number of columns ({len(self.columns)})")
            self._units = units.copy()
        else:
            raise TypeError("units must be a dict or list")
    
    @property
    def _constructor(self):
        """Return the constructor for this class (required for pandas subclassing)."""
        return DataFrame
    
    @property
    def _constructor_sliced(self):
        """Return the constructor for Series (required for pandas subclassing)."""
        return pd.Series
    
    def __finalize__(self, other, method=None, **kwargs):
        """
        Propagate metadata from other to self (required for pandas subclassing).
        """
        self = super().__finalize__(other, method, **kwargs)
        if isinstance(other, DataFrame):
            # Copy units if available
            if hasattr(other, '_units'):
                # Handle column operations that might change the structure
                if len(self.columns) == len(other.columns):
                    self._units = other._units.copy()
                else:
                    # Default to empty units for new structure
                    self._units = [""] * len(self.columns)
        else:
            # Default units for new DataFrames
            self._units = [""] * len(self.columns)
        return self


# Convenience function (alternative approach)
def dataframe_with_units(data=None, index=None, columns=None, dtype=None, copy=None, units=None):
    """
    Create a DataFrame with units using function approach.
    
    This is an alternative to the class-based approach that avoids pandas warnings.
    """
    return DataFrame(data=data, index=index, columns=columns, dtype=dtype, copy=copy, units=units)