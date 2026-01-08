from dataclasses import dataclass
import json
import os
from typing import Callable, Protocol, runtime_checkable

import pandas as pd

@runtime_checkable
class Market(Protocol):
    """Protocol for electricity market data loaders.
    
    Must return DataFrame with column: SP (spot price in local currency)
    Index must be timezone-aware datetime.
    """
    year_i: int
    year_f: int
    dT: float
    
    def load_data(self) -> pd.DataFrame:
        """Load and return market data as DataFrame."""
        ...

@dataclass
class MarketAU(Market):
    state: str = "NSW"
    year_i: int = 2019
    year_f: int = 2019
    dT: float = 0.5
    file_data: str | None = None

    def load_data(
            self,
    ) -> pd.DataFrame:
        """Load electricity spot price data."""
        if self.file_data is None:
            dir_spotprice = os.path.join(DIR_DATA, 'energy_market', "nem")
            self.file_data = os.path.join(dir_spotprice, 'NEM_TRADINGPRICE_2010-2020.PLK')
            
        df_SP = pd.read_pickle(self.file_data)
        df_sp_state = (
            df_SP[
            (df_SP.index.year >= self.year_i) & (df_SP.index.year <= self.year_f)
            ][['SP_' + self.state]]
        )
        df_sp_state.rename(
            columns={'Demand_' + self.state: 'Demand', 'SP_' + self.state: 'spot_price'},
            inplace=True
        )
        return df_sp_state
    

@dataclass
class MarketCL(Market):
    """Chilean electricity market data loader (SEN - Sistema Eléctrico Nacional)."""
    location: str = "crucero"  # Default barra
    year_i: int = 2024
    year_f: int = 2024
    dT: float = 0.5
    file_data: str | None = None
    
    def load_data(self) -> pd.DataFrame:
        """Load Chilean market spot price data."""
        if self.file_data is None:
            # Try JSON format first (datos-de-costos-marginales-en-linea.tsv)
            json_path = os.path.join(DIR_DATA, "energy_market", "sen", "datos-costos-marginales.tsv")
            
            # Fallback to CSV format
            csv_path = os.path.join(DIR_DATA, "energy_market", "sen", f"{self.year_i}_{self.location}.csv")
            
            if os.path.isfile(json_path):
                self.file_data = json_path
                return self._load_json_format()
            elif os.path.isfile(csv_path):
                self.file_data = csv_path
                return self._load_csv_format()
            else:
                raise FileNotFoundError(
                    f"Market data not found. Tried:\n  {json_path}\n  {csv_path}"
                )
        
        # Determine format from file extension
        if self.file_data.endswith('.tsv') or self.file_data.endswith('.json'):
            return self._load_json_format()
        else:
            return self._load_csv_format()
    
    def _load_json_format(self) -> pd.DataFrame:
        """Load market data from JSON format."""
        with open(self.file_data, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        df = pd.DataFrame(data, columns=["fecha", "barra", "cmg"])
        df["barra"] = df["barra"].str.lower()
        df["date"] = pd.to_datetime(df["fecha"])
        
        # Filter by location and year
        df_filtered = df[
            (df["barra"] == self.location.lower()) & 
            (df["date"].dt.year >= self.year_i) & 
            (df["date"].dt.year <= self.year_f)
        ].copy()
        
        df_filtered = df_filtered.rename(columns={"cmg": "spot_price"})
        df_filtered = df_filtered[["date", "spot_price"]].set_index("date")
        
        # Localize to Chilean timezone
        tz = 'America/Santiago'
        if df_filtered.index.tz is None:
            try:
                df_filtered.index = df_filtered.index.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
            except Exception:
                try:
                    df_filtered.index = df_filtered.index.tz_localize(tz, ambiguous=False, nonexistent="shift_forward")
                except Exception:
                    df_filtered.index = df_filtered.index.tz_localize(tz)
        
        # Resample to match dT
        df_filtered = df_filtered.resample(f'{self.dT:.1f}h').interpolate(method='time')
        
        return df_filtered
    
    def _load_csv_format(self) -> pd.DataFrame:
        """Load market data from CSV format."""
        df_market = pd.read_csv(str(self.file_data), sep=",", index_col=3, header=0)
        df_market.drop(["fecha", "hora"], inplace=True, axis=1, errors='ignore')
        df_market.index = pd.to_datetime(df_market.index)
        
        # Localize to Chilean timezone
        tz = 'America/Santiago'
        if df_market.index.tz is None:
            try:
                df_market.index = df_market.index.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
            except Exception:
                try:
                    df_market.index = df_market.index.tz_localize(tz, ambiguous=False, nonexistent="shift_forward")
                except Exception:
                    df_market.index = df_market.index.tz_localize(tz)
        
        # Filter by year
        df_market = df_market[
            (df_market.index.year >= self.year_i) & 
            (df_market.index.year <= self.year_f)
        ]
        
        return df_market