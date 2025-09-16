# Weather Test Fixtures

This directory contains sample weather data files for testing the weather.py module functionality.

## Structure

### TMY Data Files
- `tmy_*.csv`: Typical Meteorological Year data for major Australian cities
- Contains realistic annual weather patterns with hourly resolution
- Columns: GHI, DNI, DHI, temp_amb, temp_mains, RH, WS, WD, P

### METEONORM Format Files  
- `meteonorm_*.csv`: Weather data in METEONORM format (3-minute intervals)
- High-resolution data for detailed simulations
- Same variables as TMY data but with higher temporal resolution

### MERRA2 Metadata
- `merra2_*_meta.json`: Metadata files describing MERRA2-like datasets
- Contains variable information and descriptions for mocking MERRA2 data

### Special Test Files
- `sample_week_sydney.csv`: One week of data for quick testing
- `malformed_data.csv`: Intentionally corrupted data for error handling tests

## Cities Covered

- Sydney (NSW) - Temperate oceanic climate
- Melbourne (VIC) - Temperate oceanic climate  
- Brisbane (QLD) - Humid subtropical climate
- Perth (WA) - Mediterranean climate
- Adelaide (SA) - Mediterranean climate
- Darwin (NT) - Tropical savanna climate

## Data Characteristics

All fixture data includes realistic Australian weather patterns:
- **Solar irradiance**: Seasonal and daily cycles with weather variability
- **Temperature**: Climate-appropriate ranges with daily and seasonal cycles
- **Humidity**: Inverse correlation with temperature, seasonal patterns
- **Wind**: Seasonal variations with typical Australian patterns
- **Pressure**: Realistic atmospheric pressure with weather system variations

## Usage in Tests

```python
from tests.fixtures.weather import create_sample_tmy_data
import pandas as pd

# Load a fixture file
fixture_path = "tests/fixtures/weather/weather/tmy_sydney_2023.csv"
data = pd.read_csv(fixture_path, index_col=0, parse_dates=True)

# Or generate data programmatically
data = create_sample_tmy_data("Melbourne", 2024)
```

## Regeneration

To regenerate all fixture files:

```bash
cd tests/fixtures/weather
python __init__.py
```

This will create fresh datasets with the same structure but new random variations.