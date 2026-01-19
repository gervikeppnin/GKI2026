# Hot Water Demand Forecasting

Predict how much hot water a city will need over the next 3 days.

## The Problem

Reykjavik heats its buildings using geothermal hot water. The distribution network has **44 sensors** measuring flow in different parts of the network, plus a **total flow** measurement.

**Your task:** Given 4 weeks of historical data, predict the next 72 hours for all 45 sensors.

## Input

Your model receives:

1. **Sensor history** - 672 hours (4 weeks) of measurements for 45 sensors
2. **Timestamp** - The datetime of the first hour you need to predict
3. **Weather forecasts** - Forecasts for the next 72 hours (optional)
4. **Weather observations** - Past 672 hours of weather data (optional)

### Sensor History Format

```
Shape: (672, 45)

       M01    M02    M03   ...   M44    FRAMRENNSLI_TOTAL
t-672  523    891    45    ...   234    12543
t-671  518    887    42    ...   231    12489
...
t-2    512    876    48    ...   228    11089
t-1    508    871    51    ...   225    11023  <- most recent
```

### Timestamp Format

ISO 8601 format: `"2025-01-15T08:00:00"`

### Weather Forecast Format (optional)

**Matches `weather_forecasts.csv` exactly - per-station data, no aggregation.**

```
Shape: (N_rows, 11) - approximately 910-936 rows (72 hours × 13 stations)

Columns (same order as training CSV):
  0: date_time         - Forecast target time (ISO string) - when this forecast is FOR
  1: station_id        - Weather station ID (integer)
  2: temperature       - Temperature forecast (°C)
  3: windspeed         - Wind speed forecast (m/s)
  4: cloud_coverage    - Cloud coverage (%)
  5: gust              - Wind gust forecast (m/s) - mostly NaN
  6: humidity          - Relative humidity forecast (%) - mostly NaN
  7: winddirection     - Wind direction (compass string: N, NNA, A, etc.)
  8: dewpoint          - Dew point temperature (°C)
  9: rain_accumulated  - Accumulated rainfall (mm)
  10: value_date       - Forecast issue time (ISO string) - when forecast was made
```

### Weather History Format (optional)

**Matches `weather_observations.csv` exactly - per-station data, no aggregation.**

```
Shape: (N_rows, 21) - N_rows varies by time window

Columns (same order as training CSV):
  0: stod              - Weather station ID (integer)
  1: timi              - Observation timestamp (ISO string with timezone)
  2: f                 - Wind speed (m/s)
  3: fg                - Wind gust (m/s)
  4: fsdev             - Wind speed standard deviation (~36% NaN)
  5: d                 - Wind direction (degrees)
  6: dsdev             - Wind direction standard deviation (~23% NaN)
  7: t                 - Temperature (°C)
  8: tx                - Maximum temperature (°C) (~12% NaN)
  9: tn                - Minimum temperature (°C) (~12% NaN)
  10: rh               - Relative humidity (%)
  11: td               - Dew point temperature (°C)
  12: p                - Atmospheric pressure (hPa) (~62% NaN)
  13: r                - Precipitation (mm) (~35% NaN)
  14: tg               - Ground temperature (°C) (~90% NaN)
  15: tng              - Minimum ground temperature (°C) (~90% NaN)
  16: _rescued_data    - Internal metadata (can be ignored)
  17: value_date       - Data timestamp
  18: lh_created_date  - Database metadata (can be ignored)
  19: lh_modified_date - Database metadata (can be ignored)
  20: lh_is_deleted    - Database metadata (can be ignored)
```

**Note:** Weather data has the SAME structure as your training CSVs - per-station rows, all columns included. Process it the same way you process the training data.

### Handling Per-Station Weather Data

Weather arrays have variable row counts (N = number of stations × number of hours).

```python
def filter_by_station(weather_history, station_id=1):
    """Filter weather data to a single station."""
    if weather_history is None:
        return None
    STOD = 0  # station ID column
    mask = weather_history[:, STOD] == station_id
    return weather_history[mask]
```

## Output

Your model returns predictions for all 45 sensors for the next 72 hours:

```
Shape: (72, 45)

       M01    M02    M03   ...   M44    FRAMRENNSLI_TOTAL
t+1    510    873    49    ...   227    11150   <- 1 hour ahead
t+2    515    878    47    ...   229    11200
...
t+71   498    862    52    ...   218    10890
t+72   495    858    54    ...   215    10850   <- 72 hours ahead
```

## Data

### Training Data

| File | Description |
|------|-------------|
| `train.npz` | Sensor history, targets, and timestamps |
| `weather_forecasts.csv` | Weather forecasts (temperature, wind, etc.) |
| `weather_observations.csv` | Historical weather observations |
| `sensor_timeseries.csv` | Raw sensor time series (alternative format) |

```python
from utils import load_training_data, load_weather_data

# Load sensor data
X_train, y_train, timestamps, sensor_names = load_training_data()
print(f"Samples: {X_train.shape[0]}")
print(f"Input: {X_train.shape}")      # (N, 672, 45)
print(f"Output: {y_train.shape}")     # (N, 72, 45)

# Load weather data
weather_forecasts, weather_obs = load_weather_data()
```

### Data Notes

- Training data may contain missing values and quality issues typical of real-world sensor data
- Validation and test sets are cleaned and do not contain missing values
- Weather forecast data starts mid-2022
- Weather data comes from multiple stations

## Scoring

Your score is computed as:

```
Score = weighted_average_over_sensors(1 - RMSE_model / RMSE_baseline)
```

Where:
- `RMSE_model` is your prediction error
- `RMSE_baseline` is the baseline prediction error
- Sensors are weighted by `sqrt(mean_flow) / sum(sqrt(mean_flows))`

| Score | Meaning |
|-------|---------|
| 0 | Same as baseline |
| > 0 | Better than baseline |
| < 0 | Worse than baseline |
| 1 | Best current prediction in the competition |

## Getting Started

### 1. Setup

```bash
git clone <repo-url>
cd hot-water-forecasting

# Using uv
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Or using pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Explore the Data

```python
from utils import load_training_data
import matplotlib.pyplot as plt

X_train, y_train, timestamps, sensors = load_training_data()

# Plot one sample
sample_idx = 0
plt.figure(figsize=(12, 4))
plt.plot(range(-672, 0), X_train[sample_idx, :, -1], label='History')
plt.plot(range(0, 72), y_train[sample_idx, :, -1], label='Target')
plt.axvline(0, color='red', linestyle='--')
plt.xlabel('Hours from prediction time')
plt.ylabel('Flow')
plt.legend()
plt.show()
```

### 3. Develop Your Model

Edit `model.py`. Your `predict()` function receives:
- `sensor_history`: (672, 45) array
- `timestamp`: ISO format string
- `weather_forecast`: (N, 11) array (optional) - per-station data, same as training CSV
- `weather_history`: (N, 21) array (optional) - per-station data, same as training CSV

And must return a (72, 45) array.

### 4. Test Locally

```bash
python api.py
```

Visit http://localhost:8080 to verify.

### 5. Submit

1. Deploy to a VM (Azure, AWS, etc.)
2. Run `python api.py`
3. Submit your VM's IP to the competition portal

## API Endpoint

POST `/predict`

**Request:**
```json
{
  "sensor_history": [[...], [...], ...],
  "timestamp": "2025-01-15T08:00:00",
  "weather_forecast": [[...], [...], ...],
  "weather_history": [[...], [...], ...]
}
```

**Response:**
```json
{
  "predictions": [[...], [...], ...]
}
```

## File Structure

```
starter_repo/
├── api.py           # API endpoint
├── model.py         # Your model (edit this)
├── utils.py         # Helper functions
├── requirements.txt
├── data/
│   ├── train.npz
│   ├── weather_forecasts.csv
│   └── weather_observations.csv
└── README.md
```

Good luck!
