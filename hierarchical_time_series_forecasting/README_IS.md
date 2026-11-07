# Spá um heitavatnsnotkun

Spáðu fyrir um hversu mikið heitt vatn borgin þarf næstu 3 daga.

## Verkefnið

Reykjavík hitar byggingar sínar með jarðhitavatni. Dreifikerfið hefur **44 skynjara** sem mæla flæði á mismunandi stöðum, auk **heildarflæðismælingar**.

**Þitt verkefni:** Gefið 4 vikur af sögulegum gögnum, spáðu næstu 72 klukkustundir fyrir alla 45 skynjara.

## Inntak

Líkanið þitt fær:

1. **Skynjarasaga** - 672 klukkustundir (4 vikur) af mælingum fyrir 45 skynjara
2. **Tímastimpill** - Dagsetning/tími fyrstu klukkustundarinnar sem þarf að spá
3. **Veðurspár** - Spár fyrir næstu 72 klukkustundir (valkvætt)
4. **Veðurathuganir** - Síðustu 672 klukkustundir af veðurgögnum (valkvætt)

### Snið skynjarasögu

```
Stærð: (672, 45)

       M01    M02    M03   ...   M44    FRAMRENNSLI_TOTAL
t-672  523    891    45    ...   234    12543
t-671  518    887    42    ...   231    12489
...
t-2    512    876    48    ...   228    11089
t-1    508    871    51    ...   225    11023  <- nýjast
```

### Snið tímastimpils

ISO 8601 snið: `"2025-01-15T08:00:00"`

### Snið veðurspár (valkvætt)

```
Stærð: (N, 11) - N = fjöldi stöðva × fjöldi klukkustunda

Dálkar:
  0: date_time         - Tími spár
  1: station_id        - Veðurstöð
  2: temperature       - Hitastig (°C)
  3: windspeed         - Vindhraði (m/s)
  4: cloud_coverage    - Skýjahula (%)
  5: gust              - Vindhviður (m/s)
  6: humidity          - Rakastig (%)
  7: winddirection     - Vindátt (N, NNA, A, o.s.frv.)
  8: dewpoint          - Daggarmark (°C)
  9: rain_accumulated  - Úrkoma (mm)
  10: value_date       - Gildistími spár
```

### Snið veðurathuganir (valkvætt)

```
Stærð: (N, 21) - N = fjöldi stöðva × fjöldi klukkustunda

Dálkar:
  0: stod              - Veðurstöð
  1: timi              - Mælingartími
  2: f                 - Vindhraði (m/s)
  3: fg                - Vindhviður (m/s)
  4: fsdev             - Staðalfrávik vindhraða (~36% NaN)
  5: d                 - Vindátt (gráður)
  6: dsdev             - Staðalfrávik vindáttar (~23% NaN)
  7: t                 - Hitastig (°C)
  8: tx                - Hámarkshiti (°C) (~12% NaN)
  9: tn                - Lágmarkshiti (°C) (~12% NaN)
  10: rh               - Rakastig (%)
  11: td               - Daggarmark (°C)
  12: p                - Loftþrýstingur (hPa) (~62% NaN)
  13: r                - Úrkoma (mm) (~35% NaN)
  14: tg               - Jarðvegshiti (°C) (~90% NaN)
  15: tng              - Lágmarks jarðvegshiti (°C) (~90% NaN)
  16-20:               - Lýsigögn (má hunsa)
```

## Úttak

Líkanið skilar spám fyrir alla 45 skynjara næstu 72 klukkustundir:

```
Stærð: (72, 45)

       M01    M02    M03   ...   M44    FRAMRENNSLI_TOTAL
t+1    510    873    49    ...   227    11150   <- 1 klst fram
t+2    515    878    47    ...   229    11200
...
t+71   498    862    52    ...   218    10890
t+72   495    858    54    ...   215    10850   <- 72 klst fram
```

## Gögn

| Skrá | Lýsing |
|------|--------|
| `train.npz` | Skynjarasaga, markmið, og tímastimplar |
| `weather_forecasts.csv` | Veðurspár |
| `weather_observations.csv` | Veðurathuganir |
| `sensor_timeseries.csv` | Hráar skynjaratímaraðir |

```python
from utils import load_training_data, load_weather_data

X_train, y_train, timestamps, sensor_names = load_training_data()
print(f"Inntak: {X_train.shape}")   # (N, 672, 45)
print(f"Úttak: {y_train.shape}")    # (N, 72, 45)

weather_forecasts, weather_obs = load_weather_data()
```

## Einkunnagjöf

```
Einkunn = weighted_average_over_sensors(1 - RMSE_model / RMSE_baseline)
```

| Einkunn | Merking |
|---------|---------|
| 0 | Sama og grunnlíkan |
| > 0 | Betri en grunnlíkan |
| < 0 | Verri en grunnlíkan |
| 1 | Besta spá í keppninni |

## Fljótleg byrjun

```bash
pip install -r requirements.txt
python api.py
```

Þjónninn keyrir á `http://localhost:8080`

## API

```
POST /predict
{
    "sensor_history": [[...], [...], ...],  // 672 x 45
    "timestamp": "2025-01-15T08:00:00",
    "weather_forecast": [[...], ...],       // valkvætt
    "weather_history": [[...], ...]         // valkvætt
}

Svar:
{
    "predictions": [[...], [...], ...]      // 72 x 45
}
```

## Þróun líkans

Breyttu `model.py`. Fallið `predict()` fær:
- `sensor_history`: (672, 45) fylki
- `timestamp`: ISO snið strengur
- `weather_forecast`: (N, 11) fylki (valkvætt)
- `weather_history`: (N, 21) fylki (valkvætt)

Og skilar (72, 45) fylki.

Gangi þér vel!
