# thessaloniki-urban-microclimate

Analysis of urban microclimate and thermal comfort across three distinct urban morphologies in Thessaloniki, Greece, using ENVI-met simulation data.

## Study Areas

| Code | Area | Description |
|------|------|-------------|
| A | Zefkseidos | Dense canyon street (building shade) |
| B | Shaded | Vegetated campus (AUTh — tree shade + evapotranspiration) |
| C | Waterfront | Open asphalt parking lot (reference) |

## Seasons Simulated

- **15 February** — Winter
- **31 March** — Spring
- **15 August** — Summer (peak heat)
- **15 November** — Autumn

## Repository Structure

```
notebooks/
  config.py               # Shared config: paths, colours, load_csvs(), savefig()
  A_boxplots.ipynb        # Air & surface temperature distributions by area/season
  B_diurnal_profiles.ipynb# Hourly diurnal profiles — summer comparison, cooling gaps
  C_albedo_regression.ipynb# Albedo intraday dynamics and surface temperature relationship
  D_cooling_effect.ipynb  # Delta temperature (cooling benefit vs. Waterfront reference)
  E_thermal_stress.ipynb  # UTCI/PET thermal stress hours above thresholds
  F_peak_lag.ipynb        # Peak temperature timing and heat storage lag
  G_spatial_maps.ipynb    # Spatial heatmaps: temperature, albedo, shadow, SVF,
                          #   UTCI/PET, latent/sensible heat flux, diurnal j×hour maps

figures/
  A_boxplots/             # Saved PNG figures from notebook A
  B_diurnal_profiles/     # ...notebook B
  C_albedo_regression/
  D_cooling_effect/
  E_thermal_stress/
  F_peak_lag/
  G_spatial_maps/

extract_edt.py            # ENVI-met EDT/EDX binary parser → pandas DataFrame
extract_all.py            # Batch runner: extracts all 24 scenarios → output/*.parquet
requirements.txt          # Python dependencies
```

## Data

Raw ENVI-met simulation files (`.edt` / `.edx`) are **not** included in this repository due to size (~16 GB). Processed Parquet files in `output/` are also excluded but can be regenerated:

```bash
python extract_all.py
```

This reads from `data/<Area>/<Date>/` and writes `output/<Area>_<Date>_atmosphere.parquet` and `output/<Area>_<Date>_surface.parquet`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then open any notebook in `notebooks/`.

## Key Findings

- **Vegetation cooling** (Area B) reduces peak air temperature by up to ~2°C versus the open parking lot (Area C), primarily through evapotranspiration rather than shade.
- **Building canyon shade** (Area A) reduces surface temperatures effectively but has limited impact on air temperature.
- Summer surface temperatures in Area C (asphalt) exceed 50°C at peak hour; Area B stays below 35°C.
- UTCI thermal stress hours above the "strong heat stress" threshold (32°C) are significantly lower in Area B across all summer hours.
- Nocturnal heat storage: Area C surface temperatures at 22:00 remain ~5°C above Area B, indicating greater heat island intensity.
