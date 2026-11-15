# Scientific Data Audit & Provenance Compliance (v5.0)

This document certifies how **TERRA-TWIN** resolves all heuristic/synthetic data concerns and adheres to strict physical measurement standards:

## 1. Resolution of the 4 Audit Findings

| Previous Audit Finding | Scientific Resolution Implemented in `src/` (v5.0) | Physical Unit & Sensor Source |
| :--- | :--- | :--- |
| **1. Hand-assigned 6D target values** | Replaced with explicit `physical_measurements` in `metadata/moon_mars_targets.json` (`structure_diameter_km`, `mean_slope_deg`, `vertical_relief_m`, `bolometric_albedo`, `subsurface_void_depth_m`, `water_ice_wt_pct`, `solar_irradiance_kwh_m2_day`). Indices are deterministically calibrated from measured physical values. | `km`, `deg (°)`, `m`, `wt%`, `kWh/m²/d` — **NASA LRO LOLA / Diviner / LCROSS / MGS MOLA / MRO SHARAD** |
| **2. Slope/elevation derived from RGB brightness** | Removed shape-from-shading RGB elevation guessing. Connected live **Copernicus DEM GLO-90 / NASA SRTM 90m Radar Altimetry API** (`fetch_real_earth_dem_transect` in `src/data_loader.py`) to query orthometric elevations along the 10 km transect and compute true trigonometric slope $\theta = \arctan(|\Delta z / \Delta x|)$. | Orthometric height in **meters (`m` ASL)** and trigonometric slope in **degrees (`°`)** |
| **3. Synthetic image (`rng.normal`) & latitude climate (`est_temp`) fallbacks** | Completely removed `rng.normal` noise generation and `est_temp` / `default_precip` latitude formulas from `src/data_loader.py`. If an external sensor API is unreachable, the loader returns `None` and `data_status = "UNAVAILABLE"` with zero synthetic imputation. | `None` / `"UNAVAILABLE"` (Zero Imputation) |
| **4. Heuristic scoring vs. measured physical similarity** | Added `build_paired_physical_evidence_table()` in `src/matcher_engine.py`, computing similarity strictly across paired physical quantities (`Slope °`, `Relief m`, `Precipitation mm/day`, `Solar Flux kWh/m²/d`, `Temperature K`) and explicitly reporting `data_coverage_pct` (%). Visual AI (`CLIP-ViT-B/32`) is separated as an auxiliary morphological metric. | Paired SI Units + Explicit `data_coverage_pct` (%) |
