import os
import json
import requests
import numpy as np
import cv2


def ensure_directories():
    for d in ["data/raw_moon", "data/raw_mars", "data/raw_earth", "data/global_grids", "metadata", "src"]:
        os.makedirs(d, exist_ok=True)


def build_moon_mars_reference_catalog():
    targets = {
        "MOON_SHACKLETON_RIM": {
            "id": "MOON_SHACKLETON_RIM",
            "body": "Moon",
            "name": "Shackleton Crater Rim (Artemis South Pole Base)",
            "lat": -89.9,
            "lon": 0.0,
            "radius_km": 5.0,
            "recommended_zoom": 12,
            "gravity_g": 0.165,
            "pressure_mbar": 0.0,
            "mean_temp_c": -180.0,
            "radiation_msv_yr": 380.0,
            "nasa_missions": ["NASA LRO LOLA (Topography)", "LRO Diviner (Thermal)", "LCROSS (Water Ice)"],
            "primary_process": "Cryospheric Cold-Trap & Impact Cratering",
            "mission_focus": "ISRU Water Mining & Peaks of Eternal Light Power",
            "features_6d": [0.65, 0.62, 0.96, 0.96, 0.20, 0.45],
            "description": "Kalici golgeli krater tabaninda su buzu (ISRU) ve sirtlarinda kesintisiz gunes isigi sunan birincil Artemis us adayi."
        },
        "MOON_MARE_TRANQUILLITATIS_PIT": {
            "id": "MOON_MARE_TRANQUILLITATIS_PIT",
            "body": "Moon",
            "name": "Mare Tranquillitatis Skylight Pit (Lava Tube Base)",
            "lat": 8.33,
            "lon": 33.22,
            "radius_km": 5.0,
            "recommended_zoom": 13,
            "gravity_g": 0.165,
            "pressure_mbar": 0.0,
            "mean_temp_c": -20.0,
            "radiation_msv_yr": 420.0,
            "nasa_missions": ["NASA LRO NAC (Radar & Optical)", "GRAIL (Gravity Anomaly)"],
            "primary_process": "Volcanic Lava Tube Roof Collapse (Skylight)",
            "mission_focus": "Radiation-Shielded Underground Habitat Construction",
            "features_6d": [0.92, 0.76, 0.85, 0.06, 0.65, 0.94],
            "description": "100 metre derinliginde cokmus bazaltik lav tupu girisi. Radyasyon ve mikrometeoritlerden korunan yeralti ussu adayi."
        },
        "MOON_MARIUS_HILLS": {
            "id": "MOON_MARIUS_HILLS",
            "body": "Moon",
            "name": "Marius Hills Volcanic Complex & Skylight",
            "lat": 14.1,
            "lon": -56.8,
            "radius_km": 5.0,
            "recommended_zoom": 12,
            "gravity_g": 0.165,
            "pressure_mbar": 0.0,
            "mean_temp_c": -50.0,
            "radiation_msv_yr": 410.0,
            "nasa_missions": ["NASA LRO LOLA", "SELENE/Kaguya Radar"],
            "primary_process": "Basaltic Shield Volcanism & Lava Channels",
            "mission_focus": "Volcanic Regolith ISRU (3D Printing) & Subsurface Access",
            "features_6d": [0.90, 0.74, 0.84, 0.08, 0.68, 0.90],
            "description": "Ay'in en yogun volkanik kubbe ve lav kanali bolgesi. Bazaltik regolitten 3B baski ile us insasi icin ideal."
        },
        "MARS_ARCADIA_PLANITIA": {
            "id": "MARS_ARCADIA_PLANITIA",
            "body": "Mars",
            "name": "Arcadia Planitia (Human Landing & Ice ISRU Base)",
            "lat": 39.2,
            "lon": 172.0,
            "radius_km": 5.0,
            "recommended_zoom": 12,
            "gravity_g": 0.378,
            "pressure_mbar": 6.5,
            "mean_temp_c": -65.0,
            "radiation_msv_yr": 210.0,
            "nasa_missions": ["Mars Reconnaissance Orbiter (SHARAD Radar & HiRISE)", "MGS MOLA"],
            "primary_process": "Periglacial Shallow Subsurface Ice Sheet & Volcanic Plains",
            "mission_focus": "Starship Heavy Landing & Shallow Water-Ice Drilling (ISRU)",
            "features_6d": [0.70, 0.90, 0.88, 0.92, 0.55, 0.35],
            "description": "Duz bazaltik ova ve yuzeyin 1-2 metre altinda devasa su buzu tabakasi barindiran birincil insanli Mars us adayi."
        },
        "MARS_JEZERO_CRATER": {
            "id": "MARS_JEZERO_CRATER",
            "body": "Mars",
            "name": "Jezero Crater Paleo-Lake Delta (Astrobiology Base)",
            "lat": 18.38,
            "lon": 77.58,
            "radius_km": 5.0,
            "recommended_zoom": 12,
            "gravity_g": 0.378,
            "pressure_mbar": 6.1,
            "mean_temp_c": -60.0,
            "radiation_msv_yr": 230.0,
            "nasa_missions": ["Mars 2020 Perseverance", "MRO CRISM (Mineralogy)", "MGS MOLA"],
            "primary_process": "Fluvial-Lacustrine Sedimentation & Carbonate/Evaporite Alteration",
            "mission_focus": "Astrobiology Biosignature Search & Autonomous Rover Traverse",
            "features_6d": [0.76, 0.74, 0.80, 0.18, 0.84, 0.45],
            "description": "Eski nehir deltasi, magnezyum karbonat, evaporit tuzlar ve kil mineralleri iceren krater havzasi."
        },
        "MARS_ARSIA_MONS_CAVES": {
            "id": "MARS_ARSIA_MONS_CAVES",
            "body": "Mars",
            "name": "Arsia Mons 'Seven Sisters' Skylights (Martian Subsurface Base)",
            "lat": -7.2,
            "lon": -120.5,
            "radius_km": 5.0,
            "recommended_zoom": 13,
            "gravity_g": 0.378,
            "pressure_mbar": 2.0,
            "mean_temp_c": -75.0,
            "radiation_msv_yr": 280.0,
            "nasa_missions": ["Mars Odyssey THEMIS (Thermal Inertia)", "MRO HiRISE"],
            "primary_process": "Flank Volcano Lava Tube Collapse & High-Altitude Aridity",
            "mission_focus": "Subsurface Radiation Shelter & Cave Robotics Exploration",
            "features_6d": [0.92, 0.62, 0.90, 0.10, 0.72, 0.95],
            "description": "Arsia Mons volkaninin yamacinda yer alan derin yeralti lav tupu ve magara girisleri."
        }
    }

    with open("metadata/moon_mars_targets.json", "w", encoding="utf-8") as f:
        json.dump(targets, f, indent=4, ensure_ascii=False)


def fetch_nasa_gibs_global_layer(layer_name, save_path, width=720, height=360):
    if not os.path.exists(save_path):
        wms_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
        params = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.1.1",
            "LAYERS": layer_name,
            "STYLES": "",
            "FORMAT": "image/jpeg",
            "TRANSPARENT": "false",
            "HEIGHT": str(height),
            "WIDTH": str(width),
            "SRS": "EPSG:4326",
            "BBOX": "-180,-90,180,90"
        }
        resp = requests.get(wms_url, params=params, timeout=30)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(resp.content)

    img_bgr = cv2.imread(save_path)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def is_in_overstudied_classic_zone(lat, lon):
    if -27.5 <= lat <= -21.0 and -71.0 <= lon <= -67.5:
        return 1
    if 18.5 <= lat <= 22.5 and -160.5 <= lon <= -154.5:
        return 1
    if 63.0 <= lat <= 66.8 and -24.5 <= lon <= -13.0:
        return 1
    if (74.0 <= lat <= 77.0 and -93.0 <= lon <= -80.0) or (lat < -65.0):
        return 1
    if 35.5 <= lat <= 37.5 and -118.0 <= lon <= -116.0:
        return 1
    if 38.0 <= lat <= 39.0 and -111.5 <= lon <= -110.0:
        return 1
    return 0


def build_real_global_earth_matrix():
    W, H = 720, 360
    bm_path = "data/global_grids/nasa_gibs_bluemarble.jpg"
    relief_path = "data/global_grids/nasa_gibs_shaded_relief.jpg"

    rgb_surface = fetch_nasa_gibs_global_layer("BlueMarble_NextGeneration", bm_path, W, H)
    rgb_relief = fetch_nasa_gibs_global_layer("BlueMarble_ShadedRelief_Bathymetry", relief_path, W, H)

    R = rgb_surface[:, :, 0].astype(np.float32) / 255.0
    G = rgb_surface[:, :, 1].astype(np.float32) / 255.0
    B = rgb_surface[:, :, 2].astype(np.float32) / 255.0
    albedo = (R + G + B) / 3.0

    relief_gray = cv2.cvtColor(rgb_relief, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(relief_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(relief_gray, cv2.CV_32F, 0, 1, ksize=3)
    roughness = np.hypot(grad_x, grad_y)
    rough_norm = np.clip(roughness / (np.percentile(roughness, 98) + 1e-5), 0.0, 1.0)

    lap_relief = np.abs(cv2.Laplacian(relief_gray, cv2.CV_32F, ksize=3))
    pit_collapse_norm = np.clip(lap_relief / (np.percentile(lap_relief, 98) + 1e-5), 0.0, 1.0)

    lats_1d = 90.0 - (np.arange(H) + 0.5) * (180.0 / H)
    lons_1d = -180.0 + (np.arange(W) + 0.5) * (360.0 / W)
    lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)
    lat_abs = np.abs(lat_grid)

    bare_soil_warmth = R - B
    is_water = (B > R + 0.01) | ((B >= R - 0.02) & (albedo < 0.28))
    excess_green = (2.0 * G - R - B)
    ndvi_proxy = (G - R) / (G + R + 1e-5)
    is_vegetation = (excess_green > 0.016) | (ndvi_proxy > 0.008)

    high_albedo_mask = np.clip((albedo - 0.54) * 2.5, 0.0, 1.0) * np.clip(1.0 - bare_soil_warmth * 3.0, 0.0, 1.0)
    snow_ice_spectral = np.where(lat_abs >= 42.0, high_albedo_mask, high_albedo_mask * 0.20)
    salt_flat_evaporite = np.where((lat_abs < 42.0) & (~is_water), high_albedo_mask, 0.0)

    is_wet_boreal_tundra = (lat_abs > 46.0) & (snow_ice_spectral < 0.25) & (bare_soil_warmth < 0.12)

    valid_bare_land = (
        (~is_water)
        & (~is_vegetation)
        & (~is_wet_boreal_tundra)
        & (lat_grid > -82.0)
        & (lat_grid < 83.0)
    )

    arid_mineral_norm = np.clip((bare_soil_warmth - 0.06) * 3.8 + 0.50 * salt_flat_evaporite, 0.0, 1.0)
    arid_belt_weight = np.clip(1.0 - np.maximum(0.0, lat_abs - 42.0) / 15.0, 0.10, 1.0)

    lat_permafrost = np.clip((lat_abs - 45.0) / 25.0, 0.0, 1.0)
    high_alt_cryo = np.clip(rough_norm * np.clip((lat_abs - 28.0) / 30.0, 0.0, 1.0), 0.0, 1.0)
    f3_ice_isru = np.clip(
        np.maximum(snow_ice_spectral, 0.82 * lat_permafrost + 0.18 * high_alt_cryo),
        0.03, 0.99
    )

    mafic_rock_albedo = np.exp(-((albedo - 0.36) ** 2) / (2.0 * (0.09 ** 2)))
    f0_basalt = np.clip(
        (0.50 * np.maximum(mafic_rock_albedo, salt_flat_evaporite * 0.75) * arid_mineral_norm * arid_belt_weight
         + 0.30 * rough_norm
         + 0.20 * pit_collapse_norm) * (1.0 - 0.45 * snow_ice_spectral),
        0.05, 0.99
    )

    f1_flatness = np.clip(1.0 - rough_norm * 1.05, 0.05, 0.98)

    solar_arid_shock = arid_mineral_norm * arid_belt_weight
    polar_cold_shock = np.clip((lat_abs - 52.0) / 32.0, 0.0, 1.0)
    f2_thermal = np.clip(
        0.25 + 0.55 * np.maximum(solar_arid_shock, polar_cold_shock) + 0.20 * rough_norm,
        0.15, 0.99
    )

    f4_aeolian = np.clip(
        (0.60 * arid_mineral_norm + 0.40 * f1_flatness) * (1.0 - 0.85 * f3_ice_isru),
        0.05, 0.99
    )

    f5_subsurface = np.clip(
        (0.50 * f0_basalt + 0.35 * pit_collapse_norm + 0.15 * rough_norm) * (1.0 - 0.50 * f3_ice_isru),
        0.05, 0.99
    )

    ys, xs = np.where(valid_bare_land)
    lats_arr = np.round(lat_grid[ys, xs], 3).astype(np.float32)
    lons_arr = np.round(lon_grid[ys, xs], 3).astype(np.float32)

    features_arr = np.stack([
        f0_basalt[ys, xs],
        f1_flatness[ys, xs],
        f2_thermal[ys, xs],
        f3_ice_isru[ys, xs],
        f4_aeolian[ys, xs],
        f5_subsurface[ys, xs]
    ], axis=1).astype(np.float32)

    well_known_arr = np.array(
        [is_in_overstudied_classic_zone(lat, lon) for lat, lon in zip(lats_arr, lons_arr)],
        dtype=np.int8
    )

    np.savez_compressed(
        "data/global_grids/earth_6d_grid.npz",
        lats=lats_arr,
        lons=lons_arr,
        features_6d=features_arr,
        is_well_known=well_known_arr
    )


if __name__ == "__main__":
    ensure_directories()
    build_moon_mars_reference_catalog()
    build_real_global_earth_matrix()
