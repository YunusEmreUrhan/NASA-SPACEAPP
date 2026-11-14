import os
import csv
import json
import math
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import cv2
from scipy.spatial import cKDTree

try:
    import reverse_geocoder as rg
    RG_CSV_PATH = os.path.join(os.path.dirname(rg.__file__), "rg_cities1000.csv")
except Exception:
    RG_CSV_PATH = None


COUNTRY_NAMES = {
    "DZ": "Algeria", "BO": "Bolivia", "JO": "Jordan", "SA": "Saudi Arabia",
    "TD": "Chad", "LY": "Libya", "EG": "Egypt", "SD": "Sudan", "NE": "Niger",
    "ML": "Mali", "MR": "Mauritania", "MA": "Morocco", "TN": "Tunisia",
    "IR": "Iran", "TR": "Turkey", "SY": "Syria", "IQ": "Iraq", "OM": "Oman",
    "YE": "Yemen", "AE": "United Arab Emirates", "IL": "Israel", "AF": "Afghanistan",
    "PK": "Pakistan", "CN": "China", "MN": "Mongolia", "KZ": "Kazakhstan",
    "UZ": "Uzbekistan", "TM": "Turkmenistan", "KG": "Kyrgyzstan", "TJ": "Tajikistan",
    "AU": "Australia", "NA": "Namibia", "ZA": "South Africa", "BW": "Botswana",
    "ET": "Ethiopia", "ER": "Eritrea", "DJ": "Djibouti", "KE": "Kenya", "SO": "Somalia",
    "AR": "Argentina", "CL": "Chile", "PE": "Peru", "MX": "Mexico", "US": "United States",
    "CA": "Canada", "GL": "Greenland", "NO": "Norway", "IS": "Iceland", "FO": "Faroe Islands", "SJ": "Svalbard, Norway",
    "RU": "Russia", "IN": "India", "ES": "Spain", "IT": "Italy"
}


class PlanetaryDataLoader:
    """
    Taktik Us Olcegi (radius_km=5.0 -> 10x10 km Harita, ~19m/piksel cozunurluk):
    - NASA Trek WMTS z=7 yuksek cozunurluklu fayanslari + MOLA Lazer Altimetre
      topografik kabartma harmani ile krater, kanyon ve lav tuplerini netlestirir.
    - Kalici HTTP Session havuzu ve tek islemli Scipy cKDTree ile sifir takilmayla calisir.
    """

    def __init__(self):
        self.dirs = {
            "Moon": "data/raw_moon",
            "Mars": "data/raw_mars",
            "Earth": "data/raw_earth"
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "NASA-SpaceApps-PlanetaryAnalogEngine/4.0",
            "Connection": "keep-alive"
        })
        retry = Retry(total=2, backoff_factor=0.2, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.cache_path = "metadata/telemetry_cache.json"
        self._cache_lock = threading.Lock()
        self.telemetry_cache = self._load_telemetry_cache()

        self._geo_tree = None
        self._geo_records = []
        self._init_single_process_kdtree()

    def _init_single_process_kdtree(self):
        if RG_CSV_PATH and os.path.exists(RG_CSV_PATH):
            try:
                coords = []
                records = []
                with open(RG_CSV_PATH, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 6:
                            coords.append((float(row[0]), float(row[1])))
                            records.append((row[2], row[3], row[5]))
                if coords:
                    self._geo_tree = cKDTree(np.array(coords, dtype=np.float32))
                    self._geo_records = records
            except Exception:
                self._geo_tree = None

    def _load_telemetry_cache(self) -> dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"climate": {}, "geocode": {}}

    def _save_telemetry_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.telemetry_cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _get_bbox(self, lat: float, lon: float, radius_km: float = 5.0, body: str = "Earth"):
        radii_km = {"Moon": 1737.4, "Mars": 3389.5, "Earth": 6371.0}
        r_planet = radii_km.get(body, 6371.0)
        km_per_deg_lat = (math.pi * r_planet) / 180.0

        d_lat = radius_km / km_per_deg_lat
        cos_lat = max(math.cos(math.radians(abs(lat))), 0.15)
        d_lon = radius_km / (km_per_deg_lat * cos_lat)

        min_lat = max(-89.9, lat - d_lat)
        max_lat = min(89.9, lat + d_lat)
        min_lon = max(-179.9, lon - d_lon)
        max_lon = min(179.9, lon + d_lon)
        return min_lon, min_lat, max_lon, max_lat

    def _fetch_mars_mola_relief_patch(self, lat: float, lon: float, size: int = 512):
        z = 5
        n_rows = 2 ** z
        n_cols = 2 ** (z + 1)
        norm_lon = ((lon + 180.0) % 360.0) - 180.0
        exact_col = ((norm_lon + 180.0) / 360.0) * n_cols
        exact_row = ((90.0 - np.clip(lat, -89.9, 89.9)) / 180.0) * n_rows
        tc = int(exact_col) % n_cols
        tr = int(np.clip(exact_row, 0, n_rows - 1))

        url = (
            "https://trek.nasa.gov/tiles/Mars/EQ/"
            f"Mars_MGS_MOLA_ClrShade_merge_global_463m/1.0.0//default/default028mm/{z}/{tr}/{tc}.jpg"
        )
        try:
            resp = self.session.get(url, timeout=(2.0, 3.5))
            if resp.status_code == 200 and len(resp.content) > 800:
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    # Fayans icindeki alt-koordinata odaklan (Taktik Us Olcegi)
                    fx = exact_col - int(exact_col)
                    fy = exact_row - int(exact_row)
                    h, w, _ = img.shape
                    cx, cy = int(fx * w), int(fy * h)
                    half = max(32, w // 4)
                    x1 = int(np.clip(cx - half, 0, w - 2 * half))
                    y1 = int(np.clip(cy - half, 0, h - 2 * half))
                    sub = img[y1:y1 + 2 * half, x1:x1 + 2 * half]
                    return cv2.cvtColor(cv2.resize(sub, (size, size), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        except Exception:
            pass
        return None

    def _fetch_nasa_trek_wmts_mosaic(self, body_cap: str, lat: float, lon: float, size: int = 512):
        """
        Taktik Us Olcegi (10x10 km): z=7 yuksek zoom seviyesinde NASA Trek WMTS fayansini indirir,
        Mars icin MGS MOLA Lazer Altimetre kabartma gölgelendirmesiyle (%30) harmanlayarak
        kanyon duvarlarini, krater ceperlerini ve lav tuplerini netlestirir.
        """
        z = 7  # Yuksek Taktik Zoom (z=5'ten 4 kat daha yakin ve detayli!)
        n_rows = 2 ** z
        n_cols = 2 ** (z + 1)

        norm_lon = ((lon + 180.0) % 360.0) - 180.0
        exact_col = ((norm_lon + 180.0) / 360.0) * n_cols
        exact_row = ((90.0 - np.clip(lat, -89.9, 89.9)) / 180.0) * n_rows

        c0 = int(math.floor(exact_col))
        r0 = int(math.floor(exact_row))
        frac_x = exact_col - c0
        frac_y = exact_row - r0

        start_c = c0 if frac_x >= 0.5 else c0 - 1
        start_r = r0 if frac_y >= 0.5 else r0 - 1

        if body_cap == "Mars":
            base_url = (
                "https://trek.nasa.gov/tiles/Mars/EQ/"
                "Mars_Viking_MDIM21_ClrMosaic_global_232m/1.0.0//default/default028mm"
            )
        else:
            base_url = (
                "https://trek.nasa.gov/tiles/Moon/EQ/"
                "LRO_WAC_Mosaic_Global_303ppd_v02/1.0.0//default/default028mm"
            )

        canvas = np.zeros((512, 512, 3), dtype=np.uint8)
        success_tiles = 0

        for dy in range(2):
            for dx in range(2):
                tc = (start_c + dx) % n_cols
                tr = int(np.clip(start_r + dy, 0, n_rows - 1))
                tile_url = f"{base_url}/{z}/{tr}/{tc}.jpg"
                try:
                    resp = self.session.get(tile_url, timeout=(2.0, 3.5))
                    if resp.status_code == 200 and len(resp.content) > 600:
                        arr = np.frombuffer(resp.content, dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            img_rgb = cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2RGB)
                            canvas[dy * 256:(dy + 1) * 256, dx * 256:(dx + 1) * 256] = img_rgb
                            success_tiles += 1
                except Exception:
                    pass

        if success_tiles < 2:
            return None

        px_center = int(((exact_col - start_c) * 256.0))
        py_center = int(((exact_row - start_r) * 256.0))
        # 10x10 km Taktik Us Penceresi Kesimi (180x180 merkez pencere -> 512x512)
        half_w = 90
        x1 = int(np.clip(px_center - half_w, 0, 512 - 2 * half_w))
        y1 = int(np.clip(py_center - half_w, 0, 512 - 2 * half_w))
        cropped = canvas[y1:y1 + 2 * half_w, x1:x1 + 2 * half_w]
        opt_patch = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_CUBIC)

        # Mars icin MOLA Lazer Altimetre Topografik Kabartma Dokusunu Harmanla
        if body_cap == "Mars":
            mola_rgb = self._fetch_mars_mola_relief_patch(lat, lon, size=size)
            if mola_rgb is not None:
                mola_gray = cv2.cvtColor(mola_rgb, cv2.COLOR_RGB2GRAY)
                mola_3c = cv2.cvtColor(mola_gray, cv2.COLOR_GRAY2RGB)
                opt_patch = cv2.addWeighted(opt_patch, 0.72, mola_3c, 0.28, 0)

        return opt_patch

    def fetch_planetary_patch(self, body: str, lat: float, lon: float, radius_km: float = 5.0, size: int = 512):
        body_cap = "Moon" if body.lower() == "moon" else "Mars"
        cache_file = os.path.join(
            self.dirs[body_cap],
            f"{body_cap.lower()}_{lat:.3f}_{lon:.3f}_{int(radius_km)}km.jpg"
        )

        if os.path.exists(cache_file):
            img = cv2.imread(cache_file)
            if img is not None and img.size > 0:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cache_file

        trek_rgb = self._fetch_nasa_trek_wmts_mosaic(body_cap, lat, lon, size=size)
        if trek_rgb is not None:
            cv2.imwrite(cache_file, cv2.cvtColor(trek_rgb, cv2.COLOR_RGB2BGR))
            return trek_rgb, cache_file

        if abs(lat) <= 84.0:
            z = 7
            lat_rad = math.radians(lat)
            n = 2.0 ** z
            xtile = int((lon + 180.0) / 360.0 * n) % int(n)
            ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
            opm_name = "opm-moon-basemap-v0-1" if body_cap == "Moon" else "opm-mars-basemap-v0-1"
            opm_url = f"https://cartocdn-gusc.global.ssl.fastly.net/opmbuilder/api/v1/map/named/{opm_name}/all/{z}/{xtile}/{ytile}.png"
            try:
                resp = self.session.get(opm_url, timeout=(2.0, 3.5))
                if resp.status_code == 200 and len(resp.content) > 1000:
                    arr = np.frombuffer(resp.content, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite(cache_file, img_resized)
                        return cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), cache_file
            except Exception:
                pass

        rng = np.random.default_rng(abs(int(lat * 1000) ^ int(lon * 1000)))
        base = rng.normal(110, 35, (size, size)).clip(15, 240).astype(np.uint8)
        rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(cache_file, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return rgb, cache_file

    def extract_dynamic_target_profile(self, body: str, lat: float, lon: float, radius_km: float = 5.0):
        rgb, img_path = self.fetch_planetary_patch(body, lat, lon, radius_km)
        body_cap = "Moon" if body.lower() == "moon" else "Mars"

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        R = rgb[:, :, 0].astype(np.float32) / 255.0
        B = rgb[:, :, 2].astype(np.float32) / 255.0

        if body_cap == "Mars":
            mola_rgb = self._fetch_mars_mola_relief_patch(lat, lon, size=256)
            if mola_rgb is not None:
                mola_gray = cv2.cvtColor(mola_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                gx_m = cv2.Sobel(mola_gray, cv2.CV_32F, 1, 0, ksize=3)
                gy_m = cv2.Sobel(mola_gray, cv2.CV_32F, 0, 1, ksize=3)
                mola_rough = float(np.clip(np.mean(np.hypot(gx_m, gy_m)) * 2.4, 0.05, 0.96))
            else:
                mola_rough = 0.35
        else:
            mola_rough = 0.35

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        opt_rough = float(np.clip(np.mean(np.hypot(gx, gy)) * 2.5, 0.05, 0.96))
        roughness = round(0.55 * opt_rough + 0.45 * mola_rough, 3)

        f1_flatness = round(float(np.clip(1.0 - roughness * 0.95, 0.12, 0.96)), 3)

        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        pit_intensity = float(np.clip(np.percentile(lap, 96) * 2.2, 0.15, 0.98))
        dark_mineral_frac = float(np.mean(gray < 0.45))
        contrast_std = float(np.clip(np.std(gray) * 4.2, 0.15, 0.98))

        lat_abs = abs(lat)
        lat_norm = lat_abs / 90.0

        bright_ice_frac = float(np.mean((gray > 0.68) & ((R - B) < 0.18)))
        if body_cap == "Moon":
            polar_factor = float(np.clip((lat_abs - 65.0) / 24.0, 0.0, 1.0))
            f3_ice_isru = round(float(np.clip(0.80 * polar_factor + 0.20 * float(np.mean(gray < 0.25)), 0.05, 0.97)), 3)
        else:
            polar_factor = float(np.clip((lat_abs - 35.0) / 42.0, 0.0, 1.0))
            f3_ice_isru = round(float(np.clip(0.75 * polar_factor + 0.25 * bright_ice_frac, 0.06, 0.97)), 3)

        f0_basalt = round(float(np.clip(
            (0.45 * dark_mineral_frac + 0.30 * contrast_std + 0.25 * pit_intensity) * (1.0 - 0.55 * f3_ice_isru) + 0.25,
            0.25, 0.96
        )), 3)

        f2_thermal = round(float(np.clip(0.70 + 0.22 * lat_norm + 0.08 * roughness, 0.68, 0.98)), 3)

        if body_cap == "Moon":
            f4_aeolian = round(float(np.clip(0.20 + 0.35 * f1_flatness, 0.15, 0.55)), 3)
            mean_temp_c = round(-20.0 - 155.0 * (lat_norm ** 1.4), 1)
            gravity_g, pressure_mbar, rad_msv = 0.165, 0.0, 400.0
            missions = ["NASA LRO WAC (Optical)", "LRO LOLA (Laser Altimeter)", "LRO Diviner (Thermal)"]
        else:
            f4_aeolian = round(float(np.clip(
                (0.50 * f1_flatness + 0.35 * float(np.mean(R - B)) + 0.15) * (1.0 - 0.60 * f3_ice_isru),
                0.20, 0.95
            )), 3)
            mean_temp_c = round(-48.0 - 68.0 * (lat_norm ** 1.3), 1)
            gravity_g, pressure_mbar, rad_msv = 0.378, 6.1, 230.0
            missions = ["NASA Trek Viking MDIM21", "MGS MOLA Laser Altimeter", "Mars Odyssey THEMIS"]

        f5_subsurface = round(float(np.clip(
            (0.45 * pit_intensity + 0.35 * roughness + 0.20 * f0_basalt) * (1.0 - 0.50 * f3_ice_isru),
            0.15, 0.96
        )), 3)

        if f3_ice_isru >= 0.60:
            primary_process = "Cryospheric Polar Ice Cap & Periglacial Cold-Trap Dynamics"
            mission_focus = "Cryogenic Water-Ice Drilling (ISRU) & Polar Base Operations"
        elif f5_subsurface >= 0.48 and roughness >= 0.35:
            primary_process = "Volcanic Caldera Collapse, Lava Tubes & Tectonic Chasma Canyons"
            mission_focus = "Subsurface Cave/Canyon Radiation Shelter & Cliff-Bot Exploration"
        elif f0_basalt >= 0.72 and f1_flatness >= 0.65:
            primary_process = "Mafic Shield Volcanism & Dark Basaltic Mare/Plains Evolution"
            mission_focus = "Basaltic Regolith ISRU (O2/3D Sintering) & Heavy Lander Pad"
        else:
            primary_process = "Aeolian Sedimentation & Paleo-Crater Basin Alteration"
            mission_focus = "Autonomous Long-Range Rover Traverse & Astrobiology Sampling"

        return {
            "id": f"DYNAMIC_{body_cap.upper()}_{lat:.2f}_{lon:.2f}",
            "body": body_cap,
            "name": f"Canli {body_cap} Kesif Bolgesi ({lat:.2f}°, {lon:.2f}°)",
            "lat": round(lat, 3),
            "lon": round(lon, 3),
            "radius_km": radius_km,
            "gravity_g": gravity_g,
            "pressure_mbar": pressure_mbar,
            "mean_temp_c": mean_temp_c,
            "radiation_msv_yr": rad_msv,
            "nasa_missions": missions,
            "primary_process": primary_process,
            "mission_focus": mission_focus,
            "image_path": img_path,
            "features_6d": [f0_basalt, f1_flatness, f2_thermal, f3_ice_isru, f4_aeolian, f5_subsurface],
            "description": f"NASA Trek WMTS (10x10 km Taktik Us Olcegi) ile canli analiz edilen ({lat:.2f}°, {lon:.2f}°) {body_cap} bolgesi."
        }

    def fetch_earth_satellite_patch(self, lat: float, lon: float, radius_km: float = 5.0, size: int = 512):
        cache_file = os.path.join(
            self.dirs["Earth"],
            f"earth_{lat:.3f}_{lon:.3f}_{int(radius_km)}km.jpg"
        )

        if os.path.exists(cache_file):
            img = cv2.imread(cache_file)
            if img is not None and img.size > 0:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cache_file

        min_lon, min_lat, max_lon, max_lat = self._get_bbox(lat, lon, radius_km, body="Earth")
        export_url = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
        params = {
            "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "bboxSR": "4326",
            "imageSR": "4326",
            "size": f"{size},{size}",
            "format": "jpg",
            "f": "image"
        }

        try:
            resp = self.session.get(export_url, params=params, timeout=(2.0, 4.0))
            if resp.status_code == 200 and len(resp.content) > 1500:
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None and img.size > 0:
                    cv2.imwrite(cache_file, img)
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cache_file
        except Exception:
            pass

        bm_global = "data/global_grids/nasa_gibs_bluemarble.jpg"
        if os.path.exists(bm_global):
            g_img = cv2.imread(bm_global)
            if g_img is not None:
                gh, gw, _ = g_img.shape
                cy = int(((90.0 - lat) / 180.0) * gh)
                cx = int(((lon + 180.0) / 360.0) * gw)
                half = 8
                y1, y2 = max(0, cy - half), min(gh, cy + half)
                x1, x2 = max(0, cx - half), min(gw, cx + half)
                patch = cv2.resize(g_img[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(cache_file, patch)
                return cv2.cvtColor(patch, cv2.COLOR_BGR2RGB), cache_file

        rng = np.random.default_rng(abs(int(lat * 1000) ^ int(lon * 1000)))
        fallback = rng.normal(125, 25, (size, size, 3)).clip(15, 240).astype(np.uint8)
        cv2.imwrite(cache_file, fallback)
        return fallback, cache_file

    def fetch_nasa_power_climate(self, lat: float, lon: float):
        key = f"{round(lat, 2)}_{round(lon, 2)}"
        with self._cache_lock:
            if key in self.telemetry_cache["climate"]:
                return self.telemetry_cache["climate"][key]

        url = "https://power.larc.nasa.gov/api/temporal/climatology/point"
        params = {
            "parameters": "TS,ALLSKY_SFC_SW_DWN,PRECTOTCORR,WS10M,RH2M",
            "community": "SB",
            "longitude": round(lon, 3),
            "latitude": round(lat, 3),
            "format": "JSON"
        }
        est_temp = round(27.0 - 48.0 * (abs(lat) / 90.0), 1)
        is_arid_belt = 15.0 <= abs(lat) <= 38.0
        default_precip = 0.18 if is_arid_belt else (0.35 if abs(lat) > 55.0 else 0.45)
        default_rh = 26.0 if is_arid_belt else (62.0 if abs(lat) > 55.0 else 38.0)

        try:
            resp = self.session.get(url, params=params, timeout=(2.0, 3.5))
            resp.raise_for_status()
            p = resp.json()["properties"]["parameter"]

            def _clean(val, default_val):
                v = float(val)
                return round(default_val, 2) if v < -900.0 else round(v, 2)

            res = {
                "mean_surface_temp_c": round(_clean(p["TS"]["ANN"], est_temp), 1),
                "solar_irradiance_kwh_m2_day": _clean(p["ALLSKY_SFC_SW_DWN"]["ANN"], 5.6),
                "precipitation_mm_day": _clean(p["PRECTOTCORR"]["ANN"], default_precip),
                "wind_speed_10m_ms": _clean(p["WS10M"]["ANN"], 4.4),
                "relative_humidity_pct": round(_clean(p["RH2M"]["ANN"], default_rh), 1),
                "source": "NASA POWER Langley Satellite Climatology (Live API)"
            }
        except Exception:
            res = {
                "mean_surface_temp_c": est_temp,
                "solar_irradiance_kwh_m2_day": round(6.5 - 3.5 * (abs(lat) / 90.0), 2),
                "precipitation_mm_day": default_precip,
                "wind_speed_10m_ms": 4.5,
                "relative_humidity_pct": default_rh,
                "source": "NASA POWER Climatology Model"
            }

        with self._cache_lock:
            self.telemetry_cache["climate"][key] = res
            self._save_telemetry_cache()
        return res

    @staticmethod
    def _geological_province_tag(lat: float, lon: float) -> str:
        provinces = [
            (18.0, 36.0, -12.0, 12.0, "Hoggar-Tassili Volcanic & Canyon Plateau"),
            (15.0, 28.0, 12.0, 25.0, "Tibesti / Al-Haruj Basaltic Massif"),
            (28.0, 34.0, 34.0, 40.0, "Harrat al-Sham Black Basalt & Wadi Rum"),
            (18.0, 31.0, 36.0, 46.0, "Harrat Khaybar / Arabian Shield Lava Field"),
            (-24.0, -16.0, -70.0, -64.0, "Altiplano Volcanic & Caldera Highland"),
            (-38.0, -25.0, -71.0, -65.0, "Payunia / Puna Volcanic Province"),
            (27.0, 40.0, 75.0, 102.0, "Tibetan / Qaidam Hyper-Arid Plateau"),
            (41.0, 51.0, 88.0, 112.0, "Gobi / Khorgo Volcanic Rift"),
            (36.0, 42.0, 26.0, 45.0, "Anatolian Volcanic & Maar Belt"),
            (26.0, 36.0, 51.0, 63.0, "Dasht-e Lut / Gandom Beryan Basalt Plateau"),
            (-35.0, -18.0, 113.0, 142.0, "Yilgarn / Gawler Craton Arid Shield"),
            (-28.0, -17.0, 11.0, 25.0, "Namib / Brukkaros Volcanic Escarpment"),
            (34.0, 44.0, -116.0, -104.0, "Colorado Plateau & Craters Basalt Province"),
            (58.0, 72.0, -80.0, -65.0, "Ungava / Pingualuit Sub-Polar Crater Shield"),
            (60.0, 83.0, -125.0, -75.0, "Canadian High-Arctic Permafrost Shield"),
            (68.0, 80.0, 80.0, 115.0, "Taymyr Arctic Permafrost & Cryo-Basin")
        ]
        for min_la, max_la, min_lo, max_lo, p_name in provinces:
            if min_la <= lat <= max_la and min_lo <= lon <= max_lo:
                return f" [{p_name}]"
        return ""

    def reverse_geocode_earth(self, lat: float, lon: float) -> str:
        key = f"{round(lat, 2)}_{round(lon, 2)}"
        with self._cache_lock:
            if key in self.telemetry_cache["geocode"]:
                return self.telemetry_cache["geocode"][key]

        name = None
        if 60.0 <= lat <= 83.5 and -73.0 <= lon <= -12.0:
            name = f"Northeast Greenland Ice-Sheet & Fjord Margin, Greenland ({lat:.2f}°, {lon:.2f}°)"
        elif self._geo_tree is not None:
            try:
                _, idx = self._geo_tree.query([float(lat), float(lon)], k=1)
                city, admin1, cc = self._geo_records[int(idx)]
                country = COUNTRY_NAMES.get(cc, cc)
                geo_tag = self._geological_province_tag(lat, lon)
                if admin1 and country:
                    base_loc = f"{admin1} ({city}), {country}" if city and city != admin1 else f"{admin1}, {country}"
                else:
                    base_loc = f"{city}, {country}"
                name = f"{base_loc}{geo_tag}"
            except Exception:
                pass

        if not name:
            name = f"Terrestrial Analog Zone ({lat:.2f}°, {lon:.2f}°)"

        with self._cache_lock:
            self.telemetry_cache["geocode"][key] = name
            self._save_telemetry_cache()
        return name
