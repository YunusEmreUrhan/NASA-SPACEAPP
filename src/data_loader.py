import os
import csv
import json
import math
import threading
from datetime import datetime, timezone
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

# P0 Madde 1: Zorunlu Veri Durum Etiketleri
VALID_STATUS_LABELS = (
    "NASA'dan al?nd?",
    "NASA verisinden i?lendi",
    "senaryo varsay?m?",
    "sim?lasyon",
    "veri yok"
)

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
    "AR": "Argentina", "BR": "Brazil", "CL": "Chile", "PE": "Peru", "MX": "Mexico",
    "US": "United States", "CA": "Canada", "GL": "Greenland", "NO": "Norway",
    "IS": "Iceland", "FO": "Faroe Islands", "SJ": "Svalbard, Norway",
    "RU": "Russia", "IN": "India", "ES": "Spain", "IT": "Italy"
}


class PlanetaryDataLoader:
    """
    P0-P3 Bilimsel Veri Denetimi Uyumlu Veri Y?kleyici:
    1) A? hatas?nda veya ?evrimd??? durumda ASLA kural tabanl?/sentetik say? ?retmez; 'veri yok' d?ner.
    2) NASA Trek WMTS ve GIBS JPEG katmanlar?ndan fiziksel say? (e?im, r?lyef, ma?ara derinli?i) T?RETMEZ;
       g?r?nt?leri yaln?zca 'g?rsel ba?lam (visual context)' olarak sunar.
    3) Her ?l??m i?in kaynak URL, edinim zaman?, ?r?n/s?r?m, birim, koordinat/ayak izi ve kalite bayra?? ta??r.
    """

    def __init__(self, offline_mode: bool = False):
        self.offline_mode = offline_mode
        self.dirs = {
            "Moon": "data/raw_moon",
            "Mars": "data/raw_mars",
            "Earth": "data/raw_earth"
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "TERRA-TWIN-ScientificAudit/7.0",
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
                coords, records = [], []
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
                    data = json.load(f)
                for k in ("climate", "geocode", "dem_srtm"):
                    if k not in data:
                        data[k] = {}
                return data
            except Exception:
                pass
        return {"climate": {}, "geocode": {}, "dem_srtm": {}}

    def _save_telemetry_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
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

    def fetch_real_earth_dem_transect(self, lat: float, lon: float, radius_km: float = 5.0, num_samples: int = 11) -> dict:
        """
        P0 Madde 1 & 2: Yaln?zca kalibre edilmi? Copernicus GLO-90 / NASA SRTM 90m DEM servisinden
        birimli (metre) y?kseklik ?eker. A? kapal?ysa veya servis yan?t vermezse g?rselden veya sin?s
        e?risinden de?er UYDURMAZ; a??k?a status_label='veri yok' ve None d?ner.
        """
        key = f"{round(lat, 3)}_{round(lon, 3)}_{int(radius_km)}km"
        if not self.offline_mode:
            with self._cache_lock:
                if key in self.telemetry_cache.get("dem_srtm", {}):
                    cached = self.telemetry_cache["dem_srtm"][key]
                    if cached.get("elevations_m") and len(cached["elevations_m"]) >= 5:
                        cached["status_label"] = "NASA verisinden i?lendi"
                        cached["data_status"] = "MEASURED_SRTM_DEM"
                        return cached

        min_lon, min_lat, max_lon, max_lat = self._get_bbox(lat, lon, radius_km, body="Earth")
        sample_lons = np.linspace(min_lon, max_lon, num_samples)
        sample_lats = np.linspace(min_lat, max_lat, num_samples)

        lat_str = ",".join([f"{x:.4f}" for x in sample_lats])
        lon_str = ",".join([f"{x:.4f}" for x in sample_lons])
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_str}&longitude={lon_str}"

        if not self.offline_mode:
            try:
                resp = self.session.get(url, timeout=(3.0, 4.5))
                if resp.status_code == 200:
                    elevs = resp.json().get("elevation", [])
                    valid_elevs = [float(e) for e in elevs if e is not None and float(e) > -500.0]
                    if len(valid_elevs) == num_samples:
                        elev_arr = [round(e, 1) for e in valid_elevs]
                        res = {
                            "elevations_m": elev_arr,
                            "min_elev_m": round(float(np.min(elev_arr)), 1),
                            "max_elev_m": round(float(np.max(elev_arr)), 1),
                            "relief_m": round(float(np.ptp(elev_arr)), 1),
                            "uncertainty_m": 8.0,
                            "unit": "m",
                            "valid_cell_ratio": 1.0,
                            "footprint_km": round(radius_km * 2.0, 1),
                            "product_version": "Copernicus DEM GLO-90 / NASA SRTM 90m v3.0",
                            "source": "Copernicus DEM GLO-90 / NASA SRTM 90m Radar Altimetry",
                            "source_url": url,
                            "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "quality_flag": "QF-1 (Valid DEM Transect, Nodata=0%)",
                            "processing_step": "Diagonal 11-point orthometric elevation sampling along footprint",
                            "status_label": "NASA verisinden i?lendi",
                            "data_status": "MEASURED_SRTM_DEM"
                        }
                        with self._cache_lock:
                            self.telemetry_cache["dem_srtm"][key] = res
                            self._save_telemetry_cache()
                        return res
            except Exception:
                pass

        # P0 Kabul ?l??t?: Servis ba?ar?s?zsa veya ?evrimd???ysa ASLA de?er uydurma!
        return {
            "elevations_m": None,
            "min_elev_m": None,
            "max_elev_m": None,
            "relief_m": None,
            "uncertainty_m": None,
            "unit": "m",
            "valid_cell_ratio": 0.0,
            "footprint_km": round(radius_km * 2.0, 1),
            "product_version": "Copernicus GLO-90 / SRTM 90m",
            "source": "DEM Service Unreachable (Zero Imputation)",
            "source_url": url,
            "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "quality_flag": "QF-0 (Missing / Network Error - Not Imputed)",
            "processing_step": "Excluded from scoring due to missing DEM response",
            "status_label": "veri yok",
            "data_status": "UNAVAILABLE"
        }

    def _fetch_nasa_trek_wmts_mosaic(self, body_cap: str, lat: float, lon: float, size: int = 512):
        if self.offline_mode:
            return None
        z = 7
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
        half_w = 90
        x1 = int(np.clip(px_center - half_w, 0, 512 - 2 * half_w))
        y1 = int(np.clip(py_center - half_w, 0, 512 - 2 * half_w))
        cropped = canvas[y1:y1 + 2 * half_w, x1:x1 + 2 * half_w]
        return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_CUBIC)

    def fetch_planetary_patch(self, body: str, lat: float, lon: float, radius_km: float = 5.0, size: int = 512):
        """
        P0 Madde 5: NASA Trek WMTS karolar? YALNIZCA g?rsel ba?lam (visual context) i?in indirilir.
        Karo yoksa s?f?r/sentetik g?r?nt? ?retilmez, bo? matris ve 'veri yok' d?ner.
        """
        body_cap = "Moon" if body.lower() == "moon" else "Mars"
        cache_file = os.path.join(
            self.dirs[body_cap],
            f"{body_cap.lower()}_{lat:.3f}_{lon:.3f}_{int(radius_km)}km.jpg"
        )

        if not self.offline_mode and os.path.exists(cache_file):
            img = cv2.imread(cache_file)
            if img is not None and img.size > 0:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cache_file

        trek_rgb = self._fetch_nasa_trek_wmts_mosaic(body_cap, lat, lon, size=size)
        if trek_rgb is not None:
            cv2.imwrite(cache_file, cv2.cvtColor(trek_rgb, cv2.COLOR_RGB2BGR))
            return trek_rgb, cache_file

        return np.zeros((size, size, 3), dtype=np.uint8), ""

    def extract_dynamic_target_profile(self, body: str, lat: float, lon: float, radius_km: float = 5.0):
        """
        P0 Madde 2 & 5: Serbest se?ilen Ay/Mars koordinat? i?in WMTS JPEG karosunun renklerinden
        e?im, r?lyef veya ma?ara derinli?i ?IKARMAZ!
        Do?rudan ?l??m raster? y?kl? de?ilse alanlar? 'veri yok' b?rak?r; Stefan-Boltzmann hesab?n?
        ise a??k?a 'senaryo varsay?m? / analitik model' olarak etiketler ve fiziksel ?l??m skoruna sokmaz.
        """
        rgb, img_path = self.fetch_planetary_patch(body, lat, lon, radius_km)
        body_cap = "Moon" if body.lower() == "moon" else "Mars"
        has_visual_tile = bool(img_path and np.max(rgb) > 0)

        lat_abs = abs(lat)
        cos_lat = max(0.06, math.cos(math.radians(min(86.5, lat_abs))))
        sigma_sb = 5.670374419e-8
        assumed_albedo = 0.12 if body_cap == "Moon" else 0.17
        s0_wm2 = 1361.0 if body_cap == "Moon" else 589.0
        t_eq_k = (((1.0 - assumed_albedo) * s0_wm2 * cos_lat) / (4.0 * sigma_sb * 0.95)) ** 0.25
        if body_cap == "Mars":
            t_eq_k += 5.5
        mean_temp_c = round(float(t_eq_k - 273.15), 1)

        # P0 Madde 5 Kabul ?l??t?: G?rsel karodan fiziksel e?im/r?lyef ?retilmez -> None ('veri yok')
        pm = {
            "structure_diameter_km": None,
            "regional_basin_diameter_km": round(radius_km * 2.0, 1),
            "mean_slope_deg": None,
            "slope_uncertainty_deg": None,
            "vertical_relief_m": None,
            "relief_uncertainty_m": None,
            "bolometric_albedo": None,
            "measurement_source": (
                "NASA Trek WMTS (Yaln?zca G?rsel Ba?lam ? Say?sal DEM Y?kl? De?il)"
                if has_visual_tile else "WMTS Kapsamas? Yok / Veri Yok"
            ),
            "official_url": "https://trek.nasa.gov/",
            "status_label": "senaryo varsay?m?"
        }

        observations = {
            "feature_scale_km": {
                "value": round(radius_km * 2.0, 1),
                "uncertainty": 0.5,
                "unit": "km",
                "status_label": "senaryo varsay?m?",
                "product": "Kullan?c? Se?imli ?nceleme Penceresi ?ap?",
                "epoch": "N/A",
                "quality_flag": "QF-3 (User Window Assumption)",
                "source_url": "https://trek.nasa.gov/"
            },
            "mean_slope_deg": {
                "value": None,
                "uncertainty": None,
                "unit": "deg",
                "status_label": "veri yok",
                "product": "Yerel LOLA/MOLA Say?sal DEM Sorgulanmad? (WMTS JPEG'den E?im T?retilmez)",
                "epoch": "N/A",
                "quality_flag": "QF-0 (Missing Paired DEM)",
                "source_url": "https://trek.nasa.gov/"
            },
            "vertical_relief_m": {
                "value": None,
                "uncertainty": None,
                "unit": "m",
                "status_label": "veri yok",
                "product": "Yerel LOLA/MOLA Say?sal DEM Sorgulanmad?",
                "epoch": "N/A",
                "quality_flag": "QF-0 (Missing Paired DEM)",
                "source_url": "https://trek.nasa.gov/"
            },
            "surface_temp_k": {
                "value": round(float(t_eq_k), 1),
                "uncertainty": 15.0,
                "unit": "K",
                "status_label": "sim?lasyon",
                "product": "1B Stefan-Boltzmann Enlemsel Radyatif Denge Modeli (NASA ?l??m? De?ildir)",
                "epoch": "Analitik",
                "quality_flag": "QF-3 (Analytical Simulation - Excluded from NASA Score)",
                "source_url": "https://nssdc.gsfc.nasa.gov/planetary/factsheet/"
            }
        }

        lat_norm = lat_abs / 90.0
        ice_proxy = round(float(np.clip((lat_abs - 45.0) / 45.0, 0.05, 0.95)), 3)
        return {
            "id": f"DYNAMIC_{body_cap.upper()}_{lat:.2f}_{lon:.2f}",
            "body": body_cap,
            "name": f"{body_cap} Serbest ?nceleme Noktas? ({lat:.2f}?, {lon:.2f}?)",
            "lat": round(lat, 3),
            "lon": round(lon, 3),
            "radius_km": radius_km,
            "footprint_km": round(radius_km * 2.0, 1),
            "coordinate_system": f"IAU {body_cap} 2000 Planetocentric",
            "data_status": "senaryo varsay?m?",
            "gravity_g": 0.165 if body_cap == "Moon" else 0.378,
            "pressure_mbar": 0.0 if body_cap == "Moon" else 6.1,
            "mean_temp_c": mean_temp_c,
            "radiation_msv_yr": 380.0 if body_cap == "Moon" else 230.0,
            "physical_measurements": pm,
            "observations": observations,
            "nasa_missions": ["NASA Trek WMTS (Visual Context Only)"],
            "primary_process": "Kullan?c? Tan?ml? Koordinat ?n Taramas? (Senaryo Varsay?m?)",
            "mission_focus": "Ara?t?rma ?nceli?i ?n Taramas? (Say?sal DEM E?le?mesi Yok)",
            "image_path": img_path,
            "features_6d": [0.65, 0.65, round(1.0 - 0.5 * lat_norm, 3), ice_proxy, 0.50, 0.50],
            "description": (
                f"[senaryo varsay?m?] Serbest koordinat ({lat:.2f}?, {lon:.2f}?). "
                "WMTS g?r?nt? karosundan fiziksel e?im/r?lyef t?retilmemi?tir (veri yok)."
            )
        }

    def fetch_earth_satellite_patch(self, lat: float, lon: float, radius_km: float = 5.0, size: int = 512):
        cache_file = os.path.join(
            self.dirs["Earth"],
            f"earth_{lat:.3f}_{lon:.3f}_{int(radius_km)}km.jpg"
        )

        if not self.offline_mode and os.path.exists(cache_file):
            img = cv2.imread(cache_file)
            if img is not None and img.size > 0:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cache_file

        if not self.offline_mode:
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

        return np.zeros((size, size, 3), dtype=np.uint8), ""

    def fetch_nasa_power_climate(self, lat: float, lon: float):
        """
        P0 Madde 1 & 4: NASA POWER D?nya Klimatoloji API'si.
        - D?nya'ya ait 2m/y?zey meteorolojik ba?lam verisidir; Ay/Mars y?zey s?cakl???yla do?rudan
          e?de?er tutulamaz (yaln?zca D?nya saha ba?lam? olarak raporlan?r).
        - A? hatas?nda veya ?evrimd??? durumda ASLA de?er uydurmaz; status_label='veri yok' d?ner.
        """
        key = f"{round(lat, 2)}_{round(lon, 2)}"
        url = (
            f"https://power.larc.nasa.gov/api/temporal/climatology/point"
            f"?parameters=TS,ALLSKY_SFC_SW_DWN,PRECTOTCORR,WS10M,RH2M"
            f"&community=SB&longitude={round(lon, 3)}&latitude={round(lat, 3)}&format=JSON"
        )

        if not self.offline_mode:
            with self._cache_lock:
                if key in self.telemetry_cache["climate"]:
                    cached = self.telemetry_cache["climate"][key]
                    if "Live API" in cached.get("source", "") and cached.get("mean_surface_temp_c") is not None:
                        sw_c = cached.get("solar_irradiance_kwh_m2_day")
                        if sw_c is not None and float(sw_c) > 24.0:
                            cached["solar_irradiance_kwh_m2_day"] = round(float(sw_c) * 0.024, 2)
                        cached["status_label"] = "NASA'dan al?nd?"
                        cached["data_status"] = "MEASURED_NASA_POWER"
                        cached["source_url"] = url
                        cached["temporal_coverage"] = "30-Year Climatology (1991-2020)"
                        cached["quality_flag"] = "QF-1 (NASA POWER Validated - Earth Context Only)"
                        return cached

            try:
                resp = self.session.get(url, timeout=(2.5, 4.0))
                resp.raise_for_status()
                p = resp.json()["properties"]["parameter"]

                def _read_valid(val):
                    v = float(val)
                    return None if v < -900.0 else round(v, 2)

                ts = _read_valid(p["TS"]["ANN"])
                sw_raw = _read_valid(p["ALLSKY_SFC_SW_DWN"]["ANN"])
                sw = round(sw_raw * 0.024, 2) if (sw_raw is not None and sw_raw > 24.0) else sw_raw
                pr = _read_valid(p["PRECTOTCORR"]["ANN"])
                ws = _read_valid(p["WS10M"]["ANN"])
                rh = _read_valid(p["RH2M"]["ANN"])

                if ts is not None:
                    res = {
                        "mean_surface_temp_c": round(ts, 1),
                        "solar_irradiance_kwh_m2_day": sw,
                        "precipitation_mm_day": pr,
                        "wind_speed_10m_ms": ws,
                        "relative_humidity_pct": round(rh, 1) if rh is not None else None,
                        "temporal_coverage": "30-Year Climatology (1991-2020)",
                        "product_version": "NASA POWER Release 9 (MERRA-2 / GEWEX SRB)",
                        "source": "NASA POWER Langley Satellite Climatology (Live API)",
                        "source_url": url,
                        "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "quality_flag": "QF-1 (Validated Earth Meteorology - Context Only)",
                        "status_label": "NASA'dan al?nd?",
                        "data_status": "MEASURED_NASA_POWER"
                    }
                    with self._cache_lock:
                        self.telemetry_cache["climate"][key] = res
                        self._save_telemetry_cache()
                    return res
            except Exception:
                pass

        return {
            "mean_surface_temp_c": None,
            "solar_irradiance_kwh_m2_day": None,
            "precipitation_mm_day": None,
            "wind_speed_10m_ms": None,
            "relative_humidity_pct": None,
            "temporal_coverage": "N/A",
            "product_version": "NASA POWER Release 9",
            "source": "NASA POWER Unreachable (Zero Imputation)",
            "source_url": url,
            "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "quality_flag": "QF-0 (Missing / Network Error - Not Imputed)",
            "status_label": "veri yok",
            "data_status": "UNAVAILABLE"
        }

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
        if lat <= -60.0:
            name = f"Antarctic Polar Ice Sheet & Dry Valleys Sector, Antarctica ({lat:.2f}?, {lon:.2f}?)"
        elif (60.0 <= lat <= 75.0 and -58.0 <= lon <= -12.0) or (75.0 < lat <= 83.5 and -73.0 <= lon <= -12.0):
            name = f"Northeast Greenland Ice-Sheet & Fjord Margin, Greenland ({lat:.2f}?, {lon:.2f}?)"
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
            name = f"Terrestrial Analog Zone ({lat:.2f}?, {lon:.2f}?)"

        with self._cache_lock:
            self.telemetry_cache["geocode"][key] = name
            self._save_telemetry_cache()
        return name
