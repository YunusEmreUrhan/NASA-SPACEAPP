import os
import sys
import io
import json
import math
import warnings
import logging
import contextlib
from concurrent.futures import ThreadPoolExecutor

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import numpy as np
import cv2
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import PlanetaryDataLoader
from preprocessor import PlanetaryPreprocessor


DIMENSION_NAMES = [
    "Litoloji & Bazalt (Geology)",
    "Düzlük & İniş Güvenliği (Slope <5°)",
    "Termal Ekstrem (Thermal Shock)",
    "Su Buzu & Permafrost (ISRU Water)",
    "Rüzgar & Toz Süreci (Aeolian/Dust)",
    "Lav Tüpü & Mağara (Subsurface Shelter)"
]

MISSION_WEIGHT_PRESETS = {
    "BALANCED": [0.20, 0.20, 0.15, 0.15, 0.15, 0.15],
    "HABITAT_LAVA_TUBE": [0.25, 0.15, 0.15, 0.10, 0.05, 0.30],
    "ISRU_WATER_MINING": [0.15, 0.15, 0.20, 0.40, 0.05, 0.05],
    "ROVER_LANDING_OPS": [0.20, 0.30, 0.10, 0.05, 0.25, 0.10]
}


class PlanetaryMatcherEngine:
    def __init__(
        self,
        grid_path: str = "data/global_grids/earth_6d_grid.npz",
        catalog_path: str = "metadata/moon_mars_targets.json",
        clip_mode: str = "auto"
    ):
        self.loader = PlanetaryDataLoader()
        self.preprocessor = PlanetaryPreprocessor(target_size=224)

        if not os.path.exists(grid_path):
            raise FileNotFoundError(
                f"Kuresel matris bulunamadi: {grid_path}. Once build_planetary_data.py calistirin."
            )

        grid_data = np.load(grid_path)
        self.lats = grid_data["lats"]
        self.lons = grid_data["lons"]
        self.features_6d = grid_data["features_6d"]
        self.is_well_known = grid_data["is_well_known"]

        self.catalog = {}
        if os.path.exists(catalog_path):
            with open(catalog_path, "r", encoding="utf-8") as f:
                self.catalog = json.load(f)

        self.use_clip = False
        self.clip_model = None
        self.clip_processor = None
        self.device = "cpu"
        self._init_vision_backbone(clip_mode)

    def _init_vision_backbone(self, clip_mode: str):
        if clip_mode == "off":
            return

        model_id = "openai/clip-vit-base-patch32"
        try:
            import torch
            import transformers
            from transformers import CLIPProcessor, CLIPModel

            transformers.logging.set_verbosity_error()
            if hasattr(transformers.utils.logging, "disable_progress_bar"):
                transformers.utils.logging.disable_progress_bar()
            logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
            logging.getLogger("transformers").setLevel(logging.CRITICAL)

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            if clip_mode == "auto":
                try:
                    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                        self.clip_model = CLIPModel.from_pretrained(
                            model_id, local_files_only=True
                        ).to(self.device)
                        self.clip_processor = CLIPProcessor.from_pretrained(
                            model_id, local_files_only=True
                        )
                    self.clip_model.eval()
                    self.use_clip = True
                    return
                except Exception:
                    return

            if clip_mode == "force":
                with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                    self.clip_model = CLIPModel.from_pretrained(model_id).to(self.device)
                    self.clip_processor = CLIPProcessor.from_pretrained(model_id)
                self.clip_model.eval()
                self.use_clip = True
        except Exception:
            self.use_clip = False

    @staticmethod
    def _normalize_lon(lon: float) -> float:
        return ((lon + 180.0) % 360.0) - 180.0

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r_earth = 6371.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
        return 2.0 * r_earth * math.asin(min(1.0, math.sqrt(a)))

    @staticmethod
    def _to_morphology_neutral_pil(pil_img: Image.Image) -> Image.Image:
        arr = np.array(pil_img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        neutral_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(neutral_rgb)

    def _compute_clip_embedding(self, pil_img: Image.Image) -> np.ndarray:
        import torch
        neutral_pil = self._to_morphology_neutral_pil(pil_img)
        inputs = self.clip_processor(images=neutral_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat = self.clip_model.get_image_features(**inputs)
            if not isinstance(feat, torch.Tensor):
                feat = feat.pooler_output if hasattr(feat, "pooler_output") else feat[0]
            feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        return feat.cpu().numpy()[0]

    @staticmethod
    def _compute_multiscale_morphology_vec(norm_tensor: np.ndarray) -> np.ndarray:
        gray = np.mean(norm_tensor, axis=2).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        ang = (np.arctan2(gy, gx) + np.pi) / (2.0 * np.pi)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

        h_mag, _ = np.histogram(mag, bins=16, range=(0.0, 1.5), density=True)
        h_ang, _ = np.histogram(ang, bins=16, range=(0.0, 1.0), density=True)
        h_lap, _ = np.histogram(lap, bins=16, range=(0.0, 1.2), density=True)

        h, w = gray.shape
        yy, xx = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        r_grid = np.hypot(yy - cy, xx - cx)
        max_r = np.hypot(cy, cx)
        radial_bins = np.linspace(0, max_r, 17)
        radial_prof = []
        for i in range(16):
            mask = (r_grid >= radial_bins[i]) & (r_grid < radial_bins[i + 1])
            radial_prof.append(float(np.mean(mag[mask])) if np.any(mask) else 0.0)
        radial_arr = np.array(radial_prof, dtype=np.float32)

        vec = np.concatenate([h_mag, h_ang, h_lap, radial_arr]).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-6)

    @staticmethod
    def _extract_live_highres_6d_vector(
        earth_rgb: np.ndarray,
        climate: dict,
        coarse_vec: list,
        cand_lat: float
    ) -> list:
        gray = cv2.cvtColor(earth_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.hypot(gx, gy)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

        dark_mineral_frac = float(np.mean((gray > 0.10) & (gray < 0.52)))
        local_contrast = float(np.clip(np.std(gray) * 4.0, 0.18, 0.98))
        pit_collapse_hi = float(np.clip(np.percentile(lap, 96) * 2.4, 0.18, 0.98))

        live_basalt = float(np.clip(
            0.35 * dark_mineral_frac + 0.25 * local_contrast + 0.40 * coarse_vec[0],
            0.20, 0.98
        ))

        rough_hi = float(np.clip(np.mean(grad_mag) * 1.9, 0.05, 0.95))
        live_flatness = float(np.clip(0.60 * (1.0 - rough_hi) + 0.40 * coarse_vec[1], 0.10, 0.98))

        solar_kwh = float(climate.get("solar_irradiance_kwh_m2_day", 5.5))
        precip = float(climate.get("precipitation_mm_day", 0.2))
        temp_c = float(climate.get("mean_surface_temp_c", 18.0))
        aridity = float(np.clip(1.0 - precip / 1.5, 0.10, 0.99))
        live_thermal = float(np.clip(
            0.45 * (solar_kwh / 7.0) * aridity + 0.35 * coarse_vec[2] + 0.20 * (abs(temp_c - 15.0) / 30.0),
            0.20, 0.98
        ))

        if temp_c < 2.0 or abs(cand_lat) > 48.0:
            live_ice = float(np.clip(max(coarse_vec[3], (5.0 - temp_c) / 22.0), 0.15, 0.98))
        else:
            live_ice = float(np.clip(min(coarse_vec[3], 0.12), 0.03, 0.25))

        wind_ms = float(climate.get("wind_speed_10m_ms", 4.2))
        live_aeolian = float(np.clip(
            (0.45 * aridity + 0.35 * min(1.0, wind_ms / 6.0) + 0.20 * live_flatness) * (1.0 - 0.75 * live_ice),
            0.10, 0.98
        ))

        shadow_pits = float(np.clip(np.mean(gray < 0.28) * 2.8, 0.15, 0.98))
        live_subsurface = float(np.clip(
            0.40 * pit_collapse_hi + 0.35 * live_basalt + 0.25 * max(shadow_pits, coarse_vec[5]),
            0.15, 0.98
        ))

        return [
            round(live_basalt, 3),
            round(live_flatness, 3),
            round(live_thermal, 3),
            round(live_ice, 3),
            round(live_aeolian, 3),
            round(live_subsurface, 3)
        ]

    def _compute_hybrid_visual_score(
        self,
        emb_target,
        tex_target: np.ndarray,
        earth_rgb: np.ndarray
    ) -> dict:
        pil_e, ten_e = self.preprocessor.preprocess_for_clip(earth_rgb)
        tex_e = self._compute_multiscale_morphology_vec(ten_e)
        tex_score = float(np.clip(np.dot(tex_target, tex_e), 0.15, 0.99))

        if self.use_clip and emb_target is not None:
            emb_e = self._compute_clip_embedding(pil_e)
            raw_clip_cos = float(np.dot(emb_target, emb_e))
            clip_score = float(np.clip((raw_clip_cos - 0.55) / 0.40, 0.15, 0.99))
            hybrid = 0.70 * clip_score + 0.30 * tex_score
        else:
            clip_score = tex_score
            hybrid = tex_score

        return {
            "clip_similarity_pct": round(clip_score * 100.0, 1),
            "texture_similarity_pct": round(tex_score * 100.0, 1),
            "hybrid_visual_pct": round(hybrid * 100.0, 1),
            "engine_type": "CLIP-ViT-B/32 + OpenCV" if self.use_clip else "Multi-Scale Morphology CV"
        }

    @staticmethod
    def _compute_topographic_profile_match(target_cs: dict, earth_cs: dict) -> float:
        slope_diff = abs(target_cs["mean_slope_deg"] - earth_cs["mean_slope_deg"])
        safe_diff = abs(target_cs["safe_landing_pct"] - earth_cs["safe_landing_pct"])
        slope_safety_score = np.clip(100.0 - (slope_diff * 7.0 + safe_diff * 0.25), 20.0, 99.0)

        t_elev = np.array(target_cs["elevation_m"], dtype=np.float32)
        e_elev = np.array(earth_cs["elevation_m"], dtype=np.float32)
        t_norm = (t_elev - t_elev.min()) / (np.ptp(t_elev) + 1e-5)
        e_norm = (e_elev - e_elev.min()) / (np.ptp(e_elev) + 1e-5)

        l1_direct = float(np.mean(np.abs(t_norm - e_norm)))
        l1_flipped = float(np.mean(np.abs(t_norm - e_norm[::-1])))
        best_l1_diff = min(l1_direct, l1_flipped)

        curve_shape_score = np.clip((1.0 - best_l1_diff) * 100.0, 20.0, 99.0)
        return round(float(0.60 * slope_safety_score + 0.40 * curve_shape_score), 1)

    @staticmethod
    def _compute_live_nasa_power_gatekeeper(
        target_vec: np.ndarray,
        climate: dict,
        cand_lat: float = 25.0
    ):
        precip = float(climate.get("precipitation_mm_day", 0.3))
        temp_c = float(climate.get("mean_surface_temp_c", 15.0))
        rh_pct = float(climate.get("relative_humidity_pct", 30.0))

        ice_need = float(target_vec[3])
        gatekeeper_penalty = 0.0

        if ice_need < 0.35:
            aridity_score = float(np.clip(1.0 - (precip / 1.2), 0.05, 0.99))
            humidity_score = float(np.clip(1.0 - max(0.0, rh_pct - 25.0) / 60.0, 0.10, 0.99))
            thermal_score = float(np.clip(1.0 - max(0.0, 5.0 - temp_c) / 25.0, 0.15, 0.98))
            env_score = (0.55 * aridity_score + 0.25 * humidity_score + 0.20 * thermal_score) * 100.0
            if precip > 0.75 or temp_c < 0.0 or rh_pct > 65.0:
                gatekeeper_penalty = 22.0
        elif ice_need > 0.65:
            cold_score = float(np.clip(1.0 - max(0.0, temp_c + 2.0) / 22.0, 0.05, 0.99))
            aridity_score = float(np.clip(1.0 - (precip / 2.5), 0.15, 0.99))
            env_score = (0.70 * cold_score + 0.30 * aridity_score) * 100.0
            if temp_c > 8.0:
                gatekeeper_penalty = 25.0
        else:
            aridity_score = float(np.clip(1.0 - (precip / 1.8), 0.15, 0.99))
            env_score = (0.70 * aridity_score + 0.30 * 0.80) * 100.0

        return round(float(np.clip(env_score, 15.0, 99.0)), 1), gatekeeper_penalty

    @staticmethod
    def evaluate_planetary_processes(target: dict, cand_vec: list, climate: dict) -> dict:
        t_vec = target["features_6d"]
        c_basalt, c_flat, c_therm, c_ice, c_wind, c_cave = cand_vec
        wind_ms = float(climate.get("wind_speed_10m_ms", 4.2))

        earth_processes = {
            "Volcanic Basalt Effusion & Lava Tube / Skylight Collapse": round(
                (0.55 * c_cave + 0.45 * c_basalt) * 100.0, 1
            ),
            "Tectonic Chasma Canyon & Escarpment Cliff Erosion": round(
                (0.50 * (1.0 - c_flat) + 0.30 * c_cave + 0.20 * c_basalt) * 100.0, 1
            ),
            "Cryospheric Permafrost & Periglacial Ice Dynamics": round(c_ice * 100.0, 1),
            "Aeolian Dune Transport & Evaporitic Basin Alteration": round(
                (0.45 * c_wind + 0.35 * c_flat + 0.20 * min(1.0, wind_ms / 6.5)) * (1.0 - 0.30 * c_cave) * 100.0, 1
            )
        }
        dominant_earth_proc = max(earth_processes.items(), key=lambda x: x[1])
        proc_match = round(
            (1.0 - float(np.mean(np.abs(np.array(t_vec) - np.array(cand_vec))))) * 100.0, 1
        )

        return {
            "target_planetary_process": target.get("primary_process", "Volcanic & Impact Regolith Evolution"),
            "dominant_earth_process": dominant_earth_proc[0],
            "dominant_process_intensity_pct": dominant_earth_proc[1],
            "process_fidelity_pct": proc_match,
            "process_breakdown": earth_processes
        }

    @staticmethod
    def evaluate_astrobiology_and_habitability(cand_vec: list, climate: dict) -> dict:
        c_basalt, c_flat, c_therm, c_ice, c_wind, c_cave = cand_vec
        solar_kwh = float(climate.get("solar_irradiance_kwh_m2_day", 5.8))
        precip = float(climate.get("precipitation_mm_day", 0.2))
        temp_c = float(climate.get("mean_surface_temp_c", 18.0))

        aridity_factor = float(np.clip(1.0 - precip / 1.5, 0.1, 0.99))
        uv_stress_index = round(float(np.clip((solar_kwh / 7.2) * 100.0, 25.0, 99.0)), 1)

        if c_ice > 0.55 or temp_c < 0.0:
            guild = "Cryophilic & Psychrophilic Permafrost Extremophiles (Kriyo-Biyoimza Analogu)"
            target_relevance = "Mars Kuzey Düzlükleri (Arcadia/Korolev) ve Ay Kutup Soğuk Tuzakları organik molekül korunumu"
        elif c_cave > 0.68:
            guild = "Subsurface Endolithic & Chemolithoautotrophic Cave Microbiomes (Yeraltı Lav Tüpü Ekstremofilleri)"
            target_relevance = "Yüzey UV radyasyonundan kaçıp bazaltik lav tüplerine sığınan kemosentetik yaşam analogu"
        elif c_flat > 0.72 and c_wind > 0.68:
            guild = "Halophilic & Polyextremophilic Evaporite/Playa Microorganisms (Tuz Düzlüğü & Paleo-Göl Biyo-İmzası)"
            target_relevance = "Mars Jezero Krateri karbonat/sülfat çökellerinde mikrobik stromatolit ve lipid korunumu"
        else:
            guild = "Hyper-Arid UV-Resistant Hypolithic/Endolithic Desert Microbiota (Hiper-Kurak Kaya İçi Yaşam)"
            target_relevance = "Aşırı kuraklık ve yüksek UV altında bazalt gözeneklerinde biyo-mineralizasyon izleri"

        biosig_preservation = round(
            float(np.clip((0.45 * aridity_factor + 0.30 * c_basalt + 0.25 * max(c_cave, c_ice)) * 100.0, 45.0, 98.5)), 1
        )

        return {
            "extremophile_guild": guild,
            "biosignature_preservation_pct": biosig_preservation,
            "uv_radiation_stress_pct": uv_stress_index,
            "solar_irradiance_kwh_m2_day": solar_kwh,
            "astrobiology_mission_relevance": target_relevance
        }

    @staticmethod
    def evaluate_ops_and_base_feasibility(target: dict, cand_cs: dict = None, climate: dict = None) -> dict:
        vec = target["features_6d"]
        cs = cand_cs if cand_cs is not None else target.get("cross_section", {})
        body = target.get("body", "Moon")
        lat_abs = abs(float(target.get("lat", 0.0)))

        basalt, flatness, thermal, ice_isru, dust, subsurface = vec
        safe_landing_pct = float(cs.get("safe_landing_pct", flatness * 100.0))
        mean_slope = float(cs.get("mean_slope_deg", 3.5))

        if subsurface >= 0.70:
            habitat_type = "Subsurface Lava Tube / Canyon Cliff-Wall Habitat"
            shielding_req = "Doğal bazalt tavan (>5m kayaç zırhı) — GCR ve Güneş Fırtınalarına (SPE) karşı %99+ koruma"
            rad_score = round(subsurface * 100.0, 1)
        else:
            regolith_m = round(2.8 - 0.8 * basalt, 1)
            habitat_type = "Surface 3D-Printed Sintered Basalt Regolith Dome"
            shielding_req = f"Modüller üzerine en az {regolith_m} metre mikrodalga sinterlenmiş bazaltik regolit zırhı"
            rad_score = round((0.60 * basalt + 0.40 * flatness) * 100.0, 1)

        if ice_isru >= 0.55:
            isru_strategy = "Cryogenic Water-Ice Electrolysis (H2/O2 Roket Yakıtı + Su) & Regolit Metalurjisi"
            isru_score = round((0.70 * ice_isru + 0.30 * basalt) * 100.0, 1)
        else:
            isru_strategy = "Ilmenite/Basalt Molten Regolith Electrolysis (O2 Üretimi) & 3B Regolit İnşaatı"
            isru_score = round((0.75 * basalt + 0.25 * (1.0 - dust * 0.3)) * 100.0, 1)

        if body == "Moon" and lat_abs > 78.0:
            power_arch = "Dikey Güneş Kuleleri (Peaks of Eternal Light %85+ Aydınlanma) + Kriyo Yakıt Hücresi"
        elif body == "Moon":
            power_arch = "14 Günlük Ay Gecesi İçin 40 kWe Fisyon Reaktörü (NASA Kilopower) + Solar Dizi"
        else:
            power_arch = "Toz Fırtınasına Dayanıklı Hibrit Nükleer Fisyon (Kilopower) + Otonom Solar Dizi"

        rover_mobility_pct = round(float(np.clip(100.0 - mean_slope * 4.5 - dust * 12.0, 45.0, 98.0)), 1)
        trn_navigation_pct = round(float(np.clip((0.55 * basalt + 0.45 * subsurface) * 100.0, 55.0, 98.0)), 1)
        construction_3d_pct = round(float(np.clip((0.65 * basalt + 0.35 * flatness) * 100.0, 50.0, 98.0)), 1)

        overall_base_viability = round(
            0.35 * safe_landing_pct + 0.35 * rad_score + 0.30 * isru_score, 1
        )

        return {
            "overall_base_viability_pct": overall_base_viability,
            "recommended_habitat_architecture": habitat_type,
            "radiation_shielding_protocol": shielding_req,
            "isru_primary_strategy": isru_strategy,
            "power_and_illumination_architecture": power_arch,
            "landing_safety_envelope_pct": round(safe_landing_pct, 1),
            "rover_mobility_trafficability_pct": rover_mobility_pct,
            "trn_optical_navigation_pct": trn_navigation_pct,
            "regolith_construction_3d_pct": construction_3d_pct
        }

    def get_target_profile(
        self,
        body: str = "Mars",
        lat: float = 18.38,
        lon: float = 77.58,
        radius_km: float = 5.0,
        preset_id: str = None
    ) -> dict:
        if preset_id and preset_id in self.catalog:
            target = dict(self.catalog[preset_id])
            target["lon"] = round(self._normalize_lon(float(target["lon"])), 3)
            target["radius_km"] = float(target.get("radius_km", radius_km))
            rgb, img_path = self.loader.fetch_planetary_patch(
                target["body"], target["lat"], target["lon"], target["radius_km"]
            )
            target["image_path"] = img_path
            target["rgb_image"] = rgb
        else:
            norm_lon = self._normalize_lon(lon)
            target = self.loader.extract_dynamic_target_profile(body, lat, norm_lon, radius_km)
            rgb, _ = self.loader.fetch_planetary_patch(body, lat, norm_lon, radius_km)
            target["rgb_image"] = rgb

        target["cross_section"] = self.preprocessor.extract_elevation_cross_section(
            target["rgb_image"], radius_km=target.get("radius_km", radius_km)
        )
        target["base_feasibility"] = self.evaluate_ops_and_base_feasibility(target)
        return target

    def find_moon_to_mars_bridge(self, target_profile: dict) -> dict:
        current_body = target_profile.get("body", "Moon")
        opposite_body = "Mars" if current_body.lower() == "moon" else "Moon"
        t_vec = np.array(target_profile["features_6d"], dtype=np.float32)

        best_match = None
        best_sim = -1.0

        for item in self.catalog.values():
            if item.get("body", "").lower() == opposite_body.lower():
                c_vec = np.array(item["features_6d"], dtype=np.float32)
                sim = float(1.0 - np.mean(np.abs(t_vec - c_vec)))
                if sim > best_sim:
                    best_sim = sim
                    best_match = item

        if not best_match:
            return {}

        return {
            "source_body": current_body,
            "bridge_body": opposite_body,
            "bridge_target_id": best_match["id"],
            "bridge_name": best_match["name"],
            "lat": best_match["lat"],
            "lon": round(self._normalize_lon(float(best_match["lon"])), 3),
            "architectural_transfer_pct": round(best_sim * 100.0, 1),
            "shared_mission_focus": best_match.get("mission_focus", ""),
            "primary_process": best_match.get("primary_process", ""),
            "engineering_adaptation_note": (
                "Ay vakum (0 mbar, 0.165g) ortamında doğrulanan kapalı döngü yaşam ve lav tüpü/ISRU "
                "donanımı, Mars'ın 6.1 mbar CO2 atmosferi ve 0.378g yerçekimine göre toz filtreleme "
                "ve konvektif ısı yalıtımı eklenerek doğrudan ölçeklenebilir."
            )
        }

    @staticmethod
    def _assign_strategic_roles_to_top_candidates(candidates: list):
        if not candidates:
            return
        best_vis_idx = int(np.argmax([c["visual_hybrid_pct"] for c in candidates]))
        best_topo_idx = int(np.argmax([c["topographic_match_pct"] for c in candidates]))
        best_env_idx = int(np.argmax([c["live_env_verification_pct"] for c in candidates]))

        for i, c in enumerate(candidates):
            strengths_tags = []
            if i == 0:
                strengths_tags.append("🥇 Birincil Tam-Spektrum Analog (Highest Overall Fidelity)")
            if i == best_topo_idx:
                strengths_tags.append("⛰️ En Güçlü 2B Topografik Harita & Kesit İkizi (Top 30% Topo Fit)")
            if i == best_vis_idx:
                strengths_tags.append("👁️ En Yüksek Görsel & Morfolojik İkiz (Top CLIP/CV Match)")
            if i == best_env_idx:
                strengths_tags.append("🌡️ En İyi Atmosferik & İklimsel İkiz (Top NASA POWER Fit)")
            if not strengths_tags:
                strengths_tags.append("🔬 Alternatif Çok Amaçlı Donanım Doğrulama Sahası")
            c["strategic_role_badge"] = " | ".join(strengths_tags)

    def find_earth_analogs(
        self,
        target_profile: dict,
        mission_mode: str = "BALANCED",
        custom_weights: list = None,
        novelty_only: bool = True,
        top_k: int = 3,
        min_separation_km: float = 450.0
    ) -> dict:
        target_vec = np.array(target_profile["features_6d"], dtype=np.float32)
        if custom_weights is not None and len(custom_weights) == 6:
            weights = np.array(custom_weights, dtype=np.float32)
        else:
            weights = np.array(
                MISSION_WEIGHT_PRESETS.get(mission_mode, MISSION_WEIGHT_PRESETS["BALANCED"]),
                dtype=np.float32
            )
        weights = weights / (np.sum(weights) + 1e-6)

        diffs = np.abs(self.features_6d - target_vec)
        weighted_l1_dist = np.sum(diffs * weights, axis=1)
        coarse_similarity = np.clip(1.0 - weighted_l1_dist, 0.01, 0.99)

        ice_need = float(target_vec[3])
        if ice_need < 0.35:
            # Kuru volkanik/kanyon hedeflerinde kutuplari ve nemli ekvator kusagini (<15° enlem) maskele
            cryo_or_humid_mask = (self.features_6d[:, 3] > 0.35) | (np.abs(self.lats) > 45.0) | (np.abs(self.lats) < 14.0)
            coarse_similarity = np.where(cryo_or_humid_mask, coarse_similarity * 0.35, coarse_similarity)
        elif ice_need > 0.65:
            hot_mask = (self.features_6d[:, 3] < 0.50) | (np.abs(self.lats) < 48.0)
            coarse_similarity = np.where(hot_mask, coarse_similarity * 0.35, coarse_similarity)

        valid_mask = np.ones(len(self.lats), dtype=bool)
        if novelty_only:
            valid_mask = (self.is_well_known == 0)

        masked_scores = np.where(valid_mask, coarse_similarity, -1.0)
        sorted_indices = np.argsort(masked_scores)[::-1]

        pool_size = max(top_k + 5, 8)
        shortlist_indices = []
        for idx in sorted_indices:
            if masked_scores[idx] <= 0:
                break
            cand_lat, cand_lon = float(self.lats[idx]), float(self.lons[idx])
            too_close = False
            same_sector_count = 0
            for sel_idx in shortlist_indices:
                s_lat, s_lon = float(self.lats[sel_idx]), float(self.lons[sel_idx])
                dist_km = self._haversine_km(cand_lat, cand_lon, s_lat, s_lon)
                if dist_km < min_separation_km:
                    too_close = True
                    break
                if abs(cand_lat - s_lat) < 10.0 and abs(cand_lon - s_lon) < 15.0:
                    same_sector_count += 1
            if not too_close and same_sector_count < 2:
                shortlist_indices.append(int(idx))
            if len(shortlist_indices) >= pool_size:
                break

        radius_km = float(target_profile.get("radius_km", 5.0))
        target_cs = target_profile["cross_section"]

        pil_t, ten_t = self.preprocessor.preprocess_for_clip(target_profile["rgb_image"])
        emb_target = self._compute_clip_embedding(pil_t) if self.use_clip else None
        tex_target = self._compute_multiscale_morphology_vec(ten_t)

        def _fetch_io_bundle(rank_and_idx):
            pre_rank, grid_idx = rank_and_idx
            c_lat = float(self.lats[grid_idx])
            c_lon = float(self.lons[grid_idx])
            earth_rgb, earth_img_path = self.loader.fetch_earth_satellite_patch(
                c_lat, c_lon, radius_km=radius_km
            )
            climate_data = self.loader.fetch_nasa_power_climate(c_lat, c_lon)
            region_name = self.loader.reverse_geocode_earth(c_lat, c_lon)
            return pre_rank, grid_idx, earth_rgb, earth_img_path, climate_data, region_name

        indexed_shortlist = list(enumerate(shortlist_indices, 1))
        with ThreadPoolExecutor(max_workers=4) as executor:
            io_results = list(executor.map(_fetch_io_bundle, indexed_shortlist))

        evaluated_pool = []
        for pre_rank, idx, earth_rgb, earth_img_path, climate_data, region_name in io_results:
            c_lat = float(self.lats[idx])
            c_lon = float(self.lons[idx])
            coarse_vec = self.features_6d[idx].tolist()

            live_c_vec = self._extract_live_highres_6d_vector(earth_rgb, climate_data, coarse_vec, c_lat)
            live_diffs = np.abs(np.array(live_c_vec, dtype=np.float32) - target_vec)
            live_6d_pct = round(float(np.clip(1.0 - np.sum(live_diffs * weights), 0.10, 0.99)) * 100.0, 1)

            vis_metrics = self._compute_hybrid_visual_score(emb_target, tex_target, earth_rgb)
            earth_cs = self.preprocessor.extract_elevation_cross_section(earth_rgb, radius_km=radius_km)
            topo_match_pct = self._compute_topographic_profile_match(target_cs, earth_cs)
            env_verify_pct, gate_penalty = self._compute_live_nasa_power_gatekeeper(
                target_vec, climate_data, c_lat
            )

            micro_feature = self.preprocessor.pinpoint_micro_scale_feature(
                earth_rgb, c_lat, c_lon, radius_km=radius_km, mission_mode=mission_mode
            )
            hud_rgb = self.preprocessor.render_engineering_hud(
                earth_rgb, earth_cs, micro_feature, radius_km=radius_km
            )
            process_analogy = self.evaluate_planetary_processes(target_profile, live_c_vec, climate_data)
            astrobio = self.evaluate_astrobiology_and_habitability(live_c_vec, climate_data)
            ops_readiness = self.evaluate_ops_and_base_feasibility(target_profile, earth_cs, climate_data)

            # YENI AGIRLIK DENKLEMI:
            # %35 6B Kosul + %30 2B Topografik Harita/Kesit (+%10) + %20 Gorsel AI (-%10) + %15 NASA POWER
            raw_total = (
                0.35 * live_6d_pct
                + 0.30 * topo_match_pct
                + 0.20 * vis_metrics["hybrid_visual_pct"]
                + 0.15 * env_verify_pct
                - gate_penalty
            )
            total_score = round(float(np.clip(raw_total, 15.0, 99.0)), 1)

            dim_matches_pct = [
                round((1.0 - abs(float(target_vec[d]) - float(live_c_vec[d]))) * 100.0, 1)
                for d in range(6)
            ]

            evaluated_pool.append({
                "grid_index": idx,
                "pre_screen_rank": pre_rank,
                "region_name": region_name,
                "lat": round(c_lat, 3),
                "lon": round(c_lon, 3),
                "is_novel_discovery": bool(self.is_well_known[idx] == 0),
                "total_match_pct": total_score,
                "parametric_6d_pct": live_6d_pct,
                "visual_hybrid_pct": vis_metrics["hybrid_visual_pct"],
                "clip_similarity_pct": vis_metrics["clip_similarity_pct"],
                "texture_similarity_pct": vis_metrics["texture_similarity_pct"],
                "topographic_match_pct": topo_match_pct,
                "live_env_verification_pct": env_verify_pct,
                "vision_engine_used": vis_metrics["engine_type"],
                "features_6d": live_c_vec,
                "dimension_matches_pct": dim_matches_pct,
                "image_path": earth_img_path,
                "rgb_image": earth_rgb,
                "hud_image": hud_rgb,
                "cross_section": earth_cs,
                "nasa_power_climate": climate_data,
                "micro_scale_pinpoint": micro_feature,
                "process_analogy": process_analogy,
                "astrobiology_profile": astrobio,
                "operations_readiness": ops_readiness
            })

        evaluated_pool.sort(key=lambda x: x["total_match_pct"], reverse=True)
        final_candidates = evaluated_pool[:top_k]
        self._assign_strategic_roles_to_top_candidates(final_candidates)

        complementary_site = self._find_complementary_site(
            target_vec, final_candidates, novelty_only=novelty_only
        )
        planetary_bridge = self.find_moon_to_mars_bridge(target_profile)

        return {
            "target": target_profile,
            "mission_mode": mission_mode,
            "novelty_only": novelty_only,
            "min_separation_km": min_separation_km,
            "scanned_total_cells": len(self.lats),
            "shortlist_evaluated_count": len(evaluated_pool),
            "shortlist_full_pool": [
                {
                    "pre_rank": c["pre_screen_rank"],
                    "region_name": c["region_name"],
                    "lat": c["lat"],
                    "lon": c["lon"],
                    "visual_hybrid_pct": c["visual_hybrid_pct"],
                    "topographic_match_pct": c["topographic_match_pct"],
                    "final_score_pct": c["total_match_pct"]
                }
                for c in evaluated_pool
            ],
            "primary_candidates": final_candidates,
            "complementary_split_site": complementary_site,
            "moon_mars_bridge": planetary_bridge
        }

    def _find_complementary_site(
        self,
        target_vec: np.ndarray,
        final_candidates: list,
        novelty_only: bool = True
    ) -> dict:
        """
        1. sahada en zayif kalan boyutu tamamlayan, ilk 3 sahadan en az 600 km uzakta olan
        ve Canli NASA POWER kuraklik/kutup barajini gecen 4. bagimsiz sahayi bulur.
        """
        primary_cand = final_candidates[0]
        prim_vec = np.array(primary_cand["features_6d"], dtype=np.float32)
        deficits = target_vec - prim_vec
        weakest_dim = int(np.argmax(deficits))

        if deficits[weakest_dim] < 0.06:
            weakest_dim = int(np.argmin(primary_cand["dimension_matches_pct"]))

        dim_name = DIMENSION_NAMES[weakest_dim]
        target_val = float(target_vec[weakest_dim])

        dim_closeness = 1.0 - np.abs(self.features_6d[:, weakest_dim] - target_val)
        overall_closeness = 1.0 - np.mean(np.abs(self.features_6d - target_vec), axis=1)
        comp_score = 0.65 * dim_closeness + 0.35 * overall_closeness

        # Ikili sahada da kuraklik/kutup ve tropikal kusak kilidini uygula (Somali sizintisini engeller)
        ice_need = float(target_vec[3])
        if ice_need < 0.35:
            bad_mask = (self.features_6d[:, 3] > 0.35) | (np.abs(self.lats) > 45.0) | (np.abs(self.lats) < 16.0)
            comp_score = np.where(bad_mask, -1.0, comp_score)
        elif ice_need > 0.65:
            bad_mask = (self.features_6d[:, 3] < 0.50) | (np.abs(self.lats) < 48.0)
            comp_score = np.where(bad_mask, -1.0, comp_score)

        if novelty_only:
            comp_score = np.where(self.is_well_known == 0, comp_score, -1.0)

        best_idxs = np.argsort(comp_score)[::-1]
        chosen_idx = int(best_idxs[0])
        chosen_climate = None

        for idx in best_idxs[:40]:
            c_lat, c_lon = float(self.lats[idx]), float(self.lons[idx])
            far_from_all = True
            for fc in final_candidates:
                if self._haversine_km(fc["lat"], fc["lon"], c_lat, c_lon) < 600.0:
                    far_from_all = False
                    break
            if not far_from_all:
                continue

            clim = self.loader.fetch_nasa_power_climate(c_lat, c_lon)
            _, pen = self._compute_live_nasa_power_gatekeeper(target_vec, clim, c_lat)
            if pen == 0.0:
                chosen_idx = int(idx)
                chosen_climate = clim
                break

        c_lat = float(self.lats[chosen_idx])
        c_lon = float(self.lons[chosen_idx])
        c_region = self.loader.reverse_geocode_earth(c_lat, c_lon)
        if chosen_climate is None:
            chosen_climate = self.loader.fetch_nasa_power_climate(c_lat, c_lon)
        c_dim_match = round(float(dim_closeness[chosen_idx]) * 100.0, 1)

        return {
            "weakest_dimension_index": weakest_dim,
            "weakest_dimension_name": dim_name,
            "primary_site_score_in_dim_pct": primary_cand["dimension_matches_pct"][weakest_dim],
            "complementary_region_name": c_region,
            "lat": round(c_lat, 3),
            "lon": round(c_lon, 3),
            "complementary_dim_match_pct": c_dim_match,
            "features_6d": [round(float(x), 3) for x in self.features_6d[chosen_idx]],
            "nasa_power_climate": chosen_climate
        }
