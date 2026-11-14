
import os
import sys
import time
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from PIL import Image

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
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
        self._geocode_cache = {}
        self._geocode_lock = threading.Lock()

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
            print("[AI ENGINE] Ultra-Hizli Morfolojik CV Modu aktif (Zero-Download).")
            return

        model_id = "openai/clip-vit-base-patch32"
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            if clip_mode == "auto":
                try:
                    self.clip_model = CLIPModel.from_pretrained(
                        model_id, local_files_only=True
                    ).to(self.device)
                    self.clip_processor = CLIPProcessor.from_pretrained(
                        model_id, local_files_only=True
                    )
                    self.clip_model.eval()
                    self.use_clip = True
                    print(f"[AI ENGINE] Yerel onbellekten OpenAI CLIP ({self.device.upper()}) aninda yuklendi!")
                    return
                except Exception:
                    print("[AI ENGINE] CLIP onbellekte yok -> Aninda Morfolojik CV Modu aktif!")
                    return

            if clip_mode == "force":
                print(f"[AI ENGINE] OpenAI CLIP modeli ({self.device.upper()}) hazirlaniyor...")
                self.clip_model = CLIPModel.from_pretrained(model_id).to(self.device)
                self.clip_processor = CLIPProcessor.from_pretrained(model_id)
                self.clip_model.eval()
                self.use_clip = True
                print("[AI ENGINE] CLIP Derin Ogrenme Modeli hazir!")
        except Exception as e:
            print(f"[AI ENGINE] CLIP yerine Morfolojik CV motoru kullaniliyor ({e}).")
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

    def _safe_reverse_geocode(self, lat: float, lon: float) -> str:
        key = (round(lat, 2), round(lon, 2))
        with self._geocode_lock:
            if key in self._geocode_cache:
                return self._geocode_cache[key]
            time.sleep(0.35)
            name = self.loader.reverse_geocode_earth(lat, lon)
            self._geocode_cache[key] = name
            return name

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
        precip = float(climate.get("precipitation_mm_day", 0.4))
        if precip < -900.0:
            precip = 0.4

        temp_c = float(climate.get("mean_surface_temp_c", 15.0))
        if temp_c < -900.0:
            temp_c = 25.0 - 45.0 * (abs(cand_lat) / 90.0)

        rh_pct = float(climate.get("relative_humidity_pct", 30.0))
        if rh_pct < -900.0:
            rh_pct = 35.0

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

    def get_target_profile(
        self,
        body: str = "Mars",
        lat: float = 18.38,
        lon: float = 77.58,
        radius_km: float = 20.0,
        preset_id: str = None
    ) -> dict:
        if preset_id and preset_id in self.catalog:
            target = dict(self.catalog[preset_id])
            target["lon"] = round(self._normalize_lon(float(target["lon"])), 3)
            rgb, img_path = self.loader.fetch_planetary_patch(
                target["body"], target["lat"], target["lon"], target.get("radius_km", radius_km)
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
            "primary_process": best_match.get("primary_process", "")
        }

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
        param_similarity = np.clip(1.0 - weighted_l1_dist, 0.01, 0.99)

        ice_need = float(target_vec[3])
        if ice_need < 0.35:
            cryo_mask = (self.features_6d[:, 3] > 0.35) | (np.abs(self.lats) > 45.0)
            param_similarity = np.where(cryo_mask, param_similarity * 0.35, param_similarity)
        elif ice_need > 0.65:
            hot_mask = (self.features_6d[:, 3] < 0.50) | (np.abs(self.lats) < 32.0)
            param_similarity = np.where(hot_mask, param_similarity * 0.35, param_similarity)

        valid_mask = np.ones(len(self.lats), dtype=bool)
        if novelty_only:
            valid_mask = (self.is_well_known == 0)

        masked_scores = np.where(valid_mask, param_similarity, -1.0)
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

        radius_km = float(target_profile.get("radius_km", 20.0))
        target_cs = target_profile["cross_section"]

        pil_t, ten_t = self.preprocessor.preprocess_for_clip(target_profile["rgb_image"])
        emb_target = self._compute_clip_embedding(pil_t) if self.use_clip else None
        tex_target = self._compute_multiscale_morphology_vec(ten_t)

        def _fetch_io_bundle(grid_idx: int):
            c_lat = float(self.lats[grid_idx])
            c_lon = float(self.lons[grid_idx])
            earth_rgb, earth_img_path = self.loader.fetch_earth_satellite_patch(
                c_lat, c_lon, radius_km=radius_km
            )
            climate_data = self.loader.fetch_nasa_power_climate(c_lat, c_lon)
            region_name = self._safe_reverse_geocode(c_lat, c_lon)
            return grid_idx, earth_rgb, earth_img_path, climate_data, region_name

        with ThreadPoolExecutor(max_workers=4) as executor:
            io_results = list(executor.map(_fetch_io_bundle, shortlist_indices))

        evaluated_pool = []
        for idx, earth_rgb, earth_img_path, climate_data, region_name in io_results:
            c_lat = float(self.lats[idx])
            c_lon = float(self.lons[idx])
            c_vec = self.features_6d[idx].tolist()
            p_score_pct = round(float(param_similarity[idx]) * 100.0, 1)

            vis_metrics = self._compute_hybrid_visual_score(emb_target, tex_target, earth_rgb)
            earth_cs = self.preprocessor.extract_elevation_cross_section(earth_rgb, radius_km=radius_km)
            topo_match_pct = self._compute_topographic_profile_match(target_cs, earth_cs)
            env_verify_pct, gate_penalty = self._compute_live_nasa_power_gatekeeper(
                target_vec, climate_data, c_lat
            )

            raw_total = (
                0.45 * p_score_pct
                + 0.20 * vis_metrics["hybrid_visual_pct"]
                + 0.15 * topo_match_pct
                + 0.20 * env_verify_pct
                - gate_penalty
            )
            total_score = round(float(np.clip(raw_total, 15.0, 99.0)), 1)

            dim_matches_pct = [
                round((1.0 - abs(float(target_vec[d]) - float(c_vec[d]))) * 100.0, 1)
                for d in range(6)
            ]

            evaluated_pool.append({
                "grid_index": idx,
                "region_name": region_name,
                "lat": round(c_lat, 3),
                "lon": round(c_lon, 3),
                "is_novel_discovery": bool(self.is_well_known[idx] == 0),
                "total_match_pct": total_score,
                "parametric_6d_pct": p_score_pct,
                "visual_hybrid_pct": vis_metrics["hybrid_visual_pct"],
                "clip_similarity_pct": vis_metrics["clip_similarity_pct"],
                "texture_similarity_pct": vis_metrics["texture_similarity_pct"],
                "topographic_match_pct": topo_match_pct,
                "live_env_verification_pct": env_verify_pct,
                "vision_engine_used": vis_metrics["engine_type"],
                "features_6d": [round(x, 3) for x in c_vec],
                "dimension_matches_pct": dim_matches_pct,
                "image_path": earth_img_path,
                "rgb_image": earth_rgb,
                "cross_section": earth_cs,
                "nasa_power_climate": climate_data
            })

        evaluated_pool.sort(key=lambda x: x["total_match_pct"], reverse=True)
        final_candidates = evaluated_pool[:top_k]

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
        comp_score = 0.70 * dim_closeness + 0.30 * overall_closeness

        if novelty_only:
            comp_score = np.where(self.is_well_known == 0, comp_score, -1.0)

        best_idxs = np.argsort(comp_score)[::-1]
        chosen_idx = int(best_idxs[0])

        # Ilk 3 adayin HICBIRINE 600 km'den yakin olmayan 4. bagimsiz bir bolge sec
        for idx in best_idxs:
            c_lat, c_lon = float(self.lats[idx]), float(self.lons[idx])
            far_from_all = True
            for fc in final_candidates:
                if self._haversine_km(fc["lat"], fc["lon"], c_lat, c_lon) < 600.0:
                    far_from_all = False
                    break
            if far_from_all:
                chosen_idx = int(idx)
                break

        c_lat = float(self.lats[chosen_idx])
        c_lon = float(self.lons[chosen_idx])
        c_region = self._safe_reverse_geocode(c_lat, c_lon)
        c_climate = self.loader.fetch_nasa_power_climate(c_lat, c_lon)
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
            "nasa_power_climate": c_climate
        }
