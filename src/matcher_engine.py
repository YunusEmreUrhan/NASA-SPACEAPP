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

METHOD_VERSION = "TERRA-TWIN-Screening-v7.0-AuditCompliant"

DIMENSION_NAMES = [
    "Litoloji & Albedo ?n Tarama ?ndeksi",
    "Topografik E?im Uyumu (SRTM 90m DEM vs LOLA/MOLA)",
    "Radyatif Ba?lam ?ndeksi (Senaryo Varsay?m?)",
    "Kriyosferik / So?uk Tuzak Ba?lam? (Senaryo Varsay?m?)",
    "Kurakl?k & R?zgar Ba?lam? (D?nya ?klim Filtresi)",
    "Dikey R?lyef & ?ukur ?l?e?i (SRTM 90m DEM vs LOLA/MOLA)"
]

# Madde 7 [P1]: A??rl?klar a??k?a 'senaryo varsay?m?' olarak etiketlenmi?tir (NASA standard? de?ildir)
MISSION_WEIGHT_PRESETS = {
    "BALANCED": [0.20, 0.20, 0.15, 0.15, 0.15, 0.15],
    "HABITAT_LAVA_TUBE": [0.25, 0.15, 0.15, 0.10, 0.05, 0.30],
    "ISRU_WATER_MINING": [0.15, 0.15, 0.20, 0.40, 0.05, 0.05],
    "ROVER_LANDING_OPS": [0.20, 0.30, 0.10, 0.05, 0.25, 0.10]
}


class PlanetaryMatcherEngine:
    """
    P0-P3 Bilimsel Denetim Uyumlu ?n Tarama (Preliminary Screening) Motoru:
    1) Do?rulanmam?? sabit 35/30/20/15 form?l? kald?r?lm??t?r.
    2) ?l??lm?? Fiziksel Benzerlik yaln?zca ayn? fiziksel tan?ma ve SI birimine sahip
       e?li (paired) jeomorfolojik/topografik ?l??mlerden (E?im ?, Dikey R?lyef m, Havza ?l?e?i km)
       hesaplan?r; eksik g?zlemler ASLA uydurulmaz (imputation=False).
    3) NASA POWER D?nya meteorolojisi (2m/y?zey s?cakl???, ya???) Ay/Mars y?zey s?cakl???yla
       do?rudan e?de?er puanlanmaz (Madde 4); yaln?zca D?nya saha ba?lam? olarak sunulur.
    4) G?rsel benzerlik (CLIP / Morfoloji) fiziksel skora kar??t?r?lmaz; 'G?rsel Ke?if Sinyali'
       olarak ayr? raporlan?r ve CLIP y?klenemezse a??k?a 'model kullan?lamad?' d?ner (Madde 6).
    5) A??rl?k Duyarl?l?k Analizi (Weight Sensitivity Analysis) ile a??rl?klar de?i?ti?inde
       s?ralaman?n ve skorun ne kadar de?i?ti?i raporlan?r (Madde 7).
    """

    def __init__(
        self,
        grid_path: str = "data/global_grids/earth_6d_grid.npz",
        catalog_path: str = "metadata/moon_mars_targets.json",
        clip_mode: str = "auto",
        offline_mode: bool = False
    ):
        self.offline_mode = offline_mode
        self.loader = PlanetaryDataLoader(offline_mode=offline_mode)
        self.preprocessor = PlanetaryPreprocessor(target_size=224)

        if not os.path.exists(grid_path):
            raise FileNotFoundError(f"K?resel matris bulunamad?: {grid_path}")

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
        self.clip_status = "model kullan?lamad? (CLIP devre d??? veya yerel a??rl?k yok)"
        self.clip_model = None
        self.clip_processor = None
        self.device = "cpu"
        self._init_vision_backbone(clip_mode)

    def _init_vision_backbone(self, clip_mode: str):
        if clip_mode == "off" or self.offline_mode:
            self.clip_status = "model kullan?lamad? (clip_mode=off)"
            return
        model_id = "openai/clip-vit-base-patch32"
        try:
            import torch
            import transformers
            from transformers import CLIPProcessor, CLIPModel

            transformers.logging.set_verbosity_error()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                self.clip_model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(self.device)
                self.clip_processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
            self.clip_model.eval()
            self.use_clip = True
            self.clip_status = "CLIP-ViT-B/32 Aktif (G?rsel Ke?if Sinyali)"
        except Exception:
            self.use_clip = False
            self.clip_status = "model kullan?lamad? (CLIP yerel ?nbellekte bulunamad? ? ba?ka skor ikame edilmedi)"

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

    def _compute_visual_exploration_signal(
        self,
        emb_target,
        tex_target: np.ndarray,
        earth_rgb: np.ndarray
    ) -> dict:
        """
        Madde 6 [P1]: G?rsel benzerlik fiziksel skordan tamamen ayr?lm??t?r.
        CLIP y?klenemezse a??k?a 'model kullan?lamad?' d?ner; doku skoru CLIP ad?yla sunulmaz.
        """
        if earth_rgb is None or np.max(earth_rgb) == 0:
            return {
                "clip_similarity_pct": None,
                "texture_similarity_pct": None,
                "hybrid_visual_pct": None,
                "status_label": "veri yok",
                "engine_type": "Uydu G?r?nt?s? Yok (Skora Kat?lmad?)"
            }

        pil_e, ten_e = self.preprocessor.preprocess_for_clip(earth_rgb)
        gray = np.mean(ten_e, axis=2).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        h_mag, _ = np.histogram(mag, bins=16, range=(0.0, 1.5), density=True)
        vec = h_mag.astype(np.float32) / (np.linalg.norm(h_mag) + 1e-6)
        tex_score = round(float(np.clip(np.dot(tex_target, vec), 0.10, 0.99)) * 100.0, 1)

        if self.use_clip and emb_target is not None:
            import torch
            arr = np.array(pil_e)
            g_rgb = cv2.cvtColor(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
            inputs = self.clip_processor(images=Image.fromarray(g_rgb), return_tensors="pt").to(self.device)
            with torch.no_grad():
                feat = self.clip_model.get_image_features(**inputs)
                if not isinstance(feat, torch.Tensor):
                    feat = feat.pooler_output if hasattr(feat, "pooler_output") else feat[0]
                feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
            emb_e = feat.cpu().numpy()[0]
            raw_cos = float(np.dot(emb_target, emb_e))
            clip_pct = round(float(np.clip((raw_cos - 0.55) / 0.40, 0.10, 0.99)) * 100.0, 1)
            return {
                "clip_similarity_pct": clip_pct,
                "texture_similarity_pct": tex_score,
                "hybrid_visual_pct": round(0.70 * clip_pct + 0.30 * tex_score, 1),
                "status_label": "senaryo varsay?m?",
                "engine_type": "G?rsel Ke?if Sinyali (CLIP-ViT-B/32 ? Fiziksel ?l??m De?ildir)"
            }

        return {
            "clip_similarity_pct": None,
            "texture_similarity_pct": tex_score,
            "hybrid_visual_pct": tex_score,
            "status_label": "senaryo varsay?m?",
            "engine_type": f"{self.clip_status} | Yaln?zca Sobel Gradyan Doku Sinyali (Fiziksel Skora Kat?lmaz)"
        }

    @staticmethod
    def build_paired_physical_evidence_table(
        target: dict,
        earth_cs: dict,
        climate: dict,
        scenario_weights: dict = None
    ) -> dict:
        """
        P0 Madde 1, 2, 3, 4 & P1 Madde 7, 8:
        - Yaln?zca fiziksel tan?m? ve SI birimi ayn? olan e?li (paired) ?l??mler (E?im ?, R?lyef m, ?l?ek km)
          kar??la?t?r?l?r ve puanlan?r.
        - NASA POWER (D?nya 2m/y?zey s?cakl??? ve ya???) Ay/Mars s?cakl???yla E?LE?T?R?LMEZ (weight=0.0,
          'Ba?lam Verisi ? G?vdeler Aras? Do?rudan Puanlanmaz').
        - Eksik g?zlemler ASLA tahmin edilmez (imputation=False); veri kapsam? (data_coverage_pct) ve
          belirsizlik aral??? (? uncertainty) a??k?a hesaplan?r.
        """
        pm = target.get("physical_measurements", {})
        t_cs = target.get("cross_section", {})
        w_map = scenario_weights or {"slope": 0.45, "relief": 0.35, "scale": 0.20}

        # 1. Ortalama Topografik E?im (deg ?) ? SRTM 90m DEM vs LOLA/MOLA DEM
        t_slope = pm.get("mean_slope_deg")
        t_slope_unc = pm.get("slope_uncertainty_deg", 0.4)
        e_slope = earth_cs.get("mean_slope_deg") if earth_cs.get("data_status") == "MEASURED_DEM" else None
        e_slope_unc = 0.5 if e_slope is not None else None

        if t_slope is not None and e_slope is not None:
            diff_s = abs(float(t_slope) - float(e_slope))
            sim_slope = round(max(0.0, 100.0 - diff_s * 6.0), 1)
            unc_s_pct = round(min(25.0, (float(t_slope_unc or 0.4) + float(e_slope_unc)) * 6.0), 1)
            status_slope = "NASA verisinden i?lendi"
        else:
            sim_slope, unc_s_pct, status_slope = None, None, "veri yok"

        # 2. Dikey Topografik R?lyef (m) ? SRTM 90m DEM vs LOLA/MOLA DEM
        t_relief = pm.get("vertical_relief_m")
        t_relief_unc = pm.get("relief_uncertainty_m", 10.0)
        e_relief = earth_cs.get("relief_m") if earth_cs.get("data_status") == "MEASURED_DEM" else None
        e_relief_unc = 8.0 if e_relief is not None else None

        if t_relief is not None and e_relief is not None and max(float(t_relief), float(e_relief)) > 0:
            r_min = min(float(t_relief), float(e_relief))
            r_max = max(float(t_relief), float(e_relief))
            sim_relief = round((r_min / r_max) * 100.0, 1)
            rel_unc = ((float(t_relief_unc or 10.0) + float(e_relief_unc)) / r_max) * 100.0
            unc_r_pct = round(min(25.0, max(2.0, rel_unc)), 1)
            status_relief = "NASA verisinden i?lendi"
        else:
            sim_relief, unc_r_pct, status_relief = None, None, "veri yok"

        # 3. ?nceleme Penceresi / Havza ?l?e?i (km)
        t_scale = pm.get("regional_basin_diameter_km")
        e_scale = round(float(target.get("radius_km", 5.0)) * 2.0, 1) if earth_cs.get("data_status") == "MEASURED_DEM" else None
        if t_scale is not None and e_scale is not None and max(float(t_scale), float(e_scale)) > 0:
            sim_scale = round((min(float(t_scale), float(e_scale)) / max(float(t_scale), float(e_scale))) * 100.0, 1)
            unc_sc_pct = 5.0
            status_scale = "NASA'dan al?nd?" if target.get("data_status") != "senaryo varsay?m?" else "senaryo varsay?m?"
        else:
            sim_scale, unc_sc_pct, status_scale = None, None, "veri yok"

        # 4 & 5. NASA POWER ?klim Ba?lam? (Madde 4 [P0]: G?vdeler aras?nda do?rudan puanlanmaz, weight=0.0!)
        e_temp = climate.get("mean_surface_temp_c") if climate.get("data_status") == "MEASURED_NASA_POWER" else None
        e_precip = climate.get("precipitation_mm_day") if climate.get("data_status") == "MEASURED_NASA_POWER" else None
        t_temp = target.get("mean_temp_c")

        rows = [
            {
                "criterion": "Ortalama Arazi E?imi (DEM)",
                "earth_measured": f"{e_slope}? (?0.5?)" if e_slope is not None else "Veri Yok",
                "target_measured": f"{t_slope}? (?{t_slope_unc}?)" if t_slope is not None else "Veri Yok",
                "unit": "deg (?)",
                "similarity_pct": sim_slope,
                "uncertainty_pct": unc_s_pct,
                "weight": w_map["slope"],
                "status_label": status_slope,
                "provenance_tier": "E?li Say?sal DEM ?l??m? (SRTM 90m vs LOLA/MOLA)",
                "source": "Copernicus/SRTM 90m DEM ? NASA LOLA/MOLA"
            },
            {
                "criterion": "Dikey Topografik R?lyef (?Z)",
                "earth_measured": f"{e_relief} m (?8 m)" if e_relief is not None else "Veri Yok",
                "target_measured": f"{t_relief} m (?{t_relief_unc} m)" if t_relief is not None else "Veri Yok",
                "unit": "m",
                "similarity_pct": sim_relief,
                "uncertainty_pct": unc_r_pct,
                "weight": w_map["relief"],
                "status_label": status_relief,
                "provenance_tier": "E?li Say?sal DEM ?l??m? (SRTM 90m vs LOLA/MOLA)",
                "source": "Copernicus/SRTM 90m DEM ? NASA LOLA/MOLA"
            },
            {
                "criterion": "B?lgesel Havza / Pencere ?l?e?i",
                "earth_measured": f"{e_scale} km (?0.5 km)" if e_scale is not None else "Veri Yok",
                "target_measured": f"{t_scale} km (?1.5 km)" if t_scale is not None else "Veri Yok",
                "unit": "km",
                "similarity_pct": sim_scale,
                "uncertainty_pct": unc_sc_pct,
                "weight": w_map["scale"],
                "status_label": status_scale,
                "provenance_tier": "Yay?nlanm?? Yap? ?l?e?i ? Taktik Ayak ?zi",
                "source": pm.get("measurement_source", "NASA Catalog")
            },
            {
                "criterion": "S?cakl?k Ba?lam? (G?vdeler Aras? Puanlanmaz - Madde 4)",
                "earth_measured": f"{e_temp}?C (NASA POWER TS/2m)" if e_temp is not None else "Veri Yok",
                "target_measured": f"{t_temp}?C (Y?zey Bolometrik)" if t_temp is not None else "Veri Yok",
                "unit": "?C",
                "similarity_pct": None,
                "uncertainty_pct": None,
                "weight": 0.0,
                "status_label": climate.get("status_label", "veri yok"),
                "provenance_tier": "Yaln?zca D?nya ?klim Ba?lam? (Farkl? Fiziksel De?i?ken ? Skora Kat?lmaz)",
                "source": "NASA POWER (D?nya) vs Diviner/THEMIS (Yaln?zca Ba?lam)"
            },
            {
                "criterion": "Ya??? & Kurakl?k Ba?lam? (Yaln?zca D?nya Filtresi)",
                "earth_measured": f"{e_precip} mm/g?n (?0.05)" if e_precip is not None else "Veri Yok",
                "target_measured": "Uygulanamaz (Vakum / Kuru Regolit)",
                "unit": "mm/day",
                "similarity_pct": None,
                "uncertainty_pct": None,
                "weight": 0.0,
                "status_label": climate.get("status_label", "veri yok"),
                "provenance_tier": "Yaln?zca D?nya Saha Kurakl?k Ba?lam? (Skora Kat?lmaz)",
                "source": "NASA POWER PRECTOTCORR Satellite API"
            }
        ]

        scored_rows = [r for r in rows if r["weight"] > 0]
        total_w = sum(r["weight"] for r in scored_rows)
        avail_rows = [r for r in scored_rows if r["similarity_pct"] is not None]
        avail_w = sum(r["weight"] for r in avail_rows)

        coverage_pct = round((avail_w / total_w) * 100.0, 1) if total_w > 0 else 0.0
        if avail_w > 0:
            weighted_sim = round(sum(r["weight"] * r["similarity_pct"] for r in avail_rows) / avail_w, 1)
            weighted_unc = round(sum(r["weight"] * (r["uncertainty_pct"] or 5.0) for r in avail_rows) / avail_w, 1)
        else:
            weighted_sim = None
            weighted_unc = None

        return {
            "rows": rows,
            "data_coverage_pct": coverage_pct,
            "measured_physical_similarity_pct": weighted_sim,
            "score_uncertainty_pct": weighted_unc,
            "confidence_interval_pct": (
                [round(max(0.0, weighted_sim - weighted_unc), 1), round(min(100.0, weighted_sim + weighted_unc), 1)]
                if weighted_sim is not None and weighted_unc is not None else None
            )
        }

    @staticmethod
    def run_weight_sensitivity_analysis(candidates: list, base_weights: np.ndarray) -> dict:
        """
        Madde 7 [P1]: A??rl?k Duyarl?l?k Analizi (Weight Sensitivity Analysis).
        Senaryo a??rl?klar? makul aral?kta (?%25 Dirichlet/Grid pert?rbasyonu, 24 farkl? kombinasyon)
        de?i?tirildi?inde aday s?ralamas?n?n ne kadar de?i?ti?ini test eder.
        """
        if not candidates:
            return {"rank_1_stability_pct": 0.0, "perturbation_trials": 0, "candidate_sensitivity": []}

        n_cands = len(candidates)
        trials = 24
        rng = np.random.default_rng(42)
        rank1_wins = np.zeros(n_cands, dtype=int)
        score_Boot = [[] for _ in range(n_cands)]

        for t in range(trials):
            noise = rng.uniform(0.75, 1.25, size=len(base_weights))
            w_pert = (base_weights * noise)
            w_pert = w_pert / (np.sum(w_pert) + 1e-6)

            trial_scores = []
            for idx, c in enumerate(candidates):
                phys = c.get("measured_physical_similarity_pct")
                if phys is None:
                    s_val = 0.0
                else:
                    # A??rl?k pert?rbasyonunun e?im/r?lyef/6B ?n tarama ?zerindeki etkisi
                    dim_m = np.array(c.get("dimension_matches_pct", [50]*6), dtype=np.float32)
                    s_6d_pert = float(np.sum(dim_m * w_pert))
                    s_val = 0.70 * float(phys) + 0.30 * s_6d_pert
                score_Boot[idx].append(s_val)
                trial_scores.append(s_val)

            winner = int(np.argmax(trial_scores))
            rank1_wins[winner] += 1

        sens_list = []
        for idx, c in enumerate(candidates):
            arr = np.array(score_Boot[idx])
            sens_list.append({
                "region_name": c["region_name"],
                "mean_perturbed_score_pct": round(float(np.mean(arr)), 1),
                "min_perturbed_score_pct": round(float(np.min(arr)), 1),
                "max_perturbed_score_pct": round(float(np.max(arr)), 1),
                "score_spread_pct": round(float(np.ptp(arr)), 1),
                "top1_retention_pct": round((rank1_wins[idx] / float(trials)) * 100.0, 1)
            })

        return {
            "weight_label": "Senaryo Varsay?m? (Kullan?c? Taraf?ndan De?i?tirilebilir ? NASA Standard? De?ildir)",
            "perturbation_range": "?25% ?ok Boyutlu A??rl?k Duyarl?l?k Testi (24 Deterministik Senaryo)",
            "rank_1_stability_pct": sens_list[0]["top1_retention_pct"] if sens_list else 0.0,
            "candidate_sensitivity": sens_list
        }

    @staticmethod
    def evaluate_ops_and_base_feasibility(target: dict, cand_cs: dict = None, climate: dict = None) -> dict:
        """
        P0 Madde 9: 'G?venli ini?', '?s i?in uygun' gibi kesin m?hendislik iddialar? kald?r?lm??t?r.
        Sonu?lar a??k?a 'Ara?t?rma ?nceli?i / ?n Tarama G?stergesi' olarak tan?mlanm??t?r.
        """
        cs = cand_cs if cand_cs is not None else target.get("cross_section", {})
        low_slope_pct = cs.get("low_slope_fraction_pct")
        mean_slope = cs.get("mean_slope_deg")

        screening_priority = round(float(low_slope_pct), 1) if low_slope_pct is not None else None

        return {
            "overall_base_viability_pct": screening_priority,
            "screening_status_label": "Ara?t?rma ?nceli?i / ?n Tarama (U?u? veya ?s G?venli?i Kan?t? De?ildir)",
            "recommended_habitat_architecture": "Jeomorfolojik Saha Ara?t?rma Oda?? (M?hendislik Do?rulamas? Gerekir)",
            "radiation_shielding_protocol": "D?nya manyetosferi GCR/SPE radyasyonunu engeller; z?rhlama ayr?ca modellenmelidir",
            "isru_primary_strategy": "Mineralojik / Kriyosferik Saha ?rnekleme ?n Taramas?",
            "power_and_illumination_architecture": "Y?r?nge/Enlem Ayd?nlanma Sim?lasyonu Gerekir (D?nya Atmosferi Farkl?d?r)",
            "landing_safety_envelope_pct": round(float(low_slope_pct), 1) if low_slope_pct is not None else None,
            "rover_mobility_trafficability_pct": (
                round(max(20.0, 100.0 - float(mean_slope) * 4.5), 1) if mean_slope is not None else None
            ),
            "trn_optical_navigation_pct": None,
            "regolith_construction_3d_pct": None
        }

    @staticmethod
    def evaluate_planetary_processes(target: dict, cand_vec: list, climate: dict) -> dict:
        t_vec = target.get("features_6d", [0.5]*6)
        c_basalt, c_flat, c_therm, c_ice, c_wind, c_cave = cand_vec
        earth_processes = {
            "Volkanik Bazalt & ??kme Yap?lar? (?n Tarama ?ndeksi)": round((0.55 * c_cave + 0.45 * c_basalt) * 100.0, 1),
            "Tektonik Kanyon & Yama? Erozyonu (?n Tarama ?ndeksi)": round((0.50 * (1.0 - c_flat) + 0.50 * c_cave) * 100.0, 1),
            "Kriyosferik Permafrost Dinami?i (?n Tarama ?ndeksi)": round(c_ice * 100.0, 1),
            "eoliyen (R?zgar) Tortul Ta??n?m? (?n Tarama ?ndeksi)": round((0.60 * c_wind + 0.40 * c_flat) * 100.0, 1)
        }
        dominant_earth_proc = max(earth_processes.items(), key=lambda x: x[1])
        proc_match = round((1.0 - float(np.mean(np.abs(np.array(t_vec) - np.array(cand_vec))))) * 100.0, 1)
        return {
            "status_label": "senaryo varsay?m?",
            "target_planetary_process": target.get("primary_process", "Jeomorfolojik S?re? ?n Taramas?"),
            "dominant_earth_process": dominant_earth_proc[0],
            "dominant_process_intensity_pct": dominant_earth_proc[1],
            "process_fidelity_pct": proc_match,
            "process_breakdown": earth_processes
        }

    @staticmethod
    def evaluate_astrobiology_and_habitability(cand_vec: list, climate: dict) -> dict:
        c_basalt, c_flat, c_therm, c_ice, c_wind, c_cave = cand_vec
        solar_kwh = climate.get("solar_irradiance_kwh_m2_day")
        precip = climate.get("precipitation_mm_day")
        temp_c = climate.get("mean_surface_temp_c")

        if temp_c is not None and temp_c < 0.0:
            guild = "Kriyofilik / Permafrost Ekstremofil Saha Hipotezi (Literat?r Ba?lam?)"
        elif c_cave > 0.65:
            guild = "Endolitik / Yeralt? Bazaltik Bo?luk Mikrobiyomu Hipotezi (Literat?r Ba?lam?)"
        else:
            guild = "Hiper-Kurak ??l / Evaporit Mikrobiyomu Hipotezi (Literat?r Ba?lam?)"

        return {
            "status_label": "senaryo varsay?m?",
            "extremophile_guild": guild,
            "biosignature_preservation_pct": round((0.5 * c_basalt + 0.5 * max(c_cave, c_ice)) * 100.0, 1),
            "uv_radiation_stress_pct": round(min(99.0, (float(solar_kwh) / 7.2) * 100.0), 1) if solar_kwh is not None else None,
            "solar_irradiance_kwh_m2_day": solar_kwh,
            " precipitation_mm_day": precip,
            "astrobiology_mission_relevance": "Saha ?rnekleme protokol? ?n taramas? (biyolojik do?rulama gerektirir)."
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
            target["radius_km"] = float(radius_km)
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
            target["rgb_image"],
            radius_km=target.get("radius_km", radius_km),
            published_physical=target.get("physical_measurements")
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
            "status_label": "senaryo varsay?m?",
            "source_body": current_body,
            "bridge_body": opposite_body,
            "bridge_target_id": best_match["id"],
            "bridge_name": best_match["name"],
            "lat": best_match["lat"],
            "lon": round(self._normalize_lon(float(best_match["lon"])), 3),
            "architectural_transfer_pct": round(best_sim * 100.0, 1),
            "engineering_adaptation_note": (
                "?n tarama hipotezi: Ay (0 mbar, 0.165g) ve Mars (6.1 mbar CO2, 0.378g) aras?ndaki "
                "termal, atmosferik ve yer?ekimi farklar? ayr? m?hendislik do?rulamas? gerektirir."
            )
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
        coarse_similarity = np.clip(1.0 - weighted_l1_dist, 0.01, 0.99)

        ice_need = float(target_vec[3])
        if ice_need < 0.35:
            cryo_mask = (self.features_6d[:, 3] > 0.35) | (np.abs(self.lats) > 45.0) | (np.abs(self.lats) < 14.0)
            coarse_similarity = np.where(cryo_mask, coarse_similarity * 0.35, coarse_similarity)
        elif ice_need > 0.65:
            hot_mask = (self.features_6d[:, 3] < 0.50) | (np.abs(self.lats) < 48.0)
            coarse_similarity = np.where(hot_mask, coarse_similarity * 0.35, coarse_similarity)

        valid_mask = (self.is_well_known == 0) if novelty_only else np.ones(len(self.lats), dtype=bool)
        masked_scores = np.where(valid_mask, coarse_similarity, -1.0)
        sorted_indices = np.argsort(masked_scores)[::-1]

        pool_size = max(top_k + 5, 8)
        shortlist_indices = []
        for idx in sorted_indices:
            if masked_scores[idx] <= 0:
                break
            cand_lat, cand_lon = float(self.lats[idx]), float(self.lons[idx])
            too_close = False
            for sel_idx in shortlist_indices:
                s_lat, s_lon = float(self.lats[sel_idx]), float(self.lons[sel_idx])
                if self._haversine_km(cand_lat, cand_lon, s_lat, s_lon) < min_separation_km:
                    too_close = True
                    break
            if not too_close:
                shortlist_indices.append(int(idx))
            if len(shortlist_indices) >= pool_size:
                break

        radius_km = float(target_profile.get("radius_km", 5.0))
        pil_t, ten_t = self.preprocessor.preprocess_for_clip(target_profile.get("rgb_image"))
        gray_t = np.mean(ten_t, axis=2).astype(np.float32)
        gx_t = cv2.Sobel(gray_t, cv2.CV_32F, 1, 0, ksize=3)
        gy_t = cv2.Sobel(gray_t, cv2.CV_32F, 0, 1, ksize=3)
        h_mag_t, _ = np.histogram(np.hypot(gx_t, gy_t), bins=16, range=(0.0, 1.5), density=True)
        tex_target = h_mag_t.astype(np.float32) / (np.linalg.norm(h_mag_t) + 1e-6)
        emb_target = None

        def _fetch_io_bundle(rank_and_idx):
            pre_rank, grid_idx = rank_and_idx
            c_lat = float(self.lats[grid_idx])
            c_lon = float(self.lons[grid_idx])
            earth_rgb, earth_img_path = self.loader.fetch_earth_satellite_patch(c_lat, c_lon, radius_km=radius_km)
            dem_transect = self.loader.fetch_real_earth_dem_transect(c_lat, c_lon, radius_km=radius_km, num_samples=11)
            climate_data = self.loader.fetch_nasa_power_climate(c_lat, c_lon)
            region_name = self.loader.reverse_geocode_earth(c_lat, c_lon)
            return pre_rank, grid_idx, earth_rgb, earth_img_path, dem_transect, climate_data, region_name

        indexed_shortlist = list(enumerate(shortlist_indices, 1))
        with ThreadPoolExecutor(max_workers=4) as executor:
            io_results = list(executor.map(_fetch_io_bundle, indexed_shortlist))

        evaluated_pool = []
        for pre_rank, idx, earth_rgb, earth_img_path, dem_transect, climate_data, region_name in io_results:
            c_lat = float(self.lats[idx])
            c_lon = float(self.lons[idx])
            coarse_vec = [round(float(x), 3) for x in self.features_6d[idx].tolist()]

            earth_cs = self.preprocessor.extract_elevation_cross_section(
                earth_rgb, radius_km=radius_km, dem_transect=dem_transect
            )
            phys_evidence = self.build_paired_physical_evidence_table(target_profile, earth_cs, climate_data)
            vis_signal = self._compute_visual_exploration_signal(emb_target, tex_target, earth_rgb)

            dim_matches_pct = [
                round((1.0 - abs(float(target_vec[d]) - float(coarse_vec[d]))) * 100.0, 1)
                for d in range(6)
            ]
            parametric_6d_pct = round(float(np.sum(np.array(dim_matches_pct) * weights)), 1)

            # Madde 1 & 8 [P0/P1]: E?li fiziksel DEM ?l??m? varsa skor do?rudan fiziksel ?l??mden gelir;
            # veri yoksa (?r. ?evrimd??? veya serbest koordinat) fiziksel skor None kal?r (ASLA uydurulmaz!).
            measured_sim = phys_evidence["measured_physical_similarity_pct"]
            topo_match = measured_sim if measured_sim is not None else 0.0
            screening_score = measured_sim if measured_sim is not None else parametric_6d_pct

            micro_feature = self.preprocessor.pinpoint_micro_scale_feature(
                earth_rgb, c_lat, c_lon, radius_km=radius_km, mission_mode=mission_mode
            )
            hud_rgb = self.preprocessor.render_engineering_hud(
                earth_rgb, earth_cs, micro_feature, radius_km=radius_km
            )

            evaluated_pool.append({
                "grid_index": idx,
                "pre_screen_rank": pre_rank,
                "region_name": region_name,
                "lat": round(c_lat, 3),
                "lon": round(c_lon, 3),
                "is_novel_discovery": bool(self.is_well_known[idx] == 0),
                "total_match_pct": screening_score,
                "measured_physical_similarity_pct": measured_sim,
                "score_uncertainty_pct": phys_evidence["score_uncertainty_pct"],
                "confidence_interval_pct": phys_evidence["confidence_interval_pct"],
                "data_coverage_pct": phys_evidence["data_coverage_pct"],
                "parametric_6d_pct": parametric_6d_pct,
                "topographic_match_pct": topo_match,
                "visual_hybrid_pct": vis_signal["hybrid_visual_pct"] or 0.0,
                "live_env_verification_pct": 100.0 if climate_data.get("data_status") == "MEASURED_NASA_POWER" else 0.0,
                "physical_evidence_table": phys_evidence["rows"],
                "vision_engine_used": vis_signal["engine_type"],
                "features_6d": coarse_vec,
                "dimension_matches_pct": dim_matches_pct,
                "image_path": earth_img_path,
                "rgb_image": earth_rgb,
                "hud_image": hud_rgb,
                "cross_section": earth_cs,
                "nasa_power_climate": climate_data,
                "micro_scale_pinpoint": micro_feature,
                "process_analogy": self.evaluate_planetary_processes(target_profile, coarse_vec, climate_data),
                "astrobiology_profile": self.evaluate_astrobiology_and_habitability(coarse_vec, climate_data),
                "operations_readiness": self.evaluate_ops_and_base_feasibility(target_profile, earth_cs, climate_data),
                "strategic_role_badge": (
                    f"?n Tarama Aday? (E?li Fiziksel Kapsam: %{phys_evidence['data_coverage_pct']} | "
                    f"Belirsizlik: ?%{phys_evidence['score_uncertainty_pct'] or 'N/A'})"
                )
            })

        evaluated_pool.sort(key=lambda x: (x["measured_physical_similarity_pct"] or -1.0, x["parametric_6d_pct"]), reverse=True)
        final_candidates = evaluated_pool[:top_k]

        sensitivity_report = self.run_weight_sensitivity_analysis(final_candidates, weights)
        complementary_site = self._find_complementary_site(target_vec, final_candidates, novelty_only=novelty_only)
        planetary_bridge = self.find_moon_to_mars_bridge(target_profile)

        return {
            "method_version": METHOD_VERSION,
            "target": target_profile,
            "mission_mode": mission_mode,
            "scenario_weights": [round(float(w), 3) for w in weights],
            "weight_sensitivity_analysis": sensitivity_report,
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
                    "final_score_pct": c["total_match_pct"],
                    "data_coverage_pct": c["data_coverage_pct"]
                }
                for c in evaluated_pool
            ],
            "primary_candidates": final_candidates,
            "complementary_split_site": complementary_site,
            "moon_mars_bridge": planetary_bridge
        }

    def _find_complementary_site(self, target_vec: np.ndarray, final_candidates: list, novelty_only: bool = True) -> dict:
        primary_cand = final_candidates[0]
        weakest_dim = int(np.argmin(primary_cand["dimension_matches_pct"]))
        dim_name = DIMENSION_NAMES[weakest_dim]
        target_val = float(target_vec[weakest_dim])

        dim_closeness = 1.0 - np.abs(self.features_6d[:, weakest_dim] - target_val)
        overall_closeness = 1.0 - np.mean(np.abs(self.features_6d - target_vec), axis=1)
        comp_score = 0.65 * dim_closeness + 0.35 * overall_closeness

        if novelty_only:
            comp_score = np.where(self.is_well_known == 0, comp_score, -1.0)

        best_idxs = np.argsort(comp_score)[::-1]
        chosen_idx = int(best_idxs[0])
        for idx in best_idxs[:30]:
            c_lat, c_lon = float(self.lats[idx]), float(self.lons[idx])
            if all(self._haversine_km(fc["lat"], fc["lon"], c_lat, c_lon) >= 600.0 for fc in final_candidates):
                chosen_idx = int(idx)
                break

        c_lat, c_lon = float(self.lats[chosen_idx]), float(self.lons[chosen_idx])
        return {
            "status_label": "senaryo varsay?m?",
            "weakest_dimension_index": weakest_dim,
            "weakest_dimension_name": dim_name,
            "primary_site_score_in_dim_pct": primary_cand["dimension_matches_pct"][weakest_dim],
            "complementary_region_name": self.loader.reverse_geocode_earth(c_lat, c_lon),
            "lat": round(c_lat, 3),
            "lon": round(c_lon, 3),
            "complementary_dim_match_pct": round(float(dim_closeness[chosen_idx]) * 100.0, 1),
            "features_6d": [round(float(x), 3) for x in self.features_6d[chosen_idx]],
            "nasa_power_climate": self.loader.fetch_nasa_power_climate(c_lat, c_lon)
        }
