import math
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d


class PlanetaryPreprocessor:
    def __init__(self, target_size: int = 224):
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess_for_clip(self, image_rgb: np.ndarray):
        if image_rgb is None or image_rgb.size == 0:
            blank = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            return Image.fromarray(blank), blank.astype(np.float32) / 255.0

        resized = cv2.resize(
            image_rgb,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_CUBIC
        )
        lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_clahe = self.clahe.apply(l_chan)
        enhanced_rgb = cv2.cvtColor(cv2.merge((l_clahe, a_chan, b_chan)), cv2.COLOR_LAB2RGB)

        pil_img = Image.fromarray(enhanced_rgb)
        norm_tensor = enhanced_rgb.astype(np.float32) / 255.0
        return pil_img, norm_tensor

    def extract_elevation_cross_section(
        self,
        image_rgb: np.ndarray,
        radius_km: float = 5.0,
        num_points: int = 100
    ):
        span_km = radius_km * 2.0
        distances_km = np.linspace(0.0, span_km, num_points)

        if image_rgb is None or image_rgb.size == 0:
            return {
                "distances_km": np.round(distances_km, 2),
                "elevation_m": np.zeros(num_points),
                "slope_deg": np.zeros(num_points),
                "mean_slope_deg": 0.0,
                "max_slope_deg": 0.0,
                "safe_landing_pct": 100.0,
                "best_base_km": round(span_km / 2.0, 2)
            }

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        h, w = gray.shape
        band_half = max(2, h // 20)
        mid_row = h // 2
        center_band = np.mean(gray[mid_row - band_half:mid_row + band_half, :], axis=0)

        albedo_smooth = gaussian_filter1d(center_band, sigma=3.0)
        shading_deviation = center_band - gaussian_filter1d(center_band, sigma=25.0)
        integrated_relief = np.cumsum(shading_deviation)
        drift = np.linspace(integrated_relief[0], integrated_relief[-1], len(integrated_relief))
        integrated_relief = integrated_relief - drift

        x_old = np.linspace(0.0, 1.0, w)
        x_new = np.linspace(0.0, 1.0, num_points)
        prof_integ = np.interp(x_new, x_old, integrated_relief)
        prof_albedo = np.interp(x_new, x_old, albedo_smooth)

        norm_integ = (prof_integ - np.min(prof_integ)) / (np.ptp(prof_integ) + 1e-6)
        norm_alb = (prof_albedo - np.min(prof_albedo)) / (np.ptp(prof_albedo) + 1e-6)
        combined = gaussian_filter1d(0.65 * norm_integ + 0.35 * norm_alb, sigma=1.8)

        contrast_factor = float(np.clip(np.std(gray) * 4.5, 0.25, 1.5))
        elevation_m = (combined - np.min(combined)) * (95.0 * contrast_factor)

        dx_m = (span_km * 1000.0) / max(1, (num_points - 1))
        dz_dx = np.gradient(elevation_m, dx_m)
        slope_deg = np.degrees(np.arctan(np.abs(dz_dx) * 3.5))

        safe_mask = slope_deg <= 5.0
        safe_landing_pct = float(np.mean(safe_mask) * 100.0)

        window = 8
        rolling_slope = np.convolve(slope_deg, np.ones(window) / window, mode="valid")
        best_idx = int(np.argmin(rolling_slope)) + (window // 2)
        best_base_km = float(distances_km[min(best_idx, num_points - 1)])

        return {
            "distances_km": np.round(distances_km, 2),
            "elevation_m": np.round(elevation_m, 1),
            "slope_deg": np.round(slope_deg, 2),
            "mean_slope_deg": round(float(np.mean(slope_deg)), 2),
            "max_slope_deg": round(float(np.max(slope_deg)), 2),
            "safe_landing_pct": round(safe_landing_pct, 1),
            "best_base_km": round(best_base_km, 2)
        }

    def pinpoint_micro_scale_feature(
        self,
        image_rgb: np.ndarray,
        center_lat: float,
        center_lon: float,
        radius_km: float = 5.0,
        mission_mode: str = "BALANCED"
    ) -> dict:
        if image_rgb is None or image_rgb.size == 0:
            return {
                "micro_feature_type": "Local Flat Basin",
                "micro_lat": round(center_lat, 4),
                "micro_lon": round(center_lon, 4),
                "offset_from_center_km": 0.0,
                "patch_scale_km": "1.25 x 1.25 km",
                "micro_suitability_pct": 85.0,
                "pixel_box": [224, 224, 64, 64]
            }

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        h, w = gray.shape
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.hypot(gx, gy)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

        win = max(32, h // 8)
        stride = max(16, win // 2)
        best_score = -1e9
        best_y, best_x = h // 2, w // 2

        for y in range(win // 2, h - win // 2, stride):
            for x in range(win // 2, w - win // 2, stride):
                sub_gray = gray[y - win // 2:y + win // 2, x - win // 2:x + win // 2]
                sub_grad = grad[y - win // 2:y + win // 2, x - win // 2:x + win // 2]
                sub_lap = lap[y - win // 2:y + win // 2, x - win // 2:x + win // 2]

                darkness = 1.0 - float(np.mean(sub_gray))
                flatness = 1.0 - float(np.clip(np.mean(sub_grad) * 2.5, 0.0, 1.0))
                pit_rim = float(np.percentile(sub_lap, 95))

                if mission_mode == "HABITAT_LAVA_TUBE":
                    score = 0.45 * pit_rim + 0.35 * darkness + 0.20 * flatness
                elif mission_mode == "ROVER_LANDING_OPS":
                    score = 0.70 * flatness + 0.30 * (1.0 - pit_rim)
                elif mission_mode == "ISRU_WATER_MINING":
                    score = 0.45 * flatness + 0.35 * pit_rim + 0.20 * (1.0 - darkness)
                else:
                    score = 0.40 * flatness + 0.35 * pit_rim + 0.25 * darkness

                if score > best_score:
                    best_score = score
                    best_y, best_x = y, x

        norm_dy = (0.5 - (best_y / float(h))) * 2.0
        norm_dx = ((best_x / float(w)) - 0.5) * 2.0
        km_per_deg_lat = 111.32
        km_per_deg_lon = max(111.32 * math.cos(math.radians(abs(center_lat))), 20.0)

        micro_lat = center_lat + (norm_dy * radius_km) / km_per_deg_lat
        micro_lon = center_lon + (norm_dx * radius_km) / km_per_deg_lon
        offset_km = math.hypot(norm_dx * radius_km, norm_dy * radius_km)

        if mission_mode == "HABITAT_LAVA_TUBE":
            f_type = "Collapsed Volcanic Pit / Canyon Wall (Skylight Micro-Analog)"
        elif mission_mode == "ROVER_LANDING_OPS":
            f_type = "Low-Slope Alluvial/Basaltic Landing Pad (<3° Micro-Basin)"
        elif mission_mode == "ISRU_WATER_MINING":
            f_type = "Shadowed Cold-Trap Depression / Evaporite Pan"
        else:
            f_type = "Crater Rim & Channel Contact Zone (High-Contrast Micro-Site)"

        suitability = round(float(np.clip(78.0 + best_score * 22.0, 75.0, 98.8)), 1)
        patch_w_km = round((win / float(w)) * radius_km * 2.0, 2)
        return {
            "micro_feature_type": f_type,
            "micro_lat": round(micro_lat, 4),
            "micro_lon": round(micro_lon, 4),
            "offset_from_center_km": round(offset_km, 2),
            "patch_scale_km": f"{patch_w_km} x {patch_w_km} km",
            "micro_suitability_pct": suitability,
            "pixel_box": [int(best_x - win // 2), int(best_y - win // 2), int(win), int(win)]
        }

    @staticmethod
    def render_engineering_hud(
        image_rgb: np.ndarray,
        cross_section: dict,
        micro_pinpoint: dict = None,
        radius_km: float = 5.0
    ) -> np.ndarray:
        if image_rgb is None or image_rgb.size == 0:
            return image_rgb
        hud = image_rgb.copy()
        h, w, _ = hud.shape
        mid_y = h // 2

        for x in range(0, w, 16):
            cv2.line(hud, (x, mid_y), (min(x + 8, w - 1), mid_y), (255, 215, 0), 2)

        span_km = max(1.0, radius_km * 2.0)
        best_km = float(cross_section.get("best_base_km", radius_km))
        base_x = int(np.clip((best_km / span_km) * w, 16, w - 16))
        cv2.circle(hud, (base_x, mid_y), 10, (0, 255, 180), 2)
        cv2.circle(hud, (base_x, mid_y), 3, (0, 255, 180), -1)
        cv2.putText(
            hud, f"BASE PAD ({best_km:.1f}km)", (max(8, base_x - 65), max(22, mid_y - 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1, cv2.LINE_AA
        )

        if micro_pinpoint and "pixel_box" in micro_pinpoint:
            bx, by, bw, bh = micro_pinpoint["pixel_box"]
            cv2.rectangle(hud, (bx, by), (bx + bw, by + bh), (0, 212, 255), 2)
            cx, cy = bx + bw // 2, by + bh // 2
            cv2.line(hud, (cx - 8, cy), (cx + 8, cy), (0, 212, 255), 1)
            cv2.line(hud, (cx, cy - 8), (cx, cy + 8), (0, 212, 255), 1)
            cv2.putText(
                hud, f"MICRO-SITE ({micro_pinpoint.get('patch_scale_km', '1.25x1.25km')})",
                (max(6, bx - 10), max(18, by - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 212, 255), 1, cv2.LINE_AA
            )

        return hud
