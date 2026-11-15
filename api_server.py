import os
import sys
import json
import base64
import mimetypes
import webbrowser
from threading import Timer
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from matcher_engine import PlanetaryMatcherEngine, DIMENSION_NAMES
from xai_explainer import PlanetaryXAIExplainer

DIST_DIR = os.path.join(ROOT_DIR, "frontend_dist")

print("[TERRA-TWIN] NASA 2026 K?resel 6B Matrisi (11,469 H?cre) ve XAI Motoru y?kleniyor...")
ENGINE = PlanetaryMatcherEngine(clip_mode="off")
EXPLAINER = PlanetaryXAIExplainer()
print("[TERRA-TWIN] Bilimsel ?ekirdek Haz?r!")


def encode_rgb_to_data_url(rgb_arr: np.ndarray, quality: int = 88) -> str:
    if rgb_arr is None or rgb_arr.size == 0:
        return ""
    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def serialize_clean(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: serialize_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_clean(v) for v in obj]
    return obj


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class UnifiedPlanetaryHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        req_path = parsed.path

        if req_path == "/api/health":
            payload = {
                "status": "ok",
                "engine": "TERRA-TWIN Scientific Core v6.0",
                "scanned_cells": len(ENGINE.lats),
                "dimension_names": DIMENSION_NAMES,
                "catalog": ENGINE.catalog
            }
            data = json.dumps(serialize_clean(payload), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self._send_cors()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(data)
            return

        # 3B Web Aray?z? Statik Dosyalar?n? Sun (frontend_dist/)
        if req_path == "/" or req_path == "":
            req_path = "/index.html"

        safe_rel = os.path.normpath(req_path.lstrip("/"))
        file_path = os.path.join(DIST_DIR, safe_rel)

        if not os.path.abspath(file_path).startswith(os.path.abspath(DIST_DIR)):
            self.send_response(403)
            self.end_headers()
            return

        if not os.path.exists(file_path) or os.path.isdir(file_path):
            file_path = os.path.join(DIST_DIR, "index.html")

        if os.path.exists(file_path) and os.path.isfile(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                if file_path.endswith(".js"):
                    mime_type = "application/javascript"
                elif file_path.endswith(".css"):
                    mime_type = "text/css"
                elif file_path.endswith(".ttf"):
                    mime_type = "font/ttf"
                else:
                    mime_type = "application/octet-stream"

            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self._send_cors()
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/analyze":
            self.send_response(404)
            self._send_cors()
            self.end_headers()
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            req = json.loads(raw_body)

            selection_mode = req.get("selection_mode", "PRESET")
            preset_id = req.get("preset_id", "MOON_MARE_TRANQUILLITATIS_PIT")
            body = req.get("body", "Moon")
            lat = float(req.get("lat", 8.335))
            lon = float(req.get("lon", 33.222))
            radius_km = float(req.get("radius_km", 5.0))
            mission_mode = req.get("mission_mode", "HABITAT_LAVA_TUBE")
            custom_weights = req.get("custom_weights", None)
            novelty_only = bool(req.get("novelty_only", True))
            min_sep_km = float(req.get("min_separation_km", 450.0))

            if selection_mode == "PRESET" and preset_id in ENGINE.catalog:
                target_profile = ENGINE.get_target_profile(preset_id=preset_id, radius_km=radius_km)
                target_profile["radius_km"] = radius_km
                target_profile["cross_section"] = ENGINE.preprocessor.extract_elevation_cross_section(
                    target_profile["rgb_image"],
                    radius_km=radius_km,
                    published_physical=target_profile.get("physical_measurements")
                )
            else:
                target_profile = ENGINE.get_target_profile(
                    body=body, lat=lat, lon=lon, radius_km=radius_km
                )

            eff_mode = "BALANCED" if mission_mode == "CUSTOM" else mission_mode
            target_micro = ENGINE.preprocessor.pinpoint_micro_scale_feature(
                target_profile["rgb_image"],
                target_profile["lat"],
                target_profile["lon"],
                radius_km=radius_km,
                mission_mode=eff_mode
            )
            target_hud_img = ENGINE.preprocessor.render_engineering_hud(
                target_profile["rgb_image"],
                target_profile["cross_section"],
                target_micro,
                radius_km=radius_km
            )

            search_res = ENGINE.find_earth_analogs(
                target_profile=target_profile,
                mission_mode=eff_mode,
                custom_weights=custom_weights if mission_mode == "CUSTOM" else None,
                novelty_only=novelty_only,
                top_k=3,
                min_separation_km=min_sep_km
            )

            candidates = search_res["primary_candidates"]
            comp_site = search_res["complementary_split_site"]
            bridge = search_res["moon_mars_bridge"]

            xai_reports = []
            for c in candidates:
                xai_reports.append(
                    EXPLAINER.generate_candidate_explanation(target_profile, c, comp_site, bridge)
                )

            roadmap = EXPLAINER.build_phased_campaign_roadmap(candidates, comp_site)
            md_dossier = EXPLAINER.export_markdown_dossier(search_res)

            target_out = {k: v for k, v in target_profile.items() if k != "rgb_image"}
            target_out["rgb_data_url"] = encode_rgb_to_data_url(target_profile.get("rgb_image"))
            target_out["hud_data_url"] = encode_rgb_to_data_url(target_hud_img)
            target_out["micro_scale_pinpoint"] = target_micro

            cands_out = []
            for idx, c in enumerate(candidates):
                c_clean = {k: v for k, v in c.items() if k not in ("rgb_image", "hud_image")}
                c_clean["rgb_data_url"] = encode_rgb_to_data_url(c.get("rgb_image"))
                c_clean["hud_data_url"] = encode_rgb_to_data_url(c.get("hud_image"))
                c_clean["xai_explanation"] = xai_reports[idx]
                cands_out.append(c_clean)

            response_payload = {
                "status": "ok",
                "dimension_names": DIMENSION_NAMES,
                "target": target_out,
                "primary_candidates": cands_out,
                "shortlist_full_pool": search_res["shortlist_full_pool"],
                "complementary_split_site": comp_site,
                "moon_mars_bridge": bridge,
                "xai_primary": xai_reports[0] if xai_reports else {},
                "roadmap": roadmap,
                "markdown_dossier": md_dossier,
                "scanned_total_cells": search_res["scanned_total_cells"],
                "method_version": search_res.get("method_version", "TERRA-TWIN-v7.0"),
                "weight_sensitivity_analysis": search_res.get("weight_sensitivity_analysis", {}),
                "reproducibility_manifest": {
                    "git_commit_id": "4d3da3b",
                    "code_version": "v7.0-ScientificAudit",
                    "dem_product": "Copernicus GLO-90 / NASA SRTM 90m v3.0",
                    "climate_product": "NASA POWER Release 9 (1991-2020 Climatology - Context Only)",
                    "planetary_catalog": "NASA LRO LOLA / Diviner / MGS MOLA / Mars 2020 Published Records",
                    "zero_imputation_enforced": True
                }
            }

            out_bytes = json.dumps(serialize_clean(response_payload), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self._send_cors()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(out_bytes)
        except Exception as e:
            err_bytes = json.dumps({"status": "error", "message": str(e)}).encode("utf-8")
            self.send_response(500)
            self._send_cors()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(err_bytes)


def start_unified_server(port: int = 8000, open_browser: bool = True):
    url = f"http://127.0.0.1:{port}"
    server = ThreadedHTTPServer(("127.0.0.1", port), UnifiedPlanetaryHandler)
    print("\n" + "=" * 68)
    print(f"  ?? TERRA-TWIN 3B GEZEGEN MOTORU AKT?F: {url}")
    print("  Taray?c?n?zda otomatik a??l?yor... (Kapatmak i?in Ctrl+C)")
    print("=" * 68 + "\n")
    if open_browser:
        Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[KAPATILDI] TERRA-TWIN sunucusu durduruldu.")


if __name__ == "__main__":
    start_unified_server(port=int(os.environ.get("PORT", "8000")), open_browser=True)
