import os
import sys
import shutil
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))


def ensure_python_dependencies():
    required = ["numpy", "scipy", "cv2", "PIL", "requests", "reverse_geocoder"]
    missing = []
    for mod in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"[KURULUM] Eksik Python paketleri kuruluyor ({', '.join(missing)})...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def ensure_data_grid():
    grid_file = os.path.join(ROOT_DIR, "data", "global_grids", "earth_6d_grid.npz")
    if not os.path.exists(grid_file):
        print("[VER?] K?resel 6B matris olu?turuluyor...")
        subprocess.check_call([sys.executable, os.path.join("src", "build_planetary_data.py")])


def ensure_frontend_dist():
    dist_index = os.path.join(ROOT_DIR, "frontend_dist", "index.html")
    frontend_dir = os.path.join(ROOT_DIR, "frontend")
    if os.path.exists(dist_index):
        return
    if os.path.exists(frontend_dir) and shutil.which("npm"):
        print("[DERLEME] 3B Web Aray?z? (frontend_dist) derleniyor...")
        npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
        if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
            subprocess.check_call([npm_cmd, "install"], cwd=frontend_dir)
        subprocess.check_call([npm_cmd, "run", "build"], cwd=frontend_dir)


if __name__ == "__main__":
    ensure_python_dependencies()
    ensure_data_grid()
    ensure_frontend_dist()

    if "--test" in sys.argv:
        from matcher_engine import PlanetaryMatcherEngine
        eng = PlanetaryMatcherEngine(clip_mode="off")
        tp = eng.get_target_profile(preset_id="MOON_MARE_TRANQUILLITATIS_PIT")
        res = eng.find_earth_analogs(tp, mission_mode="HABITAT_LAVA_TUBE", top_k=3)
        c1 = res["primary_candidates"][0]
        print(f"[BA?ARILI] Test Ge?ti! #1 Aday: {c1['region_name']} (Skor: %{c1['total_match_pct']}, Kapsam: %{c1['data_coverage_pct']})")
    else:
        from api_server import start_unified_server
        start_unified_server(port=8000, open_browser=True)
