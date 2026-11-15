import os
import sys
import subprocess

# Calisma dizinini her zaman proje kok klasorune sabitle
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

# Eger global 6B matris eksikse otomatik olustur
grid_file = os.path.join(ROOT_DIR, "data", "global_grids", "earth_6d_grid.npz")
if not os.path.exists(grid_file):
    print("[BILGI] K?resel 6B matris bulunamad?, src/build_planetary_data.py ?al??t?r?l?yor...")
    subprocess.check_call([sys.executable, os.path.join("src", "build_planetary_data.py")])

if __name__ == "__main__":
    if "--test" in sys.argv:
        from matcher_engine import PlanetaryMatcherEngine
        eng = PlanetaryMatcherEngine(clip_mode="off")
        tp = eng.get_target_profile(preset_id="MOON_MARE_TRANQUILLITATIS_PIT")
        res = eng.find_earth_analogs(tp, mission_mode="HABITAT_LAVA_TUBE", top_k=3)
        c1 = res["primary_candidates"][0]
        print(f"[BASARILI] ?ekirdek Testi Ge?ti! #1 Aday: {c1['region_name']} (Skor: %{c1['total_match_pct']}, Kapsam: %{c1['data_coverage_pct']})")
    else:
        print("[BASLATILIYOR] TERRA-TWIN aray?z? taray?c?da a??l?yor...")
        raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py"]))
