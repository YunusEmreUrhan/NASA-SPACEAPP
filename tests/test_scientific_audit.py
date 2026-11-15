import os
import sys
import json
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from data_loader import PlanetaryDataLoader, VALID_STATUS_LABELS
from matcher_engine import PlanetaryMatcherEngine


class TestScientificAuditCompliance(unittest.TestCase):
    """
    NASA-SPACEAPP_sorunlar_ve_cozum_yontemleri.txt i?indeki P0, P1, P2, P3
    Kabul ?l??tlerini do?rulayan deterministik test paketi.
    """

    def test_p0_offline_zero_imputation(self):
        """Kabul ?l??t? 1: ?nternet kapal?yken uygulama hi?bir NASA ?l??m? veya skor girdisi uydurmuyor."""
        offline_eng = PlanetaryMatcherEngine(clip_mode="off", offline_mode=True)
        tp = offline_eng.get_target_profile(preset_id="MOON_MARE_TRANQUILLITATIS_PIT")
        res = offline_eng.find_earth_analogs(tp, top_k=2)
        cand = res["primary_candidates"][0]

        self.assertEqual(cand["nasa_power_climate"]["status_label"], "veri yok")
        self.assertIsNone(cand["nasa_power_climate"]["mean_surface_temp_c"])
        self.assertEqual(cand["cross_section"]["status_label"], "veri yok")
        self.assertIsNone(cand["cross_section"]["mean_slope_deg"])
        # DEM ?evrimd??? oldu?unda e?im ve r?lyef skoru None olmal? (uydurulmamal?)
        slope_row = cand["physical_evidence_table"][0]
        self.assertIsNone(slope_row["similarity_pct"])

    def test_p0_wmts_tile_does_not_produce_physical_numbers(self):
        """Kabul ?l??t? 2 & 5: G?rsel WMTS karosu fiziksel e?im veya r?lyef ?l??m? ?retmiyor."""
        loader = PlanetaryDataLoader(offline_mode=False)
        dyn = loader.extract_dynamic_target_profile("Mars", -13.8, -61.2, 5.0)
        self.assertIsNone(dyn["physical_measurements"]["mean_slope_deg"])
        self.assertIsNone(dyn["physical_measurements"]["vertical_relief_m"])
        self.assertEqual(dyn["observations"]["mean_slope_deg"]["status_label"], "veri yok")

    def test_p0_target_catalog_has_provenance_and_units(self):
        """Kabul ?l??t? 3: Hedef katalo?undaki her g?zlem kaynak URL, birim ve belirsizlik i?eriyor."""
        with open(os.path.join(ROOT_DIR, "metadata", "moon_mars_targets.json"), "r", encoding="utf-8") as f:
            cat = json.load(f)
        for tid, item in cat.items():
            self.assertIn("observations", item)
            for obs_key, obs_val in item["observations"].items():
                self.assertIn("unit", obs_val)
                self.assertIn("source_url", obs_val)
                self.assertIn(obs_val["status_label"], VALID_STATUS_LABELS)

    def test_p0_nasa_power_not_scored_against_planetary_surface_temp(self):
        """Kabul ?l??t? 4: D?nya hava/y?zey s?cakl??? ile Ay/Mars y?zey s?cakl??? e?de?er kabul edilip puanlanm?yor."""
        eng = PlanetaryMatcherEngine(clip_mode="off", offline_mode=False)
        tp = eng.get_target_profile(preset_id="MARS_JEZERO_CRATER")
        res = eng.find_earth_analogs(tp, top_k=2)
        rows = res["primary_candidates"][0]["physical_evidence_table"]
        temp_row = [r for r in rows if "S?cakl?k" in r["criterion"]][0]
        self.assertEqual(temp_row["weight"], 0.0)
        self.assertIsNone(temp_row["similarity_pct"])

    def test_p1_weight_sensitivity_and_reproducibility(self):
        """Kabul ?l??t? 7 & 10: A??rl?k duyarl?l?k analizi ?retiliyor ve ayn? girdilerle deterministik sonu? ??k?yor."""
        eng = PlanetaryMatcherEngine(clip_mode="off", offline_mode=False)
        tp = eng.get_target_profile(preset_id="MOON_SHACKLETON_RIM")
        r1 = eng.find_earth_analogs(tp, mission_mode="ISRU_WATER_MINING", top_k=3)
        r2 = eng.find_earth_analogs(tp, mission_mode="ISRU_WATER_MINING", top_k=3)

        self.assertEqual(r1["primary_candidates"][0]["region_name"], r2["primary_candidates"][0]["region_name"])
        self.assertIn("weight_sensitivity_analysis", r1)
        self.assertGreaterEqual(len(r1["weight_sensitivity_analysis"]["candidate_sensitivity"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
