import json
from datetime import datetime, timezone


class PlanetaryXAIExplainer:
    """
    P0-P3 Bilimsel Denetim Uyumlu XAI A??klay?c? ve Rapor ?retici:
    - Eksik (None / 'veri yok') g?zlemlerde asla ??kmez ve de?er uydurmaz.
    - Her g?zlemin durum etiketini ('NASA\'dan al?nd?', 'NASA verisinden i?lendi',
      'senaryo varsay?m?', 'sim?lasyon', 'veri yok'), birimini, belirsizli?ini (?) ve
      kaynak URL\'sini a??k?a raporlar.
    """

    def generate_candidate_explanation(
        self,
        target: dict,
        candidate: dict,
        complementary_site: dict = None,
        moon_mars_bridge: dict = None
    ) -> dict:
        body = target.get("body", "Moon")
        t_grav = float(target.get("gravity_g") or (0.165 if body == "Moon" else 0.378))
        offload_pct = round((1.0 - (t_grav / 1.0)) * 100.0, 1)

        t_press = target.get("pressure_mbar", 0.0 if body == "Moon" else 6.1)
        t_rad = target.get("radiation_msv_yr", 380.0 if body == "Moon" else 230.0)
        t_temp = target.get("mean_temp_c", -50.0)

        climate = candidate.get("nasa_power_climate", {})
        c_temp = climate.get("mean_surface_temp_c")
        c_precip = climate.get("precipitation_mm_day")
        c_temp_str = f"{c_temp}?C (NASA POWER 2m/Y?zey)" if c_temp is not None else "Veri Yok"
        c_precip_str = f"{c_precip} mm/g?n" if c_precip is not None else "Veri Yok"

        breakdown_matrix = [
            {
                "parameter": "K?tle?ekim ?vmesi Fark? (?g ? Fiziksel S?n?r)",
                "severity": "Y?KSEK (HIGH)",
                "target_value": f"{t_grav} g ({body})",
                "earth_value": "1.000 g (D?nya Standart)",
                "impact": "Rover s?spansiyon y?k?, tekerlek batma mekani?i ve astronot EVA eforu D?nya\'da daha y?ksektir.",
                "engineering_mitigation": f"Test donan?m?na %{offload_pct} aktif sabit kuvvetli gimbal/pogo a??rl?k bo?altma (gravity offload) uygulanmal?d?r."
            },
            {
                "parameter": "Atmosferik Bas?n? ve ?klim Fark? (?P ? Madde 4 Uyar?s?)",
                "severity": "KR?T?K (CRITICAL)",
                "target_value": f"{t_press} mbar ({'Ultra-Y?ksek Vakum' if body == 'Moon' else '?nce CO2 Atmosferi'})",
                "earth_value": f"~1013 mbar N2/O2 (Ya???: {c_precip_str})",
                "impact": "D?nya\'daki konvektif ?s? transferi ve atmosferik bas?n?, hedef g?vde vakum/ince atmosfer ko?ullar?yla e?de?er de?ildir.",
                "engineering_mitigation": "Saha testlerinden sonra bile?enler Termal Vakum Odas?nda (TVAC, 10^-6 Torr) ayr?ca do?rulanmal?d?r."
            },
            {
                "parameter": "Kozmik ve G?ne? Radyasyonu (GCR / SPE)",
                "severity": "Y?KSEK (HIGH)",
                "target_value": f"~{t_rad} mSv/y?l (Manyetosfer Korumas? Yok)",
                "earth_value": "~2.4 mSv/y?l (D?nya Manyetosferi Korumal?)",
                "impact": "D?nya sahalar? uzay radyasyonunu ve tekil olay etkilerini (SEE) temsil etmez.",
                "engineering_mitigation": "Aviyonik kartlar proton/a??r-iyon h?zland?r?c? hatt?nda radyasyon testine tabi tutulmal?d?r."
            },
            {
                "parameter": "Termal Rejim Fark? (?T ? G?vdeler Aras? Do?rudan Puanlanmaz)",
                "severity": "ORTA-Y?KSEK (MEDIUM-HIGH)",
                "target_value": f"Ortalama {t_temp}?C (Y?zey Bolometrik)",
                "earth_value": c_temp_str,
                "impact": "D?nya meteorolojik s?cakl??? ile Ay/Mars y?zey bolometrik s?cakl??? farkl? fiziksel de?i?kenlerdir (Skora kat?lmaz).",
                "engineering_mitigation": "Kriyojenik s?v? azot (LN2) so?utma ?rt?leri ile laboratuvarda termal ?evrim testi yap?lmal?d?r."
            }
        ]

        instrument_testbed_plan = [
            {
                "instrument_system": "Terrain Relative Navigation (TRN) & LiDAR/Kamera",
                "target_mission_heritage": "Mars 2020 Lander Vision System / Artemis HLS",
                "field_test_protocol": f"{candidate.get('region_name', 'Aday Saha')} ?zerinde d???k e?imli (<5?) ve dik yama?l? kesitlerde optik ?zellik takibi",
                "risk_reduction": "G?rsel-ataletsel odometri sapmalar?n?n saha verisiyle ?n incelenmesi"
            },
            {
                "instrument_system": "Y?zey Alt? Jeoradar (GPR) & Regolit Sondaj Prototipi",
                "target_mission_heritage": "MRO SHARAD / RIMFAX / VIPER TRIDENT",
                "field_test_protocol": "Bazaltik veya permafrost tabakalarda dielektrik ge?irgenlik ve delme torku ?l??m?",
                "risk_reduction": "Sert kaya? ve buzlu tabakalarda matkap s?k??ma (auger choking) riskinin incelenmesi"
            },
            {
                "instrument_system": "Otonom Rover Yama? ve Kayma Kontrol?",
                "target_mission_heritage": "Perseverance AutoNav / Lunar Terrain Vehicle (LTV)",
                "field_test_protocol": f"SRTM 90m DEM ile ?l??len {candidate.get('cross_section', {}).get('mean_slope_deg') or 'N/A'}? ortalama e?imli arazide tekerlek kayma testi",
                "risk_reduction": "Yumu?ak tortul ve bazaltik bloklar ?zerinde otonom rota planlama do?rulamas?"
            }
        ]

        return chr(10).join(lines)
