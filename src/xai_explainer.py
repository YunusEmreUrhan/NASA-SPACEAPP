import os
import sys
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from matcher_engine import DIMENSION_NAMES


class PlanetaryXAIExplainer:
    def __init__(self):
        self.earth_gravity_g = 1.000
        self.earth_std_pressure_mbar = 1013.25

    def build_phased_campaign_roadmap(self, candidates: list, comp_site: dict) -> list:
        """
        Kesfedilen Ilk 3 Aday ve Tamamlayici Sahayi birlestirerek
        NASA TRL 4 -> TRL 8 Asamali Saha Test Kampanyasi Takvimi uretir.
        """
        c1 = candidates[0]
        c2 = candidates[1] if len(candidates) > 1 else c1
        c3 = candidates[2] if len(candidates) > 2 else c1
        comp_name = comp_site["complementary_region_name"] if comp_site else c3["region_name"]

        return [
            {
                "phase": "FAZ A (TRL 4-5): Sensör, Spektrometre & Astrobiyoloji Kalibrasyonu",
                "assigned_sites": f"#2 {c2['region_name']} & #3 {c3['region_name']}",
                "objective": (
                    f"SHERLOC/PIXL biyo-imza spektrometrelerinin '{c2['astrobiology_profile']['extremophile_guild']}' "
                    f"üzerinde kalibrasyonu ve '{c2['process_analogy']['dominant_earth_process']}' sürecinde "
                    "optik kamera (TRN) algoritmalarının ön doğrulaması."
                )
            },
            {
                "phase": "FAZ B (TRL 6-7): Tam Ölçekli Rover Mobilite, Sondaj (ISRU) & Mikro-Hedef İniş Testi",
                "assigned_sites": f"#1 {c1['region_name']} -> Mikro-Hedef ({c1['micro_scale_pinpoint']['micro_lat']}°, {c1['micro_scale_pinpoint']['micro_lon']}°)",
                "objective": (
                    f"{c1['micro_scale_pinpoint']['patch_scale_km']} ölçeğindeki '{c1['micro_scale_pinpoint']['micro_feature_type']}' "
                    f"noktasında otonom rover sürüşü (Mobilite: %{c1['operations_readiness']['rover_mobility_trafficability_pct']}), "
                    f"ISRU kaynak çıkarma ve <5° güvenli üs kurulum ({c1['cross_section']['best_base_km']}. km) tatbikatı."
                )
            },
            {
                "phase": "FAZ C (TRL 8): İkili Saha (Split-Site) Eksik Tamamlama & TVAC Vakum Sertifikasyonu",
                "assigned_sites": f"Tamamlayıcı Saha: {comp_name} + NASA TVAC Vakum Odası",
                "objective": (
                    f"Birincil sahada zayıf kalan '{comp_site['weakest_dimension_name'] if comp_site else 'Termal'}' "
                    f"boyutunun (%{comp_site['complementary_dim_match_pct'] if comp_site else 85.0} uyum) paralel doğrulanması "
                    "ve uçuş öncesi radyasyon/vakum odası sertifikasyonu."
                )
            }
        ]

    def build_instrument_testbed_plan(self, target: dict, candidate: dict) -> list:
        body = target.get("body", "Moon")
        micro = candidate.get("micro_scale_pinpoint", {})
        astro = candidate.get("astrobiology_profile", {})
        ops = candidate.get("operations_readiness", {})
        vec = candidate.get("features_6d", [0.7] * 6)

        tests = [
            {
                "instrument_system": "Terrain Relative Navigation (TRN) & LiDAR Hazard Avoidance",
                "target_mission_heritage": "Artemis HLS / Mars 2020 Lander Vision System",
                "field_test_protocol": (
                    f"{micro.get('micro_lat', candidate['lat'])}°, {micro.get('micro_lon', candidate['lon'])}° "
                    f"alt-koordinatındaki '{micro.get('micro_feature_type', 'Mikro-Havza')}' üzerinde "
                    f"optik navigasyon kameralarının (TRN Uyumu: %{ops.get('trn_optical_navigation_pct', 85.0)}) "
                    f"düşük güneş açısı ve gölge kontrastında kalibrasyonu."
                ),
                "risk_reduction": "İniş anında krater kenarı/kayaç çarpışma riskini (Landing Slope Hazard) %80+ azaltır."
            }
        ]

        if vec[5] >= 0.70:
            tests.append({
                "instrument_system": "CADRE Otonom Mağara Mikro-Rover Sürüsü & Yeraltı GPR Radarı",
                "target_mission_heritage": "NASA CADRE / Moon Diver / Martian Skylight Robotics",
                "field_test_protocol": (
                    "Bazaltik çökme çukurları ve kanyon duvarlarında tether (kablo) destekli iniş, "
                    "GPS-bağımsız SLAM haritalama ve tavan stabilitesi radar testleri."
                ),
                "risk_reduction": "Lav tüpü girişlerinde (Skylight) rover devrilmesi ve iletişim kopması riskini bertaraf eder."
            })

        if vec[3] >= 0.55:
            tests.append({
                "instrument_system": "VIPER TRIDENT Kriyojenik Darbeli Matkap & MSolo Kütle Spektrometresi",
                "target_mission_heritage": "NASA Artemis VIPER / Mars Ice Mapper",
                "field_test_protocol": (
                    "Donmuş permafrost ve buz-regolit karışımlarında 1 metre derinlikte uçucu madde (H2O/OH) "
                    "kaybı olmadan karot alma ve su ekstraksiyon (ISRU) doğrulaması."
                ),
                "risk_reduction": "Kriyojenik sert zeminde matkap sıkışması (bit binding) ve süblimleşme kaybı riskini önler."
            })
        else:
            tests.append({
                "instrument_system": "Molten Regolith Electrolysis (MRE) & Mikrodalga 3B Sinterleme Fırını",
                "target_mission_heritage": "NASA ISRU Oxygen Extraction & MMPACT Construction",
                "field_test_protocol": (
                    f"Sahadaki mafik/bazaltik regolit örnekleriyle (İnşaat Uyumu: %{ops.get('regolith_construction_3d_pct', 82.0)}) "
                    "oksijen üretimi ve radyasyon zırhı tuğlası sinterleme dayanım testleri."
                ),
                "risk_reduction": "Yüzey üssü inşasında yerel malzeme (ISRU) çatlama ve termal şok kırılma riskini azaltır."
            })

        tests.append({
            "instrument_system": "SHERLOC Deep-UV Raman Spektrometresi & PIXL X-Işını Litokimya",
            "target_mission_heritage": "Mars Perseverance / ExoMars Rosalind Franklin",
            "field_test_protocol": (
                f"'{astro.get('extremophile_guild', 'Endolitik Mikrobiyota')}' içeren kayaç ve tuz/mineral "
                f"yüzeylerinde (Biyo-İmza Korunumu: %{astro.get('biosignature_preservation_pct', 84.0)}) "
                "organik karbon ve biyo-mineral tespit eşiğinin doğrulanması."
            ),
            "risk_reduction": f"{body} numunelerinde yanlış-pozitif (false-positive) biyo-imza okuma riskini minimize eder."
        })

        return tests

    def generate_candidate_explanation(
        self,
        target: dict,
        candidate: dict,
        complementary_site: dict = None,
        moon_mars_bridge: dict = None
    ) -> dict:
        body = target.get("body", "Moon")
        t_grav = float(target.get("gravity_g", 0.165 if body == "Moon" else 0.378))
        t_press = float(target.get("pressure_mbar", 0.0 if body == "Moon" else 6.1))
        t_temp = float(target.get("mean_temp_c", -50.0))
        t_rad = float(target.get("radiation_msv_yr", 400.0 if body == "Moon" else 220.0))

        climate = candidate.get("nasa_power_climate", {})
        c_temp = float(climate.get("mean_surface_temp_c", 18.0))
        c_precip = float(climate.get("precipitation_mm_day", 0.2))
        c_rh = float(climate.get("relative_humidity_pct", 30.0))
        c_solar = float(climate.get("solar_irradiance_kwh_m2_day", 5.8))

        dim_scores = candidate.get("dimension_matches_pct", [80.0] * 6)
        ranked_dims = sorted(
            [(DIMENSION_NAMES[i], dim_scores[i], i) for i in range(6)],
            key=lambda x: x[1],
            reverse=True
        )
        strongest_dims = ranked_dims[:3]
        weakest_dims = ranked_dims[-2:]

        cs = candidate.get("cross_section", {})
        proc = candidate.get("process_analogy", {})
        micro = candidate.get("micro_scale_pinpoint", {})
        astro = candidate.get("astrobiology_profile", {})

        strengths = [
            f"Çok Boyutlu Jeofiziksel & Süreç Uyumu (%{candidate['total_match_pct']}): "
            f"En güçlü örtüşme %{strongest_dims[0][1]} ile '{strongest_dims[0][0]}' boyutundadır. "
            f"Sahadaki '{proc.get('dominant_earth_process', '')}' süreci, hedefin '{proc.get('target_planetary_process', '')}' "
            f"süreciyle %{proc.get('process_fidelity_pct', 80.0)} oranında örtüşmektedir.",
            f"Çok Ölçekli (Multi-Scale) Nokta Atışı & Topografya (%{candidate['topographic_match_pct']}): "
            f"Geniş bölge içinde {micro.get('patch_scale_km', '5x5 km')} ölçeğindeki '{micro.get('micro_feature_type', '')}' "
            f"({micro.get('micro_lat', candidate['lat'])}°, {micro.get('micro_lon', candidate['lon'])}°) nokta atışı tespit edilmiştir; "
            f"kesitin %{cs.get('safe_landing_pct', 70.0)}'ı <5° güvenli iniş zarfındadır.",
            f"Canlı NASA POWER İklim & Astrobiyoloji Doğrulaması (%{candidate['live_env_verification_pct']}): "
            f"Yağış {c_precip} mm/gün, güneş/UV akısı {c_solar} kWh/m²/gün seviyesindedir. "
            f"Bu ekstrem ortam '{astro.get('extremophile_guild', '')}' üzerinden %{astro.get('biosignature_preservation_pct', 85.0)} "
            f"biyo-imza koruma sadakati sunar."
        ]

        offload_pct = round((1.0 - (t_grav / self.earth_gravity_g)) * 100.0, 1)
        temp_delta = round(abs(c_temp - t_temp), 1)

        breakdowns = [
            {
                "parameter": "Kütleçekim İvmesi (Reduced Gravity Limit)",
                "target_value": f"{t_grav} g ({body})",
                "earth_value": f"{self.earth_gravity_g:.3f} g (Dünya)",
                "severity": "YÜKSEK (Fiziksel Kırılma)",
                "impact": (
                    f"Dünya yerçekimi {body} yüzeyinden {round(self.earth_gravity_g / t_grav, 1)} kat büyüktür; "
                    f"rover tekerlek batmasını (sinkage), şev stabilitesini ve EVA yorgunluğunu değiştirir."
                ),
                "engineering_mitigation": (
                    f"Rover ve EVA ekipmanlarına %{offload_pct} Aktif Kütleçekim Boşaltma "
                    f"(Counter-Weight Gimbal / Pogo Offload Rig) uygulanmalıdır."
                )
            },
            {
                "parameter": "Atmosfer Basıncı ve Vakum (Atmospheric Limit)",
                "target_value": f"{t_press} mbar ({'Mutlak Vakum' if t_press == 0 else 'İnce CO2 Atmosferi'})",
                "earth_value": "~850 - 1013 mbar (N2/O2 Atmosfer)",
                "severity": "KRİTİK (Ortam Kırılması)",
                "impact": (
                    f"Dünya atmosferi konvektif ısı transferi yaratır ve elektrostatik aşındırıcı tozların "
                    f"(Lunar/Martian dust adhesion) vakumdaki yapışma davranışını nem (%{c_rh}) nedeniyle perdeler."
                ),
                "engineering_mitigation": (
                    f"Arazi testinden geçen mekanizmalar {t_press} mbar basınçlı Termal Vakum Odasında (TVAC) "
                    f"triboelektrik toz yüklemesi altında doğrulanmalıdır."
                )
            },
            {
                "parameter": "Kozmik Radyasyon (GCR/SPE) & Güneş UV Akısı",
                "target_value": f"~{t_rad} mSv/yıl (Manyetosfersiz Yüzey)",
                "earth_value": f"~2.4 mSv/yıl | Solar Akı: {c_solar} kWh/m²/gün",
                "severity": "KRİTİK (Radyasyon Kırılması)",
                "impact": (
                    "Dünya'nın manyetosferi ve kalın atmosferi galaktik kozmik ışınları (GCR) ve güneş proton "
                    "olaylarını (SPE) engeller; elektronik SEU hataları ve biyo-bozunma sahada doğal oluşmaz."
                ),
                "engineering_mitigation": (
                    "Elektronik aviyonikler proton hızlandırıcı (Radiation Beamline) testine sokulmalı; "
                    "biyo-imza deneyleri ise UV-C (200-280 nm) lambaları altında yürütülmelidir."
                )
            },
            {
                "parameter": "Termal Ekstrem Farkı (ΔT) & Biyolojik Sterillik",
                "target_value": f"{t_temp}°C (Steril Regolit)",
                "earth_value": f"{c_temp}°C (Mikrobiyal Kontaminasyon Mevcut)",
                "severity": "ORTA-YÜKSEK",
                "impact": (
                    f"{temp_delta}°C sıcaklık farkı kriyojenik kırılganlığı gizler; ayrıca Dünya toprağındaki "
                    "modern organik kalıntılar yaşam arama (SHERLOC/PIXL) sensörlerinde arka plan gürültüsü yaratır."
                ),
                "engineering_mitigation": (
                    f"Aktüatörler Sıvı Azot (LN2) ceketleriyle {t_temp}°C'ye soğutulmalı; astrobiyoloji "
                    "kalibrasyonunda karbon-izotop arka plan çıkarması (Blank Subtraction Protocol) uygulanmalıdır."
                )
            },
            {
                "parameter": f"En Zayıf 6B Boyut Sapması: {weakest_dims[-1][0]}",
                "target_value": f"Hedef İndeks: {target['features_6d'][weakest_dims[-1][2]]:.2f}",
                "earth_value": f"Saha İndeksi: {candidate['features_6d'][weakest_dims[-1][2]]:.2f} (%{weakest_dims[-1][1]} Uyum)",
                "severity": "TELAFİ EDİLEBİLİR (İkili Saha ile)",
                "impact": (
                    f"Saha ({candidate['region_name']}), '{weakest_dims[-1][0]}' boyutunda "
                    f"(%{weakest_dims[-1][1]}) kısmi sapma göstermektedir."
                ),
                "engineering_mitigation": (
                    f"İkili Saha Protokolü (Split-Site Validation): '{weakest_dims[-1][0]}' alt-sistem testleri "
                    f"tamamlayıcı saha olan {complementary_site['complementary_region_name'] if complementary_site else 'ikincil sahada'} "
                    f"(%{complementary_site['complementary_dim_match_pct'] if complementary_site else 90.0} uyum) paralel yürütülmelidir."
                )
            }
        ]

        instrument_plan = self.build_instrument_testbed_plan(target, candidate)

        return {
            "candidate_region": candidate["region_name"],
            "coordinates": f"({candidate['lat']}°, {candidate['lon']}°)",
            "strategic_role_badge": candidate.get("strategic_role_badge", ""),
            "total_match_pct": candidate["total_match_pct"],
            "strengths": strengths,
            "strongest_dimensions": [{"name": d[0], "score_pct": d[1]} for d in strongest_dims],
            "weakest_dimensions": [{"name": d[0], "score_pct": d[1]} for d in weakest_dims],
            "breakdown_matrix": breakdowns,
            "instrument_testbed_plan": instrument_plan,
            "complementary_protocol": complementary_site,
            "moon_mars_bridge": moon_mars_bridge,
            "base_feasibility": target.get("base_feasibility", {})
        }

    def export_markdown_dossier(self, full_search_result: dict) -> str:
        target = full_search_result["target"]
        cands = full_search_result["primary_candidates"]
        comp = full_search_result.get("complementary_split_site", {})
        bridge = full_search_result.get("moon_mars_bridge", {})
        base_f = target.get("base_feasibility", {})
        best = cands[0]

        xai_best = self.generate_candidate_explanation(target, best, comp, bridge)
        roadmap = self.build_phased_campaign_roadmap(cands, comp)
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            "# NASA SPACE APPS 2026 — KALICI AY/MARS ÜSSÜ VE GEZEGENSEL ANALOG GÖREV DOSYASI",
            f"**Rapor Tarihi:** {now_str} | **Keşif Motoru:** {best.get('vision_engine_used', 'Hybrid AI')}",
            f"**Taranan Küresel Hücre:** {full_search_result.get('scanned_total_cells', 0):,} gerçek kara koordinatı",
            "",
            "---",
            f"## 1. HEDEF GEZEGEN VE KALICI ÜS FİZİBİLİTESİ: {target['name']}",
            f"- **Gök Cismi:** {target['body']} | **Koordinatlar:** `{target['lat']}°, {target['lon']}°`",
            f"- **Kalıcı Üs Fizibilite Skoru:** **%{base_f.get('overall_base_viability_pct', 85.0)}** ({base_f.get('recommended_habitat_architecture', '')})",
            f"- **Radyasyon Zırhı & ISRU:** {base_f.get('radiation_shielding_protocol', '')} | {base_f.get('isru_primary_strategy', '')}",
            "",
            "---",
            "## 2. KEŞFEDİLEN İLK 3 DÜNYA ANALOGUNUN DETAYLI KARŞILAŞTIRMALI PROFİLİ"
        ]

        for idx, c in enumerate(cands, 1):
            mc = c["micro_scale_pinpoint"]
            pr = c["process_analogy"]
            ab = c["astrobiology_profile"]
            op = c["operations_readiness"]
            cl = c["nasa_power_climate"]
            lines.extend([
                f"### #{idx} ADAY: {c['region_name']} (`{c['lat']}°, {c['lon']}°`) — Toplam Uyum: **%{c['total_match_pct']}**",
                f"- **Stratejik Rol:** {c.get('strategic_role_badge', '')}",
                f"- **Alt-Metrikler:** `[Ön Eleme #{c['pre_screen_rank']} -> Final #{idx}]` | 6B Vektör: `%{c['parametric_6d_pct']}` | Görsel AI: `%{c['visual_hybrid_pct']}` | Kesit: `%{c['topographic_match_pct']}` | NASA POWER: `%{c['live_env_verification_pct']}`",
                f"- **Canlı NASA POWER İklimi:** Sıcaklık `{cl['mean_surface_temp_c']}°C` | Yağış `{cl['precipitation_mm_day']} mm/gün` | Nem `%{cl['relative_humidity_pct']}` | Rüzgar `{cl['wind_speed_10m_ms']} m/s`",
                f"- **Mikro-Ölçek (Consider Scale) Nokta Atışı:** `{mc['micro_feature_type']}` (`{mc['micro_lat']}°, {mc['micro_lon']}°` | `{mc['patch_scale_km']}` | Uyum: `%{mc['micro_suitability_pct']}`)",
                f"- **Aktif Jeofiziksel Süreç:** `{pr['dominant_earth_process']}` (**%{pr['process_fidelity_pct']} Süreç Uyumu**)",
                f"- **Astrobiyoloji & Biyo-İmza:** `{ab['extremophile_guild']}` (**%{ab['biosignature_preservation_pct']} Korunum**)",
                f"- **Operasyonel Hazırlık:** İniş `<5°: %{op['landing_safety_envelope_pct']}` | Mobilite: `%{op['rover_mobility_trafficability_pct']}` | TRN: `%{op['trn_optical_navigation_pct']}` | 3B İnşaat: `%{op['regolith_construction_3d_pct']}`",
                ""
            ])

        lines.extend([
            "---",
            "## 3. TRL 4 -> TRL 8 ENTEGRE ÇOKLU SAHA TEST KAMPANYASI YOL HARİTASI",
            "| Kampanya Fazı | Atanan Dünya Sahaları | Mühendislik ve Bilimsel Hedef |",
            "| :--- | :--- | :--- |"
        ])
        for rm in roadmap:
            lines.append(f"| **{rm['phase']}** | {rm['assigned_sites']} | {rm['objective']} |")

        lines.extend([
            "",
            "---",
            "## 4. BENZERLİĞİN KIRILDIĞI NOKTALAR (EARTH'S LIMITS) VE MÜHENDİSLİK TELAFİ REÇETESİ",
            "| Fiziksel Parametre | Hedef Değer | Dünya Sahası | Kırılma Şiddeti | Mühendislik Telafi Protokolü |",
            "| :--- | :--- | :--- | :--- | :--- |"
        ])
        for b in xai_best["breakdown_matrix"]:
            lines.append(
                f"| **{b['parameter']}** | {b['target_value']} | {b['earth_value']} | {b['severity']} | {b['engineering_mitigation']} |"
            )

        return "\n".join(lines)
