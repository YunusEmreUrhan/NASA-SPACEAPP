from datetime import datetime, timezone


class PlanetaryXAIExplainer:
    """
    P0-P3 Bilimsel Denetim Uyumlu XAI A??klay?c? ve Rapor ?retici
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
        c_temp_str = f"{c_temp}°C (NASA POWER 2m/Yüzey)" if c_temp is not None else "Veri Yok"
        c_precip_str = f"{c_precip} mm/gün" if c_precip is not None else "Veri Yok"

        breakdown_matrix = [
            {
                "parameter": "Kütleçekim İvmesi Farkı (Δg — Fiziksel Sınır)",
                "severity": "YÜKSEK (HIGH)",
                "target_value": f"{t_grav} g ({body})",
                "earth_value": "1.000 g (Dünya Standart)",
                "impact": "Rover süspansiyon yükü, tekerlek batma mekaniği ve astronot EVA eforu Dünya'da daha yüksektir.",
                "engineering_mitigation": f"Test donanımına %{offload_pct} aktif sabit kuvvetli gimbal/pogo ağırlık boşaltma (gravity offload) uygulanmalıdır."
            },
            {
                "parameter": "Atmosferik Basınç ve İklim Farkı (ΔP — P0 Madde 4)",
                "severity": "KRİTİK (CRITICAL)",
                "target_value": f"{t_press} mbar ({'Ultra-Yüksek Vakum' if body == 'Moon' else 'İnce CO2 Atmosferi'})",
                "earth_value": f"~1013 mbar N2/O2 (Yağış: {c_precip_str})",
                "impact": "Dünya'daki konvektif ısı transferi ve atmosferik basınç, hedef gövde koşullarıyla eşdeğer değildir.",
                "engineering_mitigation": "Saha testlerinden sonra bileşenler Termal Vakum Odasında (TVAC, 10^-6 Torr) ayrıca doğrulanmalıdır."
            },
            {
                "parameter": "Kozmik ve Güneş Radyasyonu (GCR / SPE)",
                "severity": "YÜKSEK (HIGH)",
                "target_value": f"~{t_rad} mSv/yıl (Manyetosfer Koruması Yok)",
                "earth_value": "~2.4 mSv/yıl (Dünya Manyetosferi Korumalı)",
                "impact": "Dünya sahaları uzay radyasyonunu ve tekil olay etkilerini (SEE) temsil etmez.",
                "engineering_mitigation": "Aviyonik kartlar proton/ağır-iyon hızlandırıcı hattında radyasyon testine tabi tutulmalıdır."
            },
            {
                "parameter": "Termal Rejim Farkı (ΔT — Gövdeler Arası Puanlanmaz)",
                "severity": "ORTA-YÜKSEK (MEDIUM-HIGH)",
                "target_value": f"Ortalama {t_temp}°C (Yüzey Bolometrik)",
                "earth_value": c_temp_str,
                "impact": "Dünya meteorolojik sıcaklığı ile Ay/Mars yüzey bolometrik sıcaklığı farklı fiziksel değişkenlerdir (Skora katılmaz).",
                "engineering_mitigation": "Kriyojenik sıvı azot (LN2) soğutma örtüleri ile laboratuvarda termal çevrim testi yapılmalıdır."
            }
        ]

        slope_val = candidate.get("cross_section", {}).get("mean_slope_deg")
        slope_str = f"{slope_val}°" if slope_val is not None else "N/A"

        instrument_testbed_plan = [
            {
                "instrument_system": "Terrain Relative Navigation (TRN) & LiDAR/Kamera",
                "target_mission_heritage": "Mars 2020 Lander Vision System / Artemis HLS",
                "field_test_protocol": f"{candidate.get('region_name', 'Aday Saha')} üzerinde düşük eğimli (<5°) ve dik yamaçlı kesitlerde optik özellik takibi",
                "risk_reduction": "Görsel-ataletsel odometri sapmalarının saha verisiyle ön incelenmesi"
            },
            {
                "instrument_system": "Yüzey Altı Jeoradar (GPR) & Regolit Sondaj Prototipi",
                "target_mission_heritage": "MRO SHARAD / RIMFAX / VIPER TRIDENT",
                "field_test_protocol": "Bazaltik veya permafrost tabakalarda dielektrik geçirgenlik ve delme torku ölçümü",
                "risk_reduction": "Sert kayaç ve buzlu tabakalarda matkap sıkışma (auger choking) riskinin incelenmesi"
            },
            {
                "instrument_system": "Otonom Rover Yamaç ve Kayma Kontrolü",
                "target_mission_heritage": "Perseverance AutoNav / Lunar Terrain Vehicle (LTV)",
                "field_test_protocol": f"SRTM 90m DEM ile ölçülen {slope_str} ortalama eğimli arazide tekerlek kayma testi",
                "risk_reduction": "Yumuşak tortul ve bazaltik bloklar üzerinde otonom rota planlama doğrulaması"
            }
        ]

        return {
            "candidate_region": candidate.get("region_name", ""),
            "screening_disclaimer": "Bu sonuç yalnızca araştırma önceliği / ön tarama (preliminary screening) amaçlıdır; uçuş veya üs güvenliği kanıtı değildir.",
            "breakdown_matrix": breakdown_matrix,
            "instrument_testbed_plan": instrument_testbed_plan,
            "complementary_split_site": complementary_site,
            "moon_mars_bridge": moon_mars_bridge
        }

    @staticmethod
    def build_phased_campaign_roadmap(candidates: list, comp_site: dict = None) -> list:
        c1 = candidates[0]["region_name"] if len(candidates) > 0 else "Aday #1"
        c2 = candidates[1]["region_name"] if len(candidates) > 1 else c1
        comp = comp_site["complementary_region_name"] if comp_site else c2
        return [
            {
                "phase": "Faz 1 (TRL 4 -> 5): Masaüstü & DEM Ön Tarama Doğrulaması",
                "assigned_sites": c1,
                "objective": "SRTM 90m DEM kesitleri ile LOLA/MOLA eğim ve rölyef dağılımlarının istatistiksel karşılaştırması."
            },
            {
                "phase": "Faz 2 (TRL 5 -> 6): Birincil Sahada Sensör ve Mobilite Saha Testi",
                "assigned_sites": f"{c1} & {c2}",
                "objective": "Optik navigasyon, jeoradar (GPR) ve rover tekerlek-zemin etkileşiminin arazi koşullarında incelenmesi."
            },
            {
                "phase": "Faz 3 (TRL 6 -> 7): İkili Saha (Split-Site) ve TVAC Laboratuvar Birleştirmesi",
                "assigned_sites": f"{c1} + {comp}",
                "objective": "Eksik çevresel boyutun tamamlayıcı sahada test edilmesi ve vakum/radyasyon farklarının TVAC odasında giderilmesi."
            }
        ]

    def export_markdown_dossier(self, search_res: dict) -> str:
        target = search_res.get("target", {})
        cands = search_res.get("primary_candidates", [])
        sens = search_res.get("weight_sensitivity_analysis", {})
        comp = search_res.get("complementary_split_site", {})
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines = [
            "# TERRA-TWIN: BİLİMSEL ÖN TARAMA VE KANIT İZLENEBİLİRLİK RAPORU",
            f"**Oluşturulma Zamanı:** `{now_str}` | **Yöntem Sürümü:** `{search_res.get('method_version', 'TERRA-TWIN-v7.0')}`",
            "**Kapsam Notu (P0 Madde 9):** Bu rapor yalnızca **araştırma önceliği ve ön tarama (preliminary screening)** amaçlıdır; iniş veya kalıcı üs güvenliği kanıtı değildir.",
            "",
            "---",
            f"## 1. Hedef Gövde Gözlem Kaydı: {target.get('name')} ({target.get('body')})",
            f"- **Koordinat ve Ayak İzi:** `{target.get('lat')}°, {target.get('lon')}°` | `{target.get('footprint_km', 10.0)} km` çaplı inceleme penceresi",
            f"- **Veri Durum Etiketi:** `{target.get('data_status', 'NASA verisinden işlendi')}`",
            f"- **Birincil Jeolojik Süreç:** {target.get('primary_process', 'N/A')}",
            ""
        ]

        obs = target.get("observations", {})
        if obs:
            lines.append("| Gözlem Değişkeni | Değer (±Belirsizlik) | Birim | Durum Etiketi | Veri Ürünü & Dönem | Kalite Bayrağı | Kaynak URL |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for k, v in obs.items():
                val_s = f"{v.get('value')} (±{v.get('uncertainty')})" if v.get("value") is not None else "Veri Yok"
                lines.append(
                    f"| `{k}` | **{val_s}** | `{v.get('unit')}` | `{v.get('status_label')}` | "
                    f"{v.get('product')} ({v.get('epoch')}) | `{v.get('quality_flag')}` | {v.get('source_url')} |"
                )
            lines.append("")

        lines.append("---")
        lines.append("## 2. Ön Tarama Sonucu: İlk 3 Dünya Aday Sahası ve Eşli Fiziksel Kanıtlar")
        for idx, c in enumerate(cands, 1):
            phys_s = f"%{c['measured_physical_similarity_pct']} (±%{c.get('score_uncertainty_pct')})" if c.get("measured_physical_similarity_pct") is not None else "Veri Yok (Eşli DEM Eksik)"
            lines.append(f"### #{idx} {c.get('region_name')} (`{c.get('lat')}°, {c.get('lon')}°`)")
            lines.append(
                f"- **Ölçülmüş Fiziksel Benzerlik (Eğim/Rölyef/Ölçek):** **{phys_s}** | "
                f"**Eşli Veri Kapsamı (Data Coverage):** **`%{c.get('data_coverage_pct')}`**"
            )
            lines.append(f"- **6B Senaryo Ön Tarama İndeksi:** `%{c.get('parametric_6d_pct')}` (`senaryo varsayımı`)")
            lines.append(f"- **Görsel Keşif Sinyali (Fiziksel Skora Katılmaz):** `%{c.get('visual_hybrid_pct')}` — *{c.get('vision_engine_used')}*")
            lines.append("")
            lines.append("| Fiziksel Ölçüt | Dünya Ölçümü (±Hata) | Hedef Ölçümü (±Hata) | SI Birimi | Benzerlik | Ağırlık | Durum Etiketi | Kaynak |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for r in c.get("physical_evidence_table", []):
                sim_str = f"%{r['similarity_pct']}" if r.get("similarity_pct") is not None else "Puanlanmaz / Veri Yok"
                lines.append(
                    f"| {r['criterion']} | {r['earth_measured']} | {r['target_measured']} | `{r['unit']}` | "
                    f"**{sim_str}** | `{r['weight']}` | `{r.get('status_label', 'N/A')}` | {r['source']} |"
                )
            lines.append("")

        if sens and sens.get("candidate_sensitivity"):
            lines.append("---")
            lines.append("## 3. Ağırlık Duyarlılık Analizi (Weight Sensitivity Analysis — Madde 7)")
            lines.append(f"- **Yöntem:** {sens.get('perturbation_range')} | **#1 Aday Sıralama Kararlılığı:** `%{sens.get('rank_1_stability_pct')}`")
            lines.append("")
            lines.append("| Aday Saha | Ortalama Skor | Min Skor | Maks Skor | Değişim Aralığı (Spread) | #1 Kalma Oranı |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for s_item in sens["candidate_sensitivity"]:
                lines.append(
                    f"| {s_item['region_name']} | %{s_item['mean_perturbed_score_pct']} | "
                    f"%{s_item['min_perturbed_score_pct']} | %{s_item['max_perturbed_score_pct']} | "
                    f"±%{s_item['score_spread_pct']} | **%{s_item['top1_retention_pct']}** |"
                )
            lines.append("")

        if comp:
            lines.append("---")
            lines.append("## 4. Tamamlayıcı İkili Saha (Split-Site) Önerisi (`senaryo varsayımı`)")
            lines.append(
                f"- Birincil sahanın en düşük uyumlu boyutu (`{comp.get('weakest_dimension_name')}`) için "
                f"önerilen tamamlayıcı saha: **{comp.get('complementary_region_name')}** (`{comp.get('lat')}°, {comp.get('lon')}°`)."
            )

        return chr(10).join(lines)
