import os, sys, subprocess
try:
    import streamlit
    import plotly
    import folium
    import streamlit_folium
    import cv2
    import reverse_geocoder
except ImportError:
    print("[KURULUM] Eksik paketler otomatik kuruluyor, l?tfen bekleyin...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")])

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from matcher_engine import PlanetaryMatcherEngine, DIMENSION_NAMES, MISSION_WEIGHT_PRESETS
from xai_explainer import PlanetaryXAIExplainer

st.set_page_config(
    page_title="TERRA-TWIN | NASA 2026 Planetary Base & Analog Engine",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NASA Görev Kontrol Merkezi (Dark Telemetry UI) CSS Stili
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.0rem; padding-bottom: 1.5rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(16,24,42,0.96), rgba(9,14,26,0.96));
        border: 1px solid rgba(0, 212, 255, 0.35);
        padding: 12px 15px;
        border-radius: 10px;
    }
    .mission-banner {
        background: linear-gradient(90deg, #091124 0%, #162447 55%, #0f3460 100%);
        border-left: 5px solid #00d4ff;
        padding: 15px 22px;
        border-radius: 8px;
        margin-bottom: 14px;
    }
    .base-card {
        background: linear-gradient(135deg, rgba(15,32,60,0.85), rgba(12,22,40,0.90));
        border: 1px solid rgba(0, 245, 212, 0.40);
        border-left: 5px solid #00f5d4;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 14px;
    }
    .cand-card {
        background-color: rgba(18, 27, 46, 0.85);
        border: 1px solid rgba(255, 190, 11, 0.35);
        border-top: 4px solid #ffbe0b;
        padding: 14px;
        border-radius: 8px;
        height: 100%;
    }
    .xai-card {
        background-color: rgba(24, 30, 48, 0.80);
        border-left: 4px solid #ff595e;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .split-card {
        background-color: rgba(14, 42, 40, 0.80);
        border-left: 4px solid #00f5d4;
        padding: 14px 18px;
        border-radius: 6px;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource(show_spinner="NASA 2026 Küresel 6B Keşif Matrisi ve CLIP/CV Motoru Yükleniyor...")
def load_engines():
    matcher = PlanetaryMatcherEngine(clip_mode="auto")
    explainer = PlanetaryXAIExplainer()
    return matcher, explainer


def build_radar_chart(target_name: str, target_vec: list, candidates: list, comp_site: dict = None):
    categories = DIMENSION_NAMES + [DIMENSION_NAMES[0]]
    t_vals = [round(v * 100, 1) for v in target_vec] + [round(target_vec[0] * 100, 1)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=t_vals,
        theta=categories,
        fill="toself",
        name=f"HEDEF: {target_name[:28]}",
        line=dict(color="#00d4ff", width=3),
        fillcolor="rgba(0, 212, 255, 0.18)"
    ))

    colors = ["#ff9f1c", "#ff006e", "#a8dadc"]
    for idx, cand in enumerate(candidates[:3]):
        c_vals = [round(v * 100, 1) for v in cand["features_6d"]] + [round(cand["features_6d"][0] * 100, 1)]
        fig.add_trace(go.Scatterpolar(
            r=c_vals,
            theta=categories,
            fill="toself" if idx == 0 else "none",
            name=f"#{idx+1}: {cand['region_name'].split('[')[0].strip()[:24]}",
            line=dict(color=colors[idx], width=2.5 if idx == 0 else 1.8),
            fillcolor="rgba(255, 159, 28, 0.16)" if idx == 0 else None
        ))

    if comp_site and "features_6d" in comp_site:
        comp_vals = [round(v * 100, 1) for v in comp_site["features_6d"]] + [
            round(comp_site["features_6d"][0] * 100, 1)
        ]
        fig.add_trace(go.Scatterpolar(
            r=comp_vals,
            theta=categories,
            name=f"İkili Saha: {comp_site['complementary_region_name'].split('[')[0].strip()[:22]}",
            line=dict(color="#00f5d4", width=2, dash="dot")
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(12,18,30,0.85)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.15)")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e6ed", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=-0.30, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=25, b=65),
        height=420
    )
    return fig


def build_cross_section_chart(target: dict, cand: dict):
    t_cs = target["cross_section"]
    e_cs = cand["cross_section"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_cs["distances_km"],
        y=t_cs["elevation_m"],
        mode="lines",
        name=f"{target['body']} Hedef Kesiti (Ort. Eğim: {t_cs['mean_slope_deg']}°)",
        line=dict(color="#00d4ff", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=e_cs["distances_km"],
        y=e_cs["elevation_m"],
        mode="lines",
        name=f"Dünya Analogu Kesiti (Ort. Eğim: {e_cs['mean_slope_deg']}° | <5° Güvenli: %{e_cs['safe_landing_pct']})",
        line=dict(color="#ff9f1c", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(255, 159, 28, 0.12)"
    ))

    best_km = e_cs["best_base_km"]
    idx = int(np.argmin(np.abs(np.array(e_cs["distances_km"]) - best_km)))
    best_elev = float(e_cs["elevation_m"][idx])

    fig.add_trace(go.Scatter(
        x=[best_km],
        y=[best_elev],
        mode="markers+text",
        name="Optimal Üs / İniş Noktası (<5° Eğim)",
        text=[f"Üs Noktası ({best_km} km)"],
        textposition="top center",
        marker=dict(size=13, color="#00f5d4", symbol="star", line=dict(width=1.5, color="#ffffff"))
    ))

    fig.update_layout(
        title="2B Fotoklinometrik Topografya Kesiti (%30 Seçim Ağırlığı) & Kalıcı Üs Konumu",
        xaxis_title=f"Taktik Üs Kesit Mesafesi (0 - {target.get('radius_km', 5.0)*2:.0f} km)",
        yaxis_title="Bağıl Topografik Kabartma (m)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,18,30,0.85)",
        font=dict(color="#e0e6ed"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.30, xanchor="center", x=0.5),
        margin=dict(l=40, r=30, t=45, b=65),
        height=420
    )
    return fig


def build_global_discovery_map(candidates: list, comp_site: dict = None):
    fig = go.Figure()

    lats = [c["lat"] for c in candidates]
    lons = [c["lon"] for c in candidates]
    labels = [f"#{i+1} {c['region_name'].split('[')[0].strip()}" for i, c in enumerate(candidates)]
    hover_texts = [
        f"<b>#{i+1} {c['region_name']}</b><br>"
        f"Toplam Uyum: <b>%{c['total_match_pct']}</b> (Ön Eleme #{c['pre_screen_rank']} -> Final #{i+1})<br>"
        f"Rol: {c.get('strategic_role_badge', '')}<br>"
        f"Mikro-Hedef (1.25km): {c['micro_scale_pinpoint']['micro_lat']}°, {c['micro_scale_pinpoint']['micro_lon']}°<br>"
        f"NASA POWER: {c['nasa_power_climate']['mean_surface_temp_c']}°C | Yağış: {c['nasa_power_climate']['precipitation_mm_day']} mm/gün"
        for i, c in enumerate(candidates)
    ]

    fig.add_trace(go.Scattergeo(
        lat=lats,
        lon=lons,
        text=labels,
        hovertext=hover_texts,
        hoverinfo="text",
        mode="markers+text",
        textposition="top center",
        textfont=dict(color="#ffbe0b", size=11),
        name="Keşfedilen İlk 3 Dünya Analogu",
        marker=dict(
            size=[18, 14, 12][:len(candidates)],
            color=["#ff006e", "#ff9f1c", "#ffbe0b"][:len(candidates)],
            line=dict(width=2, color="#ffffff"),
            symbol="circle"
        )
    ))

    if comp_site:
        fig.add_trace(go.Scattergeo(
            lat=[candidates[0]["lat"], comp_site["lat"]],
            lon=[candidates[0]["lon"], comp_site["lon"]],
            mode="lines",
            name="İkili Saha (Split-Site) Köprüsü",
            line=dict(width=2.2, color="#00f5d4", dash="dash")
        ))
        fig.add_trace(go.Scattergeo(
            lat=[comp_site["lat"]],
            lon=[comp_site["lon"]],
            text=[f"İkili Saha: {comp_site['complementary_region_name'].split('[')[0].strip()}"],
            hovertext=[
                f"<b>Tamamlayıcı 2. Saha: {comp_site['complementary_region_name']}</b><br>"
                f"Tamamlanan Boyut: {comp_site['weakest_dimension_name']} (%{comp_site['complementary_dim_match_pct']} Uyum)"
            ],
            hoverinfo="text",
            mode="markers+text",
            textposition="bottom center",
            textfont=dict(color="#00f5d4", size=11),
            name="Tamamlayıcı 2. Saha (Split-Site)",
            marker=dict(size=15, color="#00f5d4", symbol="diamond", line=dict(width=2, color="#ffffff"))
        ))

    fig.update_layout(
        geo=dict(
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(22, 30, 44)",
            showocean=True,
            oceancolor="rgb(9, 14, 24)",
            showcountries=True,
            countrycolor="rgba(255, 255, 255, 0.22)",
            coastlinecolor="rgba(0, 212, 255, 0.35)",
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e6ed"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=10, b=10),
        height=390
    )
    return fig


def build_process_and_ops_chart(cand: dict):
    proc_brk = cand["process_analogy"]["process_breakdown"]
    ops = cand["operations_readiness"]

    labels = list(proc_brk.keys()) + [
        "İniş Güvenliği (<5° Zarfı)",
        "Rover Mobilite & Trafik",
        "TRN Optik Navigasyon",
        "3B Regolit Sinterleme İnşaat"
    ]
    values = list(proc_brk.values()) + [
        ops["landing_safety_envelope_pct"],
        ops["rover_mobility_trafficability_pct"],
        ops["trn_optical_navigation_pct"],
        ops["regolith_construction_3d_pct"]
    ]
    colors = ["#ff9f1c"] * 4 + ["#00f5d4"] * 4

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colors),
        text=[f"%{v}" for v in values],
        textposition="auto"
    ))
    fig.update_layout(
        title=f"Jeofiziksel Süreç Yoğunluğu (Turuncu) & Operasyonel Hazırlık (Turkuaz) — {cand['region_name'].split('[')[0]}",
        xaxis=dict(range=[0, 105], title="Uyum / Hazırlık Skoru (%)", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,18,30,0.85)",
        font=dict(color="#e0e6ed", size=11),
        margin=dict(l=20, r=20, t=40, b=30),
        height=360
    )
    return fig


def main():
    st.markdown(
        """
        <div class="mission-banner">
            <h2 style="margin:0; color:#ffffff;">🛰️ TERRA-TWIN: NASA 2026 Permanent Moon/Mars Base & Analog Discovery Engine</h2>
            <p style="margin:4px 0 0 0; color:#a8dadc; font-size:0.94rem;">
                <b>2-Kademeli Hibrit Mimari:</b> 11,469 Küresel Hücre Ön Elemesi ➔ 8 Adayın Canlı Uydu Görüntüsü (CLIP + 64B CV),
                <b>%30 Ağırlıklı 2B Topografik Kesit</b> ve Canlı NASA POWER İklim Barajı ile Kafa Kafaya Yarışı & XAI Karar Destek Sistemi
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    engine, explainer = load_engines()

    if "dyn_lat" not in st.session_state:
        st.session_state.dyn_lat = -13.80
    if "dyn_lon" not in st.session_state:
        st.session_state.dyn_lon = -61.20

    # --- SOL MENÜ: GÖREV KONTROL PANELİ ---
    with st.sidebar:
        st.header("⚙️ Görev Kontrol Paneli")

        selection_mode = st.radio(
            "1. Gezegen Hedefi Seçim Modu:",
            options=[
                "🏛️ Resmi NASA Üs Kataloğu (6 Görev)",
                "🎯 Serbest Koordinat & Harita Tıklama (Canlı)"
            ],
            index=0
        )

        preset_id = None
        body = "Mars"
        tactical_radius_km = 5.0

        if selection_mode.startswith("🏛️"):
            catalog_options = {
                f"[{v['body'].upper()}] {v['name']}": k
                for k, v in engine.catalog.items()
            }
            selected_label = st.selectbox("NASA Kalıcı Üs Hedefi Seçin:", list(catalog_options.keys()))
            preset_id = catalog_options[selected_label]
            t_info = engine.catalog[preset_id]
            tactical_radius_km = st.slider(
                "Taktik Üs Harita Yarıçapı (km):",
                min_value=2.5, max_value=20.0, value=5.0, step=2.5,
                help="5.0 km yarıçap = 10x10 km Taktik İniş Elipsi ve Kalıcı Üs Operasyon Ölçeği (~19m/piksel)."
            )
            st.caption(
                f"📌 **Koordinat:** `{t_info['lat']}°, {t_info['lon']}°` | "
                f"**Taktik Harita:** `{tactical_radius_km*2:.0f} x {tactical_radius_km*2:.0f} km`\n\n"
                f"🔬 **Odak:** {t_info['mission_focus']}"
            )
        else:
            body = st.selectbox("Hedef Gök Cismi Seçin:", ["Mars", "Moon"], index=0)

            # Hizli Ornek Koordinat Butonlari
            st.caption("Hızlı Test Koordinatları (Veya Aşağıdan Girin / Haritaya Tıklayın):")
            q_col1, q_col2 = st.columns(2)
            with q_col1:
                if st.button("🌋 Valles Marineris", use_container_width=True):
                    st.session_state.dyn_lat, st.session_state.dyn_lon = -13.80, -61.20
                if st.button("❄️ Korolev Buzulu", use_container_width=True):
                    st.session_state.dyn_lat, st.session_state.dyn_lon = 72.80, 164.50
            with q_col2:
                if st.button("⛰️ Olympus Mons", use_container_width=True):
                    st.session_state.dyn_lat, st.session_state.dyn_lon = 18.65, -133.80
                if st.button("🌑 Syrtis Major", use_container_width=True):
                    st.session_state.dyn_lat, st.session_state.dyn_lon = 8.40, 69.50

            col_lat, col_lon = st.columns(2)
            with col_lat:
                st.session_state.dyn_lat = st.number_input(
                    "Enlem (Lat °):", -89.9, 89.9, float(st.session_state.dyn_lat), step=0.5
                )
            with col_lon:
                st.session_state.dyn_lon = st.number_input(
                    "Boylam (Lon °):", -180.0, 180.0, float(st.session_state.dyn_lon), step=0.5
                )
            tactical_radius_km = st.slider(
                "Taktik Üs Harita Yarıçapı (km):",
                min_value=2.5, max_value=20.0, value=5.0, step=2.5
            )

        st.divider()
        st.subheader("2. Görev Mimarisi ve 6B Ağırlıklar")
        mission_label_map = {
            "🏰 Kalıcı Yeraltı / Lav Tüpü Üssü (Habitat)": "HABITAT_LAVA_TUBE",
            "🧊 Su Buzu & Kriyosferik ISRU Madenciliği": "ISRU_WATER_MINING",
            "🚜 Otonom Rover & Güvenli İniş (<5° Eğim)": "ROVER_LANDING_OPS",
            "⚖️ Dengeli Çok Amaçlı Keşif (Balanced)": "BALANCED",
            "🎛️ Manuel Jüri Ağırlıklandırması (Custom 6D)": "CUSTOM"
        }
        chosen_mission_label = st.selectbox("Görev Önceliği:", list(mission_label_map.keys()), index=0)
        mission_mode = mission_label_map[chosen_mission_label]

        custom_weights = None
        if mission_mode == "CUSTOM":
            st.caption("6 Bilimsel Boyutun Ağırlığını Ayarlayın:")
            w0 = st.slider("Litoloji & Bazalt", 0.0, 1.0, 0.25, 0.05)
            w1 = st.slider("Düzlük & İniş Güvenliği (<5°)", 0.0, 1.0, 0.20, 0.05)
            w2 = st.slider("Termal Ekstrem", 0.0, 1.0, 0.15, 0.05)
            w3 = st.slider("Su Buzu & Permafrost (ISRU)", 0.0, 1.0, 0.15, 0.05)
            w4 = st.slider("Rüzgar & Toz Dinamiği", 0.0, 1.0, 0.10, 0.05)
            w5 = st.slider("Lav Tüpü & Mağara Korunumu", 0.0, 1.0, 0.15, 0.05)
            custom_weights = [w0, w1, w2, w3, w4, w5]
            mission_mode = "BALANCED"

        st.divider()
        st.subheader("3. Keşif Filtreleri & Ağırlık Dağılımı")
        novelty_only = st.toggle(
            "💎 Sadece Az Bilinen Sahaları Keşfet (Novelty Mode)",
            value=True,
            help="Atacama, Hawaii, İzlanda, Antarktika, Devon Adası, Death Valley ve Utah MDRS klasik sahalarını maskeler."
        )
        min_sep_km = st.slider("Min. Havza Ayrım Mesafesi (NMS km):", 150, 1000, 450, 50)

        st.info(
            "📐 **Aktif Hibrit Puanlama Ağırlıkları:**\n"
            "- **%35** 6B Jeofiziksel Koşul Vektörü\n"
            "- **%30** 2B Topografik Harita & Kesit\n"
            "- **%20** Görsel AI (CLIP-ViT-B/32 + 64B CV)\n"
            "- **%15** Canlı NASA POWER İklim Barajı"
        )

    # --- SERBEST MODDA İNTERAKTİF AY/MARS HARİTA TIKLAMA PANELİ ---
    if not selection_mode.startswith("🏛️"):
        with st.expander("🗺️ İnteraktif Ay / Mars Küresel Haritasından Tıklayarak Koordinat Seç", expanded=True):
            st.caption(f"**{body}** haritasında istediğiniz kratere, kanyona veya düzlüğe tıklayın; koordinatlar anında yakalanıp canlı NASA Trek WMTS görüntüsü indirilir:")
            tile_url = (
                "https://cartocdn-gusc.global.ssl.fastly.net/opmbuilder/api/v1/map/named/opm-moon-basemap-v0-1/all/{z}/{x}/{y}.png"
                if body == "Moon"
                else "https://cartocdn-gusc.global.ssl.fastly.net/opmbuilder/api/v1/map/named/opm-mars-basemap-v0-1/all/{z}/{x}/{y}.png"
            )
            m_planet = folium.Map(location=[st.session_state.dyn_lat, st.session_state.dyn_lon], zoom_start=3, tiles=None)
            folium.TileLayer(tiles=tile_url, attr="OpenPlanetary / NASA", name=f"{body} Basemap").add_to(m_planet)
            folium.Marker(
                [st.session_state.dyn_lat, st.session_state.dyn_lon],
                tooltip=f"Seçili {body} Hedefi ({st.session_state.dyn_lat:.2f}°, {st.session_state.dyn_lon:.2f}°)"
            ).add_to(m_planet)
            m_planet.add_child(folium.LatLngPopup())
            map_click = st_folium(m_planet, height=270, use_container_width=True)
            if map_click and map_click.get("last_clicked"):
                clicked_lat = round(float(map_click["last_clicked"]["lat"]), 3)
                clicked_lon = round(float(map_click["last_clicked"]["lng"]), 3)
                if (clicked_lat != st.session_state.dyn_lat) or (clicked_lon != st.session_state.dyn_lon):
                    st.session_state.dyn_lat = clicked_lat
                    st.session_state.dyn_lon = clicked_lon
                    st.rerun()

    # --- ANA HİBRİT MOTOR ÇALIŞTIRMA ---
    with st.spinner("10x10 km Taktik NASA Trek görüntüsü analiz ediliyor, 8 aday canlı uydu görüntüsüyle çarpıştırılıyor..."):
        if preset_id:
            target_profile = engine.get_target_profile(preset_id=preset_id, radius_km=tactical_radius_km)
            target_profile["radius_km"] = tactical_radius_km
            target_profile["cross_section"] = engine.preprocessor.extract_elevation_cross_section(
                target_profile["rgb_image"], radius_km=tactical_radius_km
            )
        else:
            target_profile = engine.get_target_profile(
                body=body,
                lat=float(st.session_state.dyn_lat),
                lon=float(st.session_state.dyn_lon),
                radius_km=tactical_radius_km
            )

        # Hedef gezegen goruntusu icin de HUD katmani olustur
        target_micro = engine.preprocessor.pinpoint_micro_scale_feature(
            target_profile["rgb_image"],
            target_profile["lat"],
            target_profile["lon"],
            radius_km=tactical_radius_km,
            mission_mode=mission_mode
        )
        target_hud_img = engine.preprocessor.render_engineering_hud(
            target_profile["rgb_image"],
            target_profile["cross_section"],
            target_micro,
            radius_km=tactical_radius_km
        )

        search_res = engine.find_earth_analogs(
            target_profile=target_profile,
            mission_mode=mission_mode,
            custom_weights=custom_weights,
            novelty_only=novelty_only,
            top_k=3,
            min_separation_km=float(min_sep_km)
        )

    candidates = search_res["primary_candidates"]
    best_cand = candidates[0]
    comp_site = search_res["complementary_split_site"]
    bridge = search_res["moon_mars_bridge"]
    base_f = target_profile["base_feasibility"]
    xai_data = explainer.generate_candidate_explanation(target_profile, best_cand, comp_site, bridge)
    roadmap = explainer.build_phased_campaign_roadmap(candidates, comp_site)

    # --- ÜST TELEMETRİ METRİK KARTLARI ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        f"🏛️ Kalıcı {target_profile['body']} Üssü Fizibilitesi",
        f"%{base_f['overall_base_viability_pct']} Uygunluk",
        f"{target_profile['lat']}°, {target_profile['lon']}° ({tactical_radius_km*2:.0f}x{tactical_radius_km*2:.0f} km)"
    )
    m2.metric(
        "🥇 #1 Keşfedilen Dünya Analogu",
        best_cand["region_name"].split("[")[0].strip(),
        f"%{best_cand['total_match_pct']} Toplam Uyum (Ön Eleme #{best_cand['pre_screen_rank']} ➔ #1)"
    )
    m3.metric(
        "⛰️ 2B Topografik Kesit & İniş (%30)",
        f"%{best_cand['topographic_match_pct']} Kesit Uyumu",
        f"<5° Güvenli Alan: %{best_cand['cross_section']['safe_landing_pct']} (Üs: {best_cand['cross_section']['best_base_km']} km)"
    )
    m4.metric(
        "📡 Canlı NASA POWER Doğrulaması",
        f"{best_cand['nasa_power_climate']['mean_surface_temp_c']}°C | {best_cand['nasa_power_climate']['precipitation_mm_day']} mm/gün",
        f"%{best_cand['live_env_verification_pct']} Atmosferik Baraj Uyumu"
    )

    # --- KALICI AY/MARS ÜSSÜ MÜHENDİSLİK ÖZET KARTI ---
    st.markdown(
        f"""
        <div class="base-card">
            <b>🏗️ SEÇİLEN HEDEF İÇİN KALICI ÜS MİMARİSİ ({target_profile['name']}):</b><br>
            • <b>Önerilen Habitat Yapısı:</b> {base_f['recommended_habitat_architecture']} &nbsp;|&nbsp;
            <b>Radyasyon Zırhlama:</b> {base_f['radiation_shielding_protocol']}<br>
            • <b>ISRU Kaynak Çıkarma:</b> {base_f['isru_primary_strategy']} &nbsp;|&nbsp;
            <b>Enerji & Aydınlanma:</b> {base_f['power_and_illumination_architecture']}
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- 5 ANA KARAR DESTEK SEKMESİ ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌍 1. Küresel Keşif, Canlı Yarış & HUD Uydu",
        "📐 2. 6B Radar, 2B Kesit (%30) & İlk 3 Aday Karnesi",
        "🧬 3. Süreçler, Astrobiyoloji & Robotik Operasyon",
        "⚠️ 4. XAI Kırılma Matrisi, Cihaz Testi & TRL Yol Haritası",
        "📄 5. Resmi NASA Görev Dosyası (.md / .json)"
    ])

    # ==========================================================================
    # SEKME 1: KÜRESEL HARİTA, 8 ADAY CANLI YARIŞ TABLOSU VE HUD UYDU KIYASLAMASI
    # ==========================================================================
    with tab1:
        col_left, col_right = st.columns([1.15, 1.1])

        with col_left:
            st.subheader("🌐 Küresel Keşif Haritası ve İkili Saha (Split-Site) Köprüsü")
            st.plotly_chart(build_global_discovery_map(candidates, comp_site), use_container_width=True)

            st.markdown("##### 🏆 Kademe 1 ➔ Kademe 2: Hazır Setten Seçilen 8 Adayın Canlı Görüntü Kapışması (Re-Ranking)")
            pool_rows = []
            for final_rk, item in enumerate(search_res["shortlist_full_pool"], 1):
                shift = item["pre_rank"] - final_rk
                shift_badge = f"🔼 +{shift} Yükseldi" if shift > 0 else (f"🔽 {shift} Düştü" if shift < 0 else "➖ Korudu")
                pool_rows.append({
                    "Final Sıra": f"#{final_rk}",
                    "Ön Eleme": f"#{item['pre_rank']} ({shift_badge})",
                    "Keşfedilen Dünya Bölgesi [Jeolojik Kuşak]": item["region_name"],
                    "Koordinat": f"{item['lat']}°, {item['lon']}°",
                    "2B Kesit (%30)": f"%{item['topographic_match_pct']}",
                    "Görsel AI (%20)": f"%{item['visual_hybrid_pct']}",
                    "Nihai Skor": f"%{item['final_score_pct']}"
                })
            st.dataframe(pd.DataFrame(pool_rows), use_container_width=True, hide_index=True)

        with col_right:
            st.subheader(f"🛰️ Taktik Üs Ölçeği ({tactical_radius_km*2:.0f}x{tactical_radius_km*2:.0f} km) & Mikro-Hedef HUD Kıyaslaması")

            hud_col1, hud_col2 = st.columns([1.3, 1.0])
            with hud_col1:
                selected_rank = st.radio(
                    "Görseli İncelenen Aday:",
                    options=[f"#{i+1} {c['region_name'].split('[')[0].strip()} (%{c['total_match_pct']})" for i, c in enumerate(candidates)],
                    horizontal=True
                )
            with hud_col2:
                show_hud = st.toggle("🎯 Mühendislik HUD Nişangahı", value=True)

            sel_idx = int(selected_rank.split(" ")[0].replace("#", "")) - 1
            active_cand = candidates[sel_idx]
            mc = active_cand["micro_scale_pinpoint"]

            st.info(f"**🎖️ #{sel_idx+1} Adayın Stratejik Rolü:** {active_cand.get('strategic_role_badge', '')}")

            img_c1, img_c2 = st.columns(2)
            with img_c1:
                st.image(
                    target_hud_img if show_hud else target_profile["rgb_image"],
                    caption=f"HEDEF: {target_profile['name']} ({tactical_radius_km*2:.0f}x{tactical_radius_km*2:.0f} km Taktik Ölçek)",
                    use_container_width=True
                )
            with img_c2:
                st.image(
                    active_cand["hud_image"] if show_hud else active_cand["rgb_image"],
                    caption=f"#{sel_idx+1} ANALOG: {active_cand['region_name']} ({active_cand['lat']}°, {active_cand['lon']}°)",
                    use_container_width=True
                )

            st.markdown(
                f"""
                **🎯 Mikro-Ölçek (*Consider Scale*) Nokta Atışı ({mc['patch_scale_km']}):**
                - **Tespit Edilen Spesifik Yapı:** `{mc['micro_feature_type']}` (Uygunluk: **`%{mc['micro_suitability_pct']}`**)
                - **Tam Mikro-Koordinat:** `{mc['micro_lat']}°, {mc['micro_lon']}°` *(Merkezden Ofset: {mc['offset_from_center_km']} km)*
                - **Canlı NASA POWER İklimi:** Sıcaklık `{active_cand['nasa_power_climate']['mean_surface_temp_c']}°C` | Yağış `{active_cand['nasa_power_climate']['precipitation_mm_day']} mm/gün` | Nem `%{active_cand['nasa_power_climate']['relative_humidity_pct']}` | Rüzgar `{active_cand['nasa_power_climate']['wind_speed_10m_ms']} m/s`
                """
            )

    # ==========================================================================
    # SEKME 2: 6B RADAR, 2B TOPOGRAFİK KESİT VE İLK 3 ADAYIN DETAYLI KARNESİ
    # ==========================================================================
    with tab2:
        r_col1, r_col2 = st.columns([1.0, 1.15])
        with r_col1:
            st.subheader("🕸️ 6 Boyutlu Jeofiziksel Parmak İzi Uyumu")
            st.plotly_chart(
                build_radar_chart(target_profile["name"], target_profile["features_6d"], candidates, comp_site),
                use_container_width=True
            )
        with r_col2:
            st.subheader("⛰️ 2B Topografik Kesit (%30 Seçim Ağırlığı) & Üs Konumu")
            st.plotly_chart(
                build_cross_section_chart(target_profile, best_cand),
                use_container_width=True
            )

        st.divider()
        st.subheader("📏 Bilimsel Fiziksel Ölçüm Kanıt Tablosu (SI Birimleri, Hata Payları & Veri Kapsamı)")
        st.caption(
            f"**#1 Aday ({best_cand['region_name']})** için doğrudan sensör ölçümleri karşılaştırması — "
            f"**Eşli Veri Kapsamı (Data Coverage): `%{best_cand.get('data_coverage_pct', 100.0)}`** | "
            f"**Fiziksel Ölçüm Benzerliği: `%{best_cand.get('measured_physical_similarity_pct', best_cand['total_match_pct'])}`** "
            f"*(Eksik gözlemler asla tahmin edilmez / sıfır uydurma veri)*"
        )
        if "physical_evidence_table" in best_cand:
            ev_df = pd.DataFrame(best_cand["physical_evidence_table"]).rename(columns={
                "criterion": "Fiziksel Ölçüt (Criterion)",
                "earth_measured": "Dünya Sahası Ölçümü (±Hata)",
                "target_measured": f"{target_profile['body']} Hedef Ölçümü (±Hata)",
                "unit": "SI Birimi",
                "similarity_pct": "Fiziksel Benzerlik (%)",
                "weight": "Ağırlık",
                "provenance_tier": "Veri Kanıt Seviyesi (Tier)",
                "source": "Sensör & Veri Kaynağı"
            })
            st.dataframe(ev_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("🏅 Keşfedilen İlk 3 Dünya Analogunun Tam Karşılaştırmalı Mühendislik Karnesi")
        c_cols = st.columns(3)
        for i, cand in enumerate(candidates[:3]):
            with c_cols[i]:
                mc_i = cand["micro_scale_pinpoint"]
                pr_i = cand["process_analogy"]
                ab_i = cand["astrobiology_profile"]
                op_i = cand["operations_readiness"]
                cl_i = cand["nasa_power_climate"]
                st.markdown(
                    f"""
                    <div class="cand-card">
                        <h4 style="margin:0; color:#ffbe0b;">#{i+1} {cand['region_name']}</h4>
                        <p style="margin:4px 0; color:#00f5d4; font-size:0.86rem;"><b>{cand.get('strategic_role_badge', '')}</b></p>
                        <hr style="margin:8px 0; border-color:rgba(255,255,255,0.15);">
                        <b>📊 Toplam Bilimsel Uyum: %{cand['total_match_pct']}</b> <i>(Ön Eleme #{cand['pre_screen_rank']} ➔ #{i+1})</i><br>
                        • <b>6B Vektör (%35):</b> %{cand['parametric_6d_pct']} | <b>2B Topo (%30):</b> %{cand['topographic_match_pct']}<br>
                        • <b>Görsel AI (%20):</b> %{cand['visual_hybrid_pct']} | <b>NASA POWER (%15):</b> %{cand['live_env_verification_pct']}<br><br>
                        <b>🎯 Mikro-Hedef ({mc_i['patch_scale_km']}):</b><br>
                        <code>{mc_i['micro_lat']}°, {mc_i['micro_lon']}°</code> — {mc_i['micro_feature_type']} (<b>%{mc_i['micro_suitability_pct']}</b>)<br><br>
                        <b>🌪️ Aktif Süreç (%{pr_i['process_fidelity_pct']} Uyum):</b><br>
                        {pr_i['dominant_earth_process']}<br><br>
                        <b>🧬 Astrobiyoloji & Biyo-İmza (%{ab_i['biosignature_preservation_pct']}):</b><br>
                        {ab_i['extremophile_guild']}<br><br>
                        <b>🚜 Operasyonel Hazırlık:</b><br>
                        İniş (&lt;5°): <b>%{op_i['landing_safety_envelope_pct']}</b> | Mobilite: <b>%{op_i['rover_mobility_trafficability_pct']}</b> | TRN: <b>%{op_i['trn_optical_navigation_pct']}</b><br><br>
                        <b>📡 Canlı NASA POWER:</b> {cl_i['mean_surface_temp_c']}°C | Yağış: {cl_i['precipitation_mm_day']} mm/g | Nem: %{cl_i['relative_humidity_pct']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ==========================================================================
    # SEKME 3: SÜREÇLER, ASTROBİYOLOJİ VE ROBOTİK OPERASYON HAZIRLIĞI
    # ==========================================================================
    with tab3:
        p_col1, p_col2 = st.columns([1.2, 1.0])
        with p_col1:
            st.plotly_chart(build_process_and_ops_chart(best_cand), use_container_width=True)
        with p_col2:
            ab = best_cand["astrobiology_profile"]
            pr = best_cand["process_analogy"]
            st.subheader("🧬 Astrobiyoloji, Ekstremofil & Biyo-İmza Analizi")
            st.success(f"**Baskın Ekstremofil Analog Loncası:**\n{ab['extremophile_guild']}")
            st.markdown(
                f"""
                - **Biyo-İmza Koruma Sadakati:** `%{ab['biosignature_preservation_pct']}`
                - **Güneş / UV Radyasyon Stres İndeksi:** `%{ab['uv_radiation_stress_pct']}` (`{ab['solar_irradiance_kwh_m2_day']} kWh/m²/gün`)
                - **Gezegensel Astrobiyoloji Görev Bağlantısı:** {ab['astrobiology_mission_relevance']}
                ---
                ##### 🌋 Süreç Eşleşmesi (*Processes as well as Places*)
                - **Hedef Gezegen Süreci:** `{pr['target_planetary_process']}`
                - **Dünya Sahası Aktif Süreci:** `{pr['dominant_earth_process']}` (**`%{pr['process_fidelity_pct']}` Uyum**)
                """
            )

    # ==========================================================================
    # SEKME 4: XAI KIRILMA MATRİSİ, CİHAZ TEST PLANI & TRL 4->8 YOL HARİTASI
    # ==========================================================================
    with tab4:
        st.subheader("⚠️ Benzerlik Nerede Kırılıyor? (Earth's Limits as an Analog & Mitigation)")
        for brk in xai_data["breakdown_matrix"]:
            st.markdown(
                f"""
                <div class="xai-card">
                    <b>🚨 {brk['parameter']} — Kırılma Şiddeti: {brk['severity']}</b><br>
                    <span style="color:#a8dadc;"><b>Hedef Değer:</b> {brk['target_value']} &nbsp;|&nbsp; <b>Dünya Sahası:</b> {brk['earth_value']}</span><br>
                    <b>Fiziksel Etki:</b> {brk['impact']}<br>
                    <span style="color:#00f5d4;"><b>🛠️ Mühendislik Telafi Protokolü (Mitigation):</b> {brk['engineering_mitigation']}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader("🔬 NASA Cihaz Test Yatağı ve Risk Azaltma Matrisi (Connect to Exploration)")
        inst_df = pd.DataFrame(xai_data["instrument_testbed_plan"]).rename(columns={
            "instrument_system": "Test Edilecek NASA Cihazı / Sistemi",
            "target_mission_heritage": "Görev Mirası (Heritage)",
            "field_test_protocol": "Saha Test Protokolü",
            "risk_reduction": "Azaltılan Görev Riski"
        })
        st.dataframe(inst_df, use_container_width=True, hide_index=True)

        st.subheader("🗓️ TRL 4 ➔ TRL 8 Aşamalı Çoklu Saha Test Kampanyası Yol Haritası")
        rm_df = pd.DataFrame(roadmap).rename(columns={
            "phase": "Kampanya Fazı (TRL)",
            "assigned_sites": "Atanan Dünya Analog Sahaları",
            "objective": "Mühendislik ve Bilimsel Doğrulama Hedefi"
        })
        st.dataframe(rm_df, use_container_width=True, hide_index=True)

        c_split, c_bridge = st.columns(2)
        with c_split:
            if comp_site:
                st.markdown(
                    f"""
                    <div class="split-card">
                        <h4 style="margin:0; color:#00f5d4;">🧩 İkili Saha Reçetesi (Complementary Split-Site)</h4>
                        <p style="margin:6px 0;">
                            Birincil saha olan <b>{best_cand['region_name']}</b>, <i>'{comp_site['weakest_dimension_name']}'</i>
                            boyutunda (<b>%{comp_site['primary_site_score_in_dim_pct']}</b>) kısmi sapma göstermektedir.<br><br>
                            <b>Önerilen Tamamlayıcı 2. Test Sahası:</b> <b>{comp_site['complementary_region_name']}</b>
                            (<code>{comp_site['lat']}°, {comp_site['lon']}°</code>)<br>
                            <b>Tamamlama Uyumu:</b> <b>%{comp_site['complementary_dim_match_pct']}</b>
                            (Canlı Sıcaklık: {comp_site['nasa_power_climate']['mean_surface_temp_c']}°C | Yağış: {comp_site['nasa_power_climate']['precipitation_mm_day']} mm/gün)
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        with c_bridge:
            if bridge:
                st.markdown(
                    f"""
                    <div class="split-card" style="border-left-color:#ffbe0b;">
                        <h4 style="margin:0; color:#ffbe0b;">🚀 {bridge['source_body']} ↔ {bridge['bridge_body']} Çapraz Mimari Köprüsü</h4>
                        <p style="margin:6px 0;">
                            Seçilen <b>{bridge['source_body']}</b> hedefi için doğrulanan yaşam alanı ve rover mimarisi,
                            <b>%{bridge['architectural_transfer_pct']}</b> mühendislik uyumuyla doğrudan
                            <b>{bridge['bridge_body']} — {bridge['bridge_name']}</b> (<code>{bridge['lat']}°, {bridge['lon']}°</code>)
                            görevinde yeniden kullanılabilir!<br>
                            <b>Adaptasyon Notu:</b> {bridge.get('engineering_adaptation_note', '')}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ==========================================================================
    # SEKME 5: İNDİRİLEBİLİR GÖREV DOSYASI (.MD VE .JSON)
    # ==========================================================================
    with tab5:
        st.subheader("📑 Resmi NASA 2026 Görev Dosyası (Mission Dossier Export)")
        md_dossier = explainer.export_markdown_dossier(search_res)

        export_json_obj = {
            "target": {
                k: v for k, v in target_profile.items()
                if k not in ("rgb_image", "cross_section")
            },
            "weights_formula": "35% 6D Geophysical + 30% 2D Topographic Cross-Section + 20% Visual AI (CLIP+CV) + 15% Live NASA POWER",
            "top_3_candidates": [
                {
                    k: v for k, v in c.items()
                    if k not in ("rgb_image", "hud_image", "cross_section")
                }
                for c in candidates
            ],
            "complementary_split_site": comp_site,
            "moon_mars_bridge": bridge,
            "trl_campaign_roadmap": roadmap
        }
        json_dossier = json.dumps(export_json_obj, indent=2, ensure_ascii=False)

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="📥 Resmi Görev Dosyasını İndir (Markdown .md)",
                data=md_dossier.encode("utf-8"),
                file_name=f"NASA_2026_Mission_Dossier_{target_profile['body']}_{best_cand['lat']}_{best_cand['lon']}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with dl_col2:
            st.download_button(
                label="📥 Ham Telemetri Paketini İndir (JSON .json)",
                data=json_dossier.encode("utf-8"),
                file_name=f"NASA_2026_Telemetry_{target_profile['body']}_{best_cand['lat']}_{best_cand['lon']}.json",
                mime="application/json",
                use_container_width=True
            )

        st.divider()
        st.markdown(md_dossier)


if __name__ == "__main__":
    main()
