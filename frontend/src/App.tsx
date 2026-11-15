import {
  Component,
  lazy,
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  ArrowDownToLine,
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  Check,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Compass,
  Copy,
  Crosshair,
  Database,
  Earth,
  ExternalLink,
  FileJson,
  Flag,
  FlaskConical,
  Globe2,
  Layers3,
  Link2,
  MapPin,
  Menu,
  Moon,
  Mountain,
  Orbit,
  PanelLeftClose,
  Play,
  Radio,
  ScanLine,
  Search,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Snowflake,
  SplitSquareHorizontal,
  Target as TargetIcon,
  Thermometer,
  TriangleAlert,
  Waves,
  X,
} from "lucide-react";
import { datasets, sites, targets, type Body, type Site, type Target } from "./data/catalog";
import type { CameraLink, GlobeMarkerItem } from "./components/PlanetViewport";

const PlanetViewport = lazy(() => import("./components/PlanetViewport"));

class SceneBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? (
      <div className="scene-message error">
        The 3D workspace encountered an error. All telemetry and exports remain active.
        <button onClick={() => this.setState({ failed: false })}>Retry 3D workspace</button>
      </div>
    ) : (
      this.props.children
    );
  }
}

const PRESET_CATALOG = [
  {
    id: "MOON_MARE_TRANQUILLITATIS_PIT",
    body: "Moon" as Body,
    name: "Mare Tranquillitatis Pit (Subsurface Lava Tube Skylight)",
    lat: 8.335,
    lon: 33.222,
    focus: "Subsurface Lava Tube Radiation Shelter & Mare Basalt ISRU",
  },
  {
    id: "MOON_SHACKLETON_RIM",
    body: "Moon" as Body,
    name: "Shackleton Crater Rim (Artemis Lunar South Pole Base)",
    lat: -89.9,
    lon: 0.0,
    focus: "Cryogenic Water-Ice ISRU Mining & Peaks of Eternal Light Solar Array",
  },
  {
    id: "MOON_MARIUS_HILLS",
    body: "Moon" as Body,
    name: "Marius Hills Volcanic Dome & Skylight Rille",
    lat: 14.09,
    lon: -56.77,
    focus: "Volcanic Dome Geology, Sinuous Rille Lava Tube Shelter & Basalt ISRU",
  },
  {
    id: "MARS_JEZERO_CRATER",
    body: "Mars" as Body,
    name: "Jezero Crater Paleo-Lake Delta (Mars 2020 Landing Site)",
    lat: 18.38,
    lon: 77.58,
    focus: "Paleo-Lake Biosignature Sample Return & Autonomous Rover Traverse",
  },
  {
    id: "MARS_ARCADIA_PLANITIA",
    body: "Mars" as Body,
    name: "Arcadia Planitia Shallow Subsurface Ice Sheet Base",
    lat: 39.2,
    lon: -188.0,
    focus: "Flat Heavy-Starship Landing Pad (<2° Slope) & Shallow Ice ISRU Mining",
  },
  {
    id: "MARS_ARSIA_MONS_CAVES",
    body: "Mars" as Body,
    name: "Arsia Mons 'Seven Sisters' Skylight Caves (Tharsis Montes)",
    lat: -7.22,
    lon: -120.45,
    focus: "High-Altitude Volcanic Cave Habitat & Subsurface Astrobiology",
  },
];

const MISSION_MODES = [
  { id: "HABITAT_LAVA_TUBE", label: "🏰 Kalıcı Yeraltı / Lav Tüpü Üssü (Habitat)" },
  { id: "ISRU_WATER_MINING", label: "🧊 Su Buzu & Kriyosferik ISRU Madenciliği" },
  { id: "ROVER_LANDING_OPS", label: "🚜 Otonom Rover & Güvenli İniş (<5° Eğim)" },
  { id: "BALANCED", label: "⚖️ Dengeli Çok Amaçlı Keşif (Balanced)" },
  { id: "CUSTOM", label: "🎛️ Manuel Jüri Ağırlıklandırması (Custom 6D)" },
];

const QUICK_COORDS = [
  { label: "🌋 Valles Marineris", body: "Mars" as Body, lat: -13.8, lon: -61.2 },
  { label: "❄️ Korolev Buzulu", body: "Mars" as Body, lat: 72.8, lon: 164.5 },
  { label: "⛰️ Olympus Mons", body: "Mars" as Body, lat: 18.65, lon: -133.8 },
  { label: "🌑 Syrtis Major", body: "Mars" as Body, lat: 8.4, lon: 69.5 },
  { label: "🌕 Tycho Krateri (Ay)", body: "Moon" as Body, lat: -43.31, lon: -11.36 },
];

function triggerDownload(content: string, mime: string, filename: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// 6B SVG Radar Grafigi Bileseni
function RadarChart6D({
  dimNames,
  targetName,
  targetVec,
  candidates,
  compSite,
}: {
  dimNames: string[];
  targetName: string;
  targetVec: number[];
  candidates: any[];
  compSite: any;
}) {
  const size = 340;
  const cx = size / 2;
  const cy = size / 2;
  const radius = 112;
  const labels = [
    "1. Bazalt/Albedo",
    "2. Düzlük (<5°)",
    "3. Termal/Güneş",
    "4. Su Buzu (ISRU)",
    "5. Rüzgar/Kuraklık",
    "6. Rölyef/Mağara",
  ];

  const pointAt = (idx: number, val01: number) => {
    const angle = (Math.PI * 2 * idx) / 6 - Math.PI / 2;
    const r = Math.max(0, Math.min(1, val01)) * radius;
    return `${(cx + r * Math.cos(angle)).toFixed(1)},${(cy + r * Math.sin(angle)).toFixed(1)}`;
  };

  const polyPoints = (vec: number[]) => vec.map((v, i) => pointAt(i, v)).join(" ");

  return (
    <div>
      <svg viewBox={`0 0 ${size} ${size}`} style={{ width: "100%", maxHeight: 310 }}>
        {[0.25, 0.5, 0.75, 1.0].map((lvl) => (
          <polygon
            key={lvl}
            points={[0, 1, 2, 3, 4, 5].map((i) => pointAt(i, lvl)).join(" ")}
            fill="none"
            stroke="#263544"
            strokeWidth="1"
          />
        ))}
        {[0, 1, 2, 3, 4, 5].map((i) => {
          const angle = (Math.PI * 2 * i) / 6 - Math.PI / 2;
          const x2 = cx + radius * Math.cos(angle);
          const y2 = cy + radius * Math.sin(angle);
          const lx = cx + (radius + 26) * Math.cos(angle);
          const ly = cy + (radius + 20) * Math.sin(angle);
          return (
            <g key={i}>
              <line x1={cx} y1={cy} x2={x2} y2={y2} stroke="#2b3d4f" strokeWidth="1" />
              <text
                x={lx}
                y={ly}
                fill="#9rb2c5"
                fontSize="8.5"
                textAnchor="middle"
                dominantBaseline="middle"
                style={{ fill: "#a8c0d0", fontFamily: "var(--font-mono)" }}
              >
                {labels[i] || dimNames[i]?.slice(0, 14)}
              </text>
            </g>
          );
        })}
        {/* Hedef Poligonu */}
        <polygon
          points={polyPoints(targetVec)}
          fill="rgba(0, 212, 255, 0.20)"
          stroke="#00d4ff"
          strokeWidth="2.5"
        />
        {/* #1 Aday Poligonu */}
        {candidates[0] && (
          <polygon
            points={polyPoints(candidates[0].features_6d)}
            fill="rgba(255, 159, 28, 0.18)"
            stroke="#ff9f1c"
            strokeWidth="2.2"
          />
        )}
        {/* #2 Aday Poligonu */}
        {candidates[1] && (
          <polygon
            points={polyPoints(candidates[1].features_6d)}
            fill="none"
            stroke="#ff006e"
            strokeWidth="1.6"
          />
        )}
        {/* Ikili Saha Poligonu */}
        {compSite?.features_6d && (
          <polygon
            points={polyPoints(compSite.features_6d)}
            fill="none"
            stroke="#00f5d4"
            strokeWidth="1.8"
            strokeDasharray="4 3"
          />
        )}
      </svg>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, justifyContent: "center", fontSize: 9 }}>
        <span style={{ color: "#00d4ff" }}>● HEDEF: {targetName.slice(0, 22)}</span>
        {candidates[0] && (
          <span style={{ color: "#ff9f1c" }}>
            ● #1: {candidates[0].region_name.split("[")[0].trim().slice(0, 20)}
          </span>
        )}
        {candidates[1] && (
          <span style={{ color: "#ff006e" }}>
            ● #2: {candidates[1].region_name.split("[")[0].trim().slice(0, 18)}
          </span>
        )}
        {compSite && (
          <span style={{ color: "#00f5d4" }}>
            ┈ İkili Saha: {compSite.complementary_region_name.split("[")[0].trim().slice(0, 18)}
          </span>
        )}
      </div>
    </div>
  );
}

// 2B Topografik Kesit SVG Grafigi
function CrossSectionChart2D({ target, cand }: { target: any; cand: any }) {
  const tCs = target?.cross_section;
  const eCs = cand?.cross_section;
  if (!tCs || !eCs) return null;

  const w = 460;
  const h = 250;
  const padL = 42;
  const padR = 18;
  const padT = 24;
  const padB = 34;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const tElev: number[] = tCs.elevation_m || [];
  const eElev: number[] = eCs.elevation_m || [];
  const dists: number[] = eCs.distances_km || [];
  const maxElev = Math.max(10, ...tElev, ...eElev);
  const maxDist = Math.max(1, ...(dists.length ? dists : [10]));

  const makePath = (arr: number[]) =>
    arr
      .map((v, i) => {
        const x = padL + (i / Math.max(1, arr.length - 1)) * plotW;
        const y = padT + plotH - (v / maxElev) * plotH;
        return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

  const bestKm = Number(eCs.best_base_km ?? maxDist / 2);
  const bestIdx = Math.min(
    eElev.length - 1,
    Math.max(0, Math.round((bestKm / maxDist) * (eElev.length - 1))),
  );
  const bestX = padL + (bestIdx / Math.max(1, eElev.length - 1)) * plotW;
  const bestY = padT + plotH - ((eElev[bestIdx] || 0) / maxElev) * plotH;

  return (
    <div>
      <svg viewBox={`0 0 ${w} ${h}`} style={{ width: "100%", maxHeight: 280 }}>
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const y = padT + plotH - f * plotH;
          return (
            <g key={f}>
              <line x1={padL} y1={y} x2={w - padR} y2={y} stroke="#233240" strokeWidth="1" />
              <text x={padL - 6} y={y + 3} fill="#7d95a6" fontSize="8" textAnchor="end">
                {Math.round(f * maxElev)}m
              </text>
            </g>
          );
        })}
        <path d={makePath(tElev)} fill="none" stroke="#00d4ff" strokeWidth="2.2" />
        <path d={makePath(eElev)} fill="none" stroke="#ff9f1c" strokeWidth="2.2" />
        <circle cx={bestX} cy={bestY} r="5.5" fill="#00f5d4" stroke="#ffffff" strokeWidth="1.5" />
        <text x={bestX} y={Math.max(14, bestY - 9)} fill="#00f5d4" fontSize="8.5" textAnchor="middle">
          ★ ÜS NOKTASI ({bestKm} km)
        </text>
        <text x={w / 2} y={h - 6} fill="#8ea4b5" fontSize="8.5" textAnchor="middle">
          Taktik Kesit Mesafesi (0 — {maxDist.toFixed(0)} km) · Kaynak: {eCs.dem_source?.slice(0, 38)}
        </text>
      </svg>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 14, justifyContent: "center", fontSize: 9 }}>
        <span style={{ color: "#00d4ff" }}>
          ━ {target.body} Hedef (Eğim: {tCs.mean_slope_deg}° | Rölyef: {tCs.relief_m} m)
        </span>
        <span style={{ color: "#ff9f1c" }}>
          ━ Dünya SRTM 90m DEM (Eğim: {eCs.mean_slope_deg}° | &lt;5° Güvenli: %{eCs.safe_landing_pct})
        </span>
      </div>
    </div>
  );
}

export default function App() {
  const [body, setBody] = useState<Body>("Earth");
  const [split, setSplit] = useState<boolean>(true);
  const [collapsed, setCollapsed] = useState(false);
  const [imagery, setImagery] = useState(true);
  const [grid, setGrid] = useState(true);
  const [markers, setMarkers] = useState(true);
  const [sync, setSync] = useState(false);
  const [focus, setFocus] = useState(0);
  const [language, setLanguage] = useState<"TR" | "EN">("TR");
  const [notice, setNotice] = useState("");
  const [modal, setModal] = useState<"method" | "data" | "export" | "saved" | null>(null);

  // --- APP.PY GOREV KONTROL DURUMLARI ---
  const [selectionMode, setSelectionMode] = useState<"PRESET" | "DYNAMIC">("PRESET");
  const [presetId, setPresetId] = useState<string>("MOON_MARE_TRANQUILLITATIS_PIT");
  const [dynBody, setDynBody] = useState<Body>("Mars");
  const [dynLat, setDynLat] = useState<number>(-13.8);
  const [dynLon, setDynLon] = useState<number>(-61.2);
  const [radiusKm, setRadiusKm] = useState<number>(5.0);
  const [missionMode, setMissionMode] = useState<string>("HABITAT_LAVA_TUBE");
  const [customWeights, setCustomWeights] = useState<number[]>([0.25, 0.2, 0.15, 0.15, 0.1, 0.15]);
  const [noveltyOnly, setNoveltyOnly] = useState<boolean>(true);
  const [minSepKm, setMinSepKm] = useState<number>(450);

  // --- MOTOR SONUC DURUMLARI ---
  const [apiData, setApiData] = useState<any | null>(null);
  const [loadingEngine, setLoadingEngine] = useState<boolean>(false);
  const [selectedCandIdx, setSelectedCandIdx] = useState<number>(0);
  const [showHud, setShowHud] = useState<boolean>(true);
  const [activeMainTab, setActiveMainTab] = useState<number>(1);
  const [query, setQuery] = useState<string>("");
  const [saved, setSaved] = useState<{ title: string; presetId: string; mode: string; score: number }[]>([]);

  const cameraLink = useRef<CameraLink>({
    direction: { x: 0, y: 0, z: 1 },
    distance: 6.4,
    source: "",
    revision: 0,
  });

  const tr = (en: string, tu: string) => (language === "TR" ? tu : en);

  // Canli Python API Cagrisi
  async function runEngineAnalysis(overrideParams?: Partial<{
    selection_mode: "PRESET" | "DYNAMIC";
    preset_id: string;
    body: Body;
    lat: number;
    lon: number;
    radius_km: number;
    mission_mode: string;
    custom_weights: number[];
    novelty_only: boolean;
    min_separation_km: number;
  }>) {
    setLoadingEngine(true);
    const payload = {
      selection_mode: overrideParams?.selection_mode ?? selectionMode,
      preset_id: overrideParams?.preset_id ?? presetId,
      body: overrideParams?.body ?? (selectionMode === "PRESET" ? PRESET_CATALOG.find((p) => p.id === presetId)?.body ?? "Moon" : dynBody),
      lat: overrideParams?.lat ?? dynLat,
      lon: overrideParams?.lon ?? dynLon,
      radius_km: overrideParams?.radius_km ?? radiusKm,
      mission_mode: overrideParams?.mission_mode ?? missionMode,
      custom_weights: overrideParams?.custom_weights ?? customWeights,
      novelty_only: overrideParams?.novelty_only ?? noveltyOnly,
      min_separation_km: overrideParams?.min_separation_km ?? minSepKm,
    };

    try {
      let resp: Response;
      try {
        resp = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!resp.ok) throw new Error("Proxy fallback");
      } catch {
        resp = await fetch("http://127.0.0.1:8000/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      }
      const data = await resp.json();
      if (data && data.status === "ok") {
        setApiData(data);
        setSelectedCandIdx(0);
        setFocus((n) => n + 1);
      }
    } catch (err) {
      setNotice("Python API sunucusuna bağlanılamadı (api_server.py çalışıyor mu kontrol edin).");
    } finally {
      setLoadingEngine(false);
    }
  }

  useEffect(() => {
    runEngineAnalysis();
  }, [selectionMode, presetId, dynBody, radiusKm, missionMode, noveltyOnly, minSepKm]);

  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(""), 3800);
    return () => clearTimeout(t);
  }, [notice]);

  const targetProfile = apiData?.target;
  const candidates: any[] = apiData?.primary_candidates || [];
  const activeCand = candidates[selectedCandIdx] || candidates[0] || null;
  const compSite = apiData?.complementary_split_site || null;
  const bridge = apiData?.moon_mars_bridge || null;
  const baseFeas = targetProfile?.base_feasibility || null;
  const shortlistPool: any[] = apiData?.shortlist_full_pool || [];
  const xaiData = activeCand?.xai_explanation || apiData?.xai_primary || null;
  const roadmap: any[] = apiData?.roadmap || [];
  const mdDossier: string = apiData?.markdown_dossier || "";

  // 3B Küre İçin Aktif Dünya ve Hedef Nesneleri
  const activeSiteObj: Site = useMemo(() => {
    if (activeCand) {
      return {
        id: `cand_${selectedCandIdx}`,
        name: activeCand.region_name.split("[")[0].trim(),
        region: activeCand.region_name,
        lat: activeCand.lat,
        lon: activeCand.lon,
        environment: activeCand.features_6d?.[3] > 0.5 ? "Polar" : "Volcanic",
        terrain: activeCand.strategic_role_badge || "Terrestrial Analog",
        description: activeCand.process_analogy?.dominant_earth_process || "",
        usefulness: activeCand.astrobiology_profile?.astrobiology_mission_relevance || "",
        limitation: "See XAI Breakdown Matrix",
        source: "https://power.larc.nasa.gov/",
      };
    }
    return sites[0];
  }, [activeCand, selectedCandIdx]);

  const activeTargetObj: Target = useMemo(() => {
    if (targetProfile) {
      return {
        id: targetProfile.id,
        name: targetProfile.name,
        body: (targetProfile.body === "Moon" ? "Moon" : "Mars") as "Moon" | "Mars",
        lat: targetProfile.lat,
        lon: targetProfile.lon,
        context: targetProfile.description || "",
        source: targetProfile.physical_measurements?.official_url || "https://trek.nasa.gov/",
      };
    }
    return targets[0];
  }, [targetProfile]);

  // 3B Dünya Küresi İşaretçileri (#1, #2, #3 + İkili Saha + 8'li Havuz)
  const earthGlobeMarkers: GlobeMarkerItem[] = useMemo(() => {
    const list: GlobeMarkerItem[] = [];
    const rankColors = [0xff006e, 0xff9f1c, 0xffbe0b];
    candidates.forEach((c, idx) => {
      list.push({
        id: `cand_${idx}`,
        name: c.region_name.split("[")[0].trim(),
        lat: c.lat,
        lon: c.lon,
        badge: `#${idx + 1} (%${c.total_match_pct})`,
        colorHex: rankColors[idx] ?? 0x72f2ca,
      });
    });
    if (compSite) {
      list.push({
        id: "split_site",
        name: compSite.complementary_region_name.split("[")[0].trim(),
        lat: compSite.lat,
        lon: compSite.lon,
        badge: "🧩 İkili Saha",
        colorHex: 0x00f5d4,
        isSplitSite: true,
      });
    }
    shortlistPool.slice(3).forEach((p, idx) => {
      list.push({
        id: `pool_${idx + 4}`,
        name: p.region_name.split("[")[0].trim(),
        lat: p.lat,
        lon: p.lon,
        badge: `#${idx + 4}`,
        colorHex: 0x6889a8,
      });
    });
    return list;
  }, [candidates, compSite, shortlistPool]);

  const targetGlobeMarkers: GlobeMarkerItem[] = useMemo(() => {
    const tBody = activeTargetObj.body;
    const list: GlobeMarkerItem[] = PRESET_CATALOG.filter((p) => p.body === tBody).map((p) => ({
      id: p.id,
      name: p.name.split("(")[0].trim(),
      lat: p.lat,
      lon: p.lon,
      badge: "🏛️",
      colorHex: p.id === presetId && selectionMode === "PRESET" ? 0x00d4ff : 0x9de9d3,
    }));
    if (selectionMode === "DYNAMIC" && targetProfile) {
      list.push({
        id: targetProfile.id,
        name: `${targetProfile.body} (${targetProfile.lat}°, ${targetProfile.lon}°)`,
        lat: targetProfile.lat,
        lon: targetProfile.lon,
        badge: "🎯 SEÇİLİ HEDEF",
        colorHex: 0xff006e,
      });
    }
    return list;
  }, [activeTargetObj.body, presetId, selectionMode, targetProfile]);

  // 3B Küre Üzerinde Herhangi Bir Yere Tıklandığında Çalışan Fonksiyon
  function handleGlobeCoordinateClick(clickedBody: Body, lat: number, lon: number) {
    if (clickedBody === "Moon" || clickedBody === "Mars") {
      setSelectionMode("DYNAMIC");
      setDynBody(clickedBody);
      setDynLat(lat);
      setDynLon(lon);
      setNotice(`🎯 3B ${clickedBody} küresinden (${lat}°, ${lon}°) koordinatı seçildi, analiz ediliyor...`);
      runEngineAnalysis({
        selection_mode: "DYNAMIC",
        body: clickedBody,
        lat,
        lon,
      });
    } else {
      // Dünya küresinde bir yere tıklandığında en yakın adayı seç
      if (candidates.length > 0) {
        let bestIdx = 0;
        let bestD = 1e9;
        candidates.forEach((c, i) => {
          const d = Math.hypot(c.lat - lat, c.lon - lon);
          if (d < bestD) {
            bestD = d;
            bestIdx = i;
          }
        });
        setSelectedCandIdx(bestIdx);
        setNotice(`🌍 Dünya küresinden #${bestIdx + 1} ${candidates[bestIdx].region_name.split("[")[0]} seçildi.`);
      }
    }
  }

  function handleMarkerSelect(markerId: string) {
    if (markerId.startsWith("cand_")) {
      const idx = parseInt(markerId.replace("cand_", ""), 10);
      if (!isNaN(idx) && candidates[idx]) {
        setSelectedCandIdx(idx);
        setFocus((n) => n + 1);
      }
    } else if (PRESET_CATALOG.some((p) => p.id === markerId)) {
      setSelectionMode("PRESET");
      setPresetId(markerId);
    }
  }

  const filteredPool = useMemo(() => {
    if (!query.trim()) return shortlistPool;
    return shortlistPool.filter((item) =>
      item.region_name.toLowerCase().includes(query.toLowerCase()),
    );
  }, [shortlistPool, query]);

  return (
    <div className={`app ${collapsed ? "is-collapsed" : ""}`}>
      {/* ÜST BAR */}
      <header className="topbar">
        <a
          className="identity"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setBody("Earth");
            setSplit(true);
          }}
        >
          <span className="brand-mark">
            <Orbit size={31} strokeWidth={1.4} />
          </span>
          <span>
            TERRA-TWIN
            <span className="brand-sub">
              NASA 2026 ANALOG ENGINE <b>v6.0</b>
            </span>
          </span>
        </a>

        <nav className="planet-nav" aria-label="Planetary arenas">
          {(["Earth", "Moon", "Mars"] as Body[]).map((b, i) => (
            <button
              key={b}
              className={body === b && !split ? "active" : ""}
              onClick={() => {
                setBody(b);
                setSplit(false);
                if (b === "Moon" || b === "Mars") {
                  setDynBody(b);
                }
              }}
            >
              <span className={`planet-dot dot-${b.toLowerCase()}`} />
              {tr(b, ["Dünya (3B)", "Ay (3B)", "Mars (3B)"][i])}
              <span className="nav-num">0{i + 1}</span>
            </button>
          ))}
          <button
            className={split ? "active" : ""}
            onClick={() => {
              setSplit(true);
              setFocus((n) => n + 1);
            }}
          >
            <SplitSquareHorizontal size={15} />
            {tr("Split 3D Worlds", "İkili 3B Küre Kıyaslama")}
          </button>
        </nav>

        <div className="top-status">
          <span className="status-dot" />
          {loadingEngine ? "ANALİZ EDİLİYOR..." : "11,469 KÜRESEL HÜCRE AKTİF"}
          <span className="status-divider" />
          <Database size={13} />
          SRTM 90m DEM + NASA POWER + LOLA/MOLA
        </div>

        <div className="top-actions">
          <button
            className="language"
            onClick={() => setLanguage((l) => (l === "TR" ? "EN" : "TR"))}
          >
            <Globe2 size={14} />
            {language}
            <ChevronDown size={11} />
          </button>
          <button className="icon-button" onClick={() => setModal("method")} title="Bilimsel Metodoloji">
            <CircleHelp size={18} />
          </button>
          <button className="demo-button" onClick={() => setActiveMainTab(5)}>
            <ArrowDownToLine size={13} />
            {tr("Export Dossier", "Görev Dosyası İndir")}
          </button>
        </div>
      </header>

      <div className="app-body">
        {/* SOL KENAR ÇUBUĞU */}
        <aside className="sidebar" style={{ width: collapsed ? 68 : 238 }}>
          <div className="workspace-name">
            <span className="workspace-icon">
              <FlaskConical size={17} />
            </span>
            <div>
              <b>{tr("Mission Control", "Görev Kontrol Merkezi")}</b>
              <small>NASA SPACE APPS 2026</small>
            </div>
            <button aria-label="Collapse sidebar" onClick={() => setCollapsed(!collapsed)}>
              {collapsed ? <Menu size={17} /> : <PanelLeftClose size={16} />}
            </button>
          </div>

          <div className="sidebar-scroll">
            <div className="nav-group">
              <label>{tr("5 DECISION TABS", "5 KARAR DESTEK SEKMESİ")}</label>
              {[
                { id: 1, label: "1. Küresel Keşif & HUD Uydu", Icon: Globe2 },
                { id: 2, label: "2. 6B Radar, 2B Kesit & SI Kanıt", Icon: Mountain },
                { id: 3, label: "3. Süreç, Astrobiyoloji & Rover", Icon: FlaskConical },
                { id: 4, label: "4. XAI Kırılma & TRL Yol Haritası", Icon: TriangleAlert },
                { id: 5, label: "5. Görev Dosyası (.md/.json/.csv)", Icon: ArrowDownToLine },
              ].map(({ id, label, Icon }) => (
                <button
                  key={id}
                  className={`side-link ${activeMainTab === id ? "selected" : ""}`}
                  onClick={() => {
                    setActiveMainTab(id);
                    document.getElementById("analysis-tabs-anchor")?.scrollIntoView({ behavior: "smooth" });
                  }}
                >
                  <Icon size={15} />
                  <span>{label}</span>
                  <i />
                </button>
              ))}
            </div>

            <div className="nav-group">
              <label>{tr("3D PLANETARY ARENAS", "3B GEZEGEN KÜRELERİ")}</label>
              <button
                className={`side-link ${split ? "selected" : ""}`}
                onClick={() => setSplit(true)}
              >
                <SplitSquareHorizontal size={16} />
                <span>Dünya ↔ {activeTargetObj.body} İkili 3B</span>
                <i />
              </button>
              <button
                className={`side-link ${body === "Earth" && !split ? "selected" : ""}`}
                onClick={() => {
                  setBody("Earth");
                  setSplit(false);
                }}
              >
                <Earth size={16} />
                <span>3B Dünya Analog Küresi</span>
              </button>
              <button
                className={`side-link ${body === "Moon" && !split ? "selected" : ""}`}
                onClick={() => {
                  setBody("Moon");
                  setDynBody("Moon");
                  setSplit(false);
                }}
              >
                <Moon size={16} />
                <span>3B Ay Küresi (Tıkla & Seç)</span>
              </button>
              <button
                className={`side-link ${body === "Mars" && !split ? "selected" : ""}`}
                onClick={() => {
                  setBody("Mars");
                  setDynBody("Mars");
                  setSplit(false);
                }}
              >
                <Orbit size={16} />
                <span>3B Mars Küresi (Tıkla & Seç)</span>
              </button>
            </div>

            <div className="nav-group layer-group">
              <label>
                {tr("ACTIVE SENSOR LAYERS", "AKTİF SENSÖR KATMANLARI")}
                <button onClick={() => setModal("data")}>
                  <Settings2 size={13} />
                </button>
              </label>
              <label className="layer-control">
                <Layers3 size={15} />
                <span>Yüzey Uydu Dokusu (2K)</span>
                <input
                  type="checkbox"
                  checked={imagery}
                  onChange={(e) => setImagery(e.target.checked)}
                />
                <i />
              </label>
              <label className="layer-control">
                <ScanLine size={15} />
                <span>Gezegensel Koordinat Ağı</span>
                <input
                  type="checkbox"
                  checked={grid}
                  onChange={(e) => setGrid(e.target.checked)}
                />
                <i />
              </label>
              <label className="layer-control">
                <MapPin size={15} />
                <span>3B Analog & İkili Saha Pinleri</span>
                <input
                  type="checkbox"
                  checked={markers}
                  onChange={(e) => setMarkers(e.target.checked)}
                />
                <i />
              </label>
              {[
                ["SRTM 90m / LOLA DEM", Mountain, "AKTİF"],
                ["2B Eğim & <5° İniş Zarfı", SlidersHorizontal, "AKTİF"],
                ["NASA POWER İklim & Termal", Thermometer, "AKTİF"],
                ["Kriyosferik Su Buzu (ISRU)", Snowflake, "AKTİF"],
                ["Lav Tüpü & Mağara Rölyefi", Waves, "AKTİF"],
              ].map(([label, Icon, badge]) => {
                const I = Icon as typeof Mountain;
                return (
                  <div className="layer-control" key={String(label)} style={{ color: "#9de9d3" }}>
                    <I size={15} />
                    <span>{String(label)}</span>
                    <span style={{ marginLeft: "auto", fontSize: 8, fontFamily: "var(--font-mono)" }}>
                      {String(badge)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="sidebar-bottom">
            <div className="challenge-badge">
              <span className="challenge-orbit">
                <Orbit size={23} />
              </span>
              <div>
                NASA SPACE APPS<small>CHALLENGE 2026</small>
              </div>
              <ArrowUpRight size={14} />
            </div>
            <p>80% Fiziksel/Topoğrafik · 20% Görsel AI</p>
            <div>
              <span className="tiny-dot" /> %100 SI KANIT KAPSAMI <span>v6.0</span>
            </div>
          </div>
        </aside>

        {/* ANA İÇERİK ALANI */}
        <main className="main">
          {/* 1. GÖREV KONTROL VE KOORDİNAT SEÇİM PANELİ (APP.PY SIDEBAR ÖZELLİKLERİ) */}
          <section className="tt-control-panel">
            <div className="tt-control-row">
              <div className="tt-field" style={{ maxWidth: 250 }}>
                <label>1. GEZEGEN HEDEFİ SEÇİM MODU</label>
                <select
                  value={selectionMode}
                  onChange={(e) => setSelectionMode(e.target.value as "PRESET" | "DYNAMIC")}
                >
                  <option value="PRESET">🏛️ Resmi NASA Üs Kataloğu (6 Görev)</option>
                  <option value="DYNAMIC">🎯 Serbest Koordinat & 3B Küre Tıklama (Canlı)</option>
                </select>
              </div>

              {selectionMode === "PRESET" ? (
                <div className="tt-field" style={{ flex: 2 }}>
                  <label>NASA KALICI AY / MARS ÜS HEDEFİ SEÇİN</label>
                  <select value={presetId} onChange={(e) => setPresetId(e.target.value)}>
                    {PRESET_CATALOG.map((p) => (
                      <option key={p.id} value={p.id}>
                        [{p.body.toUpperCase()}] {p.name} ({p.lat}°, {p.lon}°)
                      </option>
                    ))}
                  </select>
                </div>
              ) : (
                <>
                  <div className="tt-field" style={{ maxWidth: 120 }}>
                    <label>GÖK CİSMİ</label>
                    <select value={dynBody} onChange={(e) => setDynBody(e.target.value as Body)}>
                      <option value="Mars">Mars</option>
                      <option value="Moon">Moon (Ay)</option>
                    </select>
                  </div>
                  <div className="tt-field" style={{ maxWidth: 115 }}>
                    <label>ENLEM (LAT °)</label>
                    <input
                      type="number"
                      step="0.5"
                      min="-89.9"
                      max="89.9"
                      value={dynLat}
                      onChange={(e) => setDynLat(Number(e.target.value))}
                    />
                  </div>
                  <div className="tt-field" style={{ maxWidth: 115 }}>
                    <label>BOYLAM (LON °)</label>
                    <input
                      type="number"
                      step="0.5"
                      min="-180"
                      max="180"
                      value={dynLon}
                      onChange={(e) => setDynLon(Number(e.target.value))}
                    />
                  </div>
                  <button
                    className="primary-button"
                    style={{ alignSelf: "flex-end", padding: "8px 14px" }}
                    onClick={() => runEngineAnalysis()}
                  >
                    <Crosshair size={14} /> Analiz Et
                  </button>
                </>
              )}

              <div className="tt-field" style={{ flex: 1.4 }}>
                <label>2. GÖREV MİMARİSİ VE ÖNCELİĞİ</label>
                <select value={missionMode} onChange={(e) => setMissionMode(e.target.value)}>
                  {MISSION_MODES.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Serbest Mod Hızlı Test Koordinatları */}
            {selectionMode === "DYNAMIC" && (
              <div className="tt-control-row" style={{ justifyContent: "space-between" }}>
                <div className="tt-quick-btns">
                  <span style={{ fontSize: 9, color: "#8496a7", alignSelf: "center", marginRight: 4 }}>
                    Hızlı Test Koordinatları (Veya 3B Ay/Mars Küresine Tıklayın):
                  </span>
                  {QUICK_COORDS.map((q) => (
                    <button
                      key={q.label}
                      className={`tt-quick-btn ${dynLat === q.lat && dynLon === q.lon ? "active" : ""}`}
                      onClick={() => {
                        setDynBody(q.body);
                        setDynLat(q.lat);
                        setDynLon(q.lon);
                        runEngineAnalysis({
                          selection_mode: "DYNAMIC",
                          body: q.body,
                          lat: q.lat,
                          lon: q.lon,
                        });
                      }}
                    >
                      {q.label} ({q.lat}°, {q.lon}°)
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* 3. Keşif Filtreleri: Yaygın Yer vs Yeni Keşif (Novelty Mode), Taktik Ölçek ve NMS */}
            <div className="tt-control-row" style={{ borderTop: "1px solid #232d38", paddingTop: 10 }}>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <span style={{ fontSize: 9, fontFamily: "var(--font-mono)", color: "#8496a7" }}>
                  3. SAHA KEŞİF FİLTRESİ:
                </span>
                <button
                  className={`tt-quick-btn ${noveltyOnly ? "active" : ""}`}
                  onClick={() => setNoveltyOnly(true)}
                  title="Atacama, Hawaii, İzlanda, Antarktika, Devon Adası, Death Valley ve Utah MDRS klasik sahalarını maskeler."
                >
                  💎 Sadece Az Bilinen Yeni Sahaları Keşfet (Novelty Mode)
                </button>
                <button
                  className={`tt-quick-btn ${!noveltyOnly ? "active" : ""}`}
                  onClick={() => setNoveltyOnly(false)}
                  title="Yaygın/klasik NASA test sahalarını da (Atacama, Haughton, Death Valley vb.) aramaya dahil eder."
                >
                  🏛️ Yaygın / Klasik Sahaları da Dahil Et
                </button>
              </div>

              <div style={{ display: "flex", gap: 14, alignItems: "center", marginLeft: "auto" }}>
                <label style={{ fontSize: 9, fontFamily: "var(--font-mono)", color: "#9de9d3" }}>
                  Taktik Üs Ölçeği: <b>{(radiusKm * 2).toFixed(0)}x{(radiusKm * 2).toFixed(0)} km</b> (r={radiusKm}km)
                  <input
                    type="range"
                    min="2.5"
                    max="20"
                    step="2.5"
                    value={radiusKm}
                    onChange={(e) => setRadiusKm(Number(e.target.value))}
                    style={{ width: 90, marginLeft: 8, accentColor: "#8ce9d0" }}
                  />
                </label>

                <label style={{ fontSize: 9, fontFamily: "var(--font-mono)", color: "#9de9d3" }}>
                  Min. Havza Ayrımı (NMS): <b>{minSepKm} km</b>
                  <input
                    type="range"
                    min="150"
                    max="1000"
                    step="50"
                    value={minSepKm}
                    onChange={(e) => setMinSepKm(Number(e.target.value))}
                    style={{ width: 90, marginLeft: 8, accentColor: "#8ce9d0" }}
                  />
                </label>
              </div>
            </div>

            {/* Manuel Jüri 6B Ağırlık Kaydırıcıları */}
            {missionMode === "CUSTOM" && (
              <div className="tt-weights-grid">
                {[
                  "1. Litoloji & Bazalt",
                  "2. Düzlük & İniş (<5°)",
                  "3. Termal & Güneş",
                  "4. Su Buzu (ISRU)",
                  "5. Rüzgar & Toz",
                  "6. Lav Tüpü & Mağara",
                ].map((lbl, idx) => (
                  <div className="tt-weight-item" key={lbl}>
                    <div style={{ display: "flex", justifyContent: "space-between", color: "#b9d0df" }}>
                      <span>{lbl}</span>
                      <b>{customWeights[idx].toFixed(2)}</b>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={customWeights[idx]}
                      onChange={(e) => {
                        const next = [...customWeights];
                        next[idx] = Number(e.target.value);
                        setCustomWeights(next);
                      }}
                      onMouseUp={() => runEngineAnalysis({ custom_weights: customWeights })}
                    />
                  </div>
                ))}
              </div>
            )}
          </section>

          {/* 2. ÜST TELEMETRİ METRİK KARTLARI (4 KART) */}
          {targetProfile && activeCand && baseFeas && (
            <>
              <div className="tt-telemetry-grid">
                <div className="tt-metric-card">
                  <small>🏛️ KALICI {targetProfile.body.toUpperCase()} ÜSSÜ FİZİBİLİTESİ</small>
                  <strong>%{baseFeas.overall_base_viability_pct} Uygunluk</strong>
                  <span>
                    {targetProfile.lat}°, {targetProfile.lon}° ({(radiusKm * 2).toFixed(0)}x{(radiusKm * 2).toFixed(0)} km)
                  </span>
                </div>
                <div className="tt-metric-card">
                  <small>🥇 #{selectedCandIdx + 1} KEŞFEDİLEN DÜNYA ANALOGU</small>
                  <strong>{activeCand.region_name.split("[")[0].trim()}</strong>
                  <span>
                    %{activeCand.total_match_pct} Uyum (Ön Eleme #{activeCand.pre_screen_rank} ➔ #{selectedCandIdx + 1})
                  </span>
                </div>
                <div className="tt-metric-card">
                  <small>⛰️ 2B TOPOGRAFİK KESİT & İNİŞ (%30)</small>
                  <strong>%{activeCand.topographic_match_pct} Kesit Uyumu</strong>
                  <span>
                    &lt;5° Güvenli: %{activeCand.cross_section?.safe_landing_pct} (Üs: {activeCand.cross_section?.best_base_km} km)
                  </span>
                </div>
                <div className="tt-metric-card">
                  <small>📡 CANLI NASA POWER & SI KAPSAMI</small>
                  <strong>
                    {activeCand.nasa_power_climate?.mean_surface_temp_c}°C | {activeCand.nasa_power_climate?.precipitation_mm_day} mm/g
                  </strong>
                  <span>
                    %{activeCand.data_coverage_pct} SI Kanıt Kapsamı · Fiziksel: %{activeCand.measured_physical_similarity_pct}
                  </span>
                </div>
              </div>

              <div className="tt-base-banner">
                <b>🏗️ SEÇİLEN HEDEF İÇİN KALICI ÜS MİMARİSİ ({targetProfile.name}):</b>
                <br />• <b>Önerilen Habitat Yapısı:</b> {baseFeas.recommended_habitat_architecture} &nbsp;|&nbsp;{" "}
                <b>Radyasyon Zırhlama:</b> {baseFeas.radiation_shielding_protocol}
                <br />• <b>ISRU Kaynak Çıkarma:</b> {baseFeas.isru_primary_strategy} &nbsp;|&nbsp;{" "}
                <b>Enerji & Aydınlanma:</b> {baseFeas.power_and_illumination_architecture}
              </div>
            </>
          )}

          {/* 3. 3 BOYUTLU KÜRE ÇALIŞMA ALANI VE SAĞ İNCELEYİCİ (INSPECTOR) */}
          <div className="scientific-workspace">
            <section className="exploration-column">
              <div className="scene-frame">
                <div className="scene-topline">
                  <div className="scene-tabs">
                    <button
                      className={split ? "active" : ""}
                      onClick={() => {
                        setSplit(true);
                        setFocus((n) => n + 1);
                      }}
                    >
                      <SplitSquareHorizontal size={14} />
                      İkili 3B Küre (Dünya ↔ {activeTargetObj.body})
                    </button>
                    <button
                      className={!split && body === "Earth" ? "active" : ""}
                      onClick={() => {
                        setSplit(false);
                        setBody("Earth");
                      }}
                    >
                      <Earth size={14} />
                      3B Dünya Küresi
                    </button>
                    <button
                      className={!split && body === "Moon" ? "active" : ""}
                      onClick={() => {
                        setSplit(false);
                        setBody("Moon");
                        setDynBody("Moon");
                      }}
                    >
                      <Moon size={14} />
                      3B Ay Küresi (Tıkla & Seç)
                    </button>
                    <button
                      className={!split && body === "Mars" ? "active" : ""}
                      onClick={() => {
                        setSplit(false);
                        setBody("Mars");
                        setDynBody("Mars");
                      }}
                    >
                      <Orbit size={14} />
                      3B Mars Küresi (Tıkla & Seç)
                    </button>
                  </div>
                  <button className="scene-source" onClick={() => setModal("data")}>
                    <Database size={12} /> Sensör Kaynakları <ArrowUpRight size={12} />
                  </button>
                </div>

                <div className={split ? "viewport-grid split" : "viewport-grid"}>
                  <SceneBoundary>
                    <Suspense fallback={<div className="canvas-loading">3B Gezegen Motoru Yükleniyor…</div>}>
                      <PlanetViewport
                        body={split ? "Earth" : body}
                        site={activeSiteObj}
                        target={activeTargetObj}
                        imagery={imagery}
                        grid={grid}
                        markers={markers}
                        onSelect={handleMarkerSelect}
                        onGlobeCoordinateClick={handleGlobeCoordinateClick}
                        customMarkers={(split ? "Earth" : body) === "Earth" ? earthGlobeMarkers : targetGlobeMarkers}
                        splitBridgeCoords={
                          compSite && candidates[0]
                            ? {
                                fromLat: candidates[0].lat,
                                fromLon: candidates[0].lon,
                                toLat: compSite.lat,
                                toLon: compSite.lon,
                              }
                            : null
                        }
                        focus={focus}
                        sync={sync}
                        cameraLink={cameraLink}
                      />
                      {split && (
                        <PlanetViewport
                          body={activeTargetObj.body}
                          site={activeSiteObj}
                          target={activeTargetObj}
                          imagery={imagery}
                          grid={grid}
                          markers={markers}
                          onSelect={handleMarkerSelect}
                          onGlobeCoordinateClick={handleGlobeCoordinateClick}
                          customMarkers={targetGlobeMarkers}
                          focus={focus}
                          sync={sync}
                          cameraLink={cameraLink}
                        />
                      )}
                    </Suspense>
                  </SceneBoundary>

                  {split && (
                    <button
                      className={`sync-button ${sync ? "enabled" : ""}`}
                      onClick={() => setSync(!sync)}
                    >
                      <Link2 size={13} />
                      {sync ? "3B Küre Dönüşü Senkronize" : "3B Küre Dönüşünü Bağla"}
                    </button>
                  )}
                </div>

                <div className="scene-footer">
                  <span>
                    <span className="tiny-dot" />
                    11,469 DÜNYA HÜCRESİ TARANDI <span className="footer-separator">/</span> 8 ADAY CANLI UYDU İLE ÇARPIŞTIRILDI
                  </span>
                  <span>
                    Döndürmek için sürükleyin · Yakınlaştırmak için kaydırın · Koordinat seçmek için küreye tıklayın
                  </span>
                </div>
              </div>

              {/* 4. APP.PY'DEKİ 5 ANA KARAR DESTEK SEKMESİ */}
              <section className="candidate-section" id="analysis-tabs-anchor">
                <div className="section-heading">
                  <div className="section-tabs" style={{ gap: 14, flexWrap: "wrap" }}>
                    {[
                      { id: 1, title: "🌍 1. Küresel Keşif, 8'li Yarış & HUD Uydu" },
                      { id: 2, title: "📐 2. 6B Radar, 2B Kesit (%30) & SI Kanıt" },
                      { id: 3, title: "🧬 3. Süreçler, Astrobiyoloji & Operasyon" },
                      { id: 4, title: "⚠️ 4. XAI Kırılma, Cihaz Testi & TRL" },
                      { id: 5, title: "📄 5. Görev Dosyası (.md / .json / .csv)" },
                    ].map((t) => (
                      <button
                        key={t.id}
                        className={activeMainTab === t.id ? "active" : ""}
                        onClick={() => setActiveMainTab(t.id)}
                      >
                        {t.title}
                      </button>
                    ))}
                  </div>
                </div>

                {/* ==================== SEKME 1: 8'Lİ CANLI YARIŞ & HUD UYDU ==================== */}
                {activeMainTab === 1 && activeCand && targetProfile && (
                  <div style={{ marginTop: 14 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
                      <div style={{ display: "flex", gap: 8 }}>
                        {candidates.map((c, i) => (
                          <button
                            key={c.grid_index}
                            className={`tt-quick-btn ${selectedCandIdx === i ? "active" : ""}`}
                            onClick={() => setSelectedCandIdx(i)}
                          >
                            #{i + 1} {c.region_name.split("[")[0].trim()} (%{c.total_match_pct}){" "}
                            {c.is_novel_discovery ? "💎 Yeni Keşif" : "🏛️ Klasik"}
                          </button>
                        ))}
                      </div>
                      <label style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 10, color: "#9de9d3", cursor: "pointer" }}>
                        <input
                          type="checkbox"
                          checked={showHud}
                          onChange={(e) => setShowHud(e.target.checked)}
                        />
                        🎯 Mühendislik HUD Nişangahı & 1.25 km Mikro-Hedef Kutusu
                      </label>
                    </div>

                    <div className="tt-hud-grid">
                      <div className="tt-hud-box">
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontSize: 9.5 }}>
                          <b style={{ color: "#00d4ff" }}>
                            HEDEF: {targetProfile.name} ({(radiusKm * 2).toFixed(0)}x{(radiusKm * 2).toFixed(0)} km)
                          </b>
                          <span style={{ color: "#8496a7" }}>
                            {targetProfile.lat}°, {targetProfile.lon}°
                          </span>
                        </div>
                        <img
                          src={showHud ? targetProfile.hud_data_url : targetProfile.rgb_data_url}
                          alt={targetProfile.name}
                        />
                        <div style={{ marginTop: 8, fontSize: 9, color: "#9ab0c1" }}>
                          • <b>Ölçüm Kaynağı:</b> {targetProfile.physical_measurements?.measurement_source}
                          <br />• <b>Hedef Eğim / Rölyef:</b> {targetProfile.cross_section?.mean_slope_deg}° (±0.3°) |{" "}
                          {targetProfile.cross_section?.relief_m} m | <b>Sıcaklık:</b> {targetProfile.mean_temp_c}°C
                        </div>
                      </div>

                      <div className="tt-hud-box">
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontSize: 9.5 }}>
                          <b style={{ color: "#ff9f1c" }}>
                            #{selectedCandIdx + 1} ANALOG: {activeCand.region_name}
                          </b>
                          <span className={activeCand.is_novel_discovery ? "tt-badge-novel" : "tt-badge-classic"}>
                            {activeCand.is_novel_discovery ? "💎 AZ BİLİNEN YENİ KEŞİF" : "🏛️ YAYGIN KLASİK SAHA"}
                          </span>
                        </div>
                        <img
                          src={showHud ? activeCand.hud_data_url : activeCand.rgb_data_url}
                          alt={activeCand.region_name}
                        />
                        <div style={{ marginTop: 8, fontSize: 9, color: "#9ab0c1" }}>
                          • <b>🎯 Mikro-Ölçek ({activeCand.micro_scale_pinpoint?.patch_scale_km}):</b>{" "}
                          <code>
                            {activeCand.micro_scale_pinpoint?.micro_lat}°, {activeCand.micro_scale_pinpoint?.micro_lon}°
                          </code>{" "}
                          — {activeCand.micro_scale_pinpoint?.micro_feature_type} (
                          <b style={{ color: "#9de9d3" }}>%{activeCand.micro_scale_pinpoint?.micro_suitability_pct}</b>)
                          <br />• <b>Canlı NASA POWER:</b> {activeCand.nasa_power_climate?.mean_surface_temp_c}°C | Yağış:{" "}
                          {activeCand.nasa_power_climate?.precipitation_mm_day} mm/gün | Nem: %
                          {activeCand.nasa_power_climate?.relative_humidity_pct} | Güneş:{" "}
                          {activeCand.nasa_power_climate?.solar_irradiance_kwh_m2_day} kWh/m²/g
                        </div>
                      </div>
                    </div>

                    {/* Kademe 1 -> Kademe 2: 8 Adayın Canlı Görüntü Kapışması Tablosu */}
                    <div style={{ marginTop: 18 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                        <h4 style={{ fontSize: 11, color: "#d5e5f0" }}>
                          🏆 Kademe 1 ➔ Kademe 2: Ön Elemeden Gelen 8 Adayın Canlı Uydu Görüntüsü & SRTM DEM Kapışması (Re-Ranking)
                        </h4>
                        <label className="search-input" style={{ maxWidth: 240, height: 27 }}>
                          <Search size={13} />
                          <input
                            placeholder="Havza veya ülke ara…"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                          />
                        </label>
                      </div>
                      <div className="evidence-table-wrap">
                        <table className="evidence-table">
                          <thead>
                            <tr>
                              <th>Final Sıra</th>
                              <th>Ön Eleme Sırası</th>
                              <th>Keşfedilen Dünya Bölgesi [Jeolojik Kuşak]</th>
                              <th>Koordinat</th>
                              <th>2B Topo Kesit (%30)</th>
                              <th>Görsel AI (%20)</th>
                              <th>Nihai Hibrit Skor</th>
                            </tr>
                          </thead>
                          <tbody>
                            {filteredPool.map((item, idx) => {
                              const shift = item.pre_rank - (idx + 1);
                              const shiftBadge =
                                shift > 0 ? `🔼 +${shift} Yükseldi` : shift < 0 ? `🔽 ${shift} Düştü` : "➖ Korudu";
                              return (
                                <tr
                                  key={`${item.lat}_${item.lon}`}
                                  style={{
                                    cursor: idx < 3 ? "pointer" : "default",
                                    background: idx === selectedCandIdx ? "rgba(157, 233, 211, 0.08)" : "transparent",
                                  }}
                                  onClick={() => {
                                    if (idx < 3) setSelectedCandIdx(idx);
                                  }}
                                >
                                  <td>
                                    <b>#{idx + 1}</b>
                                  </td>
                                  <td>
                                    #{item.pre_rank} ({shiftBadge})
                                  </td>
                                  <td>{item.region_name}</td>
                                  <td>
                                    {item.lat}°, {item.lon}°
                                  </td>
                                  <td>%{item.topographic_match_pct}</td>
                                  <td>%{item.visual_hybrid_pct}</td>
                                  <td>
                                    <b style={{ color: "#9de9d3" }}>%{item.final_score_pct}</b>
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                )}

                {/* ==================== SEKME 2: 6B RADAR, 2B KESİT, SI KANIT & İLK 3 KARNE ==================== */}
                {activeMainTab === 2 && activeCand && targetProfile && (
                  <div style={{ marginTop: 14 }}>
                    <div className="tt-chart-grid">
                      <div className="tt-chart-box">
                        <h4 style={{ fontSize: 11, color: "#d5e5f0", marginBottom: 8 }}>
                          🕸️ 6 Boyutlu Jeofiziksel Parmak İzi Uyumu (%35 Ağırlık)
                        </h4>
                        <RadarChart6D
                          dimNames={apiData?.dimension_names || []}
                          targetName={targetProfile.name}
                          targetVec={targetProfile.features_6d || [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}
                          candidates={candidates}
                          compSite={compSite}
                        />
                      </div>

                      <div className="tt-chart-box">
                        <h4 style={{ fontSize: 11, color: "#d5e5f0", marginBottom: 8 }}>
                          ⛰️ 2B Gerçek SRTM 90m DEM vs LOLA/MOLA Topografik Kesit (%30 Ağırlık)
                        </h4>
                        <CrossSectionChart2D target={targetProfile} cand={activeCand} />
                      </div>
                    </div>

                    {/* Bilimsel Fiziksel Ölçüm Kanıt Tablosu */}
                    <div style={{ marginTop: 18 }}>
                      <h4 style={{ fontSize: 11.5, color: "#d5e5f0" }}>
                        📏 Bilimsel Fiziksel Ölçüm Kanıt Tablosu (SI Birimleri, ± Hata Payları & %{activeCand.data_coverage_pct} Veri Kapsamı)
                      </h4>
                      <p style={{ fontSize: 9.5, color: "#8ba1b2", marginTop: 4 }}>
                        Seçili Aday: <b>{activeCand.region_name}</b> — Eşli Veri Kapsamı:{" "}
                        <b style={{ color: "#9de9d3" }}>%{activeCand.data_coverage_pct}</b> | Ölçülmüş Fiziksel Benzerlik:{" "}
                        <b style={{ color: "#ffbe0b" }}>%{activeCand.measured_physical_similarity_pct}</b>{" "}
                        <i>(Eksik gözlemler asla tahmin edilmez / sıfır uydurma veri)</i>
                      </p>
                      <div className="evidence-table-wrap">
                        <table className="evidence-table">
                          <thead>
                            <tr>
                              <th>Fiziksel Ölçüt (Criterion)</th>
                              <th>Dünya Sahası Ölçümü (±Hata)</th>
                              <th>{targetProfile.body} Hedef Ölçümü (±Hata)</th>
                              <th>SI Birimi</th>
                              <th>Fiziksel Benzerlik</th>
                              <th>Ağırlık</th>
                              <th>Kanıt Seviyesi</th>
                              <th>Sensör & Veri Kaynağı</th>
                            </tr>
                          </thead>
                          <tbody>
                            {(activeCand.physical_evidence_table || []).map((r: any) => (
                              <tr key={r.criterion}>
                                <td>{r.criterion}</td>
                                <td>{r.earth_measured}</td>
                                <td>{r.target_measured}</td>
                                <td>{r.unit}</td>
                                <td>
                                  <b style={{ color: "#9de9d3" }}>
                                    {r.similarity_pct !== null ? `%${r.similarity_pct}` : "Missing"}
                                  </b>
                                </td>
                                <td>{r.weight}</td>
                                <td>{r.provenance_tier}</td>
                                <td>{r.source}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {/* Keşfedilen İlk 3 Dünya Analogunun Tam Mühendislik Karnesi */}
                    <div style={{ marginTop: 18 }}>
                      <h4 style={{ fontSize: 11.5, color: "#d5e5f0", marginBottom: 10 }}>
                        🏅 Keşfedilen İlk 3 Dünya Analogunun Tam Karşılaştırmalı Mühendislik Karnesi
                      </h4>
                      <div className="candidate-grid">
                        {candidates.map((cand, i) => (
                          <div
                            key={cand.grid_index}
                            className={`candidate-card ${selectedCandIdx === i ? "active" : ""}`}
                            onClick={() => setSelectedCandIdx(i)}
                            style={{ cursor: "pointer" }}
                          >
                            <div className="candidate-card-top">
                              <span className={cand.is_novel_discovery ? "tt-badge-novel" : "tt-badge-classic"}>
                                {cand.is_novel_discovery ? "💎 YENİ KEŞİF" : "🏛️ KLASİK SAHA"}
                              </span>
                              <span className="candidate-index">#{i + 1}</span>
                            </div>
                            <h3>{cand.region_name}</h3>
                            <p style={{ color: "#00f5d4", fontSize: 8.5, whiteSpace: "normal", marginTop: 4 }}>
                              {cand.strategic_role_badge}
                            </p>
                            <div style={{ fontSize: 9, color: "#b8cad8", lineHeight: 1.75, marginTop: 10 }}>
                              • <b>Toplam Bilimsel Uyum: %{cand.total_match_pct}</b> (Ön Eleme #{cand.pre_screen_rank} ➔ #{i + 1})
                              <br />• <b>6B Vektör (%35):</b> %{cand.parametric_6d_pct} | <b>2B Topo (%30):</b> %{cand.topographic_match_pct}
                              <br />• <b>Görsel AI (%20):</b> %{cand.visual_hybrid_pct} | <b>NASA POWER (%15):</b> %{cand.live_env_verification_pct}
                              <br />• <b>Mikro-Hedef ({cand.micro_scale_pinpoint?.patch_scale_km}):</b>{" "}
                              <code>
                                {cand.micro_scale_pinpoint?.micro_lat}°, {cand.micro_scale_pinpoint?.micro_lon}°
                              </code>{" "}
                              (%{cand.micro_scale_pinpoint?.micro_suitability_pct})
                              <br />• <b>Aktif Süreç (%{cand.process_analogy?.process_fidelity_pct}):</b>{" "}
                              {cand.process_analogy?.dominant_earth_process}
                              <br />• <b>Biyo-İmza Korunumu:</b> %{cand.astrobiology_profile?.biosignature_preservation_pct}
                              <br />• <b>Operasyon:</b> İniş (&lt;5°): %{cand.operations_readiness?.landing_safety_envelope_pct} | Rover: %
                              {cand.operations_readiness?.rover_mobility_trafficability_pct}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* ==================== SEKME 3: SÜREÇLER, ASTROBİYOLOJİ VE ROBOTİK OPERASYON ==================== */}
                {activeMainTab === 3 && activeCand && (
                  <div className="tt-chart-grid" style={{ marginTop: 14 }}>
                    <div className="tt-chart-box">
                      <h4 style={{ fontSize: 11, color: "#d5e5f0", marginBottom: 12 }}>
                        🌋 Jeofiziksel Süreç Yoğunluğu (Turuncu) & Operasyonel Hazırlık (Turkuaz)
                      </h4>
                      {Object.entries(activeCand.process_analogy?.process_breakdown || {}).map(([k, v]: any) => (
                        <div className="criterion-chart-row" key={k} style={{ gridTemplateColumns: "210px 1fr 48px" }}>
                          <span>{k}</span>
                          <div>
                            <i style={{ width: `${v}%`, background: "#ff9f1c" }} />
                          </div>
                          <b>%{v}</b>
                        </div>
                      ))}
                      {[
                        ["İniş Güvenliği (<5° Eğim Zarfı)", activeCand.operations_readiness?.landing_safety_envelope_pct],
                        ["Rover Mobilite & Trafiklenebilirlik", activeCand.operations_readiness?.rover_mobility_trafficability_pct],
                        ["TRN Optik Navigasyon Uygunluğu", activeCand.operations_readiness?.trn_optical_navigation_pct],
                        ["3B Regolit Sinterleme İnşaat", activeCand.operations_readiness?.regolith_construction_3d_pct],
                      ].map(([lbl, val]: any) => (
                        <div className="criterion-chart-row" key={lbl} style={{ gridTemplateColumns: "210px 1fr 48px" }}>
                          <span>{lbl}</span>
                          <div>
                            <i style={{ width: `${val}%`, background: "#00f5d4" }} />
                          </div>
                          <b>%{val}</b>
                        </div>
                      ))}
                    </div>

                    <div className="tt-chart-box">
                      <h4 style={{ fontSize: 11, color: "#d5e5f0", marginBottom: 10 }}>
                        🧬 Astrobiyoloji, Ekstremofil Loncası & Biyo-İmza Analizi
                      </h4>
                      <div className="tt-split-box" style={{ marginBottom: 12 }}>
                        <b>Baskın Ekstremofil Analog Loncası:</b>
                        <br />
                        {activeCand.astrobiology_profile?.extremophile_guild}
                      </div>
                      <div style={{ fontSize: 10.5, lineHeight: 1.85, color: "#c2d4e2" }}>
                        • <b>Biyo-İmza Koruma Sadakati:</b>{" "}
                        <b style={{ color: "#9de9d3" }}>%{activeCand.astrobiology_profile?.biosignature_preservation_pct}</b>
                        <br />• <b>Güneş / UV Radyasyon Stres İndeksi:</b> %{activeCand.astrobiology_profile?.uv_radiation_stress_pct} (
                        {activeCand.astrobiology_profile?.solar_irradiance_kwh_m2_day} kWh/m²/gün)
                        <br />• <b>Gezegensel Astrobiyoloji Görev Bağlantısı:</b>{" "}
                        {activeCand.astrobiology_profile?.astrobiology_mission_relevance}
                        <hr style={{ margin: "12px 0", borderColor: "#253442" }} />
                        <b>🌪️ Süreç Eşleşmesi (Processes as well as Places):</b>
                        <br />• <b>Hedef Gezegen Süreci:</b> {activeCand.process_analogy?.target_planetary_process}
                        <br />• <b>Dünya Sahası Aktif Süreci:</b> {activeCand.process_analogy?.dominant_earth_process} (
                        <b style={{ color: "#ffbe0b" }}>%{activeCand.process_analogy?.process_fidelity_pct} Uyum</b>)
                      </div>
                    </div>
                  </div>
                )}

                {/* ==================== SEKME 4: XAI KIRILMA MATRİSİ, CİHAZ TESTİ & TRL YOL HARİTASI ==================== */}
                {activeMainTab === 4 && xaiData && (
                  <div style={{ marginTop: 14 }}>
                    <h4 style={{ fontSize: 11.5, color: "#d5e5f0", marginBottom: 10 }}>
                      ⚠️ Benzerlik Nerede Kırılıyor? (Earth's Limits as an Analog & Mühendislik Telafi Protokolleri)
                    </h4>
                    {(xaiData.breakdown_matrix || []).map((brk: any) => (
                      <div className="tt-xai-card" key={brk.parameter}>
                        <b style={{ color: "#ff8589" }}>
                          🚨 {brk.parameter} — Kırılma Şiddeti: {brk.severity}
                        </b>
                        <br />
                        <span style={{ color: "#9de9d3" }}>
                          <b>Hedef Değer:</b> {brk.target_value} &nbsp;|&nbsp; <b>Dünya Sahası:</b> {brk.earth_value}
                        </span>
                        <br />
                        <b>Fiziksel Etki:</b> {brk.impact}
                        <br />
                        <span style={{ color: "#00f5d4" }}>
                          <b>🛠️ Mühendislik Telafi Protokolü (Mitigation):</b> {brk.engineering_mitigation}
                        </span>
                      </div>
                    ))}

                    <h4 style={{ fontSize: 11.5, color: "#d5e5f0", margin: "18px 0 8px" }}>
                      🔬 NASA Cihaz Test Yatağı ve Risk Azaltma Matrisi (Connect to Exploration)
                    </h4>
                    <div className="evidence-table-wrap">
                      <table className="evidence-table">
                        <thead>
                          <tr>
                            <th>Test Edilecek NASA Cihazı / Sistemi</th>
                            <th>Görev Mirası (Heritage)</th>
                            <th>Saha Test Protokolü</th>
                            <th>Azaltılan Görev Riski</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(xaiData.instrument_testbed_plan || []).map((inst: any) => (
                            <tr key={inst.instrument_system}>
                              <td>{inst.instrument_system}</td>
                              <td>{inst.target_mission_heritage}</td>
                              <td>{inst.field_test_protocol}</td>
                              <td>{inst.risk_reduction}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    <h4 style={{ fontSize: 11.5, color: "#d5e5f0", margin: "18px 0 8px" }}>
                      🗓️ TRL 4 ➔ TRL 8 Aşamalı Çoklu Saha Test Kampanyası Yol Haritası
                    </h4>
                    <div className="evidence-table-wrap">
                      <table className="evidence-table">
                        <thead>
                          <tr>
                            <th>Kampanya Fazı (TRL)</th>
                            <th>Atanan Dünya Analog Sahaları</th>
                            <th>Mühendislik ve Bilimsel Doğrulama Hedefi</th>
                          </tr>
                        </thead>
                        <tbody>
                          {roadmap.map((rm: any) => (
                            <tr key={rm.phase}>
                              <td>
                                <b>{rm.phase}</b>
                              </td>
                              <td>{rm.assigned_sites}</td>
                              <td>{rm.objective}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    <div className="tt-hud-grid" style={{ marginTop: 14 }}>
                      {compSite && (
                        <div className="tt-split-box">
                          <h4 style={{ color: "#00f5d4", fontSize: 11, marginBottom: 6 }}>
                            🧩 İkili Saha Reçetesi (Complementary Split-Site)
                          </h4>
                          Birincil saha olan <b>{candidates[0]?.region_name}</b>,{" "}
                          <i>'{compSite.weakest_dimension_name}'</i> boyutunda (
                          <b>%{compSite.primary_site_score_in_dim_pct}</b>) kısmi sapma göstermektedir.
                          <br />
                          <b>Önerilen Tamamlayıcı 2. Test Sahası:</b> <b>{compSite.complementary_region_name}</b> (
                          <code>
                            {compSite.lat}°, {compSite.lon}°
                          </code>
                          ) — Tamamlama Uyumu: <b>%{compSite.complementary_dim_match_pct}</b>
                        </div>
                      )}
                      {bridge && (
                        <div className="tt-split-box" style={{ borderLeftColor: "#ffbe0b" }}>
                          <h4 style={{ color: "#ffbe0b", fontSize: 11, marginBottom: 6 }}>
                            🚀 {bridge.source_body} ↔ {bridge.bridge_body} Çapraz Mimari Köprüsü
                          </h4>
                          Seçilen <b>{bridge.source_body}</b> hedefi için doğrulanan yaşam alanı ve rover mimarisi,{" "}
                          <b>%{bridge.architectural_transfer_pct}</b> mühendislik uyumuyla doğrudan{" "}
                          <b>
                            {bridge.bridge_body} — {bridge.bridge_name}
                          </b>{" "}
                          (<code>{bridge.lat}°, {bridge.lon}°</code>) görevinde yeniden kullanılabilir!
                          <br />
                          <span style={{ color: "#9ab0c1" }}>{bridge.engineering_adaptation_note}</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* ==================== SEKME 5: GÖREV DOSYASI VE VERİ İNDİRME MERKEZİ ==================== */}
                {activeMainTab === 5 && activeCand && targetProfile && (
                  <div style={{ marginTop: 14 }}>
                    <h4 style={{ fontSize: 12, color: "#d5e5f0", marginBottom: 10 }}>
                      📑 Resmi NASA 2026 Görev Dosyası ve Bilimsel Veri İndirme Merkezi
                    </h4>
                    <div className="export-options" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                      <button
                        onClick={() => {
                          triggerDownload(
                            mdDossier,
                            "text/markdown;charset=utf-8",
                            `NASA_2026_Mission_Dossier_${targetProfile.body}_${activeCand.lat}_${activeCand.lon}.md`,
                          );
                          setNotice("Resmi Görev Dosyası (.md) indirildi");
                        }}
                      >
                        <BookOpen size={22} />
                        <span>
                          <b>Resmi Görev Dosyası (.md)</b>
                          <small>Tam XAI kırılma matrisi, ilk 3 aday ve TRL yol haritası.</small>
                        </span>
                        <ArrowDownToLine size={18} />
                      </button>

                      <button
                        onClick={() => {
                          const cleanJson = JSON.stringify(apiData, null, 2);
                          triggerDownload(
                            cleanJson,
                            "application/json;charset=utf-8",
                            `NASA_2026_Telemetry_${targetProfile.body}_${activeCand.lat}_${activeCand.lon}.json`,
                          );
                          setNotice("Ham Telemetri Paketi (.json) indirildi");
                        }}
                      >
                        <FileJson size={22} />
                        <span>
                          <b>Ham Telemetri Paketi (.json)</b>
                          <small>6B vektörler, SRTM DEM kesiti ve NASA POWER ölçümleri.</small>
                        </span>
                        <ArrowDownToLine size={18} />
                      </button>

                      <button
                        onClick={() => {
                          const rows = activeCand.physical_evidence_table || [];
                          const csvLines = [
                            ["Criterion", "Earth_Measured", "Target_Measured", "SI_Unit", "Similarity_Pct", "Weight", "Tier", "Source"].join(","),
                            ...rows.map((r: any) =>
                              [
                                `"${r.criterion}"`,
                                `"${r.earth_measured}"`,
                                `"${r.target_measured}"`,
                                `"${r.unit}"`,
                                r.similarity_pct,
                                r.weight,
                                `"${r.provenance_tier}"`,
                                `"${r.source}"`,
                              ].join(","),
                            ),
                          ].join("\r\n");
                          triggerDownload(
                            csvLines,
                            "text/csv;charset=utf-8",
                            `NASA_2026_Physical_Evidence_${activeCand.lat}_${activeCand.lon}.csv`,
                          );
                          setNotice("SI Fiziksel Kanıt Tablosu (.csv) indirildi");
                        }}
                      >
                        <Database size={22} />
                        <span>
                          <b>Fiziksel SI Kanıt Tablosu (.csv)</b>
                          <small>Excel/Python uyumlu SI birimleri ve ± hata payları.</small>
                        </span>
                        <ArrowDownToLine size={18} />
                      </button>

                      <button
                        onClick={() => {
                          const htmlDoc = `<!doctype html><html><head><meta charset="utf-8"><title>TERRA-TWIN NASA 2026 Report</title><style>body{font-family:system-ui,sans-serif;max-width:960px;margin:40px auto;line-height:1.65;color:#111;padding:20px}pre{white-space:pre-wrap;background:#f4f6f8;padding:18px;border-radius:8px}</style></head><body><h1>TERRA-TWIN — NASA 2026 Mission Dossier</h1><pre>${mdDossier.replace(/</g, "&lt;")}</pre></body></html>`;
                          triggerDownload(
                            htmlDoc,
                            "text/html;charset=utf-8",
                            `NASA_2026_Printable_Report_${targetProfile.body}.html`,
                          );
                          setNotice("Yazdırılabilir HTML/PDF Raporu indirildi");
                        }}
                      >
                        <FlaskConical size={22} />
                        <span>
                          <b>Yazdırılabilir Rapor (.html / PDF)</b>
                          <small>Tarayıcıda açıp Ctrl+P ile PDF olarak kaydedebilirsiniz.</small>
                        </span>
                        <ArrowDownToLine size={18} />
                      </button>
                    </div>

                    <div
                      style={{
                        marginTop: 16,
                        background: "#0e141c",
                        border: "1px solid #2a3948",
                        borderRadius: 6,
                        padding: 16,
                        maxHeight: 420,
                        overflowY: "auto",
                        fontFamily: "var(--font-mono)",
                        fontSize: 10,
                        lineHeight: 1.75,
                        whiteSpace: "pre-wrap",
                        color: "#c6d7e4",
                      }}
                    >
                      {mdDossier}
                    </div>
                  </div>
                )}
              </section>
            </section>

            {/* SAĞ İNCELEYİCİ (SITE INSPECTOR) */}
            <aside className="inspector">
              <div className="inspector-top">
                <span>
                  <ScanLine size={13} />
                  CANLI ADAY İNCELEYİCİSİ
                </span>
                <span className="earth-label">
                  #{selectedCandIdx + 1} / {candidates.length}
                </span>
              </div>

              {activeCand && targetProfile ? (
                <>
                  <div
                    className="site-image"
                    style={{
                      backgroundImage: `url(${showHud ? activeCand.hud_data_url : activeCand.rgb_data_url})`,
                      backgroundSize: "cover",
                      backgroundPosition: "center",
                    }}
                  >
                    <span>
                      <MapPin size={11} />
                      {activeCand.lat.toFixed(2)}°, {activeCand.lon.toFixed(2)}°
                    </span>
                    <span className="image-credit">
                      {activeCand.is_novel_discovery ? "💎 NOVEL DISCOVERY" : "🏛️ CLASSIC SITE"}
                    </span>
                  </div>

                  <div className="site-title">
                    <div className="eyebrow">{activeCand.strategic_role_badge}</div>
                    <h2 style={{ fontSize: 18 }}>{activeCand.region_name.split("[")[0].trim()}</h2>
                    <p>
                      <MapPin size={12} />
                      {activeCand.region_name.includes("[")
                        ? activeCand.region_name.split("[")[1].replace("]", "")
                        : "Terrestrial Analog Basin"}
                    </p>
                    <div className="site-tags">
                      <span>SRTM Eğim: {activeCand.cross_section?.mean_slope_deg}°</span>
                      <span>Rölyef: {activeCand.cross_section?.relief_m} m</span>
                    </div>
                  </div>

                  <div className="inspector-content">
                    <div className="inspector-section">
                      <div className="inspector-section-heading">
                        <h4>BİLİMSEL HİBRİT UYUM SKORU</h4>
                        <button onClick={() => setModal("method")}>
                          <CircleHelp size={13} />
                        </button>
                      </div>
                      <div className="score-block">
                        <div
                          className="score-ring"
                          style={{ "--score": `${activeCand.total_match_pct}%` } as React.CSSProperties}
                        >
                          <span>
                            {activeCand.total_match_pct.toFixed(1)}
                            <small>/ 100</small>
                          </span>
                        </div>
                        <div>
                          <b>Fiziksel Uyum: %{activeCand.measured_physical_similarity_pct}</b>
                          <p>Hedef: {targetProfile.name.slice(0, 26)}</p>
                          <span className="limited-badge" style={{ color: "#8ce9d0", borderColor: "#3e7c6a" }}>
                            TIER-1 SENSÖR DOĞRULAMALI
                          </span>
                        </div>
                      </div>

                      <div className="inspector-coverage">
                        <span>Eşli Fiziksel Veri Kapsamı (SI Coverage)</span>
                        <b>%{activeCand.data_coverage_pct}</b>
                      </div>
                      <div className="progress-track">
                        <i style={{ width: `${activeCand.data_coverage_pct}%` }} />
                      </div>
                    </div>

                    <div className="inspector-section">
                      <h4>MİKRO-HEDEF (1.25 x 1.25 KM) & NASA POWER</h4>
                      <div className="measurement">
                        <Crosshair size={15} />
                        <span>
                          Mikro-Koordinat
                          <small>{activeCand.micro_scale_pinpoint?.micro_feature_type}</small>
                        </span>
                        <b>
                          {activeCand.micro_scale_pinpoint?.micro_lat}°, {activeCand.micro_scale_pinpoint?.micro_lon}°
                        </b>
                      </div>
                      <div className="measurement">
                        <Thermometer size={15} />
                        <span>
                          Yüzey Sıcaklığı & Yağış
                          <small>NASA POWER Langley Canlı API</small>
                        </span>
                        <b>
                          {activeCand.nasa_power_climate?.mean_surface_temp_c}°C ·{" "}
                          {activeCand.nasa_power_climate?.precipitation_mm_day} mm/g
                        </b>
                      </div>
                      <div className="measurement">
                        <Mountain size={15} />
                        <span>
                          &lt;5° Güvenli İniş Zarfı
                          <small>Copernicus / NASA SRTM 90m DEM</small>
                        </span>
                        <b>%{activeCand.cross_section?.safe_landing_pct}</b>
                      </div>
                    </div>

                    <div className="inspector-section difference-section">
                      <h4>
                        <TriangleAlert size={12} />
                        İKİLİ SAHA (SPLIT-SITE) TAMAMLAYICISI
                      </h4>
                      <p>
                        {compSite
                          ? `En zayıf boyut (${compSite.weakest_dimension_name}) için önerilen 2. saha: ${compSite.complementary_region_name} (%${compSite.complementary_dim_match_pct} uyum).`
                          : "Tüm boyutlar dengeli."}
                      </p>
                    </div>
                  </div>

                  <div className="inspector-actions">
                    <button
                      className="primary-button"
                      onClick={() => {
                        setSplit(true);
                        setFocus((n) => n + 1);
                      }}
                    >
                      <SplitSquareHorizontal size={15} />
                      3B Kürede Hedef İle Kıyasla
                      <ArrowRight size={15} />
                    </button>
                    <div>
                      <button
                        className="outline-button"
                        onClick={() => {
                          setSaved((prev) => [
                            ...prev,
                            {
                              title: `${activeCand.region_name.split("[")[0].trim()} ↔ ${targetProfile.name}`,
                              presetId,
                              mode: missionMode,
                              score: activeCand.total_match_pct,
                            },
                          ]);
                          setNotice("Karşılaştırma oturuma kaydedildi.");
                        }}
                      >
                        <BookOpen size={13} /> Oturuma Kaydet ({saved.length})
                      </button>
                      <button
                        className="outline-button"
                        onClick={() => setActiveMainTab(5)}
                        title="Rapor İndir"
                      >
                        <ArrowDownToLine size={15} />
                      </button>
                    </div>
                  </div>
                </>
              ) : (
                <div style={{ padding: 24, color: "#8ba0b2" }}>Veriler yükleniyor…</div>
              )}
            </aside>
          </div>

          <footer className="main-footer">
            <span>
              <ShieldCheck size={12} />
              TERRA-TWIN v6.0 · %100 Ölçülmüş Fiziksel Sensör Verisi (SRTM 90m DEM + NASA POWER + LOLA/MOLA)
            </span>
            <button onClick={() => setModal("data")}>
              VERİ KAYNAKLARI & KANIT SEVİYELERİ <ArrowUpRight size={11} />
            </button>
            <span>2026 NASA Space Apps Challenge — Permanent Moon Base Locations and Mars</span>
          </footer>
        </main>
      </div>

      {notice && (
        <div className="toast" role="status">
          <Check size={15} />
          {notice}
          <button onClick={() => setNotice("")}>
            <X size={14} />
          </button>
        </div>
      )}

      {modal && (
        <div
          className="modal-backdrop"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) setModal(null);
          }}
        >
          <section className="modal" role="dialog" tabIndex={-1}>
            <div className="modal-heading">
              <div>
                <div className="eyebrow">TERRA-TWIN BİLİMSEL METODOLOJİ VE VERİ MERKEZİ</div>
                <h2>
                  {modal === "method"
                    ? "2-Kademeli Hibrit Fiziksel Eşleştirme Denklemi"
                    : "Bağlı NASA ve Copernicus Sensör Veri Kaynakları"}
                </h2>
              </div>
              <button className="icon-button" onClick={() => setModal(null)}>
                <X size={21} />
              </button>
            </div>

            {modal === "method" ? (
              <div className="method-content">
                <div className="method-equation">
                  S_total = 0.35 S_6D + 0.30 S_topo(SRTM) + 0.20 S_vis(CLIP) + 0.15 S_env(POWER) - Φ_gate
                  <small>%80 Fiziksel/Topografik Ölçüm + %20 Görsel Morfoloji · Sıfır Sentetik Veri</small>
                </div>
                <h3>1. Kademe 1 — 11,469 Küresel Kara Hücresi Taraması</h3>
                <p>
                  NASA GIBS küresel katmanlarından okyanuslar, bitki örtüsü ve ıslak tundralar elenerek 11,469 çıplak kaya,
                  hiper-kurak ve kriyosferik hücre indekslenir. 450 km Haversine NMS kısıtlamasıyla birbirinden bağımsız
                  en iyi 8 jeolojik havza seçilir.
                </p>
                <h3>2. Kademe 2 — Canlı SRTM 90m DEM, NASA POWER ve NASA Trek WMTS Çarpıştırması</h3>
                <p>
                  8 aday için Copernicus GLO-90 / NASA SRTM 90m Radar Yükselti API'sinden gerçek metrik kesit (metre ve derece),
                  NASA Langley POWER uydusundan canlı meteoroloji (°C, mm/gün, kWh/m²/gün) ve NASA Solar System Treks WMTS
                  servisinden 10x10 km taktik uydu yamaları çekilerek kafa kafaya yarıştırılır.
                </p>
              </div>
            ) : (
              <div className="dataset-list">
                {datasets.map((d) => (
                  <article key={d.id}>
                    <div className="dataset-heading">
                      <span className="dataset-icon">
                        <Database size={19} />
                      </span>
                      <div>
                        <h3>{d.title}</h3>
                        <p>
                          {d.provider} · {d.mission}
                        </p>
                      </div>
                      <span className="data-badge">{d.dataStatus}</span>
                    </div>
                    <p className="dataset-limit">{d.limitations}</p>
                    <div className="dataset-links">
                      <a href={d.officialDocumentationUrl} target="_blank" rel="noreferrer">
                        Resmi Dokümantasyon <ExternalLink size={11} />
                      </a>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
