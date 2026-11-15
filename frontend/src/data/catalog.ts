export type Body = "Earth" | "Moon" | "Mars";
export type DataStatus = "CACHED" | "PROCESSED" | "UNAVAILABLE";
export interface DatasetMetadata {
  id: string;
  title: string;
  provider: string;
  mission: string;
  planetaryBody: Body;
  variables: string[];
  units: string;
  spatialResolution: string;
  temporalCoverage: string;
  geographicCoverage: string;
  license: string;
  attribution: string;
  officialDocumentationUrl: string;
  dataAccessUrl: string;
  authenticationRequired: boolean;
  lastRetrievedAt: string;
  dataStatus: DataStatus;
  limitations: string;
}
export const gibsUrl =
  "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=BlueMarble_NextGeneration&STYLES=&FORMAT=image/jpeg&CRS=EPSG:4326&BBOX=-90,-180,90,180&WIDTH=2048&HEIGHT=1024";
export const retrieved = "2026-11-14";
export const datasets: DatasetMetadata[] = [
  {
    id: "gibs",
    title: "Blue Marble Next Generation",
    provider: "NASA Earthdata / GIBS",
    mission: "Terra / MODIS",
    planetaryBody: "Earth",
    variables: ["Surface imagery"],
    units: "RGB visualization; not a measurement raster",
    spatialResolution: "Rendered at 0.1758°/pixel; original composite ~500 m",
    temporalCoverage: "2004 composite; not current conditions",
    geographicCoverage: "Global",
    license: "NASA imagery-use guidelines; credit NASA",
    attribution: "NASA Earth Observatory / Reto Stöckli",
    officialDocumentationUrl: "https://nasa-gibs.github.io/gibs-api-docs/access-basics/",
    dataAccessUrl: gibsUrl,
    authenticationRequired: false,
    lastRetrievedAt: retrieved,
    dataStatus: "CACHED",
    limitations:
      "Resampled global context only. No quantitative reflectance, elevation, or temperature is inferred. Worldview is explicitly listed in the official challenge Resources; GIBS supplies its imagery.",
  },
  {
    id: "haughton",
    title: "Haughton field-site characterization",
    provider: "NASA Ames",
    mission: "Haughton Mars Project / Icebreaker",
    planetaryBody: "Earth",
    variables: ["Impact-structure diameter"],
    units: "km",
    spatialResolution: "Whole impact structure; rounded published value",
    temporalCoverage: "Published August 2014; observation epoch unspecified",
    geographicCoverage: "Devon Island, Nunavut, Canada",
    license: "NASA content-use guidelines",
    attribution: "NASA Ames / Icebreaker field team",
    officialDocumentationUrl:
      "https://www.nasa.gov/?blogs-migration=hmp-2014-field-team-leaves-for-haughton-crater-tests",
    dataAccessUrl: "https://www.nasa.gov/wp-json/wp/v2/blogs-migration/763490",
    authenticationRequired: false,
    lastRetrievedAt: retrieved,
    dataStatus: "CACHED",
    limitations:
      "NASA field note describes a 20-km structure. Other delineations report different diameters. Formal error bars and measurement method are not provided.",
  },
  {
    id: "shackleton",
    title: "Shackleton crater characterization",
    provider: "NASA / ShadowCam",
    mission: "Danuri / ShadowCam",
    planetaryBody: "Moon",
    variables: ["Crater diameter"],
    units: "km",
    spatialResolution: "Whole crater; rounded published value",
    temporalCoverage: "Published May 5, 2023; image acquisition unspecified",
    geographicCoverage: "Lunar south pole",
    license: "NASA text-use guidelines; external image rights separate",
    attribution: "NASA APOD; ShadowCam NASA/KARI/ASU",
    officialDocumentationUrl:
      "https://science.nasa.gov/image-article/apod-2023-may-5-shackleton-from-shadowcam/",
    dataAccessUrl: "https://science.nasa.gov/wp-json/wp/v2/image-article/1230816",
    authenticationRequired: false,
    lastRetrievedAt: retrieved,
    dataStatus: "CACHED",
    limitations:
      "21-km published diameter. No uncertainty given. Not a LOLA extraction. Crater-scale similarity is not a base-safety assessment.",
  },
  {
    id: "jezero",
    title: "Jezero landing-site characterization",
    provider: "NASA / JPL-Caltech",
    mission: "Mars 2020",
    planetaryBody: "Mars",
    variables: ["Crater diameter"],
    units: "km",
    spatialResolution: "Whole crater; rounded published value",
    temporalCoverage: "Published November 19, 2018; measurement epoch unspecified",
    geographicCoverage: "Jezero Crater, Mars",
    license: "NASA content-use guidelines",
    attribution: "NASA / JPL-Caltech",
    officialDocumentationUrl:
      "https://www.nasa.gov/news-release/nasa-announces-landing-site-for-mars-2020-rover/",
    dataAccessUrl:
      "https://www.nasa.gov/news-release/nasa-announces-landing-site-for-mars-2020-rover/",
    authenticationRequired: false,
    lastRetrievedAt: retrieved,
    dataStatus: "CACHED",
    limitations:
      "45-km published crater diameter, not a local terrain statistic or MOLA extraction. Formal measurement uncertainty unspecified.",
  },
  {
    id: "textures",
    title: "Moon & Mars global texture maps",
    provider: "Solar System Scope",
    mission: "NASA-derived cartographic visualization",
    planetaryBody: "Mars",
    variables: ["Context imagery"],
    units: "RGB",
    spatialResolution: "2048 × 1024 pixels; not native mission resolution",
    temporalCoverage: "Multi-epoch mosaic; dates not supplied",
    geographicCoverage: "Global Moon and Mars",
    license: "CC BY 4.0 — Solar System Scope",
    attribution: "Solar System Scope; source imagery NASA",
    officialDocumentationUrl: "https://www.solarsystemscope.com/textures/",
    dataAccessUrl: "https://www.solarsystemscope.com/textures/download/2k_mars.jpg",
    authenticationRequired: false,
    lastRetrievedAt: retrieved,
    dataStatus: "CACHED",
    limitations:
      "Context only; not a verified native MOLA/LOLA product. Files retrieved successfully; provider landing page returned HTTP 403 during audit, so upstream mission provenance remains unverified.",
  },
];
export interface Observation {
  value: number;
  unit: "km";
  datasetId: string;
  kind: "published";
  uncertainty: null;
}
export interface Site {
  id: string;
  name: string;
  region: string;
  lat: number;
  lon: number;
  environment: "Polar" | "Arid" | "Volcanic";
  terrain: string;
  description: string;
  usefulness: string;
  limitation: string;
  diameter?: Observation;
  source: string;
}
export const sites: Site[] = [
  {
    id: "haughton",
    name: "Haughton Crater",
    region: "Nunavut, Canada",
    lat: 75.38,
    lon: -89.68,
    environment: "Polar",
    terrain: "Polar desert · Impact structure",
    description:
      "An ancient impact structure in the Canadian High Arctic. Ice-cemented impact breccia makes this a documented field laboratory for planetary exploration.",
    usefulness:
      "Practice impact-geology fieldwork, remote operations, and prototype drilling in ice-cemented ground.",
    limitation:
      "Eroded sedimentary target rocks differ from lunar regolith. Earth’s atmosphere, gravity, biology, and seasonal ice cannot reproduce lunar conditions.",
    diameter: {
      value: 20,
      unit: "km",
      datasetId: "haughton",
      kind: "published",
      uncertainty: null,
    },
    source: datasets[1].officialDocumentationUrl,
  },
  {
    id: "atacama",
    name: "Atacama Desert",
    region: "Antofagasta, Chile",
    lat: -24.5,
    lon: -69.25,
    environment: "Arid",
    terrain: "Hyperarid desert · Salt-rich soils",
    description:
      "A well-known terrestrial environment for studying aridity and the limits of life. The marker is a regional reference, not a uniform measurement window.",
    usefulness: "Investigate biosignature preservation and field protocols in arid soils.",
    limitation:
      "No harmonized numerical observations are included for this region. It is not eligible for a score or ranking.",
    source:
      "https://www.spaceappschallenge.org/2026/challenges/permanent-moon-base-locations-and-mars/",
  },
  {
    id: "death-valley",
    name: "Death Valley",
    region: "California, United States",
    lat: 36.24,
    lon: -116.82,
    environment: "Arid",
    terrain: "Arid basin · Evaporite deposits",
    description:
      "A documented analog candidate named in the official challenge. Basin and evaporite environments offer geological context, not a validated match.",
    usefulness: "Study sedimentary environments and desert field operations.",
    limitation:
      "No compatible temperature, roughness, or mineralogical measurements are included. Extreme heat alone does not establish a Mars analogy.",
    source:
      "https://www.spaceappschallenge.org/2026/challenges/permanent-moon-base-locations-and-mars/",
  },
];
export interface Target {
  id: string;
  name: string;
  body: "Moon" | "Mars";
  lat: number;
  lon: number;
  context: string;
  diameter?: Observation;
  source: string;
}
export const targets: Target[] = [
  {
    id: "shackleton",
    name: "Shackleton Crater",
    body: "Moon",
    lat: -89.9,
    lon: 0,
    context:
      "South-polar impact crater with a permanently shadowed interior. Earth cannot reproduce its vacuum or polar illumination regime.",
    diameter: {
      value: 21,
      unit: "km",
      datasetId: "shackleton",
      kind: "published",
      uncertainty: null,
    },
    source: datasets[2].officialDocumentationUrl,
  },
  {
    id: "jezero",
    name: "Jezero Crater",
    body: "Mars",
    lat: 18.38,
    lon: 77.58,
    context:
      "Perseverance exploration region: a 45-km impact basin with an ancient lake and delta. Only published crater diameter is compared; local terrain and thermal observations are not harmonized.",
    diameter: { value: 45, unit: "km", datasetId: "jezero", kind: "published", uncertainty: null },
    source: "https://www.nasa.gov/news-release/nasa-announces-landing-site-for-mars-2020-rover/",
  },
];
export const criteria = [
  {
    id: "diameter",
    label: "Impact-structure scale",
    unit: "km",
    reason: "Crater-scale field traverses and morphology; not local trafficability.",
  },
  {
    id: "slope",
    label: "Slope & roughness",
    unit: "°",
    reason: "Mobility and terrain safety; compatible DEM windows required.",
  },
  {
    id: "geology",
    label: "Surface materials",
    unit: "class",
    reason: "Substrate and instrument response; mineralogical observations required.",
  },
  {
    id: "thermal",
    label: "Thermal environment",
    unit: "K",
    reason: "Equipment operation; comparable diurnal observations required.",
  },
  {
    id: "ice",
    label: "Water / ice evidence",
    unit: "indicator",
    reason: "Resource investigation; comparable depth and detection limits required.",
  },
] as const;
export type CriterionId = (typeof criteria)[number]["id"];
export interface Profile {
  id: string;
  label: string;
  weights: Record<CriterionId, number>;
}
export const profiles: Profile[] = [
  {
    id: "base",
    label: "Permanent lunar base analog",
    weights: { diameter: 1, slope: 1, geology: 1, thermal: 1, ice: 0 },
  },
  {
    id: "polar",
    label: "Lunar polar operations",
    weights: { diameter: 1, slope: 2, geology: 1, thermal: 3, ice: 3 },
  },
  {
    id: "landing",
    label: "Mars landing preparation",
    weights: { diameter: 1, slope: 4, geology: 2, thermal: 2, ice: 1 },
  },
  {
    id: "rover",
    label: "Rover & terrain navigation",
    weights: { diameter: 1, slope: 5, geology: 2, thermal: 1, ice: 0 },
  },
  {
    id: "geology",
    label: "Geological field research",
    weights: { diameter: 3, slope: 2, geology: 4, thermal: 0, ice: 0 },
  },
  {
    id: "ice",
    label: "Water / ice investigation",
    weights: { diameter: 0, slope: 1, geology: 2, thermal: 3, ice: 4 },
  },
];
