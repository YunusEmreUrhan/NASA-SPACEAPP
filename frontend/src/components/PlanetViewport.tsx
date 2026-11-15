import { useEffect, useRef, useState, type MutableRefObject } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import gsap from "gsap";
import { Crosshair, Maximize, Minus, Plus, RotateCcw } from "lucide-react";
import { sites, type Body, type Site, type Target } from "../data/catalog";

export interface CameraLink {
  direction: { x: number; y: number; z: number };
  distance: number;
  source: string;
  revision: number;
}

export interface GlobeMarkerItem {
  id: string;
  name: string;
  lat: number;
  lon: number;
  badge?: string;
  colorHex?: number;
  isSplitSite?: boolean;
}

interface Props {
  body: Body;
  site: Site;
  target: Target;
  imagery: boolean;
  grid: boolean;
  markers: boolean;
  onSelect: (id: string) => void;
  onGlobeCoordinateClick?: (body: Body, lat: number, lon: number) => void;
  customMarkers?: GlobeMarkerItem[];
  splitBridgeCoords?: { fromLat: number; fromLon: number; toLat: number; toLon: number } | null;
  focus: number;
  sync: boolean;
  cameraLink: MutableRefObject<CameraLink>;
}

export function geographicPosition(lat: number, lon: number, radius = 2) {
  const phi = THREE.MathUtils.degToRad(90 - lat),
    theta = THREE.MathUtils.degToRad(lon + 180);
  return new THREE.Vector3(
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta),
  );
}

export function vectorToGeographic(point: THREE.Vector3): { lat: number; lon: number } {
  const r = point.length() || 1;
  const sinLat = Math.max(-1, Math.min(1, point.y / r));
  const lat = THREE.MathUtils.radToDeg(Math.asin(sinLat));
  const thetaDeg = (THREE.MathUtils.radToDeg(Math.atan2(point.z, -point.x)) + 360) % 360;
  const lon = thetaDeg - 180;
  return {
    lat: Math.round(lat * 100) / 100,
    lon: Math.round(lon * 100) / 100,
  };
}

export default function PlanetViewport({
  body,
  site,
  target,
  imagery,
  grid,
  markers,
  onSelect,
  onGlobeCoordinateClick,
  customMarkers,
  splitBridgeCoords,
  focus,
  sync,
  cameraLink,
}: Props) {
  const host = useRef<HTMLDivElement>(null),
    shell = useRef<HTMLDivElement>(null);
  const api = useRef<{ zoom: (n: number) => void; reset: () => void } | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error" | "webgl">("loading");
  const [retry, setRetry] = useState(0);
  const [hoverCoords, setHoverCoords] = useState<{ lat: number; lon: number } | null>(null);
  const [fullscreenError, setFullscreenError] = useState(false);

  const latest = useRef({
    site,
    target,
    imagery,
    grid,
    markers,
    onSelect,
    onGlobeCoordinateClick,
    sync,
  });
  latest.current = {
    site,
    target,
    imagery,
    grid,
    markers,
    onSelect,
    onGlobeCoordinateClick,
    sync,
  };

  const focusFn = useRef<() => void>(() => {});

  useEffect(() => {
    const container = host.current;
    if (!container) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: "high-performance",
      });
    } catch {
      setStatus("webgl");
      return;
    }
    setStatus("loading");
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);
    renderer.domElement.setAttribute(
      "aria-label",
      `${body} 3D Interactive Planetary Globe. Click anywhere on the surface to select coordinates.`,
    );
    renderer.domElement.setAttribute("role", "img");

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
    const focusLat = body === "Earth" ? site.lat : target.lat;
    const focusLon = body === "Earth" ? site.lon : target.lon;
    const initial = geographicPosition(focusLat, focusLon, 6.2);
    camera.position.copy(initial);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.075;
    controls.enablePan = true;
    controls.screenSpacePanning = true;
    controls.minDistance = 2.65;
    controls.maxDistance = 12;
    controls.rotateSpeed = 0.55;
    controls.zoomSpeed = 0.7;

    scene.add(new THREE.AmbientLight(0xb5c8de, 1.9));
    const sun = new THREE.DirectionalLight(0xffffff, 2.8);
    sun.position.set(-4, 6, 8);
    scene.add(sun);

    const geo = new THREE.SphereGeometry(2, 96, 64);
    const mat = new THREE.MeshStandardMaterial({
      color: body === "Mars" ? 0xbb795b : 0xffffff,
      roughness: 0.95,
    });
    const globe = new THREE.Mesh(geo, mat);
    scene.add(globe);

    let disposed = false,
      texture: THREE.Texture | undefined;
    const textureUrl =
      body === "Earth" ? "/textures/earth-gibs.jpg" : `/textures/${body.toLowerCase()}.jpg`;
    new THREE.TextureLoader().load(
      textureUrl,
      (t) => {
        if (disposed) {
          t.dispose();
          return;
        }
        texture = t;
        t.colorSpace = THREE.SRGBColorSpace;
        t.anisotropy = renderer.capabilities.getMaxAnisotropy();
        mat.map = latest.current.imagery ? t : null;
        mat.color.set(0xffffff);
        mat.needsUpdate = true;
        setStatus("ready");
      },
      undefined,
      () => {
        if (!disposed) setStatus("error");
      },
    );

    const lines = new THREE.Group();
    const gridMat = new THREE.LineBasicMaterial({
      color: 0x8ac6d9,
      transparent: true,
      opacity: 0.14,
      depthWrite: false,
    });
    for (let lat = -60; lat <= 60; lat += 30) {
      const p = [];
      for (let lon = -180; lon <= 180; lon += 3) p.push(geographicPosition(lat, lon, 2.006));
      lines.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(p), gridMat));
    }
    for (let lon = 0; lon < 360; lon += 30) {
      const p = [];
      for (let lat = -90; lat <= 90; lat += 3) p.push(geographicPosition(lat, lon, 2.006));
      lines.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(p), gridMat));
    }
    scene.add(lines);

    const atmosphereMat = new THREE.ShaderMaterial({
      uniforms: {
        glowColor: {
          value: new THREE.Color(
            body === "Earth" ? 0x468ed2 : body === "Mars" ? 0xd97348 : 0x818e9f,
          ),
        },
      },
      vertexShader:
        "varying vec3 vNormal; varying vec3 vPosition; void main(){vNormal=normalize(normalMatrix*normal); vec4 p=modelViewMatrix*vec4(position,1.0);vPosition=p.xyz;gl_Position=projectionMatrix*p;}",
      fragmentShader:
        "uniform vec3 glowColor; varying vec3 vNormal; varying vec3 vPosition; void main(){float d=abs(dot(normalize(vNormal),normalize(-vPosition))); float a=pow(1.0-d,4.0)*0.34;gl_FragColor=vec4(glowColor,a);}",
      transparent: true,
      side: THREE.BackSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const atmosphere = new THREE.Mesh(new THREE.SphereGeometry(2.045, 64, 48), atmosphereMat);
    scene.add(atmosphere);

    const markerGroup = new THREE.Group();
    scene.add(markerGroup);
    const markerMeshes: THREE.Mesh[] = [];
    const labelElements: { el: HTMLButtonElement; pos: THREE.Vector3; id: string }[] = [];

    const activeList: GlobeMarkerItem[] =
      customMarkers && customMarkers.length > 0
        ? customMarkers
        : body === "Earth"
          ? sites.map((s) => ({ id: s.id, name: s.name, lat: s.lat, lon: s.lon }))
          : [{ id: target.id, name: target.name, lat: target.lat, lon: target.lon }];

    const markerGeometry = new THREE.SphereGeometry(0.028, 14, 14);
    const ringGeometry = new THREE.RingGeometry(0.048, 0.062, 36);

    activeList.forEach((item) => {
      const col = item.colorHex ?? (item.isSplitSite ? 0x00f5d4 : 0x72f2ca);
      const mMat = new THREE.MeshBasicMaterial({ color: col });
      const rMat = new THREE.MeshBasicMaterial({
        color: col,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.85,
      });

      const pos = geographicPosition(item.lat, item.lon, 2.024);
      const mesh = new THREE.Mesh(markerGeometry, mMat);
      mesh.position.copy(pos);
      mesh.userData.id = item.id;
      markerGroup.add(mesh);
      markerMeshes.push(mesh);

      const ring = new THREE.Mesh(ringGeometry, rMat);
      ring.position.copy(pos.clone().multiplyScalar(1.002));
      ring.lookAt(pos.clone().multiplyScalar(2));
      markerGroup.add(ring);

      const el = document.createElement("button");
      el.className = "globe-label";
      el.textContent = item.badge ? `${item.badge} ${item.name}` : item.name;
      el.title = `${item.name} · (${item.lat.toFixed(2)}°, ${item.lon.toFixed(2)}°)`;
      el.onclick = (ev) => {
        ev.stopPropagation();
        latest.current.onSelect(item.id);
      };
      container.appendChild(el);
      labelElements.push({ el, pos, id: item.id });
    });

    if (body === "Earth" && splitBridgeCoords) {
      const pStart = geographicPosition(splitBridgeCoords.fromLat, splitBridgeCoords.fromLon, 2.02);
      const pEnd = geographicPosition(splitBridgeCoords.toLat, splitBridgeCoords.toLon, 2.02);
      const arcPoints: THREE.Vector3[] = [];
      for (let i = 0; i <= 48; i++) {
        const t = i / 48;
        const interp = new THREE.Vector3().lerpVectors(pStart, pEnd, t).normalize();
        const elev = 2.02 + Math.sin(t * Math.PI) * 0.14;
        arcPoints.push(interp.multiplyScalar(elev));
      }
      const arcGeo = new THREE.BufferGeometry().setFromPoints(arcPoints);
      const arcMat = new THREE.LineBasicMaterial({ color: 0x00f5d4, transparent: true, opacity: 0.9 });
      markerGroup.add(new THREE.Line(arcGeo, arcMat));
    }

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let tween: gsap.core.Tween | undefined;
    const animateCamera = (to: THREE.Vector3) => {
      tween?.kill();
      controls.target.set(0, 0, 0);
      controls.enabled = false;
      tween = gsap.to(camera.position, {
        x: to.x,
        y: to.y,
        z: to.z,
        duration: reduced ? 0 : 1.15,
        ease: "power3.inOut",
        onUpdate: () => {
          camera.lookAt(0, 0, 0);
          publish();
        },
        onComplete: () => {
          controls.enabled = true;
          controls.update();
        },
      });
    };

    const publish = () => {
      if (latest.current.sync) {
        const link = cameraLink.current;
        link.direction = camera.position.clone().normalize();
        link.distance = camera.position.length();
        link.source = body;
        link.revision++;
      }
    };
    controls.addEventListener("change", publish);
    controls.addEventListener("start", () => {
      tween?.kill();
      controls.enabled = true;
    });

    focusFn.current = () => {
      const item = body === "Earth" ? latest.current.site : latest.current.target;
      animateCamera(geographicPosition(item.lat, item.lon, 5.2));
    };
    api.current = {
      zoom: (n) => animateCamera(camera.position.clone().multiplyScalar(n).clampLength(2.7, 12)),
      reset: () => animateCamera(initial),
    };

    const resize = new ResizeObserver(() => {
      const w = container.clientWidth,
        h = container.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    });
    resize.observe(container);

    const raycaster = new THREE.Raycaster(),
      pointer = new THREE.Vector2();
    let startX = 0,
      startY = 0;

    const down = (e: PointerEvent) => {
      startX = e.clientX;
      startY = e.clientY;
    };

    const move = (e: PointerEvent) => {
      const b = renderer.domElement.getBoundingClientRect();
      pointer.set(
        ((e.clientX - b.left) / b.width) * 2 - 1,
        (-(e.clientY - b.top) / b.height) * 2 + 1,
      );
      raycaster.setFromCamera(pointer, camera);
      const hitGlobe = raycaster.intersectObject(globe)[0];
      if (hitGlobe) {
        setHoverCoords(vectorToGeographic(hitGlobe.point));
      } else {
        setHoverCoords(null);
      }
    };

    const click = (e: PointerEvent) => {
      if (Math.hypot(e.clientX - startX, e.clientY - startY) > 5) return;
      const b = renderer.domElement.getBoundingClientRect();
      pointer.set(
        ((e.clientX - b.left) / b.width) * 2 - 1,
        (-(e.clientY - b.top) / b.height) * 2 + 1,
      );
      raycaster.setFromCamera(pointer, camera);

      const hitMarker = raycaster.intersectObjects(markerMeshes)[0];
      if (hitMarker) {
        latest.current.onSelect(hitMarker.object.userData.id);
        return;
      }

      const hitGlobe = raycaster.intersectObject(globe)[0];
      if (hitGlobe && latest.current.onGlobeCoordinateClick) {
        const coords = vectorToGeographic(hitGlobe.point);
        latest.current.onGlobeCoordinateClick(body, coords.lat, coords.lon);
      }
    };

    renderer.domElement.addEventListener("pointerdown", down);
    renderer.domElement.addEventListener("pointermove", move);
    renderer.domElement.addEventListener("pointerup", click);

    let frame = 0,
      lastRevision = -1,
      wasSynced = false;
    const render = () => {
      if (disposed) return;
      const props = latest.current,
        link = cameraLink.current;
      if (props.sync && !wasSynced) {
        controls.target.set(0, 0, 0);
        if (body === "Earth") publish();
      }
      wasSynced = props.sync;
      controls.enablePan = !props.sync;
      if (
        props.sync &&
        link.source !== "" &&
        link.source !== body &&
        link.revision !== lastRevision &&
        !tween?.isActive()
      ) {
        camera.position.copy(link.direction).multiplyScalar(link.distance);
        lastRevision = link.revision;
      }
      controls.update();
      lines.visible = props.grid;
      markerGroup.visible = props.markers;
      if (mat.map !== (props.imagery ? (texture ?? null) : null)) {
        mat.map = props.imagery ? (texture ?? null) : null;
        mat.needsUpdate = true;
      }
      const direction = camera.position.clone().normalize();
      labelElements.forEach(({ el, pos, id }) => {
        const visible = props.markers && pos.clone().normalize().dot(direction) > 0.22;
        el.style.display = visible ? "block" : "none";
        if (!visible) return;
        const projected = pos.clone().project(camera);
        el.style.left = `${(projected.x * 0.5 + 0.5) * container.clientWidth}px`;
        el.style.top = `${(-projected.y * 0.5 + 0.5) * container.clientHeight}px`;
        el.classList.toggle(
          "selected",
          id === (body === "Earth" ? props.site.id : props.target.id),
        );
      });
      renderer.render(scene, camera);
      frame = requestAnimationFrame(render);
    };
    render();

    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      tween?.kill();
      resize.disconnect();
      controls.dispose();
      renderer.domElement.removeEventListener("pointerdown", down);
      renderer.domElement.removeEventListener("pointermove", move);
      renderer.domElement.removeEventListener("pointerup", click);
      scene.traverse((obj) => {
        if (obj instanceof THREE.Mesh || obj instanceof THREE.Line) {
          obj.geometry?.dispose();
        }
      });
      mat.dispose();
      gridMat.dispose();
      atmosphereMat.dispose();
      markerGeometry.dispose();
      ringGeometry.dispose();
      texture?.dispose();
      renderer.dispose();
      labelElements.forEach((x) => x.el.remove());
      renderer.domElement.remove();
      api.current = null;
    };
  }, [body, retry, cameraLink, customMarkers, splitBridgeCoords]);

  useEffect(() => {
    if (focus > 0) focusFn.current();
  }, [focus, site.id, target.id]);

  async function fullscreen() {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await shell.current?.requestFullscreen();
    } catch {
      setFullscreenError(true);
    }
  }

  return (
    <div className={`planet-viewport planet-${body.toLowerCase()}`} ref={shell}>
      <div className="planet-canvas" ref={host} />
      <div className="viewport-caption">
        <span className="tiny-dot" />
        {body === "Earth" ? "TERRESTRIAL ANALOG ARENA (11,469 BARE-LAND CELLS)" : `${body.toUpperCase()} SURFACE TELEMETRY LAB`}
        {hoverCoords && (
          <span style={{ marginLeft: 10, color: "#9de9d3", fontWeight: 600 }}>
            🎯 İMLEÇ: {hoverCoords.lat.toFixed(2)}°, {hoverCoords.lon.toFixed(2)}° (Küreye Tıkla & Analiz Et)
          </span>
        )}
        <span className="view-mode">3D INTERACTIVE GLOBE</span>
      </div>
      {status === "loading" && (
        <div className="scene-message">Loading high-resolution {body.toLowerCase()} surface…</div>
      )}
      {fullscreenError && (
        <div className="scene-message">
          Fullscreen is blocked.<button onClick={() => setFullscreenError(false)}>Dismiss</button>
        </div>
      )}
      {(status === "error" || status === "webgl") && (
        <div className="scene-message error">
          {status === "webgl"
            ? "WebGL is unavailable. All telemetry, comparisons, and exports remain accessible below."
            : "Surface texture could not load."}
          <button onClick={() => setRetry((n) => n + 1)}>Retry 3D renderer</button>
        </div>
      )}
      <div className="viewport-tools">
        <button
          title="Seçili saha/hedefe odaklan"
          aria-label={`Focus ${body} site`}
          onClick={() => focusFn.current()}
        >
          <Crosshair size={17} />
        </button>
        <button
          title="Yakınlaştır"
          aria-label={`Zoom in ${body}`}
          onClick={() => api.current?.zoom(0.8)}
        >
          <Plus size={18} />
        </button>
        <button
          title="Uzaklaştır"
          aria-label={`Zoom out ${body}`}
          onClick={() => api.current?.zoom(1.2)}
        >
          <Minus size={18} />
        </button>
        <span />
        <button
          title="Kamerayı sıfırla"
          aria-label={`Reset ${body} camera`}
          onClick={() => api.current?.reset()}
        >
          <RotateCcw size={16} />
        </button>
        <button title="Tam ekran" aria-label={`Fullscreen ${body}`} onClick={fullscreen}>
          <Maximize size={16} />
        </button>
      </div>
      <div className="globe-orientation">
        <span>N</span>
        <div>
          <Plus size={20} strokeWidth={1} />
        </div>
        <small>
          W <span>E</span>
        </small>
      </div>
      <div className="viewport-bottom">
        <span>
          <i /> {body === "Earth" ? "NASA GIBS · SRTM 90m DEM + NASA POWER" : `NASA TREK WMTS · ${body === "Moon" ? "LRO WAC / LOLA" : "VIKING MDIM21 / MGS MOLA"}`}
        </span>
        <span>3B KÜRE ÜZERİNE TIKLAYARAK HEDEF VEYA ADAY SEÇEBİLİRSİNİZ</span>
      </div>
    </div>
  );
}
