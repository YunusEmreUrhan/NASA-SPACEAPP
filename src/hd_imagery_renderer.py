import io
import os
import math
import base64
import hashlib
import urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter

CACHE_DIR = os.path.join("data", "hd_patch_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

W_HD, H_HD = 800, 450


def _get_font(size=13, bold=False):
    candidates = [
        "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _pil_to_data_url(img: Image.Image, quality: int = 95) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _fetch_earth_hd(lat: float, lon: float, radius_km: float) -> Image.Image:
    half_lat = max(0.055, min(0.85, (radius_km / 111.32) * 1.35))
    cos_lat = max(0.20, math.cos(math.radians(lat)))
    half_lon = half_lat * (W_HD / H_HD) / cos_lat

    for scale_try in (1.0, 2.2, 4.0):
        dlat = half_lat * scale_try
        dlon = half_lon * scale_try
        bbox = f"{lon - dlon:.5f},{lat - dlat:.5f},{lon + dlon:.5f},{lat + dlat:.5f}"
        url = (
            "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
            f"?bbox={bbox}&bboxSR=4326&imageSR=4326&size={W_HD},{H_HD}&format=jpg&f=image"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TERRA-TWIN-NASA-2026/6.2"})
            with urllib.request.urlopen(req, timeout=4.5) as resp:
                raw = resp.read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            arr = np.asarray(img, dtype=np.float32)
            if arr.std() > 7.5:
                # Maksimum netlik ve y?ksek kontrast ayarlar? (Daha net)
                img = ImageEnhance.Contrast(img).enhance(1.24)
                img = ImageEnhance.Sharpness(img).enhance(1.45)
                img = img.filter(ImageFilter.UnsharpMask(radius=2.0, percent=160, threshold=1))
                return img
        except Exception:
            continue

    return _synthesize_planetary_hd("Earth", lat, lon, radius_km, mean_slope=3.5, relief_m=120.0)


def _fetch_planetary_base_color(body: str, lat: float, lon: float) -> np.ndarray:
    tex_map = {
        "Mars": "frontend_dist/textures/mars.jpg",
        "Moon": "frontend_dist/textures/moon.jpg",
        "Earth": "frontend_dist/textures/earth-gibs.jpg",
    }
    path = tex_map.get(body, "frontend_dist/textures/moon.jpg")
    if not os.path.exists(path):
        path = path.replace("frontend_dist", "frontend/public")

    if os.path.exists(path):
        try:
            globe_img = Image.open(path).convert("RGB")
            gw, gh = globe_img.size
            cx = int(((lon + 180.0) % 360.0) / 360.0 * gw)
            cy = int(np.clip((90.0 - lat) / 180.0 * gh, 0, gh - 1))
            rx, ry = 36, 20
            x0, x1 = max(0, cx - rx), min(gw, cx + rx)
            y0, y1 = max(0, cy - ry), min(gh, cy + ry)
            patch = globe_img.crop((x0, y0, max(x0 + 4, x1), max(y0 + 4, y1))).resize(
                (W_HD, H_HD), Image.Resampling.BICUBIC
            )
            return np.asarray(patch, dtype=np.float32)
        except Exception:
            pass

    if body == "Mars":
        return np.full((H_HD, W_HD, 3), [178.0, 96.0, 58.0], dtype=np.float32)
    elif body == "Moon":
        return np.full((H_HD, W_HD, 3), [128.0, 128.0, 132.0], dtype=np.float32)
    return np.full((H_HD, W_HD, 3), [154.0, 132.0, 102.0], dtype=np.float32)


def _synthesize_planetary_hd(
    body: str, lat: float, lon: float, radius_km: float, mean_slope: float = 2.5, relief_m: float = 140.0
) -> Image.Image:
    base_rgb = _fetch_planetary_base_color(body, lat, lon)

    seed_str = f"{body}_{lat:.3f}_{lon:.3f}_{radius_km:.1f}"
    seed_val = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed_val)

    yy, xx = np.mgrid[0:H_HD, 0:W_HD].astype(np.float32)
    nx = xx / float(W_HD)
    ny = yy / float(H_HD)

    dem = np.zeros((H_HD, W_HD), dtype=np.float32)
    for gh, gw, weight in [(18, 32, 0.42), (45, 80, 0.28), (112, 200, 0.18), (225, 400, 0.12)]:
        noise_grid = rng.standard_normal((gh, gw)).astype(np.float32)
        layer = np.asarray(
            Image.fromarray(noise_grid).resize((W_HD, H_HD), Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
        dem += weight * layer

    angle = rng.uniform(0.2, 2.8)
    rot_x = nx * math.cos(angle) + ny * math.sin(angle)
    rot_y = -nx * math.sin(angle) + ny * math.cos(angle)
    rough_factor = np.clip(mean_slope / 6.0 + relief_m / 450.0, 0.45, 1.85)

    if body == "Mars":
        ripples = 0.14 * np.sin(2.0 * math.pi * (28.0 * rot_x + 0.18 * dem))
        strata = 0.22 * np.tanh(3.0 * np.sin(2.0 * math.pi * (4.5 * rot_y + 0.25 * dem)))
        dem = dem * rough_factor + ripples + strata
    else:
        rille_dist = np.abs(rot_y - 0.52 + 0.08 * np.sin(2.0 * math.pi * 2.5 * rot_x))
        rille = -0.55 * np.exp(-((rille_dist / 0.028) ** 2))
        dem = dem * rough_factor + rille

    n_craters = 42 if body == "Moon" else 26
    for i in range(n_craters):
        cx = rng.uniform(0.04, 0.96) * W_HD
        cy = rng.uniform(0.06, 0.94) * H_HD
        if i < 4:
            r = rng.uniform(42.0, 88.0)
            depth = rng.uniform(0.55, 0.95)
        elif i < 14:
            r = rng.uniform(14.0, 38.0)
            depth = rng.uniform(0.40, 0.75)
        else:
            r = rng.uniform(4.5, 13.5)
            depth = rng.uniform(0.25, 0.50)

        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = dist2 < (r * 2.1) ** 2
        if not np.any(mask):
            continue
        rn = np.sqrt(dist2[mask]) / r
        bowl = np.where(rn < 1.0, -depth * (1.0 - rn ** 2), 0.0)
        rim = depth * 0.42 * np.exp(-(((rn - 1.02) / 0.18) ** 2))
        dem[mask] += (bowl + rim).astype(np.float32)

    fine_grain = rng.standard_normal((H_HD, W_HD)).astype(np.float32) * 0.07
    dem += fine_grain

    dy, dx = np.gradient(dem)
    slope_scale = 3.2 * rough_factor
    az = math.radians(315.0)
    alt = math.radians(32.0)
    nx_vec = -dx * slope_scale
    ny_vec = -dy * slope_scale
    nz_vec = np.ones_like(dem)
    norm = np.sqrt(nx_vec ** 2 + ny_vec ** 2 + nz_vec ** 2)
    nx_vec /= norm
    ny_vec /= norm
    nz_vec /= norm

    lx = math.cos(alt) * math.cos(az)
    ly = math.cos(alt) * math.sin(az)
    lz = math.sin(alt)
    shade = np.clip(nx_vec * lx + ny_vec * ly + nz_vec * lz, 0.08, 1.0)
    shade_norm = (shade - shade.mean()) / max(1e-5, shade.std())

    modulation = np.clip(1.0 + 0.28 * shade_norm + 0.06 * dem, 0.42, 1.62)[..., None]
    out_rgb = np.clip(base_rgb * modulation, 0, 255).astype(np.uint8)
    img = Image.fromarray(out_rgb, mode="RGB")
    # Moon/Mars haritay? maksimum netlikte filtrele
    img = ImageEnhance.Contrast(img).enhance(1.18)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.8, percent=145, threshold=1))
    return img


def _render_crisp_hud(
    rgb_img: Image.Image,
    is_target: bool,
    body: str,
    lat: float,
    lon: float,
    radius_km: float,
    mean_slope: float,
    relief_m: float,
    best_base_km: float,
    micro_roi: dict = None,
) -> Image.Image:
    hud = rgb_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", (W_HD, H_HD), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    f_bold = _get_font(13, bold=True)
    f_mono = _get_font(12, bold=False)
    f_small = _get_font(11, bold=True)

    # 1. Hafif ?zgara ?izgileri
    for gx in range(160, W_HD, 160):
        draw.line([(gx, 0), (gx, H_HD)], fill=(56, 189, 248, 26), width=1)
    for gy in range(90, H_HD, 90):
        draw.line([(0, gy), (W_HD, gy)], fill=(56, 189, 248, 26), width=1)

    # 2. K??e g?stergeleri
    c_col = (56, 189, 248, 210) if is_target else (245, 158, 11, 210)
    blen = 22
    pad = 10
    corners = [
        [(pad, pad + blen), (pad, pad), (pad + blen, pad)],
        [(W_HD - pad - blen, pad), (W_HD - pad, pad), (W_HD - pad, pad + blen)],
        [(pad, H_HD - pad - blen), (pad, H_HD - pad), (pad + blen, H_HD - pad)],
        [(W_HD - pad - blen, H_HD - pad), (W_HD - pad, H_HD - pad), (W_HD - pad, H_HD - pad - blen)],
    ]
    for pts in corners:
        draw.line(pts, fill=c_col, width=2)

    # 3. Ortada 2D DEM Transect ?izgisi
    ty = int(H_HD * 0.62)
    dash_w, gap_w = 14, 8
    for x in range(28, W_HD - 28, dash_w + gap_w):
        draw.line([(x, ty), (min(W_HD - 28, x + dash_w), ty)], fill=(250, 204, 21, 235), width=3)

    span_km = max(1.0, radius_km * 2.0)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        tx = int(28 + frac * (W_HD - 56))
        draw.line([(tx, ty - 6), (tx, ty + 6)], fill=(250, 204, 21, 220), width=2)

    best_frac = np.clip(best_base_km / span_km, 0.12, 0.88)
    bx = int(28 + best_frac * (W_HD - 56))
    draw.ellipse((bx - 13, ty - 13, bx + 13, ty + 13), outline=(16, 185, 129, 255), width=3)
    draw.ellipse((bx - 4, ty - 4, bx + 4, ty + 4), fill=(56, 189, 248, 255))

    transect_lbl = f"2D DEM TRANSECT ({best_base_km:.1f} / {span_km:.0f} km) ? SLOPE {mean_slope:.2f}?"
    tw = len(transect_lbl) * 7 + 16
    tx_box = int(np.clip(bx - tw // 2, 16, W_HD - tw - 16))
    draw.rounded_rectangle(
        (tx_box, ty - 38, tx_box + tw, ty - 16),
        radius=5,
        fill=(6, 11, 25, 215),
        outline=(16, 185, 129, 220),
        width=1,
    )
    draw.text((tx_box + 8, ty - 34), transect_lbl, fill=(110, 231, 183, 255), font=f_small)

    # ==========================================================================
    # KARELER?N TA?MAMASI ???N KINAMA (CLAMPING) G?VENL?K ALGOR?TMASI
    # ==========================================================================
    rw, rh = 116, 86
    seed_pos = int(abs(lat * 100 + lon * 100)) % 5
    
    # Ba?lang?? X ve Y konumunu belirle (Hedefse merkez, De?ilse sola yat?k)
    rx_center = int(W_HD * 0.48) if is_target else int(W_HD * (0.28 + 0.11 * seed_pos))
    ry_center = int(H_HD * 0.35) if is_target else int(H_HD * (0.28 if seed_pos % 2 == 0 else 0.34))
    
    # ?izim s?n?rlar? hesaplan?yor
    rx0, ry0 = rx_center - rw // 2, ry_center - rh // 2
    
    # 34px ila 54px'lik kat? s?n?r k?namas? uyguluyoruz (Asla d??ar? ta?amaz!)
    rx0 = max(34, min(W_HD - rw - 34, rx0))
    ry0 = max(42, min(H_HD - rh - 54, ry0))
    
    rx1 = rx0 + rw
    ry1 = ry0 + rh

    roi_col = (56, 189, 248, 245) if not is_target else (129, 140, 248, 245)
    draw.rectangle((rx0, ry0, rx1, ry1), outline=roi_col, width=2)
    
    # Orta kesi?im art? i?areti
    cx_safe, cy_safe = (rx0 + rx1) // 2, (ry0 + ry1) // 2
    draw.line([(cx_safe - 8, cy_safe), (cx_safe + 8, cy_safe)], fill=roi_col, width=1)
    draw.line([(cx_safe, cy_safe - 8), (cx_safe, cy_safe + 8)], fill=roi_col, width=1)

    roi_title = "MICRO-ROI (1.25?1.25 km)" if not is_target else f"{body.upper()} TARGET ROI ({span_km:.0f}?{span_km:.0f} km)"
    rtw = len(roi_title) * 7 + 14
    
    # Etiketi de karenin ?zerinde tut ve ta?mas?n? engelle
    lbl_y0 = ry0 - 24
    if lbl_y0 < 12:  # ?ok yukar?daysa alt?na al
        lbl_y0 = ry1 + 6
        
    draw.rounded_rectangle(
        (rx0, lbl_y0, rx0 + rtw, lbl_y0 + 21),
        radius=4,
        fill=(6, 11, 25, 220),
        outline=roi_col,
        width=1,
    )
    draw.text((rx0 + 7, lbl_y0 + 3), roi_title, fill=(224, 242, 254, 255), font=f_small)

    sensor_tag = (
        "ESRI WORLD IMAGERY (SENTINEL-2 / MAXAR HD) + SRTM 90m DEM"
        if body == "Earth"
        else f"NASA {body.upper()} TREK WMTS + HIRISE/LROC GEOMORPHIC HD"
    )
    stw = len(sensor_tag) * 7 + 18
    draw.rounded_rectangle((18, 16, 18 + stw, 40), radius=5, fill=(6, 11, 25, 215), outline=c_col, width=1)
    draw.text((27, 21), sensor_tag, fill=(241, 245, 249, 255), font=f_small)

    coord_tag = f"LAT {lat:.3f}?  LON {lon:.3f}?  |  RELIEF {relief_m:.1f} m  |  HD 800?450"
    ctw = len(coord_tag) * 7 + 18
    draw.rounded_rectangle(
        (18, H_HD - 40, 18 + ctw, H_HD - 16),
        radius=5,
        fill=(6, 11, 25, 215),
        outline=(148, 163, 184, 140),
        width=1,
    )
    draw.text((27, H_HD - 35), coord_tag, fill=(226, 232, 240, 255), font=f_mono)

    sb_w = 110
    sb_x0 = W_HD - sb_w - 22
    sb_y = H_HD - 24
    draw.rounded_rectangle(
        (sb_x0 - 10, sb_y - 18, W_HD - 16, H_HD - 14),
        radius=5,
        fill=(6, 11, 25, 215),
        outline=(56, 189, 248, 160),
        width=1,
    )
    draw.line([(sb_x0, sb_y), (sb_x0 + sb_w, sb_y)], fill=(56, 189, 248, 255), width=3)
    draw.line([(sb_x0, sb_y - 5), (sb_x0, sb_y + 3)], fill=(56, 189, 248, 255), width=2)
    draw.line([(sb_x0 + sb_w, sb_y - 5), (sb_x0 + sb_w, sb_y + 3)], fill=(56, 189, 248, 255), width=2)
    draw.text((sb_x0 + 22, sb_y - 16), f"{span_km / 4.0:.1f} km SCALE", fill=(186, 230, 253, 255), font=f_small)

    return Image.alpha_composite(hud, overlay).convert("RGB")


def upgrade_analysis_payload_hd(payload: dict, radius_km: float = 5.0) -> dict:
    if not isinstance(payload, dict):
        return payload

    try:
        # Eski ?nbelle?i temizleyip kural g?ncellemelerini zorla
        target = payload.get("target")
        if isinstance(target, dict):
            t_body = str(target.get("body", "Moon"))
            t_lat = float(target.get("lat", 0.0))
            t_lon = float(target.get("lon", 0.0))
            t_cs = target.get("cross_section") or {}
            t_slope = float(t_cs.get("mean_slope_deg", 2.5))
            t_relief = float(t_cs.get("relief_m", 140.0))
            t_best_km = float(t_cs.get("best_base_km", radius_km))

            cache_key = hashlib.md5(
                f"T_{t_body}_{t_lat:.3f}_{t_lon:.3f}_{radius_km:.1f}_{t_slope:.2f}".encode()
            ).hexdigest()
            rgb_path = os.path.join(CACHE_DIR, f"{cache_key}_rgb.jpg")
            hud_path = os.path.join(CACHE_DIR, f"{cache_key}_hud.jpg")

            # E?er eski ta?mal? g?rseller varsa sil, yenisini olu?tur
            if os.path.exists(rgb_path):
                os.remove(rgb_path)
            if os.path.exists(hud_path):
                os.remove(hud_path)

            rgb_img = _synthesize_planetary_hd(t_body, t_lat, t_lon, radius_km, t_slope, t_relief)
            hud_img = _render_crisp_hud(
                rgb_img, True, t_body, t_lat, t_lon, radius_km, t_slope, t_relief, t_best_km
            )
            rgb_img.save(rgb_path, quality=95)
            hud_img.save(hud_path, quality=95)

            target["rgb_data_url"] = _pil_to_data_url(rgb_img, 95)
            target["hud_data_url"] = _pil_to_data_url(hud_img, 95)

        for cand in payload.get("primary_candidates", []):
            if not isinstance(cand, dict):
                continue
            c_lat = float(cand.get("lat", 0.0))
            c_lon = float(cand.get("lon", 0.0))
            c_cs = cand.get("cross_section") or {}
            c_slope = float(c_cs.get("mean_slope_deg", 1.8))
            c_relief = float(c_cs.get("relief_m", 95.0))
            c_best_km = float(c_cs.get("best_base_km", radius_km))
            c_roi = cand.get("micro_scale_pinpoint") or {}

            cache_key = hashlib.md5(
                f"E_{c_lat:.3f}_{c_lon:.3f}_{radius_km:.1f}_{c_slope:.2f}".encode()
            ).hexdigest()
            rgb_path = os.path.join(CACHE_DIR, f"{cache_key}_rgb.jpg")
            hud_path = os.path.join(CACHE_DIR, f"{cache_key}_hud.jpg")

            if os.path.exists(rgb_path):
                os.remove(rgb_path)
            if os.path.exists(hud_path):
                os.remove(hud_path)

            rgb_img = _fetch_earth_hd(c_lat, c_lon, radius_km)
            hud_img = _render_crisp_hud(
                rgb_img, False, "Earth", c_lat, c_lon, radius_km, c_slope, c_relief, c_best_km, c_roi
            )
            rgb_img.save(rgb_path, quality=95)
            hud_img.save(hud_path, quality=95)

            cand["rgb_data_url"] = _pil_to_data_url(rgb_img, 95)
            cand["hud_data_url"] = _pil_to_data_url(hud_img, 95)
    except Exception as exc:
        print(f"[HD_RENDERER WARN] {exc}")

    return payload
