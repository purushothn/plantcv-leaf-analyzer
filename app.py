"""
Leaf Phenotyping Analyzer — Streamlit App
University of Maryland Eastern Shore | Natarajan Lab
Uses OpenCV + NumPy (no PlantCV dependency) for Streamlit Cloud compatibility.
Extracts the same morphological and color traits as a PlantCV pipeline.
"""

import io
import warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlantCV Leaf Analyzer",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stSidebar"] {background: #1a2e14;}
  [data-testid="stSidebar"] * {color: #d4e8c2 !important;}
  .metric-card {
    background: #f0f7e8; border-radius: 10px; padding: 1rem 1.25rem;
    border-left: 4px solid #639922; margin-bottom: .5rem;
  }
  .metric-card h4 {margin:0;font-size:12px;color:#5a6e4a;text-transform:uppercase;letter-spacing:.05em;}
  .metric-card p  {margin:4px 0 0;font-size:26px;font-weight:600;color:#27500a;}
  .section-head {font-size:14px;font-weight:600;color:#3B6D11;
    border-bottom:1px solid #c0dd97;padding-bottom:4px;margin:1.2rem 0 .75rem;}
  .info-callout {
    background:#f7fbf0;border:1px solid #c0dd97;border-radius:8px;
    padding:.75rem 1rem;font-size:13px;color:#3B6D11;margin:.75rem 0;
  }
  .stDownloadButton > button {
    background:#639922 !important;color:white !important;
    border:none !important;border-radius:8px !important;
  }
  .stDownloadButton > button:hover {background:#3B6D11 !important;}
</style>
""", unsafe_allow_html=True)


# ── Core analysis functions (pure OpenCV) ─────────────────────────────────────

def pil_to_bgr(pil_img):
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def bgr_to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def segment_leaf(bgr, method, thresh_val, fill_size):
    """Segment leaf from dark background; return binary mask."""
    if method == "HSV (saturation channel)":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gray = hsv[:, :, 1]
        _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    elif method == "LAB (green-magenta channel)":
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        gray = lab[:, :, 1]
        _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    elif method == "Otsu auto-threshold":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gray = hsv[:, :, 1]
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:  # Gaussian adaptive
        gray_v = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(gray_v, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 21, -thresh_val // 4)

    # morphological cleanup — remove noise, fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # keep only largest contour (the leaf)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        clean = np.zeros_like(mask)
        cv2.drawContours(clean, [largest], -1, 255, -1)
        return clean
    return mask


def extract_traits(bgr, mask, scale):
    """Extract morphological and color traits using OpenCV."""
    traits = {}
    s2 = scale ** 2  # px² → cm²

    # ── Morphology ──
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {"error": "No leaf contour found — try adjusting threshold"}

    cnt = max(cnts, key=cv2.contourArea)
    area_px      = cv2.contourArea(cnt)
    perim_px     = cv2.arcLength(cnt, True)
    x, y, w, h   = cv2.boundingRect(cnt)
    hull         = cv2.convexHull(cnt)
    hull_area_px = cv2.contourArea(hull)

    traits["area_cm2"]           = round(area_px * s2, 4)
    traits["perimeter_cm"]       = round(perim_px * scale, 4)
    traits["width_cm"]           = round(w * scale, 4)
    traits["height_cm"]          = round(h * scale, 4)
    traits["convex_hull_area_cm2"] = round(hull_area_px * s2, 4)
    traits["aspect_ratio"]       = round(w / h, 4) if h else 0
    traits["circularity"]        = round(4 * np.pi * area_px / (perim_px ** 2), 4) if perim_px else 0
    traits["solidity"]           = round(area_px / hull_area_px, 4) if hull_area_px else 0

    # ellipse fit
    if len(cnt) >= 5:
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
        traits["ellipse_major_cm"] = round(MA * scale, 4)
        traits["ellipse_minor_cm"] = round(ma * scale, 4)
        traits["ellipse_angle"]    = round(angle, 2)
        ecc = np.sqrt(1 - (min(MA, ma) / max(MA, ma)) ** 2) if max(MA, ma) else 0
        traits["eccentricity"]     = round(float(ecc), 4)
    else:
        traits["eccentricity"] = np.nan

    # ── Color ──
    leaf_pixels = bgr[mask == 255]
    if len(leaf_pixels):
        b, g, r = leaf_pixels[:, 0], leaf_pixels[:, 1], leaf_pixels[:, 2]
        traits["red_mean"]   = round(float(r.mean()), 2)
        traits["green_mean"] = round(float(g.mean()), 2)
        traits["blue_mean"]  = round(float(b.mean()), 2)
        traits["red_std"]    = round(float(r.std()), 2)
        traits["green_std"]  = round(float(g.std()), 2)
        traits["blue_std"]   = round(float(b.std()), 2)

        # HSV
        hsv_px = cv2.cvtColor(leaf_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        traits["hue_mean"]  = round(float(hsv_px[:, 0].mean()), 2)
        traits["sat_mean"]  = round(float(hsv_px[:, 1].mean()), 2)
        traits["val_mean"]  = round(float(hsv_px[:, 2].mean()), 2)

        # LAB
        lab_px = cv2.cvtColor(leaf_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        traits["L_mean"]  = round(float(lab_px[:, 0].mean()), 2)
        traits["a_mean"]  = round(float(lab_px[:, 1].mean() - 128), 2)  # center around 0
        traits["b_lab_mean"] = round(float(lab_px[:, 2].mean() - 128), 2)

        # Greenness index (useful proxy for chlorophyll)
        total = traits["red_mean"] + traits["green_mean"] + traits["blue_mean"]
        traits["greenness_index"] = round(traits["green_mean"] / total, 4) if total else 0

    return traits


def overlay_mask(bgr, mask):
    overlay = bgr_to_rgb(bgr.copy())
    green_layer = np.zeros_like(overlay)
    green_layer[:, :, 1] = 160
    alpha = (mask > 0)[:, :, np.newaxis].astype(float) * 0.38
    overlay = (overlay * (1 - alpha) + green_layer * alpha).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (50, 200, 50), 2)
    return overlay


def make_bar_chart(df, col, label):
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = plt.cm.Greens(np.linspace(0.4, 0.85, len(df)))
    bars = ax.barh(df["filename"], df[col], color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel(label, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#f9fdf5"); fig.patch.set_facecolor("#f9fdf5")
    mx = df[col].max() if df[col].max() else 1
    for bar, val in zip(bars, df[col]):
        ax.text(val + mx * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8, color="#3B6D11")
    plt.tight_layout()
    return fig


def make_color_chart(df):
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(df)); w = 0.25
    for i, (col, color, lab) in enumerate(
            zip(["red_mean","green_mean","blue_mean"],
                ["#e05252","#52a852","#5285e0"],
                ["Red","Green","Blue"])):
        ax.bar(x + i*w, df[col], width=w, color=color, label=lab, alpha=0.85, edgecolor="white")
    ax.set_xticks(x + w)
    ax.set_xticklabels(df["filename"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean channel (0–255)", fontsize=9)
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_facecolor("#f9fdf5"); fig.patch.set_facecolor("#f9fdf5")
    plt.tight_layout()
    return fig


def to_csv(df): return df.to_csv(index=False).encode("utf-8")

def to_excel(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Leaf Traits")
        ws = w.sheets["Leaf Traits"]
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = 18
    return buf.getvalue()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image("https://www.umes.edu/wp-content/uploads/2023/06/UMES-logo-white.png",
                 use_container_width=True)
    except Exception:
        st.markdown("**UMES**")
    st.markdown("## 🍃 Leaf Analyzer")
    st.markdown("**Natarajan Lab · UMES DAFRS**")
    st.markdown("---")

    st.markdown("### Segmentation")
    seg_method = st.selectbox("Method", [
        "HSV (saturation channel)",
        "LAB (green-magenta channel)",
        "Otsu auto-threshold",
        "Gaussian adaptive"
    ], help="HSV works best for green leaves on black background.")
    thresh_val = st.slider("Threshold value", 20, 200, 55)
    fill_sz    = st.slider("Fill/clean iterations", 1, 5, 2)

    st.markdown("### Scale")
    mode = st.radio("Input", ["From print DPI", "Manual"])
    if mode == "Manual":
        scale = st.number_input("cm per pixel", value=0.0264, step=0.001, format="%.4f")
    else:
        dpi = st.selectbox("Print DPI", [72, 96, 150, 300], index=3)
        scale = 2.54 / dpi
        st.caption(f"→ {scale:.5f} cm/px")

    show_mask   = st.checkbox("Show overlay", value=True)
    show_charts = st.checkbox("Show charts",  value=True)
    do_excel    = st.checkbox("Excel export", value=True)

    st.markdown("---")
    st.markdown("""<div style='font-size:11px;color:#9bbf80;line-height:1.7'>
    <b>Pipeline (OpenCV)</b><br>
    cvtColor → threshold<br>
    → morphologyEx → findContours<br>
    → contourArea / arcLength<br>
    → fitEllipse → color stats
    </div>""", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🍃 Leaf Phenotyping Analyzer")
st.markdown(
    "Upload leaf images photographed on the **black matte background with ruler**. "
    "The app segments each leaf and extracts morphological and color traits, "
    "then exports a results table ready for R or SPSS analysis."
)
st.markdown('<div class="info-callout">📋 <b>Tip:</b> Place leaf flat on the printed black background, '
            'photograph from directly above in good light. One image = one leaf.</div>',
            unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload leaf images",
    type=["jpg","jpeg","png","tif","tiff","bmp"],
    accept_multiple_files=True
)

if not uploaded:
    st.info("👆 Upload leaf images to begin.")
    col1,col2,col3,col4 = st.columns(4)
    with col1: st.markdown("**1. Photograph** leaf on black background")
    with col2: st.markdown("**2. Upload** images above")
    with col3: st.markdown("**3. Adjust** settings in sidebar")
    with col4: st.markdown("**4. Download** CSV/Excel for analysis")
    st.stop()

st.markdown("---")
st.markdown(f"**{len(uploaded)} image(s)** · Scale: {scale:.5f} cm/px · Method: {seg_method}")

results = []
prog = st.progress(0, text="Starting…")

for i, uf in enumerate(uploaded[:20]):
    prog.progress(i / len(uploaded), text=f"Processing {uf.name} ({i+1}/{len(uploaded)})…")
    pil_img = Image.open(uf)
    bgr = pil_to_bgr(pil_img)
    try:
        mask   = segment_leaf(bgr, seg_method, thresh_val, fill_sz)
        traits = extract_traits(bgr, mask, scale)
        traits.update({"filename": uf.name, "status": "OK",
                        "_bgr": bgr, "_mask": mask})
    except Exception as e:
        traits = {"filename": uf.name, "status": f"ERROR: {e}"}
    results.append(traits)

prog.progress(1.0, text="Done ✓")

# ── Summary metrics ───────────────────────────────────────────────────────────
ok = [r for r in results if r.get("status") == "OK"]
if ok:
    areas = [r["area_cm2"] for r in ok if "area_cm2" in r]
    circs = [r["circularity"] for r in ok if "circularity" in r]
    solds = [r["solidity"] for r in ok if "solidity" in r]
    gi    = [r["greenness_index"] for r in ok if "greenness_index" in r]

    m1,m2,m3,m4 = st.columns(4)
    with m1: st.markdown(f'<div class="metric-card"><h4>leaves</h4><p>{len(ok)}</p></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="metric-card"><h4>mean area</h4><p>{np.mean(areas):.2f} <span style="font-size:14px">cm²</span></p></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="metric-card"><h4>mean circularity</h4><p>{np.mean(circs):.3f}</p></div>', unsafe_allow_html=True)
    with m4: st.markdown(f'<div class="metric-card"><h4>greenness index</h4><p>{np.mean(gi):.3f}</p></div>', unsafe_allow_html=True)

# ── Per-leaf ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">Per-leaf results</div>', unsafe_allow_html=True)

DISPLAY = {
    "area_cm2":"Leaf area (cm²)", "perimeter_cm":"Perimeter (cm)",
    "width_cm":"Width (cm)", "height_cm":"Height (cm)",
    "aspect_ratio":"Aspect ratio", "circularity":"Circularity",
    "solidity":"Solidity", "eccentricity":"Eccentricity",
    "convex_hull_area_cm2":"Convex hull area (cm²)",
    "ellipse_major_cm":"Ellipse major axis (cm)",
    "ellipse_minor_cm":"Ellipse minor axis (cm)",
    "red_mean":"Red mean", "green_mean":"Green mean", "blue_mean":"Blue mean",
    "red_std":"Red std", "green_std":"Green std", "blue_std":"Blue std",
    "hue_mean":"Hue mean", "sat_mean":"Saturation mean", "val_mean":"Value mean",
    "L_mean":"L* (lightness)", "a_mean":"a* (green-magenta)", "b_lab_mean":"b* (blue-yellow)",
    "greenness_index":"Greenness index (G/RGB)",
}

for r in results:
    area_label = f"{r['area_cm2']} cm²" if "area_cm2" in r else r.get("status","")
    with st.expander(f"🍃 {r['filename']}  —  {area_label}", expanded=False):
        if r.get("status") != "OK":
            st.error(r["status"]); continue

        c1, c2, c3 = st.columns([1, 1, 1.8])
        with c1:
            st.caption("Original")
            st.image(bgr_to_rgb(r["_bgr"]), use_container_width=True)
        with c2:
            if show_mask:
                st.caption("Segmentation overlay")
                st.image(overlay_mask(r["_bgr"], r["_mask"]), use_container_width=True)
        with c3:
            st.caption("Extracted traits")
            rows = [{"Trait": lbl, "Value": r[k]}
                    for k, lbl in DISPLAY.items() if k in r]
            if rows:
                st.dataframe(pd.DataFrame(rows).set_index("Trait"),
                             use_container_width=True, height=360)

# ── Charts ────────────────────────────────────────────────────────────────────
if show_charts and ok:
    st.markdown('<div class="section-head">Summary charts</div>', unsafe_allow_html=True)
    df_p = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_") and k!="status"}
                          for r in ok])
    df_p["filename"] = df_p["filename"].apply(lambda x: x[:18]+"…" if len(x)>18 else x)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Leaf area by sample")
        fig = make_bar_chart(df_p, "area_cm2", "Area (cm²)")
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    with c2:
        st.caption("RGB channel means")
        fig = make_color_chart(df_p)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    c3, c4 = st.columns(2)
    with c3:
        st.caption("Circularity")
        fig = make_bar_chart(df_p, "circularity", "Circularity (0–1)")
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    with c4:
        st.caption("Solidity")
        fig = make_bar_chart(df_p, "solidity", "Solidity (0–1)")
        st.pyplot(fig, use_container_width=True); plt.close(fig)

# ── Full table + export ───────────────────────────────────────────────────────
if ok:
    st.markdown('<div class="section-head">Full results table</div>', unsafe_allow_html=True)
    df_out = pd.DataFrame([{k:v for k,v in r.items()
                             if not k.startswith("_") and k != "status"}
                            for r in ok])
    cols = ["filename"] + [c for c in df_out.columns if c != "filename"]
    df_out = df_out[cols]
    st.dataframe(df_out, use_container_width=True, height=260)

    d1, d2, _ = st.columns([1,1,4])
    with d1:
        st.download_button("⬇ CSV", to_csv(df_out),
                           "leaf_results.csv", "text/csv")
    if do_excel:
        with d2:
            st.download_button("⬇ Excel", to_excel(df_out),
                               "leaf_results.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ── Trait guide ───────────────────────────────────────────────────────────────
with st.expander("📚 Trait guide for students", expanded=False):
    st.markdown(f"""
**Morphological traits**

| Trait | Meaning | Typical grape leaf range |
|-------|---------|--------------------------|
| Area (cm²) | Total leaf surface area | 15–80 cm² |
| Circularity | How circular (1 = perfect circle) | 0.45–0.75 |
| Solidity | Leaf area / convex hull area — lower = more lobed | 0.80–0.96 |
| Eccentricity | 0 = circle, 1 = line — measures elongation | 0.60–0.85 |
| Aspect ratio | Width / height | 0.9–1.3 |

**Color traits**

| Trait | Meaning |
|-------|---------|
| Green mean | Higher = greener / healthier leaf |
| a* (LAB) | Negative = green; closer to 0 or positive = yellowing/stress |
| Greenness index | Green / (R+G+B) — simple chlorophyll proxy |
| Saturation mean | Higher = more vivid color |

**Scale used:** {scale:.5f} cm/px  
Area and length values are only accurate when images are taken at a consistent height above the printed ruler background.

**Pipeline equivalent (PlantCV functions):**  
`readimage → rgb2gray_hsv → threshold.binary → fill → dilate → analyze.size → analyze.color`
    """)

st.markdown("---")
st.caption("Leaf Phenotyping Analyzer · UMES DAFRS · pnatarajan@umes.edu")
