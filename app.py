"""
PlantCV Leaf Analyzer — Streamlit App
University of Maryland Eastern Shore | Natarajan Lab
Developed for student-facing plant phenotyping workflows (grape leaf CV)
"""

import io
import warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")
import plantcv.plantcv as pcv
pcv.params.debug = None
pcv.params.verbose = False

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlantCV Leaf Analyzer",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] {background: #1a2e14;}
  [data-testid="stSidebar"] * {color: #d4e8c2 !important;}
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stCheckbox label,
  [data-testid="stSidebar"] .stNumberInput label {color: #d4e8c2 !important;}
  .metric-card {
    background: #f0f7e8; border-radius: 10px; padding: 1rem 1.25rem;
    border-left: 4px solid #639922; margin-bottom: .5rem;
  }
  .metric-card h4 {margin: 0; font-size: 12px; color: #5a6e4a; text-transform: uppercase; letter-spacing: .05em;}
  .metric-card p  {margin: 4px 0 0; font-size: 26px; font-weight: 600; color: #27500a;}
  .section-head {font-size: 14px; font-weight: 600; color: #3B6D11;
    border-bottom: 1px solid #c0dd97; padding-bottom: 4px; margin: 1.2rem 0 .75rem;}
  .pipeline-badge {
    display: inline-block; background: #EAF3DE; color: #3B6D11;
    border-radius: 12px; padding: 2px 10px; font-size: 12px; margin: 2px;
  }
  .info-callout {
    background: #f7fbf0; border: 1px solid #c0dd97; border-radius: 8px;
    padding: .75rem 1rem; font-size: 13px; color: #3B6D11; margin: .75rem 0;
  }
  .stDownloadButton > button {
    background: #639922 !important; color: white !important;
    border: none !important; border-radius: 8px !important;
  }
  .stDownloadButton > button:hover {background: #3B6D11 !important;}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def segment_leaf(bgr_img, method: str, threshold_val: int, fill_size: int):
    """Return binary mask using chosen segmentation strategy."""
    pcv.outputs.clear()
    if method == "HSV (saturation channel)":
        gray = pcv.rgb2gray_hsv(rgb_img=bgr_img, channel="s")
        thresh = pcv.threshold.binary(gray_img=gray, threshold=threshold_val, object_type="light")
    elif method == "LAB (green-magenta channel)":
        gray = pcv.rgb2gray_lab(rgb_img=bgr_img, channel="a")
        thresh = pcv.threshold.binary(gray_img=gray, threshold=threshold_val, object_type="dark")
    elif method == "Otsu auto-threshold":
        gray = pcv.rgb2gray_hsv(rgb_img=bgr_img, channel="s")
        thresh = pcv.threshold.otsu(gray_img=gray, object_type="light")
    else:  # Gaussian adaptive
        gray = pcv.rgb2gray_hsv(rgb_img=bgr_img, channel="v")
        thresh = pcv.threshold.gaussian(gray_img=gray, ksize=21, offset=threshold_val,
                                         object_type="light")
    # morphological cleanup
    filled = pcv.fill(bin_img=thresh, size=fill_size)
    dilated = pcv.dilate(gray_img=filled, ksize=3, i=1)
    return dilated


def extract_traits(bgr_img, mask, scale_cm_px: float):
    """Run PlantCV analysis pipeline; return trait dict."""
    pcv.outputs.clear()
    traits = {}
    try:
        pcv.analyze.size(img=bgr_img, labeled_mask=mask, n_labels=1)
        sz = pcv.outputs.observations.get("default_1", {})
        px_scale = scale_cm_px ** 2

        def get(key, default=np.nan):
            v = sz.get(key, {})
            return v.get("value", default) if isinstance(v, dict) else default

        traits["area_cm2"]        = round(get("area", 0) * px_scale, 4)
        traits["perimeter_cm"]    = round(get("perimeter", 0) * scale_cm_px, 4)
        traits["width_cm"]        = round(get("width", 0) * scale_cm_px, 4)
        traits["height_cm"]       = round(get("height", 0) * scale_cm_px, 4)
        traits["longest_path_cm"] = round(get("longest_path", 0) * scale_cm_px, 4)
        traits["solidity"]        = round(get("solidity", np.nan), 4)
        traits["eccentricity"]    = round(get("ellipse_eccentricity", np.nan), 4)
        traits["convex_hull_area_cm2"] = round(get("convex_hull_area", 0) * px_scale, 4)

        if traits["width_cm"] and traits["height_cm"]:
            traits["aspect_ratio"] = round(traits["width_cm"] / traits["height_cm"], 4)
        else:
            traits["aspect_ratio"] = np.nan

        if traits["perimeter_cm"] and traits["area_cm2"]:
            traits["circularity"] = round(
                4 * np.pi * traits["area_cm2"] / (traits["perimeter_cm"] ** 2), 4)
        else:
            traits["circularity"] = np.nan

    except Exception as e:
        traits["_size_error"] = str(e)

    # ── Color analysis ──
    try:
        pcv.outputs.clear()
        pcv.analyze.color(rgb_img=bgr_img, labeled_mask=mask, n_labels=1, colorspaces="all")
        co = pcv.outputs.observations.get("default_1", {})

        def chan_mean(key):
            v = co.get(key, {})
            freqs = v.get("value", []) if isinstance(v, dict) else []
            if not freqs:
                return np.nan
            bins = np.arange(len(freqs))
            total = sum(freqs)
            return round(float(np.dot(bins, freqs) / total) if total else np.nan, 2)

        traits["red_mean"]   = chan_mean("red_frequencies")
        traits["green_mean"] = chan_mean("green_frequencies")
        traits["blue_mean"]  = chan_mean("blue_frequencies")
        traits["hue_mean"]   = chan_mean("hue_frequencies")
        traits["sat_mean"]   = chan_mean("saturation_frequencies")
        traits["light_mean"] = chan_mean("lightness_frequencies")
        traits["green_mag_mean"] = chan_mean("green-magenta_frequencies")
        traits["blue_yel_mean"]  = chan_mean("blue-yellow_frequencies")
    except Exception as e:
        traits["_color_error"] = str(e)

    return traits


def overlay_mask(bgr_img, mask):
    """Return RGB image with green mask overlay."""
    overlay = bgr_to_rgb(bgr_img.copy())
    green = np.zeros_like(overlay)
    green[:, :, 1] = 180
    alpha = (mask > 0).astype(np.uint8)[:, :, np.newaxis] * 0.35
    overlay = (overlay * (1 - alpha) + green * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (60, 180, 60), 2)
    return overlay


def make_summary_bar_chart(df: pd.DataFrame, col: str, label: str):
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = plt.cm.Greens(np.linspace(0.4, 0.85, len(df)))
    bars = ax.barh(df["filename"], df[col], color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel(label, fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#f9fdf5")
    fig.patch.set_facecolor("#f9fdf5")
    for bar, val in zip(bars, df[col]):
        ax.text(val + max(df[col]) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8, color="#3B6D11")
    plt.tight_layout()
    return fig


def make_color_hist(df: pd.DataFrame):
    channels = ["red_mean", "green_mean", "blue_mean"]
    colors   = ["#e05252", "#52a852", "#5285e0"]
    labels   = ["Red", "Green", "Blue"]
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(df))
    w = 0.25
    for i, (ch, col, lab) in enumerate(zip(channels, colors, labels)):
        ax.bar(x + i * w, df[ch], width=w, color=col, label=lab, alpha=0.85, edgecolor="white")
    ax.set_xticks(x + w)
    ax.set_xticklabels(df["filename"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean channel value (0–255)", fontsize=9)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#f9fdf5")
    fig.patch.set_facecolor("#f9fdf5")
    plt.tight_layout()
    return fig


def df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def df_to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="PlantCV Results")
        ws = writer.sheets["PlantCV Results"]
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = 18
    return buf.getvalue()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://www.umes.edu/wp-content/uploads/2023/06/UMES-logo-white.png",
             use_container_width=True)
    st.markdown("## 🍃 PlantCV Leaf Analyzer")
    st.markdown("**Natarajan Lab · UMES DAFRS**")
    st.markdown("---")

    st.markdown("### Segmentation")
    seg_method = st.selectbox(
        "Method",
        ["HSV (saturation channel)", "LAB (green-magenta channel)",
         "Otsu auto-threshold", "Gaussian adaptive"],
        help="HSV works best for green leaves on dark/black background. LAB suits stressed/yellowing leaves."
    )
    thresh_val = st.slider("Threshold value", 20, 200, 60,
                           help="Ignored for Otsu (auto). For HSV: higher = stricter green selection.")
    fill_sz    = st.slider("Fill noise size (px)", 10, 500, 150,
                           help="Remove objects smaller than this (noise elimination)")

    st.markdown("### Scale calibration")
    scale_mode = st.radio("Scale input", ["Use ruler background (cm/px)", "Enter manually"])
    if scale_mode == "Enter manually":
        scale_cm_px = st.number_input("cm per pixel", value=0.0264, step=0.001, format="%.4f",
                                      help="Measure a known distance in pixels on your image to calculate this.")
    else:
        ruler_dpi = st.selectbox("Print DPI of background PDF", [72, 96, 150, 300], index=3)
        scale_cm_px = 2.54 / ruler_dpi
        st.caption(f"→ {scale_cm_px:.5f} cm/px at {ruler_dpi} DPI")

    st.markdown("### Output")
    show_mask    = st.checkbox("Show segmentation overlay", value=True)
    show_charts  = st.checkbox("Show summary charts", value=True)
    export_excel = st.checkbox("Include Excel export", value=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#9bbf80; line-height:1.6'>
    <b>Pipeline (PlantCV v4)</b><br>
    readimage → rgb2gray → threshold<br>
    → fill → dilate → analyze.size<br>
    → analyze.color → outputs
    </div>
    """, unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🍃 PlantCV Leaf Phenotyping Tool")
st.markdown(
    "Upload leaf images photographed on the **black matte background with ruler**. "
    "The app segments each leaf, extracts morphological and color traits via PlantCV, "
    "and exports a results table for statistical analysis."
)

st.markdown('<div class="info-callout">📋 <b>Tip for students:</b> Place your leaf flat on the printed black background sheet, photograph from directly above in good lighting, then upload here. Each image = one leaf.</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload leaf images",
    type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
    accept_multiple_files=True,
    help="Up to 20 images per batch. Use the black ruler background for accurate scale."
)

if not uploaded:
    st.info("👆 Upload one or more leaf images to begin analysis.")
    st.markdown("---")
    st.markdown("### How it works")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**1. Photograph** leaf on black background with ruler")
    with col2:
        st.markdown("**2. Upload** image(s) above")
    with col3:
        st.markdown("**3. Adjust** segmentation settings in sidebar")
    with col4:
        st.markdown("**4. Download** CSV/Excel results for R/SPSS analysis")
    st.stop()

# ── Run analysis ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"**{len(uploaded)} image(s) uploaded** · Scale: {scale_cm_px:.5f} cm/px · Method: {seg_method}")

all_results = []
progress = st.progress(0, text="Starting analysis...")

for i, uf in enumerate(uploaded[:20]):
    progress.progress((i) / len(uploaded), text=f"Processing {uf.name} ({i+1}/{len(uploaded)})…")
    pil_img = Image.open(uf)
    bgr     = pil_to_bgr(pil_img)

    try:
        mask   = segment_leaf(bgr, seg_method, thresh_val, fill_sz)
        traits = extract_traits(bgr, mask, scale_cm_px)
        traits["filename"] = uf.name
        traits["status"]   = "OK"
        traits["_bgr"]     = bgr
        traits["_mask"]    = mask
    except Exception as e:
        traits = {"filename": uf.name, "status": f"ERROR: {e}"}

    all_results.append(traits)

progress.progress(1.0, text="Analysis complete ✓")

# ── Metrics strip ─────────────────────────────────────────────────────────────
ok_results = [r for r in all_results if r.get("status") == "OK"]
if ok_results:
    areas = [r["area_cm2"] for r in ok_results if "area_cm2" in r]
    circs = [r["circularity"] for r in ok_results if "circularity" in r]
    solds = [r["solidity"] for r in ok_results if "solidity" in r]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><h4>leaves analyzed</h4><p>{len(ok_results)}</p></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><h4>mean leaf area</h4><p>{np.mean(areas):.2f} <span style="font-size:14px">cm²</span></p></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><h4>mean circularity</h4><p>{np.mean(circs):.3f}</p></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><h4>mean solidity</h4><p>{np.mean(solds):.3f}</p></div>', unsafe_allow_html=True)

# ── Per-leaf results ──────────────────────────────────────────────────────────
st.markdown('<div class="section-head">Per-leaf analysis</div>', unsafe_allow_html=True)

for r in all_results:
    fname = r["filename"]
    with st.expander(f"🍃 {fname}  —  {r.get('area_cm2', 'N/A')} cm²", expanded=False):
        if r.get("status") != "OK":
            st.error(r["status"])
            continue

        col_img, col_mask, col_data = st.columns([1, 1, 1.8])

        with col_img:
            st.caption("Original")
            st.image(bgr_to_rgb(r["_bgr"]), use_container_width=True)

        with col_mask:
            if show_mask:
                st.caption("Segmentation overlay")
                overlay = overlay_mask(r["_bgr"], r["_mask"])
                st.image(overlay, use_container_width=True)

        with col_data:
            st.caption("Extracted traits")
            display_keys = {
                "area_cm2": "Leaf area (cm²)",
                "perimeter_cm": "Perimeter (cm)",
                "width_cm": "Width (cm)",
                "height_cm": "Height (cm)",
                "longest_path_cm": "Longest path (cm)",
                "aspect_ratio": "Aspect ratio",
                "circularity": "Circularity",
                "solidity": "Solidity",
                "eccentricity": "Eccentricity",
                "convex_hull_area_cm2": "Convex hull area (cm²)",
                "red_mean": "Red channel mean",
                "green_mean": "Green channel mean",
                "blue_mean": "Blue channel mean",
                "hue_mean": "Hue mean",
                "sat_mean": "Saturation mean",
                "light_mean": "Lightness mean",
                "green_mag_mean": "Green-magenta (LAB a*) mean",
                "blue_yel_mean": "Blue-yellow (LAB b*) mean",
            }
            rows = []
            for k, label in display_keys.items():
                if k in r:
                    rows.append({"Trait": label, "Value": r[k]})
            if rows:
                tdf = pd.DataFrame(rows).set_index("Trait")
                st.dataframe(tdf, use_container_width=True, height=340)

# ── Summary charts ─────────────────────────────────────────────────────────────
if show_charts and ok_results:
    st.markdown('<div class="section-head">Summary charts</div>', unsafe_allow_html=True)

    df_plot = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_") and k != "status"}
        for r in ok_results
    ])
    # shorten filenames for plot labels
    df_plot["filename"] = df_plot["filename"].apply(lambda x: x[:20] + "…" if len(x) > 20 else x)

    ch1, ch2 = st.columns(2)
    with ch1:
        st.caption("Leaf area by sample")
        fig1 = make_summary_bar_chart(df_plot, "area_cm2", "Area (cm²)")
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)
    with ch2:
        st.caption("RGB channel means")
        fig2 = make_color_hist(df_plot)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    ch3, ch4 = st.columns(2)
    with ch3:
        st.caption("Circularity by sample")
        fig3 = make_summary_bar_chart(df_plot, "circularity", "Circularity (0–1)")
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)
    with ch4:
        st.caption("Solidity by sample")
        fig4 = make_summary_bar_chart(df_plot, "solidity", "Solidity (0–1)")
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

# ── Results table + export ────────────────────────────────────────────────────
if ok_results:
    st.markdown('<div class="section-head">Full results table</div>', unsafe_allow_html=True)

    df_out = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_") and k not in ("status",)}
        for r in ok_results
    ])
    # reorder: filename first
    cols = ["filename"] + [c for c in df_out.columns if c != "filename"]
    df_out = df_out[cols]

    st.dataframe(df_out, use_container_width=True, height=280)

    dl1, dl2, dl3 = st.columns([1, 1, 4])
    with dl1:
        st.download_button(
            "⬇ Download CSV",
            data=df_to_csv(df_out),
            file_name="plantcv_leaf_results.csv",
            mime="text/csv",
        )
    if export_excel:
        with dl2:
            st.download_button(
                "⬇ Download Excel",
                data=df_to_excel(df_out),
                file_name="plantcv_leaf_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ── PlantCV pipeline reference ────────────────────────────────────────────────
with st.expander("📚 PlantCV pipeline reference (for students)", expanded=False):
    st.markdown("""
**This app runs the following PlantCV functions on each image:**

| Step | PlantCV function | Purpose |
|------|-----------------|---------|
| 1 | `pcv.readimage()` | Load image into array |
| 2 | `pcv.rgb2gray_hsv(channel='s')` | Convert to saturation grayscale |
| 3 | `pcv.threshold.binary()` / `otsu()` | Separate leaf from background |
| 4 | `pcv.fill(size=N)` | Remove noise pixels |
| 5 | `pcv.dilate(ksize=3)` | Smooth mask edges |
| 6 | `pcv.analyze.size()` | Extract morphological traits |
| 7 | `pcv.analyze.color()` | Extract color channel statistics |
| 8 | `pcv.outputs.observations` | Collect all trait values |

**Trait guide:**
- **Circularity** (0–1): 1 = perfect circle. Grape leaves: 0.5–0.75 typical.
- **Solidity**: leaf area / convex hull area. Lower = more lobed/serrated leaf margins.
- **Eccentricity** (0–1): 0 = circle, 1 = line. Indicates leaf elongation.
- **Green-magenta (a*)**: negative values = greener leaf; more negative = higher chlorophyll proxy.
- **Aspect ratio**: width/height. > 1 = wider than tall.

**Scale:** Measured at {:.5f} cm/px. Area and length values are only accurate if images were taken at a fixed, consistent distance from the background sheet.
    """.format(scale_cm_px))

st.markdown("---")
st.caption("PlantCV Leaf Analyzer · UMES Department of Agriculture, Food & Resource Sciences · pnatarajan@umes.edu")
