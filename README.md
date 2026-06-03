# PlantCV Leaf Analyzer — Streamlit App

**University of Maryland Eastern Shore | Natarajan Lab**
Department of Agriculture, Food and Resource Sciences

---

## What it does
Upload grape leaf images photographed on the black matte background with ruler.
The app segments each leaf and extracts:

**Morphological traits**
- Leaf area (cm²), perimeter, width, height, longest path
- Aspect ratio, circularity, solidity, eccentricity
- Convex hull area

**Color traits**
- RGB channel means
- HSV (hue, saturation, value) means
- CIE LAB (lightness, green-magenta, blue-yellow) means

Results export to CSV or Excel for downstream analysis in R or SPSS.

---

## Local development

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy to Streamlit Community Cloud

1. Push this folder to a **public GitHub repo** (e.g. `plantcv-leaf-analyzer`)
2. Go to https://share.streamlit.io → New app
3. Set:
   - Repository: `purushothn/plantcv-leaf-analyzer`
   - Branch: `main`
   - Main file: `app.py`
4. Click **Deploy** — done!

> **Note:** Streamlit Community Cloud has a 1 GB memory limit.
> For batches >10 high-resolution images, consider running locally or on a server.

---

## Sidebar settings explained

| Setting | Recommendation |
|---------|---------------|
| HSV (saturation channel) | Best for healthy green leaves on black background |
| LAB (green-magenta) | Better for yellowing/stressed leaves |
| Otsu auto-threshold | Use when lighting varies across images |
| Threshold value | Lower = include more leaf area; higher = stricter |
| Fill noise size | 150–300 px removes small background noise particles |
| Scale (cm/px) | 300 DPI print → 0.00847 cm/px; adjust for your setup |

---

## Using with the ruler background PDF

1. Print `grape_leaf_background.pdf` at **100% size, 300 DPI** on matte paper
2. Photograph leaf flat on background from directly above (~50 cm height)
3. Keep consistent distance for all images in a batch
4. Upload images → sidebar scale is auto-set for 300 DPI print

---

## Contact
Purushothaman Natarajan, PhD  
Assistant Professor of Bioinformatics & Statistics  
pnatarajan@umes.edu | github.com/purushothn
