"""
app.py — CropDoctor Streamlit Web Application
==============================================
Run:
    streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st
from PIL import Image

from advice import get_advice, get_advice_dict
from predictor import CropDoctorPredictor

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropDoctor — AI Crop Disease Detection",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Constants ─────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
VALID_EXTENSIONS     = {"jpg", "jpeg", "png"}
MODELS_DIR           = Path("models")
CHECKPOINT           = MODELS_DIR / "best_model.pth"
CLASS_JSON           = MODELS_DIR / "class_names.json"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container {max-width: 740px; padding: 1.5rem 1.5rem 3rem;}

    h1 {font-size: 1.85rem !important; font-weight: 700;}
    h2 {font-size: 1.25rem !important; font-weight: 600;}
    h3 {font-size: 1.05rem !important; font-weight: 600;}

    /* ── Cards ── */
    .card {
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin: 0.6rem 0;
    }
    .card-disease  { background:#fff4f4; border-left:4px solid #c0392b; }
    .card-healthy  { background:#f0faf4; border-left:4px solid #2d8a4e; }
    .card-warning  { background:#fff8ec; border-left:4px solid #e6900a; }
    .card-error    { background:#fff0f0; border-left:4px solid #c0392b; }
    .card-info     { background:#f0f7ff; border-left:4px solid #3b82d4; }

    .card h2 { margin:0 0 0.4rem !important; }
    .card p  { margin:0; font-size:0.92rem; line-height:1.6; color:#333; }

    /* ── Confidence badge ── */
    .badge {
        display:inline-block; border-radius:20px;
        padding:0.2rem 0.75rem; font-size:0.82rem;
        font-weight:600; margin-left:0.5rem; vertical-align:middle;
    }
    .badge-green  { background:#d4edda; color:#155724; }
    .badge-amber  { background:#fff3cd; color:#856404; }
    .badge-red    { background:#f8d7da; color:#721c24; }

    /* ── Probability bars ── */
    .prob-row {
        display:flex; align-items:center; gap:0.65rem; margin-bottom:0.5rem;
    }
    .prob-label {
        width:210px; font-size:0.82rem; color:#333;
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
        flex-shrink:0;
    }
    .prob-track {
        flex:1; background:#e9ecef; border-radius:6px; height:11px; overflow:hidden;
    }
    .prob-fill  { height:11px; border-radius:6px; }
    .prob-pct   {
        width:40px; text-align:right; font-size:0.79rem;
        font-weight:600; color:#444; flex-shrink:0;
    }

    /* ── Advice tips ── */
    .tip {
        display:flex; align-items:flex-start; gap:0.5rem;
        background:#f0faf4; border:1px solid #c8ead6;
        border-radius:7px; padding:0.4rem 0.7rem;
        margin:0.22rem 0; font-size:0.84rem; color:#1a3d2b; line-height:1.5;
    }
    .tip-icon { flex-shrink:0; }

    /* ── Advice panel ── */
    .advice-hdr {
        background:linear-gradient(90deg,#1a6b3c,#2d8a4e);
        color:#fff; border-radius:9px 9px 0 0;
        padding:0.75rem 1.2rem; font-size:0.97rem; font-weight:600;
        margin-top:1.3rem;
    }
    .advice-body {
        background:#fff; border:1px solid #dde; border-top:none;
        border-radius:0 0 9px 9px; padding:1.1rem 1.2rem;
    }
    .section-lbl {
        font-size:0.88rem; font-weight:700; color:#1a6b3c;
        margin:1rem 0 0.4rem; text-transform:uppercase; letter-spacing:0.4px;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        font-size:0.78rem; color:#666; background:#fafafa;
        border:1px solid #e0e0e0; border-radius:8px;
        padding:0.8rem 1rem; margin-top:1.8rem; line-height:1.6;
    }

    /* ── Predict button ── */
    .stButton > button[kind="primary"] {
        background:linear-gradient(135deg,#1a6b3c,#2d8a4e) !important;
        border:none !important; border-radius:9px !important;
        font-size:1rem !important; font-weight:600 !important;
        padding:0.6rem 1.4rem !important;
        box-shadow:0 3px 10px rgba(45,138,78,0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow:0 5px 16px rgba(45,138,78,0.42) !important;
        transform:translateY(-1px) !important;
    }

    /* ── Uploaded image ── */
    .stImage img { border-radius:10px; box-shadow:0 2px 10px rgba(0,0,0,0.1); }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] { background:#f4faf6; }
</style>
""", unsafe_allow_html=True)


# ── Cached loader ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CropDoctor model ...")
def load_predictor() -> CropDoctorPredictor:
    return CropDoctorPredictor(
        checkpoint_path=str(CHECKPOINT),
        class_json_path=str(CLASS_JSON),
    )


def model_is_ready() -> bool:
    return CHECKPOINT.exists() and CLASS_JSON.exists()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _fmt(cls: str) -> str:
    return cls.replace("___", " - ").replace("_", " ").strip()

def _badge(prob: float) -> str:
    cls = "badge-green" if prob >= 0.70 else "badge-amber" if prob >= 0.40 else "badge-red"
    return f'<span class="badge {cls}">{prob*100:.1f}%</span>'

def _bar_color(prob: float) -> str:
    return "#2d8a4e" if prob >= 0.70 else "#e6900a" if prob >= 0.40 else "#c0392b"

def render_prob_bars(top_k: list):
    html = ""
    for i, entry in enumerate(top_k):
        pct   = entry["probability"] * 100
        color = _bar_color(entry["probability"])
        bold  = "font-weight:600;" if i == 0 else ""
        html += (
            f'<div class="prob-row">'
            f'<div class="prob-label" style="{bold}">{_fmt(entry["class"])}</div>'
            f'<div class="prob-track">'
            f'<div class="prob-fill" style="width:{pct:.1f}%;background:{color}"></div>'
            f'</div>'
            f'<div class="prob-pct">{pct:.1f}%</div>'
            f'</div>'
        )
    st.markdown(html, unsafe_allow_html=True)


def render_advice(class_name: str, is_healthy: bool):
    info = get_advice_dict(class_name)
    if not info:
        st.markdown(get_advice(class_name, use_llm=True))
        return

    icon = "🌱" if is_healthy else "🩺"
    st.markdown(
        f'<div class="advice-hdr">{icon}&nbsp; Treatment &amp; Prevention</div>'
        '<div class="advice-body">',
        unsafe_allow_html=True,
    )

    if info.get("symptoms") and not is_healthy:
        st.markdown('<div class="section-lbl">Symptoms</div>', unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-size:0.87rem;color:#333;line-height:1.65;margin:0'>"
            f"{info['symptoms']}</p>",
            unsafe_allow_html=True,
        )

    cultural = info.get("cultural_controls", [])
    if cultural:
        label = "Maintenance Tips" if is_healthy else "Non-Chemical Controls (try first)"
        st.markdown(f'<div class="section-lbl">{label}</div>', unsafe_allow_html=True)
        tips = "".join(
            f'<div class="tip"><span class="tip-icon">✓</span><span>{t}</span></div>'
            for t in cultural
        )
        st.markdown(tips, unsafe_allow_html=True)

    chem = info.get("chemical_guidance")
    if chem and not is_healthy:
        st.markdown('<div class="section-lbl">Chemical Guidance (general only)</div>', unsafe_allow_html=True)
        st.markdown(
            f"<div style='background:#fffbf0;border:1px solid #edd87a;border-radius:7px;"
            f"padding:0.75rem 1rem;font-size:0.84rem;color:#5a4000;line-height:1.6'>{chem}</div>",
            unsafe_allow_html=True,
        )

    seek = info.get("when_to_seek_help")
    if seek:
        st.markdown('<div class="section-lbl">When to Seek Professional Help</div>', unsafe_allow_html=True)
        st.markdown(
            f"<div style='background:#fff0f0;border:1px solid #f0b8b8;border-radius:7px;"
            f"padding:0.75rem 1rem;font-size:0.84rem;color:#6b1a1a;line-height:1.6'>{seek}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🌿 About CropDoctor")
        st.markdown(
            "Uses **MobileNetV2** trained on the "
            "[PlantVillage dataset](https://plantvillage.psu.edu/) "
            "to detect common leaf diseases."
        )
        st.markdown("**Supported crops**")
        for line in [
            "🍅 Tomato — Early Blight, Late Blight",
            "🥔 Potato — Early Blight, Late Blight",
            "🫑 Bell Pepper — Bacterial Spot",
            "🌽 Maize — Common Rust, N. Leaf Blight",
        ]:
            st.markdown(f"- {line}")
        st.divider()
        st.markdown(
            f"**Confidence threshold:** {CONFIDENCE_THRESHOLD*100:.0f}%  \n"
            "Results below this are shown as low-confidence."
        )
        st.divider()
        st.caption("SDG 2 — Zero Hunger · MobileNetV2 · CPU inference")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # Header
    st.title("🌿 CropDoctor")
    st.markdown(
        "**AI-Based Crop Disease Detection** &nbsp;·&nbsp; "
        "*SDG 2 — Zero Hunger*",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Upload a clear photo of a single crop leaf to detect disease "
        "and receive practical care guidance."
    )
    st.divider()

    # Model check
    if not model_is_ready():
        st.markdown("""
        <div class="card card-warning">
        <h2>⚠️ Model Not Trained Yet</h2>
        <p>Run training first, then refresh this page:</p>
        </div>
        """, unsafe_allow_html=True)
        st.code(
            "python train.py --data_dir data/PlantVillage --epochs 10 --samples_per_class 200",
            language="bash",
        )
        st.stop()

    predictor = load_predictor()

    # Upload
    st.subheader("📷 Upload a Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG photo of a single crop leaf",
        type=list(VALID_EXTENSIONS),
        help="Best results: clear, well-lit, single leaf filling the frame.",
    )

    if uploaded_file is None:
        st.markdown("""
        <div class="card card-info">
        <p>
        👆 <strong>Upload an image above to begin.</strong><br><br>
        For best results:<br>
        &nbsp; • Use natural daylight or bright indoor lighting<br>
        &nbsp; • Show one leaf with the symptom clearly visible<br>
        &nbsp; • Keep the image sharp and in focus<br>
        &nbsp; • Avoid backgrounds with many other leaves
        </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Validate extension
    suffix = Path(uploaded_file.name).suffix.lstrip(".").lower()
    if suffix not in VALID_EXTENSIONS:
        st.markdown(
            f'<div class="card card-error"><h2>❌ Invalid File Type</h2>'
            f'<p>Please upload a JPG or PNG image (got <code>.{suffix}</code>).</p></div>',
            unsafe_allow_html=True,
        )
        return

    # Open image
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.markdown(
            f'<div class="card card-error"><h2>❌ Could Not Read Image</h2>'
            f'<p>{exc}</p></div>',
            unsafe_allow_html=True,
        )
        return

    # Display image + metadata
    col_img, col_meta = st.columns([2, 1])
    with col_img:
        st.image(pil_image, caption=uploaded_file.name, width="stretch")
    with col_meta:
        w, h = pil_image.size
        st.markdown(f"**File**  \n`{uploaded_file.name}`")
        st.markdown(f"**Size**  \n{w} × {h} px")
        st.markdown(f"**Format**  \n{suffix.upper()}")

    st.divider()

    # Predict button
    predict_clicked = st.button(
        "🔍  Predict Disease", type="primary", width="stretch"
    )

    if not predict_clicked:
        return

    # Inference
    with st.spinner("Analysing leaf image ..."):
        result = predictor.predict(pil_image, top_k=3)

    if result.get("error"):
        st.markdown(
            f'<div class="card card-error"><h2>❌ Prediction Error</h2>'
            f'<p>{result["error"]}</p></div>',
            unsafe_allow_html=True,
        )
        return

    predicted_class = result["predicted_class"]
    confidence      = result["confidence"]
    top_k           = result["top_k"]
    is_healthy      = result["is_healthy"]
    display_name    = _fmt(predicted_class)
    conf_pct        = confidence * 100

    # Diagnosis card
    st.subheader("📊 Results")

    if confidence < CONFIDENCE_THRESHOLD:
        st.markdown(f"""
        <div class="card card-warning">
          <h2>⚠️ Low-Confidence Prediction</h2>
          <p>
            The model is only <strong>{conf_pct:.1f}%</strong> confident
            (threshold: {CONFIDENCE_THRESHOLD*100:.0f}%) — not enough for a reliable diagnosis.<br><br>
            <strong>Try:</strong> retake the photo in good, even lighting with the leaf
            filling the frame.  If the problem persists, contact your local agriculture officer.
          </p>
        </div>
        """, unsafe_allow_html=True)

    elif is_healthy:
        st.markdown(f"""
        <div class="card card-healthy">
          <h2>✅ Leaf Appears Healthy</h2>
          <p>Identified as <strong>{display_name}</strong> {_badge(confidence)}</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="card card-disease">
          <h2>🔴 Disease Detected</h2>
          <p>Predicted condition: <strong>{display_name}</strong> {_badge(confidence)}</p>
        </div>
        """, unsafe_allow_html=True)

    # Probability bars
    st.markdown(
        "<div style='font-size:0.82rem;font-weight:600;color:#555;"
        "text-transform:uppercase;letter-spacing:0.4px;margin:1rem 0 0.4rem'>"
        "Top Predictions</div>",
        unsafe_allow_html=True,
    )
    render_prob_bars(top_k)

    # Advice
    if confidence >= CONFIDENCE_THRESHOLD:
        render_advice(predicted_class, is_healthy)


if __name__ == "__main__":
    main()
