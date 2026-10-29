"""
app.py — Streamlit Glaucoma OCT Simulation App
Jalankan: streamlit run app.py
"""

import io
import time
import warnings
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

import pipeline as P

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "Glaucoma OCT Simulation",
    page_icon   = "🔬",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.stApp { background-color: #0c0c18; color: white; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b4b 0%, #0c0c18 100%);
    border-right: 1px solid #2244aa;
}
[data-testid="stSidebar"] * { color: #dde8ff !important; }
.oct-header {
    background: linear-gradient(90deg, #003399, #0055cc);
    padding: 12px 20px; border-radius: 8px;
    margin-bottom: 16px; border: 1px solid #2255cc;
}
.oct-header h1 { color: white; font-size: 1.4rem; margin:0; }
.oct-header p  { color: #88aaff; margin:0; font-size:0.85rem; }
.metric-card {
    background: #111122; border: 1px solid #2244aa;
    border-radius: 10px; padding: 14px 16px;
    text-align: center; height: 100%;
}
.metric-card .label { color: #88aaff; font-size:0.75rem; margin-bottom:4px; }
.metric-card .value { color: white; font-size:1.6rem; font-weight:700; }
.metric-card .sub   { color: #aaaaaa; font-size:0.7rem; margin-top:2px; }
.risk-normal   { color:#228822; font-weight:700; font-size:1.1rem; }
.risk-border   { color:#dd8800; font-weight:700; font-size:1.1rem; }
.risk-suspect  { color:#cc4400; font-weight:700; font-size:1.1rem; }
.risk-high     { color:#cc0000; font-weight:700; font-size:1.1rem; }
.result-glau {
    background: linear-gradient(90deg,#3a0000,#660000);
    border: 2px solid #cc0000;
    border-radius:10px; padding:16px 24px; text-align:center;
}
.result-normal {
    background: linear-gradient(90deg,#002200,#004400);
    border: 2px solid #00aa00;
    border-radius:10px; padding:16px 24px; text-align:center;
}
.result-glau h2, .result-normal h2 { margin:0; font-size:2rem; }
hr { border-color: #2244aa !important; }
.stSpinner > div { border-top-color: #4488ff !important; }
.stTabs [data-baseweb="tab"] {
    background: #111122; color: #aaaacc;
    border-radius: 6px 6px 0 0;
}
.stTabs [aria-selected="true"] {
    background: #1a2a66 !important; color: white !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading segmentation model...")
def load_seg(path: str):
    return P.load_seg_model(path)

@st.cache_resource(show_spinner="Loading classifier...")
def load_cls(path: str, backbone: str):
    return P.load_cls_model(path, backbone)

def fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()

def risk_html(risk: str) -> str:
    cls_map = {
        "Normal":       "risk-normal",
        "Borderline":   "risk-border",
        "Suspect":      "risk-suspect",
        "High Suspect": "risk-high",
    }
    css_cls = cls_map.get(risk, "risk-high")
    return '<span class="' + css_cls + '">' + risk + '</span>'

BACKBONE_OPTIONS = {
    "EfficientNet-B4"  : "efficientnet_b4",
    "ResNet-50"        : "resnet50",
    "DenseNet-121"     : "densenet121",
    "ConvNeXt-Tiny"    : "convnext_tiny",
    "Ensemble (semua)" : "ensemble",
}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔬 Glaucoma OCT Sim")
    st.markdown("---")

    st.markdown("### 📁 Path Model")
    seg_path = st.text_input(
        "Segmentation model (.pth)",
        value="models/best_model_fold3.pth",
        help="Path ke file best_model_fold3.pth",
    )

    st.markdown("### 🤖 Classifier Model")
    model_choice = st.selectbox(
        "Pilih model klasifikasi:",
        list(BACKBONE_OPTIONS.keys()),
        index=0,
    )
    selected_backbone = BACKBONE_OPTIONS[model_choice]

    cls_paths = {}
    if selected_backbone == "ensemble":
        st.markdown("**Path file per model:**")
        for display, bname in BACKBONE_OPTIONS.items():
            if bname == "ensemble":
                continue
            cls_paths[bname] = st.text_input(
                display + " (.pth)",
                value="models/cls_" + bname + "_best.pth",
                key="path_" + bname,
            )
    else:
        cls_paths[selected_backbone] = st.text_input(
            "Classifier path (.pth)",
            value="models/cls_" + selected_backbone + "_best.pth",
        )

    st.markdown("---")
    st.markdown("### ⚙️ Setting")
    threshold = st.slider(
        "Threshold glaukoma (%)", 30, 80, 50, 5,
        help="Probabilitas minimum untuk diklasifikasikan sebagai glaukoma"
    ) / 100.0

    show_gradcam = st.checkbox("Tampilkan Grad-CAM", value=True)
    download_fig = st.checkbox("Enable download hasil", value=True)

    st.markdown("---")
    st.markdown("### 📖 Referensi")
    st.markdown("""
- Garway-Heath et al. 2000
- Jonas et al. 2004
- Hood et al. 2013
- Cheng et al. 2013
    """)
    st.markdown("---")
    st.caption("v4.0 | UNet mit_b3 | OCT Sim RS-3000")


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="oct-header">
  <h1>🔬 Glaucoma Detection — OCT Simulation RS-3000</h1>
  <p>Simulasi output OCT Nidek RS-3000 dari fundus image menggunakan deep learning</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
col_up, col_info = st.columns([1, 1])

with col_up:
    st.markdown("#### 📤 Upload Fundus Image")
    uploaded = st.file_uploader(
        "Drag & drop atau klik untuk upload",
        type=["jpg","jpeg","png","bmp","tiff"],
        label_visibility="collapsed",
    )

with col_info:
    st.markdown("#### ℹ️ Cara Penggunaan")
    st.markdown("""
1. Isi path model di **sidebar kiri**
2. Pilih model klasifikasi
3. Upload gambar fundus mata
4. Klik **Analisis** dan tunggu hasil
5. Download gambar OCT simulation
    """)
    st.info("💡 Model segmentasi: **UNet mit_b3** Fold 3")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if uploaded is not None:
    pil_img  = Image.open(uploaded).convert("RGB")
    img_rgb  = np.array(pil_img)
    filename = uploaded.name

    prev_col, btn_col = st.columns([3, 1])
    with prev_col:
        st.markdown("**File:** `" + filename + "`  |  **Size:** " +
                    str(img_rgb.shape[1]) + "×" + str(img_rgb.shape[0]) + " px")
    with btn_col:
        run_btn = st.button("🔍 Analisis", type="primary", use_container_width=True)

    if run_btn:
        # Validasi seg model
        if not Path(seg_path).exists():
            st.error("❌ File segmentasi tidak ditemukan: " + seg_path)
            st.stop()

        progress = st.progress(0, text="Memuat model segmentasi...")
        t0 = time.time()

        # Load segmentasi
        try:
            seg_model = load_seg(seg_path)
        except Exception as e:
            st.error("❌ Gagal load seg model: " + str(e))
            st.stop()

        progress.progress(15, text="Segmentasi disc & cup...")

        # Pipeline
        try:
            result = P.run_pipeline(img_rgb, seg_model)
        except Exception as e:
            st.error("❌ Pipeline error: " + str(e))
            st.stop()

        progress.progress(50, text="Membangun visualisasi OCT...")

        # Visualisasi
        try:
            fig_oct = P.build_oct_figure(result, filename)
        except Exception as e:
            st.error("❌ Visualisasi error: " + str(e))
            st.stop()

        progress.progress(65, text="Menjalankan klasifikasi...")

        # Klasifikasi
        prob_glau_list  = []
        cls_detail_rows = []

        def run_single_cls(bname, bpath):
            if not Path(bpath).exists():
                return None, None, None
            try:
                m, cm, cs = load_cls(bpath, bname)
                pg, lbl, pn = P.classify(result, m, cm, cs)
                return pg, lbl, pn
            except Exception as e:
                st.warning("⚠️ Model " + bname + " gagal: " + str(e))
                return None, None, None

        if selected_backbone == "ensemble":
            for bname, bpath in cls_paths.items():
                pg, lbl, pn = run_single_cls(bname, bpath)
                if pg is not None:
                    prob_glau_list.append(pg)
                    display_name = [k for k,v in BACKBONE_OPTIONS.items() if v == bname][0]
                    cls_detail_rows.append((display_name, pg, lbl))
            if not prob_glau_list:
                st.warning("⚠️ Tidak ada model classifier yang berhasil dimuat.")
                prob_glau  = None
                pred_label = "N/A"
            else:
                prob_glau  = float(np.mean(prob_glau_list))
                pred_label = "GLAUKOMA" if prob_glau >= threshold else "NORMAL"
        else:
            bpath = list(cls_paths.values())[0]
            pg, lbl, pn = run_single_cls(selected_backbone, bpath)
            if pg is not None:
                prob_glau  = pg
                pred_label = "GLAUKOMA" if pg >= threshold else "NORMAL"
                cls_detail_rows.append((model_choice, pg, lbl))
            else:
                prob_glau  = None
                pred_label = "N/A"

        progress.progress(80, text="Grad-CAM...")

        # Grad-CAM
        gradcam_data = None
        if show_gradcam and cls_detail_rows:
            try:
                if selected_backbone != "ensemble":
                    bname_cam = selected_backbone
                else:
                    bname_cam = list(cls_paths.keys())[0]
                bpath_cam = cls_paths[bname_cam]
                if Path(bpath_cam).exists():
                    m_cam, cm_cam, cs_cam = load_cls(bpath_cam, bname_cam)
                    gradcam_data = P.compute_gradcam(result, m_cam, cm_cam, cs_cam)
            except Exception as e:
                st.warning("⚠️ Grad-CAM gagal: " + str(e))

        progress.progress(95, text="Menyusun hasil...")
        elapsed = time.time() - t0
        progress.progress(100, text="Selesai dalam " + str(round(elapsed,1)) + "s ✅")
        time.sleep(0.3)
        progress.empty()

        st.markdown("---")

        # ── Metric cards ──────────────────────────────────────────────────────
        st.markdown("### 📊 Ringkasan Hasil")
        c1, c2, c3, c4, c5, c6 = st.columns(6)

        eye = result["eye_side"]
        eye_color = "#4488ff" if eye == "OD" else "#ff8844"
        eye_label = "Kanan" if eye == "OD" else "Kiri"
        with c1:
            st.markdown(
                '<div class="metric-card">'
                '<div class="label">Mata (OD/OS)</div>'
                '<div class="value" style="color:' + eye_color + '">' + eye + '</div>'
                '<div class="sub">' + eye_label + ' — ' + str(int(result["eye_conf"]*100)) + '% conf</div>'
                '</div>', unsafe_allow_html=True)

        with c2:
            st.markdown(
                '<div class="metric-card">'
                '<div class="label">C/D Horizontal</div>'
                '<div class="value" style="color:' + result["cdr_col"] + '">' + str(round(result["cdr_h"],3)) + '</div>'
                '<div class="sub">via ellipse fitting</div>'
                '</div>', unsafe_allow_html=True)

        with c3:
            st.markdown(
                '<div class="metric-card">'
                '<div class="label">C/D Vertikal</div>'
                '<div class="value" style="color:' + result["cdr_col"] + '">' + str(round(result["cdr_v"],3)) + '</div>'
                '<div class="sub">' + result["cdr_risk"] + '</div>'
                '</div>', unsafe_allow_html=True)

        with c4:
            st.markdown(
                '<div class="metric-card">'
                '<div class="label">RNFL Whole</div>'
                '<div class="value">' + str(int(result["whole_mean"])) + '</div>'
                '<div class="sub">µm (proxy)</div>'
                '</div>', unsafe_allow_html=True)

        si = result["SI_ratio"]
        si_col = "#228822" if 0.8 < si < 1.2 else "#cc4400"
        with c5:
            st.markdown(
                '<div class="metric-card">'
                '<div class="label">S/I Ratio</div>'
                '<div class="value" style="color:' + si_col + '">' + str(round(si,3)) + '</div>'
                '<div class="sub">Normal: 0.8-1.2</div>'
                '</div>', unsafe_allow_html=True)

        with c6:
            if prob_glau is not None:
                pg_pct = prob_glau * 100
                pg_col = "#cc0000" if pg_pct > 70 else ("#dd8800" if pg_pct > 40 else "#228822")
                risk_lbl = "🔴 TINGGI" if pg_pct > 70 else ("🟡 SEDANG" if pg_pct > 40 else "🟢 RENDAH")
                st.markdown(
                    '<div class="metric-card">'
                    '<div class="label">Prob Glaukoma</div>'
                    '<div class="value" style="color:' + pg_col + '">' + str(round(pg_pct,1)) + '%</div>'
                    '<div class="sub">' + risk_lbl + '</div>'
                    '</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="metric-card">'
                    '<div class="label">Prob Glaukoma</div>'
                    '<div class="value" style="color:#888">N/A</div>'
                    '<div class="sub">Model tidak dimuat</div>'
                    '</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Diagnosis banner ──────────────────────────────────────────────────
        if prob_glau is not None:
            if pred_label == "GLAUKOMA":
                st.markdown(
                    '<div class="result-glau">'
                    '<h2>🔴 DIAGNOSA: GLAUKOMA</h2>'
                    '<p style="color:#ffaaaa;margin-top:6px">'
                    'Probabilitas: <b>' + str(round(prob_glau*100,1)) + '%</b> | '
                    'CDR-V: <b>' + str(round(result["cdr_v"],3)) + '</b> | '
                    'Status: <b>' + result["cdr_risk"] + '</b> | '
                    'Model: <b>' + model_choice + '</b>'
                    '</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="result-normal">'
                    '<h2>🟢 DIAGNOSA: NORMAL</h2>'
                    '<p style="color:#aaffaa;margin-top:6px">'
                    'Probabilitas glaukoma: <b>' + str(round(prob_glau*100,1)) + '%</b> | '
                    'CDR-V: <b>' + str(round(result["cdr_v"],3)) + '</b> | '
                    'Status: <b>' + result["cdr_risk"] + '</b> | '
                    'Model: <b>' + model_choice + '</b>'
                    '</p></div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── TABS ─────────────────────────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs([
            "🖥️ OCT Simulation",
            "📋 Detail Klinis",
            "🔥 Grad-CAM XAI",
        ])

        with tab1:
            st.markdown("#### Simulasi Output OCT Nidek RS-3000")
            oct_bytes = fig_to_bytes(fig_oct)
            st.image(oct_bytes, use_container_width=True)
            plt.close(fig_oct)
            if download_fig:
                st.download_button(
                    label="⬇️ Download OCT Simulation (PNG)",
                    data=oct_bytes,
                    file_name="oct_sim_" + Path(filename).stem + ".png",
                    mime="image/png",
                    use_container_width=True,
                )

        with tab2:
            import pandas as pd
            st.markdown("#### 📋 Parameter Klinis Lengkap")
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**🎯 CDR & Disc**")
                cdr_data = {
                    "Parameter": [
                        "C/D Horizontal","C/D Vertikal","CDR Area","CDR Status",
                        "Disc Area (px)","Cup Area (px)","Ring Area (px)",
                        "R_disc (px)","Semi-a (px)","Semi-b (px)",
                        "inner_r (px)","outer_r (px)",
                    ],
                    "Nilai": [
                        str(round(result["cdr_h"],3)),
                        str(round(result["cdr_v"],3)),
                        str(round(result["cdr_area"],3)),
                        result["cdr_risk"],
                        str(result["disc_area"]),
                        str(result["cup_area"]),
                        str(result["ring_area"]),
                        str(round(result["R_disc"],1)),
                        str(round(result["semi_a"],1)),
                        str(round(result["semi_b"],1)),
                        str(round(result["inner_r"],1)),
                        str(round(result["outer_r"],1)),
                    ],
                }
                st.dataframe(pd.DataFrame(cdr_data), use_container_width=True, hide_index=True)

            with col_b:
                st.markdown("**🔄 RNFL per Region (µm)**")
                rnfl_data = {
                    "Region": ["Whole","Superior","Inferior","Temporal","Nasal","S/I Ratio"],
                    "µm (proxy)": [
                        str(round(result["whole_mean"],1)),
                        str(round(result["S_mean"],1)),
                        str(round(result["I_mean"],1)),
                        str(round(result["T_mean"],1)),
                        str(round(result["N_mean"],1)),
                        str(round(result["SI_ratio"],3)),
                    ],
                }
                st.dataframe(pd.DataFrame(rnfl_data), use_container_width=True, hide_index=True)

                st.markdown("**🕐 ClockHour (µm)**")
                clock_data = {
                    "Jam"      : ["Jam " + str(i+1) for i in range(12)],
                    "µm"       : [str(round(result["features"][i]["pseudo_thick"],1)) for i in range(12)],
                }
                st.dataframe(pd.DataFrame(clock_data), use_container_width=True, hide_index=True)

            st.markdown("**👁️ Deteksi Mata (OD/OS)**")
            vc1, vc2, vc3, vc4 = st.columns(4)
            votes = result["votes"]
            with vc1:
                st.metric("Hasil", result["eye_side"],
                          delta="Kanan" if result["eye_side"]=="OD" else "Kiri")
            with vc2:
                st.metric("Confidence", str(int(result["eye_conf"]*100)) + "%")
            with vc3:
                st.metric("Posisi", votes[0],
                          delta="✓" if votes[0]==result["eye_side"] else "✗")
            with vc4:
                st.metric("Vascular", votes[2],
                          delta="✓" if votes[2]==result["eye_side"] else "✗")

            if cls_detail_rows:
                st.markdown("**🤖 Hasil per Model**")
                model_data = {
                    "Model"         : [r[0] for r in cls_detail_rows],
                    "Prob Glaukoma" : [str(round(r[1]*100,1)) + "%" for r in cls_detail_rows],
                    "Prediksi"      : [r[2] for r in cls_detail_rows],
                }
                st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)
                if len(cls_detail_rows) > 1 and prob_glau is not None:
                    st.metric("Ensemble (rata-rata)",
                              str(round(prob_glau*100,1)) + "%",
                              delta=pred_label)

        with tab3:
            if not show_gradcam:
                st.info("Aktifkan 'Tampilkan Grad-CAM' di sidebar.")
            elif gradcam_data is None:
                st.warning("⚠️ Grad-CAM tidak berhasil. Pastikan path model classifier tersedia.")
            else:
                import matplotlib.cm as cm_module
                crop_disp, cam_up, overlay, grad_pred = gradcam_data
                st.markdown("#### 🔥 Grad-CAM — Region yang mempengaruhi keputusan model")
                g1, g2, g3 = st.columns(3)
                with g1:
                    st.image(crop_disp, caption="Crop Ring Area", use_container_width=True)
                with g2:
                    hm_rgba = cm_module.jet(cam_up)
                    hm_uint = (hm_rgba[:,:,:3]*255).astype(np.uint8)
                    st.image(hm_uint, caption="Grad-CAM Heatmap", use_container_width=True)
                with g3:
                    lbl_txt = "🔴 GLAUKOMA" if grad_pred==1 else "🟢 NORMAL"
                    st.image(overlay, caption="Overlay | " + lbl_txt, use_container_width=True)
                st.markdown(
                    "> **Interpretasi:** Area merah/kuning = region paling berpengaruh. "
                    "Pada glaukoma biasanya area **inferior** dan **superior** disc."
                )

    else:
        st.image(np.array(pil_img), caption="Preview: " + filename,
                 use_container_width=False, width=400)

else:
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:40px; color:#666688">
        <div style="font-size:4rem">🔬</div>
        <h3 style="color:#8888aa">Upload gambar fundus mata untuk memulai analisis</h3>
        <p>Mendukung format: JPG, PNG, BMP, TIFF</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔄 Pipeline")
    cols = st.columns(7)
    steps = [
        ("📸","Upload\nFundus"),
        ("⚗️","Preprocessing\nCLAHE"),
        ("🎯","Segmentasi\nOD & Cup"),
        ("💍","RNFL Ring\nExtraction"),
        ("📊","TSNIT\nAnalysis"),
        ("🤖","Deep Learning\nClassifier"),
        ("🔥","Grad-CAM\nXAI"),
    ]
    for col, (icon, lbl) in zip(cols, steps):
        with col:
            st.markdown(
                '<div style="text-align:center;padding:12px;background:#111122;'
                'border-radius:8px;border:1px solid #2244aa">'
                '<div style="font-size:1.5rem">' + icon + '</div>'
                '<div style="font-size:0.7rem;color:#aaaacc;margin-top:4px">' + lbl + '</div>'
                '</div>', unsafe_allow_html=True)
