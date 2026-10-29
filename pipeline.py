"""
pipeline.py — Core logic for Glaucoma OCT Simulation
Extracted from Kaggle notebook (v4)

Referensi:
- Garway-Heath et al. (2000) Ophthalmology — RNFL ring
- Jonas et al. (2004) Ophthalmology — CDR ellipse fitting
- Hood et al. (2013) — TSNIT 12-sector
- Cheng et al. (2013) MICCAI — fundus proxy RNFL
"""

import cv2
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import albumentations as A
import timm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.patches import Wedge, FancyBboxPatch
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from skimage.measure import regionprops, label as sklabel
from scipy.ndimage import binary_fill_holes
from scipy.interpolate import make_interp_spline

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    "seg_img_size": 512,
    "seg_encoder":  "mit_b3",
    "seg_classes":  3,
    "inner_ratio":  1.0,
    "outer_ratio":  2.0,
    "n_sectors":    12,
    "thick_min":    32,
    "thick_max":    250,
    "thick_red":    70,
    "thick_yellow": 100,
    "dropout":      0.4,
    "cls_img_size": 224,
}

CLN_KEYS = [
    "cdr_h", "cdr_v", "cdr_area", "R_disc", "ring_area",
    "semi_a", "semi_b", "whole_mean", "S_mean",
    "I_mean", "T_mean", "N_mean", "SI_ratio",
]

TSNIT_OD  = [9,10,11, 0,1,2, 3,4,5, 6,7,8, 9]
TSNIT_OS  = [3, 2, 1, 0,11,10, 9,8,7, 6,5,4, 3]
CLOCK_LBL = ["12","1","2","3","4","5","6","7","8","9","10","11"]

OCT_CMAP = LinearSegmentedColormap.from_list(
    "oct_rs3000",
    ["#0000cc","#0055ff","#00aaff","#00ffff",
     "#00ff88","#00cc00","#88ff00",
     "#ffff00","#ffaa00","#ff5500","#cc0000"],
    N=512,
)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class GlaucomaClassifier(nn.Module):
    def __init__(self, backbone="efficientnet_b4", dropout=0.4, n_cln=13):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = timm.create_model(
            backbone, pretrained=False, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.cln = nn.Sequential(
            nn.Linear(n_cln, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout/2),
            nn.Linear(128, 64),    nn.BatchNorm1d(64),  nn.GELU(), nn.Dropout(dropout/3),
            nn.Linear(64, 32),     nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim+32, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 128),         nn.GELU(),           nn.Dropout(dropout/2),
            nn.Linear(128, 2),
        )

    def forward(self, img, cln):
        return self.head(torch.cat([self.backbone(img), self.cln(cln)], dim=1))


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_seg_model(model_path: str):
    state = torch.load(model_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    for arch in [smp.UnetPlusPlus, smp.Unet]:
        for enc in ["mit_b3", "mit_b2", "mit_b1", "resnet50", "efficientnet-b4"]:
            for ch in [3, 1]:
                try:
                    model = arch(
                        encoder_name=enc, encoder_weights=None,
                        in_channels=ch, classes=CFG["seg_classes"]
                    )
                    model.load_state_dict(state, strict=True)
                    print(f"✅ Loaded: {arch.__name__} enc={enc} ch={ch}")
                    return model.to(DEVICE).eval()
                except Exception:
                    continue
    raise RuntimeError("Gagal load model segmentasi")


def load_cls_model(model_path: str, backbone: str):
    """Load classification model checkpoint."""
    ckpt  = torch.load(model_path, map_location=DEVICE)
    model = GlaucomaClassifier(backbone, CFG["dropout"], len(CLN_KEYS))
    model.load_state_dict(ckpt["state"])
    model = model.to(DEVICE).eval()
    return model, ckpt["cm"], ckpt["cs"]


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING & SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

SEG_TF = A.Compose([
    A.Resize(CFG["seg_img_size"], CFG["seg_img_size"]),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

SEG_TF_GRAY = A.Compose([
    A.Resize(CFG["seg_img_size"], CFG["seg_img_size"]),
    A.Normalize(mean=(0.485,), std=(0.229,)),
    ToTensorV2(),
])

CLS_TF = A.Compose([
    A.Resize(CFG["cls_img_size"], CFG["cls_img_size"]),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

def preprocess_fundus(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def clean_mask(mask: np.ndarray) -> np.ndarray:
    labeled = sklabel(mask)
    props   = regionprops(labeled)
    if not props:
        return mask
    largest = max(props, key=lambda p: p.area)
    clean   = (labeled == largest.label).astype(np.uint8)
    return binary_fill_holes(clean).astype(np.uint8)

@torch.no_grad()
def segment_image(img_rgb: np.ndarray, seg_model) -> tuple:
    orig_h, orig_w = img_rgb.shape[:2]
    enhanced = preprocess_fundus(img_rgb)
    input_channels = seg_model.encoder.patch_embed1.proj.in_channels if hasattr(seg_model.encoder, "patch_embed1") else 3
    if input_channels == 1:
        gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        tensor = SEG_TF_GRAY(image=gray)["image"].unsqueeze(0).to(DEVICE)
    else:
        tensor = SEG_TF(image=enhanced)["image"].unsqueeze(0).to(DEVICE)
    pred     = torch.argmax(seg_model(tensor), dim=1).squeeze().cpu().numpy().astype(np.uint8)
    pred_orig= cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    disc_mask = clean_mask((pred_orig == 1).astype(np.uint8))
    cup_mask  = clean_mask((pred_orig == 2).astype(np.uint8))
    return disc_mask, cup_mask


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY — ELLIPSE FITTING & CDR
# ══════════════════════════════════════════════════════════════════════════════

def fit_ellipse_safe(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        H, W = mask.shape
        return W//2, H//2, 40.0, 40.0, 0.0
    c = max(contours, key=cv2.contourArea)
    if len(c) >= 5:
        (cx,cy),(dw,dh),angle = cv2.fitEllipse(c)
        return float(cx), float(cy), float(dw/2), float(dh/2), float(angle)
    x, y, w, h = cv2.boundingRect(c)
    return x+w/2, y+h/2, w/2, h/2, 0.0

def compute_cdr(disc_mask: np.ndarray, cup_mask: np.ndarray):
    _, _, da, db, _ = fit_ellipse_safe(disc_mask)
    _, _, ca, cb, _ = fit_ellipse_safe(cup_mask)
    disc_w, disc_h  = 2*da, 2*db
    cup_w,  cup_h   = 2*ca, 2*cb
    cdr_h    = round(cup_w  / (disc_w  + 1e-8), 3)
    cdr_v    = round(cup_h  / (disc_h  + 1e-8), 3)
    cdr_area = round(cup_mask.sum() / (disc_mask.sum() + 1e-8), 3)
    return cdr_h, cdr_v, cdr_area

def cdr_risk_label(cdr_v: float):
    if   cdr_v < 0.5: return "Normal",       "#228822"
    elif cdr_v < 0.7: return "Borderline",   "#dd8800"
    elif cdr_v < 0.9: return "Suspect",      "#cc4400"
    else:             return "High Suspect",  "#cc0000"

def get_disc_params(disc_mask: np.ndarray):
    cx, cy, a, b, angle = fit_ellipse_safe(disc_mask)
    R_disc = (a + b) / 2
    return int(cx), int(cy), float(a), float(b), float(R_disc), float(angle), int(disc_mask.sum())

def detect_eye_side(img_rgb: np.ndarray, disc_mask: np.ndarray, cx: int, cy: int):
    H, W = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(float)
    # Vote 1: disc position
    vote1 = "OD" if cx < W * 0.5 else "OS"
    # Vote 2: brightness asymmetry
    l_bright = gray[:, :W//3].mean()
    r_bright = gray[:, 2*W//3:].mean()
    vote2 = "OD" if r_bright < l_bright else "OS"
    # Vote 3: vascular density (std green)
    g_ch   = img_rgb[:,:,1].astype(float)
    margin = int(min(H,W) * 0.15)
    x1,x2  = max(0,cx-4*margin), min(W,cx+4*margin)
    y1,y2  = max(0,cy-2*margin), min(H,cy+2*margin)
    region  = g_ch[y1:y2, x1:x2]
    mid     = region.shape[1]//2
    vote3   = "OD" if region[:, mid:].std() > region[:, :mid].std() else "OS"
    votes      = [vote1, vote2, vote3]
    od_count   = votes.count("OD")
    result     = "OD" if od_count >= 2 else "OS"
    confidence = od_count/3 if result=="OD" else (3-od_count)/3
    return result, confidence, votes


# ══════════════════════════════════════════════════════════════════════════════
# RNFL RING & FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def create_rnfl_ring(img_shape, cx, cy, R_disc):
    H, W    = img_shape[:2]
    Y, X    = np.ogrid[:H, :W]
    dist    = np.sqrt((X-cx)**2 + (Y-cy)**2)
    inner_r = CFG["inner_ratio"] * R_disc
    outer_r = CFG["outer_ratio"] * R_disc
    ring    = ((dist >= inner_r) & (dist < outer_r)).astype(np.uint8)
    return ring, float(inner_r), float(outer_r), dist

def divide_sectors(img_shape, cx, cy, eye_side="OD", n=12):
    H, W   = img_shape[:2]
    Y, X   = np.mgrid[:H, :W]
    angle  = (np.degrees(np.arctan2(X-cx, cy-Y)) + 360) % 360
    if eye_side == "OS":
        angle = (360 - angle) % 360
    return (angle / (360/n)).astype(int) % n

def extract_rnfl_features(img_rgb, ring_mask, sector_map, dist_map, R_disc, n=12):
    R_ch    = img_rgb[:,:,0].astype(float)
    G_ch    = img_rgb[:,:,1].astype(float)
    inner_r = CFG["inner_ratio"] * R_disc
    outer_r = CFG["outer_ratio"] * R_disc
    features = []
    for s in range(n):
        mask_s = (ring_mask==1) & (sector_map==s)
        n_px   = mask_s.sum()
        if n_px < 10:
            features.append({"mean_g":0,"std_g":0,"mean_r":0,
                              "gr_ratio":1.0,"pseudo_thick":CFG["thick_min"],"n_pixels":0})
            continue
        g_vals = G_ch[mask_s]; r_vals = R_ch[mask_s]
        d_vals = dist_map[mask_s]
        w = 1.0 - (d_vals - inner_r) / (outer_r - inner_r + 1e-8)
        w = np.clip(w, 0.1, 1.0); w = w/w.sum()
        mg = float(np.dot(w, g_vals))
        mr = float(np.dot(w, r_vals))
        sg = float(g_vals.std())
        gr = mg / (mr + 1e-8)
        pt = CFG["thick_min"] + (mg/255.0) * (CFG["thick_max"] - CFG["thick_min"])
        features.append({"mean_g":mg,"std_g":sg,"mean_r":mr,
                          "gr_ratio":gr,"pseudo_thick":float(pt),"n_pixels":int(n_px)})
    return features


# ══════════════════════════════════════════════════════════════════════════════
# COLOR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def thick2color(t):
    return OCT_CMAP(np.clip((t - CFG["thick_min"]) / (CFG["thick_max"] - CFG["thick_min"]), 0, 1))

def thick2normcolor(t):
    if   t < CFG["thick_red"]:    return np.array([200, 30,  30], np.uint8), 0.60
    elif t < CFG["thick_yellow"]: return np.array([255,165,   0], np.uint8), 0.55
    else:                          return np.array([ 20,180,  40], np.uint8), 0.45

def sector_bar_color(t):
    n = np.clip((t - CFG["thick_min"]) / (CFG["thick_max"] - CFG["thick_min"]), 0, 1)
    if n < 0.30: return "#cc1100"
    if n < 0.50: return "#dd8800"
    if n < 0.65: return "#aacc00"
    return "#228822"

def make_normative_overlay(img_rgb, ring_mask, sector_map, features, disc_mask):
    H, W    = img_rgb.shape[:2]
    overlay = img_rgb.copy().astype(float)
    for s, feat in enumerate(features):
        mask_s = (ring_mask==1) & (sector_map==s)
        if not mask_s.any(): continue
        col, alpha = thick2normcolor(feat["pseudo_thick"])
        for c in range(3):
            overlay[:,:,c][mask_s] = overlay[:,:,c][mask_s]*(1-alpha) + col[c]*alpha
    for c in range(3):
        overlay[:,:,c][disc_mask==1] = overlay[:,:,c][disc_mask==1]*0.35 + 80*0.65
    return np.clip(overlay, 0, 255).astype(np.uint8)

def make_rnfl_color_map(img_shape, ring_mask, sector_map, features):
    H, W      = img_shape[:2]
    color_img = np.zeros((H,W,4), dtype=np.float32)
    for s, feat in enumerate(features):
        mask_s = (ring_mask==1) & (sector_map==s)
        if not mask_s.any(): continue
        color_img[mask_s] = thick2color(feat["pseudo_thick"])
    return color_img

def smooth_curve(x, y, n=300):
    try:
        spl  = make_interp_spline(x, y, k=3)
        xs   = np.linspace(x[0], x[-1], n)
        return xs, spl(xs)
    except:
        xs = np.linspace(x[0], x[-1], n)
        return xs, np.interp(xs, x, y)


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(img_rgb: np.ndarray, seg_model) -> dict:
    """
    Jalankan pipeline lengkap dari numpy RGB image.
    Return: dict berisi semua hasil (fitur, mask, parameter klinis).
    """
    # 1. Segmentasi
    disc_mask, cup_mask = segment_image(img_rgb, seg_model)

    # 2. Geometri
    cx, cy, semi_a, semi_b, R_disc, angle_deg, disc_area_px = get_disc_params(disc_mask)

    # 3. CDR
    cdr_h, cdr_v, cdr_area = compute_cdr(disc_mask, cup_mask)

    # 4. OD/OS
    eye_side, eye_conf, votes = detect_eye_side(img_rgb, disc_mask, cx, cy)

    # 5. RNFL ring
    ring_mask, inner_r, outer_r, dist_map = create_rnfl_ring(img_rgb.shape, cx, cy, R_disc)

    # 6. Sektor
    sector_map = divide_sectors(img_rgb.shape, cx, cy, eye_side, CFG["n_sectors"])

    # 7. Features
    features   = extract_rnfl_features(img_rgb, ring_mask, sector_map, dist_map, R_disc)

    # 8. TSNIT stats
    thick_arr  = np.array([f["pseudo_thick"] for f in features])
    cdr_risk, cdr_col = cdr_risk_label(cdr_v)

    result = {
        "img_rgb":    img_rgb,
        "disc_mask":  disc_mask,
        "cup_mask":   cup_mask,
        "ring_mask":  ring_mask,
        "sector_map": sector_map,
        "dist_map":   dist_map,
        "features":   features,
        "thick_arr":  thick_arr,
        "eye_side":   eye_side,
        "eye_conf":   eye_conf,
        "votes":      votes,
        "cx": cx, "cy": cy,
        "semi_a":     semi_a,
        "semi_b":     semi_b,
        "R_disc":     R_disc,
        "angle_deg":  angle_deg,
        "inner_r":    inner_r,
        "outer_r":    outer_r,
        "cdr_h":      cdr_h,
        "cdr_v":      cdr_v,
        "cdr_area":   cdr_area,
        "cdr_risk":   cdr_risk,
        "cdr_col":    cdr_col,
        "disc_area":  disc_area_px,
        "cup_area":   int(cup_mask.sum()),
        "ring_area":  int(ring_mask.sum()),
        "whole_mean": float(thick_arr.mean()),
        "S_mean":     float(thick_arr[[0,1,2]].mean()),
        "I_mean":     float(thick_arr[[6,7,8]].mean()),
        "T_mean":     float(thick_arr[[9,10,11]].mean()),
        "N_mean":     float(thick_arr[[3,4,5]].mean()),
        "SI_ratio":   float(thick_arr[[0,1,2]].mean() / (thick_arr[[6,7,8]].mean()+1e-8)),
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify(result: dict, cls_model, cm: np.ndarray, cs: np.ndarray) -> tuple:
    """
    Jalankan classifier pada hasil pipeline.
    Return: (prob_glaukoma, pred_label, prob_normal)
    """
    img_rgb = result["img_rgb"]
    cx, cy  = result["cx"], result["cy"]
    pad     = int(result["R_disc"] * CFG["outer_ratio"] * 1.2)
    H, W    = img_rgb.shape[:2]
    crop    = img_rgb[max(0,cy-pad):min(H,cy+pad), max(0,cx-pad):min(W,cx+pad)]
    if crop.size == 0:
        crop = img_rgb

    inp = CLS_TF(image=crop)["image"].unsqueeze(0).to(DEVICE)

    raw = np.array([result[k] for k in CLN_KEYS], dtype=np.float32)
    cln = torch.tensor((raw - cm) / cs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(cls_model(inp, cln), dim=1)[0].cpu().numpy()

    prob_glau  = float(probs[1])
    prob_norm  = float(probs[0])
    pred_label = "GLAUKOMA" if prob_glau >= 0.5 else "NORMAL"
    return prob_glau, pred_label, prob_norm


# ══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    def __init__(self, model, layer):
        self.model = model; self.grads = None; self.acts = None
        layer.register_forward_hook( lambda m,i,o: setattr(self,"acts",o.detach()))
        layer.register_backward_hook(lambda m,gi,go: setattr(self,"grads",go[0].detach()))

    def run(self, img_t, cln_t):
        self.model.eval()
        img_t  = img_t.clone().requires_grad_(True)
        logits = self.model(img_t, cln_t)
        cls    = logits.argmax(1).item()
        self.model.zero_grad(); logits[0,cls].backward()
        w   = self.grads.mean([2,3], keepdim=True)
        cam = F.relu((w * self.acts).sum(1)).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, cls

def get_target_layer(model):
    n = model.backbone_name
    if "efficientnet" in n: return model.backbone.blocks[-1]
    if "resnet"       in n: return model.backbone.layer4[-1]
    if "densenet"     in n: return model.backbone.features.denseblock4
    if "convnext"     in n: return model.backbone.stages[-1]
    return list(model.backbone.children())[-2]

def compute_gradcam(result: dict, cls_model, cm, cs, sz=224) -> np.ndarray:
    """Compute Grad-CAM heatmap, return overlay RGB (H,W,3) uint8."""
    img_rgb = result["img_rgb"]
    cx, cy  = result["cx"], result["cy"]
    pad     = int(result["R_disc"] * CFG["outer_ratio"] * 1.2)
    H, W    = img_rgb.shape[:2]
    crop    = img_rgb[max(0,cy-pad):min(H,cy+pad), max(0,cx-pad):min(W,cx+pad)]
    if crop.size == 0: crop = img_rgb
    crop_disp = cv2.resize(crop, (sz, sz))

    inp  = CLS_TF(image=crop_disp)["image"].unsqueeze(0).to(DEVICE)
    raw  = np.array([result[k] for k in CLN_KEYS], dtype=np.float32)
    cln  = torch.tensor((raw-cm)/cs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    gcam = GradCAM(cls_model, get_target_layer(cls_model))
    cam, pred = gcam.run(inp, cln)

    cam_up  = cv2.resize(cam, (sz, sz))
    hm      = cv2.applyColorMap((cam_up*255).astype(np.uint8), cv2.COLORMAP_JET)
    hm_rgb  = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(crop_disp, 0.55, hm_rgb, 0.45, 0)
    return crop_disp, cam_up, overlay, pred


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION — OCT RS-3000 STYLE
# ══════════════════════════════════════════════════════════════════════════════

def build_oct_figure(result: dict, filename: str = "") -> plt.Figure:
    """
    Buat figure matplotlib layout OCT RS-3000.
    Return: matplotlib Figure (bisa disimpan atau ditampilkan di Streamlit).
    """
    BG    = "#0c0c18"
    PANEL = "#111122"

    r          = result
    n          = CFG["n_sectors"]
    tord       = TSNIT_OD if r["eye_side"]=="OD" else TSNIT_OS
    thick_arr  = r["thick_arr"]
    tsnit_vals = [thick_arr[s] for s in tord]
    tsnit_x    = np.arange(len(tsnit_vals), dtype=float)

    whole_mean = r["whole_mean"]
    S_mean, I_mean = r["S_mean"], r["I_mean"]
    T_mean, N_mean = r["T_mean"], r["N_mean"]
    cx, cy     = r["cx"], r["cy"]
    semi_a, semi_b = r["semi_a"], r["semi_b"]
    R_disc     = r["R_disc"]
    inner_r, outer_r = r["inner_r"], r["outer_r"]
    cdr_h, cdr_v, cdr_area = r["cdr_h"], r["cdr_v"], r["cdr_area"]
    cdr_risk, cdr_col = r["cdr_risk"], r["cdr_col"]
    eye_side   = r["eye_side"]
    eye_conf   = r["eye_conf"]
    votes      = r["votes"]

    features   = r["features"]
    img_rgb    = r["img_rgb"]
    disc_mask  = r["disc_mask"]
    cup_mask   = r["cup_mask"]
    ring_mask  = r["ring_mask"]
    sector_map = r["sector_map"]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(26,15), facecolor=BG, dpi=110)

    # ── TITLE BAR ─────────────────────────────────────────────────────────────
    t_ax = fig.add_axes([0, 0.956, 1, 0.044])
    t_ax.set_facecolor("#003399"); t_ax.axis("off")
    t_ax.text(0.005, 0.55,
        "OCT Setting: DISC MAP  |  Simulasi RS-3000",
        color="white", fontsize=12, fontweight="bold",
        va="center", transform=t_ax.transAxes)
    t_ax.text(0.30, 0.55,
        f"Eye: {eye_side}  |  C/D(H)={cdr_h:.2f}  C/D(V)={cdr_v:.2f}  |  "
        f"Risk: {cdr_risk}  |  {filename}",
        color="#ffdd88", fontsize=9.5, va="center", transform=t_ax.transAxes)
    t_ax.text(0.97, 0.55, f"Eye:{eye_side}",
        color="white", fontsize=11, fontweight="bold",
        va="center", ha="right", transform=t_ax.transAxes,
        bbox=dict(facecolor="#2255cc" if eye_side=="OD" else "#cc4422",
                  edgecolor="white", boxstyle="round,pad=0.3"))
    t_ax.text(0.99, 0.55, "RS-3000 Sim",
        color="#88aaff", fontsize=8, va="center", ha="right", transform=t_ax.transAxes)

    # ── GRID ──────────────────────────────────────────────────────────────────
    gs = gridspec.GridSpec(
        3, 6, figure=fig,
        left=0.02, right=0.985, top=0.952, bottom=0.03,
        wspace=0.07, hspace=0.30,
    )
    AX = dict(facecolor=PANEL)

    # ── A: Normative Database ──────────────────────────────────────────────────
    ax_norm = fig.add_subplot(gs[0,0], **AX)
    norm_img = make_normative_overlay(img_rgb, ring_mask, sector_map, features, disc_mask)
    ax_norm.imshow(norm_img)
    ell = matplotlib.patches.Ellipse(
        (cx,cy), 2*semi_a, 2*semi_b, angle=r["angle_deg"],
        fill=False, edgecolor="red", lw=1.5)
    ax_norm.add_patch(ell)
    cup_cont,_ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cup_cont:
        ax_norm.plot(c[:,0,0], c[:,0,1], "r-", lw=0.9, alpha=0.8)
    ax_norm.set_title("Normative Database", color="white", fontsize=8, pad=3)
    ax_norm.axis("off")
    for pct,col,yp in [(100,"#00bb00",0.04),(95,"#88cc00",0.20),
                        (5,"#ffaa00",0.76),(1,"#dd2200",0.93)]:
        ax_norm.text(-0.12, yp, str(pct), color=col, fontsize=7,
                     transform=ax_norm.transAxes, ha="right", va="center")
    ax_norm.text(-0.18, 0.5, "[%]", color="#aaaaaa", fontsize=6,
                 transform=ax_norm.transAxes, ha="right", va="center", rotation=90)
    ax_norm.text(0.5, -0.04,
        "For reference only (Ethnic group not available)",
        color="#555555", fontsize=5, ha="center", transform=ax_norm.transAxes)

    # ── B: RNFLT Map ───────────────────────────────────────────────────────────
    ax_rnfl = fig.add_subplot(gs[0,1:3], **AX)
    gray3      = cv2.cvtColor(cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    base       = gray3.astype(float) / 255.0
    rnfl_rgba  = make_rnfl_color_map(img_rgb.shape, ring_mask, sector_map, features)
    alpha3     = np.stack([rnfl_rgba[:,:,3]]*3, axis=-1)
    blended    = base*(1-alpha3) + rnfl_rgba[:,:,:3]*alpha3
    ax_rnfl.imshow(np.clip(blended, 0, 1))
    for rv, ls in [(inner_r,"--"),(outer_r,"-")]:
        circ = plt.Circle((cx,cy), rv, color="red", fill=False, lw=1.3, ls=ls)
        ax_rnfl.add_patch(circ)
    fov_x = np.clip(cx + (3.0*R_disc if eye_side=="OD" else -3.0*R_disc),
                    0, img_rgb.shape[1]-1)
    ax_rnfl.plot(fov_x, cy, "o", color="#ffee00", ms=7, zorder=5,
                 markeredgecolor="white", markeredgewidth=0.5)
    ell2 = matplotlib.patches.Ellipse(
        (cx,cy), 2*semi_a, 2*semi_b, angle=r["angle_deg"],
        fill=False, edgecolor="red", lw=1.0, ls=":")
    ax_rnfl.add_patch(ell2)
    ax_rnfl.set_title("RNFLT Map  (ILM-NFL/GCL)", color="white", fontsize=9, pad=3)
    ax_rnfl.axis("off")
    # Colorbar
    cb_ax = fig.add_axes([0.342, 0.665, 0.006, 0.262])
    ColorbarBase(cb_ax, cmap=OCT_CMAP, norm=Normalize(32,250), orientation="vertical")
    cb_ax.yaxis.set_tick_params(labelcolor="white", labelsize=6)
    for sp in cb_ax.spines.values(): sp.set_edgecolor("#555")
    cb_ax.set_ylabel("[um]", color="white", fontsize=6, rotation=90, labelpad=2)

    # ── C: Fundus Photo ────────────────────────────────────────────────────────
    ax_foto = fig.add_subplot(gs[0,3:5], **AX)
    ax_foto.imshow(img_rgb)
    disc_cont,_ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in disc_cont:
        ax_foto.plot(c[:,0,0], c[:,0,1], color="red", lw=1.5)
    for c in cup_cont:
        ax_foto.plot(c[:,0,0], c[:,0,1], color="#ff8800", lw=0.9, ls="--")
    ax_foto.set_title("Fundus Photo", color="white", fontsize=9, pad=3)
    ax_foto.axis("off")

    # ── D: Deteksi OD/OS ───────────────────────────────────────────────────────
    ax_od = fig.add_subplot(gs[0,5], **AX)
    ax_od.set_xlim(0,10); ax_od.set_ylim(0,10); ax_od.axis("off")
    eye_box_col = "#1a3a88" if eye_side=="OD" else "#882211"
    rect = FancyBboxPatch((0.3,6.3), 9.4, 3.3,
        boxstyle="round,pad=0.2", facecolor=eye_box_col, edgecolor="white", lw=1.2)
    ax_od.add_patch(rect)
    ax_od.text(5, 8.1, f"EYE: {eye_side}", color="white", fontsize=17,
               fontweight="bold", ha="center", va="center")
    ax_od.text(5, 6.9, f"({'Kanan' if eye_side=='OD' else 'Kiri'})",
               color="#aaccff", fontsize=9, ha="center", va="center")
    ax_od.text(5, 5.6, f"Confidence: {eye_conf*100:.0f}%",
               color="#88ff88", fontsize=8.5, ha="center", va="center")
    vote_mnames = ["Posisi", "Brightness", "Vascular"]
    for i,(m,v) in enumerate(zip(vote_mnames, votes)):
        col = "#66ff66" if v==eye_side else "#ff8888"
        sym = "✓" if v==eye_side else "✗"
        ax_od.text(5, 4.5-i*0.95, f"{m}: {v} {sym}",
                   color=col, fontsize=7.5, ha="center", va="center")
    ax_od.set_title("Deteksi Mata", color="white", fontsize=8, pad=3)

    # ── E: Whole / S / I + TSNIT donut ────────────────────────────────────────
    ax_wsi = fig.add_subplot(gs[1,0], **AX)
    ax_wsi.set_xlim(0,10); ax_wsi.set_ylim(0,10); ax_wsi.axis("off")

    def draw_ring(ax, cx_, cy_, r_out, r_in, val, lbl="", fs=11):
        t   = np.clip((val-32)/(250-32), 0, 1)
        col = "#cc1100" if t<0.35 else ("#dd8800" if t<0.55 else "#228822")
        ax.add_patch(plt.Circle((cx_,cy_), r_out, color=col, zorder=2))
        ax.add_patch(plt.Circle((cx_,cy_), r_in,  color=PANEL, zorder=3))
        ax.text(cx_, cy_, f"{int(val)}", color="white", ha="center", va="center",
                fontsize=fs, fontweight="bold", zorder=4)
        if lbl:
            ax.text(cx_, cy_-r_out-0.25, lbl, color="#aaaaaa", ha="center",
                    va="top", fontsize=7)

    ax_wsi.text(1.3,9.6,"Whole",color="white",fontsize=8,ha="center")
    ax_wsi.text(4.7,9.6,"S/I",  color="white",fontsize=8,ha="center")
    ax_wsi.text(8.0,9.6,"TSNIT",color="white",fontsize=8,ha="center")
    draw_ring(ax_wsi, 1.3, 6.0, 1.3, 0.5, whole_mean)
    draw_ring(ax_wsi, 4.7, 7.5, 0.95, 0.38, S_mean, "S", 9)
    draw_ring(ax_wsi, 4.7, 4.5, 0.95, 0.38, I_mean, "I", 9)

    # TSNIT donut
    tsnit4  = {"T":T_mean,"S":S_mean,"N":N_mean,"I":I_mean}
    ang_map = {"S":90,"N":0,"I":270,"T":180}
    for lbl,val in tsnit4.items():
        ang  = ang_map[lbl]
        t    = np.clip((val-32)/(250-32),0,1)
        wcol = "#cc1100" if t<0.35 else ("#dd8800" if t<0.55 else "#228822")
        ax_wsi.add_patch(Wedge((8.0,6.0), 1.45, ang-44, ang+44, width=0.65,
                               color=wcol, zorder=2))
        rx = 8.0 + 0.85*math.cos(math.radians(ang))
        ry = 6.0 + 0.85*math.sin(math.radians(ang))
        ax_wsi.text(rx, ry, f"{int(val)}", color="white", ha="center", va="center",
                    fontsize=7, fontweight="bold", zorder=3)
    for lbl,ang in ang_map.items():
        rx = 8.0 + 1.75*math.cos(math.radians(ang))
        ry = 6.0 + 1.75*math.sin(math.radians(ang))
        ax_wsi.text(rx, ry, lbl, color="#bbbbbb", ha="center", va="center", fontsize=7)

    # ── F: Tabel klinis ────────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2,0], **AX)
    ax_tbl.axis("off")
    rows = [
        ("Item",           "Value",            "white"),
        ("C/D(Horizon)",   f"{cdr_h:.3f}",     cdr_col),
        ("C/D(Vertical)",  f"{cdr_v:.3f}",     cdr_col),
        ("CDR(Area)",      f"{cdr_area:.3f}",  cdr_col),
        ("CDR Status",     cdr_risk,            cdr_col),
        ("DiscArea[px]",   f"{r['disc_area']:,}","#dddddd"),
        ("CupArea[px]",    f"{r['cup_area']:,}","#dddddd"),
        ("RingArea[px]",   f"{r['ring_area']:,}","#dddddd"),
        ("R_disc[px]",     f"{R_disc:.1f}",    "#aaddff"),
        ("Semi-a[px]",     f"{semi_a:.1f}",    "#aaddff"),
        ("Semi-b[px]",     f"{semi_b:.1f}",    "#aaddff"),
        ("inner_r[px]",    f"{inner_r:.1f}",   "#aaddff"),
        ("outer_r[px]",    f"{outer_r:.1f}",   "#aaddff"),
        ("Eye Side",       eye_side,            "#ffdd88"),
    ]
    for ri,(item,val,col) in enumerate(rows):
        y  = 0.97 - ri*0.068
        fw = "bold" if ri==0 else "normal"
        ic = "#88ccff" if ri==0 else "white"
        ax_tbl.text(0.02, y, item, color=ic, fontsize=7.5, fontweight=fw,
                    transform=ax_tbl.transAxes, va="top")
        ax_tbl.text(0.60, y, val, color=col, fontsize=7.5, fontweight=fw,
                    transform=ax_tbl.transAxes, va="top")

    # ── G: ClockHour Polar ─────────────────────────────────────────────────────
    ax_clock = fig.add_subplot(gs[1,1], projection="polar", facecolor=PANEL)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    width  = 2*np.pi/n * 0.92
    for i,(ang,feat) in enumerate(zip(angles, features)):
        ax_clock.bar(ang, feat["pseudo_thick"], width=width, bottom=20,
                     color=sector_bar_color(feat["pseudo_thick"]),
                     alpha=0.92, edgecolor="#222", lw=0.5)
    ax_clock.set_theta_zero_location("N")
    ax_clock.set_theta_direction(-1)
    ax_clock.set_xticks(angles)
    ax_clock.set_xticklabels(CLOCK_LBL, color="white", fontsize=7)
    ax_clock.set_yticklabels([])
    ax_clock.tick_params(colors="white")
    ax_clock.spines["polar"].set_color("#444")
    ax_clock.set_title("ClockHour", color="white", fontsize=8, pad=8)
    for i,(ang,feat) in enumerate(zip(angles, features)):
        ax_clock.text(ang, feat["pseudo_thick"]+30,
                      f"{int(feat['pseudo_thick'])}",
                      color="#ffdd88", fontsize=5.5, ha="center", va="center")

    # ── H: Segmentasi OD & OC ──────────────────────────────────────────────────
    ax_seg = fig.add_subplot(gs[1,2], **AX)
    seg_vis = np.zeros((*disc_mask.shape,3), dtype=np.uint8)
    seg_vis[disc_mask==1] = [30,180,50]
    seg_vis[cup_mask==1]  = [200,50,50]
    seg_vis[ring_mask==1] = np.maximum(seg_vis[ring_mask==1], np.array([200,200,0],np.uint8))
    ax_seg.imshow(seg_vis)
    ell3 = matplotlib.patches.Ellipse(
        (cx,cy), 2*semi_a, 2*semi_b, angle=r["angle_deg"],
        fill=False, edgecolor="white", lw=1.5, ls="--")
    ax_seg.add_patch(ell3)
    ax_seg.legend(handles=[
        mpatches.Patch(color="#1eb432",label="Optic Disc"),
        mpatches.Patch(color="#c83232",label="Optic Cup"),
        mpatches.Patch(color="#c8c800",label="RNFL Ring"),
    ], fontsize=6, loc="lower right", facecolor="#111")
    ax_seg.set_title(
        f"Segmentasi OD & OC  |  CDR-V={cdr_v:.3f}",
        color="white", fontsize=8, pad=3)
    ax_seg.axis("off")

    # ── I: TSNIT Curve ─────────────────────────────────────────────────────────
    ax_tsnit = fig.add_subplot(gs[1:,3:6], facecolor=PANEL)
    ymax = 270
    ax_tsnit.axhspan(0,   CFG["thick_red"],    alpha=0.28, color="#cc1100", zorder=0)
    ax_tsnit.axhspan(CFG["thick_red"],
                     CFG["thick_yellow"],        alpha=0.28, color="#cc7700", zorder=0)
    ax_tsnit.axhspan(CFG["thick_yellow"], ymax,  alpha=0.18, color="#228822", zorder=0)
    ax_tsnit.axhline(CFG["thick_red"],   color="#ff4444", lw=1.2, ls="--", alpha=0.9,
                     label=f"1%  ({CFG['thick_red']}µm)")
    ax_tsnit.axhline(CFG["thick_yellow"],color="#ffaa00", lw=1.2, ls="--", alpha=0.9,
                     label=f"5%  ({CFG['thick_yellow']}µm)")
    ax_tsnit.axhline(160, color="#ffff44", lw=1.0, ls="--", alpha=0.7, label="95% (160µm)")

    xs, ys = smooth_curve(tsnit_x, np.array(tsnit_vals))
    ax_tsnit.fill_between(xs, ys, alpha=0.50, color="#00cc44", zorder=2)
    ax_tsnit.plot(xs, ys, color="#00ff66", lw=2.5, zorder=3)
    ax_tsnit.plot(tsnit_x, tsnit_vals, "o", color="white", ms=4.5, zorder=4,
                  markeredgecolor="#00ff66", markeredgewidth=0.8)
    ax_tsnit.plot(xs, ys*0.25+8, color="#003322", lw=1.2, alpha=0.55, zorder=1)
    for xi, yi in zip(tsnit_x, tsnit_vals):
        if xi in [0,3,6,9,12]: continue
        ax_tsnit.text(xi, yi+9, f"{int(yi)}", color="#ccffcc",
                      fontsize=6.5, ha="center", va="bottom", zorder=5)
    ax_tsnit.set_xticks([0,3,6,9,12])
    ax_tsnit.set_xticklabels(["T","S","N","I","T"], color="white", fontsize=14, fontweight="bold")
    ax_tsnit.set_yticks([0,50,100,150,200,250])
    ax_tsnit.yaxis.set_tick_params(labelcolor="white", labelsize=8)
    ax_tsnit.set_ylabel("[µm]", color="white", fontsize=10)
    ax_tsnit.set_ylim(0, ymax); ax_tsnit.set_xlim(-0.3, 12.3)
    ax_tsnit.spines[:].set_color("#444")
    ax_tsnit.set_title("TSNIT Curve  (Pseudo-Thickness Proxy)",
                        color="white", fontsize=10, pad=5)
    ax_tsnit.legend(fontsize=8, labelcolor="white", facecolor="#111",
                    edgecolor="#444", loc="upper right")
    for xmid,lbl in [(1.5,"T"),(4.5,"S"),(7.5,"N"),(10.5,"I")]:
        ax_tsnit.text(xmid, 4, lbl, color="#555555", fontsize=9,
                      ha="center", va="bottom", alpha=0.7)
    ax_tsnit.text(0.01, 0.02, "[um] T", color="#777777",
                  fontsize=7, transform=ax_tsnit.transAxes)

    # ── J: B-scan ──────────────────────────────────────────────────────────────
    ax_bscan = fig.add_subplot(gs[2,1:3], facecolor="black")
    row_g    = img_rgb[cy,:,1].astype(float)
    bscan_h  = 90
    bscan    = np.zeros((bscan_h, img_rgb.shape[1]), dtype=float)
    for xi in range(img_rgb.shape[1]):
        g_val = row_g[xi]
        depth = int((1 - g_val/255) * (bscan_h-15)) + 5
        bscan[depth,   xi] = g_val / 255
        bscan[depth+1, xi] = g_val / 255 * 0.7
        bscan[depth+2, xi] = g_val / 255 * 0.3
    ax_bscan.imshow(bscan, cmap="gray", aspect="auto", vmin=0, vmax=1)
    ax_bscan.axvline(cx, color="#ffaa00", lw=0.8, ls="--", alpha=0.7)
    for x_px in range(0, img_rgb.shape[1], 4):
        if ring_mask[cy, x_px]:
            ax_bscan.axvspan(x_px, x_px+4, alpha=0.08, color="#ffff00")
    ax_bscan.set_title("B-scan Simulasi  (Horizontal cross-section)",
                        color="white", fontsize=8, pad=3)
    ax_bscan.set_facecolor("black"); ax_bscan.axis("off")
    ax_bscan.text(0.01, 0.08, "R Retinal +5",
                  color="#ffdd00", fontsize=6, transform=ax_bscan.transAxes)

    fig.tight_layout(rect=[0,0,1,0.956])
    return fig
