# 🔬 Glaucoma OCT Simulation — Streamlit App

Simulasi output OCT Nidek RS-3000 dari fundus image menggunakan deep learning.

## Struktur Folder

```
glaucoma_streamlit/
├── app.py              ← Main Streamlit app
├── pipeline.py         ← Core pipeline (segmentasi, RNFL, visualisasi)
├── requirements.txt    ← Dependencies
├── README.md
└── models/             ← Buat folder ini & isi dengan file .pth
    ├── best_model_fold3.pth          ← Seg model (dari Kaggle training)
    ├── cls_efficientnet_b4_best.pth  ← Classifier EfficientNet-B4
    ├── cls_resnet50_best.pth         ← Classifier ResNet-50
    ├── cls_densenet121_best.pth      ← Classifier DenseNet-121
    └── cls_convnext_tiny_best.pth    ← Classifier ConvNeXt-Tiny
```

## Cara Jalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Siapkan model
Buat folder `models/` dan copy file `.pth` hasil training dari Kaggle:
```bash
mkdir models
# Copy best_model_fold3.pth ke folder models/
# Copy file classifier ke folder models/
```

### 3. Jalankan app
```bash
streamlit run app.py
```

App akan terbuka di browser: `http://localhost:8501`

---

## Deploy agar dapat diakses lewat link

Gunakan **Streamlit Community Cloud**. Aplikasi akan mendapat alamat
`https://nama-aplikasi.streamlit.app/` setelah deploy berhasil.

### 1. Siapkan model yang dibutuhkan

File `models/best_model_fold3.pth` sudah ada. File classifier belum ada di
folder `models/`; tanpa classifier, bagian klasifikasi akan menampilkan `N/A`.
Untuk hasil klasifikasi lengkap, tambahkan setidaknya satu file classifier
dengan nama yang tercantum pada bagian **Nama File Classifier** di bawah.

### 2. Upload proyek ke GitHub

File model segmentasi berukuran sekitar 181 MB, melampaui batas file GitHub
biasa. Instal **Git LFS** pada komputer, lalu jalankan dari folder proyek:

```bash
git lfs install
git init
git add .gitattributes .gitignore README.md app.py pipeline.py requirements.txt models/
git commit -m "Prepare Streamlit deployment"
git branch -M main
git remote add origin https://github.com/USERNAME/NAMA_REPO.git
git push -u origin main
```

Ganti `USERNAME/NAMA_REPO` dengan repo GitHub milik Anda yang telah dibuat.
`.gitattributes` sudah mengatur agar file `models/*.pth` dikirim melalui Git LFS.
Jika folder ini sudah menjadi repo Git, lewati `git init` dan gunakan remote
yang sudah ada. Jangan upload folder `venv/`.

### 3. Deploy di Streamlit Community Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io/) dan login dengan GitHub.
2. Pilih **Create app** → **Yup, I have an app**.
3. Pilih repo dan branch `main`, lalu isi **Main file path** dengan `app.py`.
4. Pilih versi Python **3.12** pada **Advanced settings**, lalu klik **Deploy**.
5. Setelah berhasil, salin URL `*.streamlit.app` untuk dibagikan.

Jika deploy gagal, buka **Manage app → Logs** pada halaman aplikasi untuk
melihat error. Pastikan file model tersedia di repo dan benar-benar terunggah
sebagai objek Git LFS, bukan hanya pointer LFS.

---

## Cara Pakai di App

1. **Sidebar** → isi path model (atau biarkan default jika struktur folder sama)
2. **Pilih model** classifier dari dropdown:
   - EfficientNet-B4
   - ResNet-50
   - DenseNet-121
   - ConvNeXt-Tiny
   - Ensemble (semua model digabung)
3. **Upload** gambar fundus (JPG/PNG)
4. Klik **Analisis**
5. Lihat hasil di 3 tab:
   - **OCT Simulation** → visualisasi lengkap mirip RS-3000
   - **Detail Klinis** → tabel CDR, RNFL per sektor, ClockHour
   - **Grad-CAM XAI** → highlight region berpengaruh

---

## Nama File Classifier

File classifier dari Kaggle notebook perlu di-rename:

| Backbone | Nama file expected |
|---|---|
| EfficientNet-B4 | `cls_efficientnet_b4_best.pth` |
| ResNet-50 | `cls_resnet50_best.pth` |
| DenseNet-121 | `cls_densenet121_best.pth` |
| ConvNeXt-Tiny | `cls_convnext_tiny_best.pth` |

Atau bisa ganti path langsung di sidebar app.

---

## Pipeline

```
Fundus Image
    ↓ CLAHE Preprocessing
    ↓ Segmentasi OD & Cup (UNet mit_b3, Fold 3; input grayscale)
    ↓ Ellipse Fitting → R_disc, semi-a, semi-b
    ↓ CDR Calculation (H/V/Area via ellipse)
    ↓ Deteksi OD/OS (3-method voting)
    ↓ RNFL Ring (inner=R_disc, outer=2×R_disc)
    ↓ 12 Sektor ClockHour
    ↓ Feature Extraction (green channel weighted)
    ↓ TSNIT Curve Reorder
    ↓ Visualisasi OCT RS-3000 Style
    ↓ Klasifikasi (backbone + clinical features)
    ↓ Grad-CAM XAI
    ↓ Output Diagnosis
```

## Referensi

- Garway-Heath et al. (2000) *Ophthalmology* — RNFL ring standard
- Jonas et al. (2004) *Ophthalmology* — CDR via ellipse fitting
- Hood et al. (2013) — TSNIT 12-sector convention
- Cheng et al. (2013) *MICCAI* — fundus proxy untuk RNFL
- Selvaraju et al. (2017) — Grad-CAM
