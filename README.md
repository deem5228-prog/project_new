# Egg Yolk Color DSM Fan Score Prediction (Complete Final Project)

โครงงานระบบปัญญาประดิษฐ์และแอปพลิเคชันมือถือสำหรับประเมินระดับสีไข่แดง (DSM Yolk Color Fan Score 1–15) แบบ **On-Device AI 100% (Edge AI)** ครบวงจรตั้งแต่กระบวนการเตรียมข้อมูล การสกัดฟีเจอร์สี การแข่งขันโมเดล การส่งออกน้ำหนักพารามิเตอร์ ไปจนถึงการทำงานจริงบนสมาร์ตโฟน

---

## 📁 โครงสร้างโฟลเดอร์โปรเจกต์ (Project Structure)

```text
egg_yolk_final_project/
├── README.md                          # เอกสารสรุปโครงงานและคู่มือการใช้งาน
│
├── 🧠 model_dev/                      # กระบวนการพัฒนาโมเดล AI และ Data Pipeline
│   ├── batch_auto_crop.py             # ขั้นตอนที่ 1: ตรวจจับและครอบตัดภาพไข่แดงทั้งชุดข้อมูลอัตโนมัติ
│   ├── build_features.py              # ขั้นตอนที่ 2: สกัดค่าสี RGB และ CIELAB (Circular Mask 42%) บันทึกเป็น CSV
│   ├── color_service.py               # โมดูลฟังก์ชันหลักสำหรับการคำนวณสี HSV และ CIELAB (CIE D65)
│   ├── compare_crop_experiments.py    # สคริปต์เปรียบเทียบผลการทดลอง Auto-Crop vs ภาพดั้งเดิม
│   ├── train_compare_models.py        # ขั้นตอนที่ 3: แข่งขันและเปรียบเทียบโมเดล Machine Learning 6 อัลกอริทึม
│   ├── export_best_model.py           # ขั้นตอนที่ 4: ส่งออกโมเดล SVR เป็น model_weights.json และ model.pkl
│   │
│   ├── test_hsv_equivalence.py        # ชุดทดสอบ A: ยืนยันความตรงกันของ HSV Mask กับ OpenCV 100%
│   ├── test_lab_equivalence.py        # ชุดทดสอบ B: ยืนยันความตรงกันของ CIELAB กับ Scikit-image (diff < 0.001)
│   ├── test_crop_consistency.py       # ชุดทดสอบ C: ยืนยันความแม่นยำของการครอปภาพจริง 5 ภาพ
│   ├── test_score_comparison.py       # ชุดทดสอบ D: ยืนยันคะแนนทำนาย SVR ตรงกัน 20 ภาพ (diff < 0.1)
│   │
│   ├── data/                          # ชุดข้อมูลและตารางฟีเจอร์ (features.csv, crop_comparison_results.csv)
│   ├── plots/                         # กราฟสรุปผลการวิจัย 12 รูปภาพ สำหรับนำไปใส่เล่มรายงาน
│   ├── model.pkl                      # โมเดล Scikit-Learn Pipeline ที่เทรนสมบูรณ์แล้ว
│   ├── model_weights.json             # ค่าพารามิเตอร์คณิตศาสตร์ SVR สำหรับ Mobile App
│   ├── ml_presentation.html           # หน้าเว็บสไลด์นำเสนอผลงานวิจัยแบบ Interactive
│   ├── ML_PIPELINE_TUTORIAL.md        # คู่มืออธิบายหลักการทางคณิตศาสตร์อย่างละเอียด
│   └── requirements.txt               # ไลบรารี Python ที่จำเป็น (opencv, scikit-learn, scikit-image)
│
├── 🌐 egg_api/                        # ระบบ Backend REST API (FastAPI) และเว็บแอปพลิเคชัน
│   ├── main.py                        # จุดเริ่มต้นเซิร์ฟเวอร์ FastAPI พร้อม CORS & Swagger Docs
│   ├── routers/predict.py             # API Endpoint POST /api/predict ทำนายคะแนนจากรูปภาพ
│   ├── schemas.py                     # โครงสร้าง Pydantic Data Model สำหรับ Request/Response
│   ├── services/                      # เซอร์วิสประมวลผลภาพ สกัดสี และทำนายผลด้วยโมเดล SVR
│   ├── web_app.html                   # เว็บแอปพลิเคชัน HTML5/JS สำหรับทดสอบทำนายผ่าน Browser
│   └── requirements.txt               # ไลบรารี Python สำหรับรัน FastAPI Backend
│
└── 📱 egg_yolk_app/                   # แอปพลิเคชันมือถือ Flutter (100% On-Device AI)
    ├── assets/
    │   └── model_weights.json         # โมเดลสมองกล SVR 107 KB ฝังในตัวแอป
    ├── lib/
    ├── test/                          # ชุด Unit Test บน Flutter (ผ่าน 100%)
    └── pubspec.yaml                   # การกำหนดค่าและ Assets ของแอป Flutter
```

---

## 🔬 4 ขั้นตอน Machine Learning Pipeline (`model_dev/`)

1. **Step 1: Data Preparation & Auto-Cropping**
   - รันคำสั่ง: `python batch_auto_crop.py`
   - ระบบใช้ OpenCV แปลงสี BGR ➔ HSV กรองสีไข่แดงด้วยเกณฑ์ $H \in [10, 28], S \in [80, 255], V \in [80, 250]$
   - ใช้ Morphological Close/Open กรองสัญญาณรบกวน ล็อกเป้าชิ้นส่วนที่ใหญ่ที่สุด (Largest Blob) และครอบตัดเฉพาะไข่แดง $r \times 1.08$

2. **Step 2: Feature Extraction**
   - รันคำสั่ง: `python build_features.py`
   - ใช้ Center Circular Mask $42\%$ เพื่อดูดเฉพาะเนื้อไข่แดงบริสุทธิ์ (ตัดขอบไข่ขาวและพื้นหลังทิ้ง)
   - แปลงสี RGB เป็น CIELAB ($L^*, a^*, b^*$) ตามมาตรฐาน CIE 1976 D65 แล้วบันทึกลง `data/features.csv`

3. **Step 3: Model Training & Evaluation**
   - รันคำสั่ง: `python train_compare_models.py`
   - แข่งขัน 6 อัลกอริทึมด้วย 5-Fold Cross Validation:
     - **SVR (RBF Kernel)** ได้รับชัยชนะ: $R^2 = 0.9185$, $\text{MAE} = 0.6403$, ความแม่นยำ $\pm 1$ ระดับ $= 92.7\%$

4. **Step 4: Model Export to On-Device**
   - รันคำสั่ง: `python export_best_model.py`
   - สกัดพารามิเตอร์ของ SVR (Support Vectors 556 จุด, Scaler Mean/Scale, Dual Coefficients Alphas, Gamma, Intercept) ออกมาเป็นไฟล์ `model_weights.json` (ขนาด 107 KB)
   - นำไปฝังลงใน `egg_yolk_app/assets/model_weights.json` ทำให้แอปพลิเคชันมือถือสามารถคำนวณสมการ SVR ได้โดยตรงโดยไม่ต้องมีเซิร์ฟเวอร์

---

## 📱 วิธีเปิดใช้งานและทดสอบ Mobile App (`egg_yolk_app/`)

1. เข้าไปที่โฟลเดอร์แอป:
   ```bash
   cd egg_yolk_app
   ```
2. ติดตั้งแพ็กเกจ:
   ```bash
   flutter pub get
   ```
3. รันการทดสอบ Unit Tests:
   ```bash
   flutter test
   ```
   *(ผลลัพธ์: ผ่านการทดสอบทั้งหมด 100%)*
4. เสียบสายโทรศัพท์มือถือ หรือเปิด Emulator แล้วสั่งรันแอป:
   ```bash
   flutter run
   ```

---

## 🧪 สรุปผลการทดสอบความเที่ยงตรง (Verification Benchmarks)

* ✅ **Test A (HSV Equivalence):** ตรวจจับพิกเซลตรงกับ OpenCV $100.0\%$ (413,280 / 413,280 พิกเซล, Mismatch = 0)
* ✅ **Test B (CIELAB Equivalence):** ค่าสี $L^*, a^*, b^*$ ตรงกับ Scikit-image $(\text{diff} = 0.000)$
* ✅ **Test C (Crop Consistency):** ค่าสีจากการครอปภาพจริง 5 ภาพ ต่างกัน $\text{diff} = 0.0$
* ✅ **Test D (Score Comparison):** คะแนนทำนาย DSM Fan Score ต่างกัน $\text{diff} = 0.0$ ทั้ง 20 ภาพ
