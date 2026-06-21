# 🩺 Thyroid Cancer Recurrence Prediction
### Dự đoán Tái phát Ung thư Tuyến giáp

[English](#english) | [Tiếng Việt](#tiếng-việt)

---

<a name="english"></a>
## English

### 📌 Overview

This project predicts whether a patient with **Differentiated Thyroid Cancer** is likely to experience **recurrence** after initial treatment, based on clinical and demographic features available at diagnosis. The project follows a complete end-to-end Machine Learning workflow: EDA → preprocessing → handling class imbalance → model training & comparison → leakage analysis → deployment as an interactive web app.

**Dataset:** [Differentiated Thyroid Cancer Recurrence](https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence) (UCI Machine Learning Repository)
- 383 patients, 16 clinical features, collected over 15 years with at least 10 years of follow-up
- No missing values
- Target: `Recurred` (Yes / No) — 28.2% positive class (mild imbalance)

### 🎯 Key Highlights

- ⚠️ **Data leakage investigation**: identified and quantified leakage risk from the `Response` (treatment response) feature, and built a model that excludes it for realistic *early* prediction
- ⚖️ **Imbalance handling comparison**: SMOTE vs `class_weight='balanced'`, tested across multiple models
- 🧮 Manual verification of **ordinal encoding logic** for clinically ordered variables (Risk, TNM staging)
- 🖥️ **Interactive Streamlit app** with probability visualization and feature importance

### 📂 Project Structure

```
thyroid-cancer-recurrence/
│
├── data/
│   ├── raw/                       # Original, untouched dataset
│   │   └── Thyroid_Diff.csv
│   └── processed/                 # Cleaned & encoded data, train/test splits
│       ├── thyroid_processed.csv
│       ├── X_train.csv / X_test.csv
│       └── y_train.csv / y_test.csv
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   └── 02_preprocessing_and_split.ipynb
│
├── src/
│   └── preprocessing.py            # Reusable preprocessing functions
│
├── models/
│   ├── best_model.pkl              # Trained Random Forest (final model)
│   ├── feature_columns.pkl         # Expected feature order for inference
│   └── encoding_info.pkl           # Encoding maps/orders used at training time
│
├── app/
│   └── app.py                      # Streamlit web app
│
├── requirements.txt
└── README.md
```

### 🔬 Methodology

#### 1. Exploratory Data Analysis (EDA)
- Target distribution: 71.8% No / 28.2% Yes
- Identified multicollinearity among `T`, `N`, `M`, `Stage` (TNM staging system — `Stage` is clinically derived from T+N+M)
- Identified rare categories in `Adenopathy` (e.g. `Posterior` with only 2 samples)
- Identified suspiciously strong correlation between `Response`, `Risk`, and the target

#### 2. Preprocessing
| Feature type | Method | Columns |
|---|---|---|
| Binary | Manual mapping (0/1) | Gender, Smoking, Hx Smoking, Hx Radiothreapy, Focality, M |
| Ordinal | `OrdinalEncoder` with manually defined clinical order | Risk, T, N, Stage |
| Nominal | One-Hot Encoding | Thyroid Function, Physical Examination, Adenopathy, Pathology, Response |
| Rare category | Manual grouping | Adenopathy (6 → 3 groups) |

#### 3. Train/Test Split
- 80/20 split, `stratify=y` to preserve class proportions, `random_state=42`

#### 4. Imbalance Handling
Two strategies were tested and compared:
- **SMOTE** (Synthetic Minority Over-sampling) — applied **only** to the training set, never to the test set
- **`class_weight='balanced'`** — no synthetic samples, reweights the loss function

#### 5. Data Leakage Analysis ⚠️

A crosstab analysis revealed extremely strong associations:

| Response | % Recurred |
|---|---|
| Excellent | 0.5% |
| Indeterminate | 11.5% |
| Biochemical Incomplete | 47.8% |
| Structural Incomplete | 97.8% |

`Response` reflects treatment outcome **observed after follow-up**, making it a near-restatement of the target rather than an independent predictor available at diagnosis time. Two models were therefore trained and compared:

| Model | Accuracy | Precision (Recurred) | Recall (Recurred) |
|---|---|---|---|
| Random Forest **with** `Response` | 0.97 | 1.00 | 0.91 |
| Random Forest **without** `Response` | 0.94 | 0.90 | 0.86 |

The drop is moderate (not collapse), confirming the remaining clinical features (`Risk`, `T`/`N`/`M`/`Stage`, `Age`...) retain strong genuine predictive value. **The model without `Response` was selected for deployment**, as it reflects a realistic early-prediction scenario (predicting before treatment-response data exists).

### 📊 Model Comparison & Remarks

| Model | Imbalance method | Accuracy | Precision (1) | Recall (1) | F1 (1) |
|---|---|---|---|---|---|
| Logistic Regression | class_weight | 0.97 | 1.00 | 0.91 | 0.95 |
| Logistic Regression | SMOTE | 0.97 | 1.00 | 0.91 | 0.95 |
| Random Forest | class_weight | 0.96 | 1.00 | 0.86 | 0.93 |
| Random Forest | SMOTE | 0.97 | 1.00 | 0.91 | 0.95 |
| **Random Forest (no `Response`, final)** | class_weight | **0.94** | **0.90** | **0.86** | **0.88** |

**Remarks:**
- Logistic Regression performed identically under both imbalance strategies — `class_weight` and SMOTE converge to similar decision boundaries for linear models on this dataset.
- For Random Forest, SMOTE outperformed `class_weight` on minority-class recall (0.91 vs 0.86) — synthetic oversampling helped the tree ensemble see more minority-class splits.
- All models still miss 2–3 out of ~22 recurrence cases in the test set — in a clinical context, this is an important limitation to communicate, since missing a recurrence (false negative) is costlier than a false alarm.

### 🖥️ Interactive Demo

The Streamlit app provides:
- A clinical input form (age, gender, smoking history, TNM staging, pathology, etc.)
- Predicted class (Recurred / Not Recurred) with probability scores
- A probability bar chart
- Top-10 feature importance chart

> ⚠️ **Disclaimer:** This app is built for educational/portfolio purposes only. It is **not** a certified medical diagnostic tool. Always consult a qualified physician for real clinical decisions.

### ⚙️ Installation & Usage

#### 1. Clone the repository
```bash
git clone https://github.com/MinhTran110/thyroid-cancer-recurrence-prediction.git
cd thyroid-cancer-recurrence-prediction
```

#### 2. Install dependencies
```bash
pip install -r requirements.txt
```

#### 3. (Optional) Re-run preprocessing
```bash
python src/preprocessing.py
```

#### 4. Run the Streamlit app
```bash
cd app
streamlit run app.py
```
Then open the local URL shown in the terminal (default: `http://localhost:8501`).

### 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| Data handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Model persistence | joblib |
| Web app | streamlit |

---
---

<a name="tiếng-việt"></a>
## Tiếng Việt

### 📌 Tổng quan

Dự án dự đoán khả năng **tái phát** của bệnh nhân **ung thư tuyến giáp đã biệt hóa tốt** sau điều trị ban đầu, dựa trên các đặc trưng lâm sàng và nhân khẩu học có sẵn tại thời điểm chẩn đoán. Dự án triển khai đầy đủ quy trình Machine Learning end-to-end: EDA → tiền xử lý → xử lý mất cân bằng lớp → huấn luyện & so sánh model → phân tích data leakage → triển khai thành web app tương tác.

**Bộ dữ liệu:** [Differentiated Thyroid Cancer Recurrence](https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence) (UCI Machine Learning Repository)
- 383 bệnh nhân, 16 đặc trưng lâm sàng, thu thập trong 15 năm với thời gian theo dõi tối thiểu 10 năm
- Không có giá trị thiếu
- Target: `Recurred` (Có/Không tái phát) — 28.2% là lớp dương (mất cân bằng nhẹ)

### 🎯 Điểm nổi bật

- ⚠️ **Phân tích data leakage**: phát hiện và định lượng nguy cơ leakage từ đặc trưng `Response` (phản ứng điều trị), xây dựng model loại bỏ biến này để mô phỏng dự đoán sớm thực tế
- ⚖️ **So sánh xử lý mất cân bằng**: SMOTE vs `class_weight='balanced'`, kiểm thử trên nhiều model
- 🧮 Tự kiểm chứng logic **encode biến có thứ tự** cho các biến phân loại y khoa (Risk, hệ thống TNM)
- 🖥️ **App Streamlit tương tác** với biểu đồ xác suất và mức độ quan trọng đặc trưng

### 📂 Cấu trúc dự án

```
thyroid-cancer-recurrence/
│
├── data/
│   ├── raw/                       # Dữ liệu gốc, không chỉnh sửa
│   │   └── Thyroid_Diff.csv
│   └── processed/                 # Dữ liệu đã làm sạch & encode, train/test
│       ├── thyroid_processed.csv
│       ├── X_train.csv / X_test.csv
│       └── y_train.csv / y_test.csv
│
├── notebooks/
│   ├── 01_eda.ipynb                # Phân tích khám phá dữ liệu
│   └── 02_preprocessing_and_split.ipynb
│
├── src/
│   └── preprocessing.py            # Các hàm xử lý dữ liệu tái sử dụng
│
├── models/
│   ├── best_model.pkl              # Random Forest đã train (model cuối)
│   ├── feature_columns.pkl         # Thứ tự cột model cần khi dự đoán
│   └── encoding_info.pkl           # Mapping/thứ tự encode đã dùng lúc train
│
├── app/
│   └── app.py                      # Web app Streamlit
│
├── requirements.txt
└── README.md
```

### 🔬 Phương pháp thực hiện

#### 1. Phân tích khám phá dữ liệu (EDA)
- Phân bố target: 71.8% Không / 28.2% Có tái phát
- Phát hiện đa cộng tuyến giữa `T`, `N`, `M`, `Stage` (hệ thống phân loại TNM — `Stage` được tính trực tiếp từ T+N+M theo quy tắc y khoa)
- Phát hiện nhóm giá trị hiếm trong `Adenopathy` (ví dụ `Posterior` chỉ có 2 mẫu)
- Phát hiện tương quan đáng nghi ngờ giữa `Response`, `Risk` và target

#### 2. Tiền xử lý
| Loại đặc trưng | Phương pháp | Các cột |
|---|---|---|
| Nhị phân | Map tay (0/1) | Gender, Smoking, Hx Smoking, Hx Radiothreapy, Focality, M |
| Có thứ tự | `OrdinalEncoder` với thứ tự y khoa tự định nghĩa | Risk, T, N, Stage |
| Không thứ tự | One-Hot Encoding | Thyroid Function, Physical Examination, Adenopathy, Pathology, Response |
| Giá trị hiếm | Gộp nhóm thủ công | Adenopathy (6 → 3 nhóm) |

#### 3. Train/Test Split
- Chia 80/20, dùng `stratify=y` để giữ đúng tỷ lệ lớp, `random_state=42`

#### 4. Xử lý mất cân bằng
Đã thử nghiệm và so sánh 2 chiến lược:
- **SMOTE** (sinh mẫu tổng hợp cho lớp thiểu số) — **chỉ** áp dụng trên tập train, không áp dụng lên tập test
- **`class_weight='balanced'`** — không sinh mẫu mới, điều chỉnh trọng số hàm loss

#### 5. Phân tích Data Leakage ⚠️

Phân tích crosstab cho thấy mối liên hệ rất mạnh:

| Response | % Tái phát |
|---|---|
| Excellent | 0.5% |
| Indeterminate | 11.5% |
| Biochemical Incomplete | 47.8% |
| Structural Incomplete | 97.8% |

`Response` phản ánh kết quả điều trị **quan sát được sau thời gian theo dõi**, gần như là một cách diễn đạt khác của outcome, không phải thông tin độc lập có sẵn tại thời điểm chẩn đoán. Vì vậy đã huấn luyện và so sánh 2 phiên bản model:

| Model | Accuracy | Precision (Tái phát) | Recall (Tái phát) |
|---|---|---|---|
| Random Forest **có** `Response` | 0.97 | 1.00 | 0.91 |
| Random Forest **không có** `Response` | 0.94 | 0.90 | 0.86 |

Mức giảm là vừa phải (không sụp đổ), xác nhận các đặc trưng lâm sàng còn lại (`Risk`, `T`/`N`/`M`/`Stage`, `Age`...) vẫn mang giá trị dự đoán thực sự đáng kể. **Model không có `Response` được chọn để triển khai**, vì phản ánh đúng tình huống dự đoán sớm thực tế (dự đoán trước khi có dữ liệu phản ứng điều trị).

### 📊 So sánh mô hình & Nhận xét

| Model | Cách xử lý imbalance | Accuracy | Precision (1) | Recall (1) | F1 (1) |
|---|---|---|---|---|---|
| Logistic Regression | class_weight | 0.97 | 1.00 | 0.91 | 0.95 |
| Logistic Regression | SMOTE | 0.97 | 1.00 | 0.91 | 0.95 |
| Random Forest | class_weight | 0.96 | 1.00 | 0.86 | 0.93 |
| Random Forest | SMOTE | 0.97 | 1.00 | 0.91 | 0.95 |
| **Random Forest (không Response, bản cuối)** | class_weight | **0.94** | **0.90** | **0.86** | **0.88** |

**Nhận xét:**
- Logistic Regression cho kết quả giống nhau ở cả 2 cách xử lý imbalance — `class_weight` và SMOTE hội tụ về ranh giới quyết định tương tự cho model tuyến tính trên bộ dữ liệu này.
- Với Random Forest, SMOTE vượt trội hơn `class_weight` về recall lớp thiểu số (0.91 vs 0.86) — việc sinh mẫu tổng hợp giúp tree ensemble "thấy" nhiều điểm chia của lớp thiểu số hơn.
- Mọi model vẫn bỏ sót 2-3/~22 ca tái phát trong tập test — trong bối cảnh y tế, đây là hạn chế quan trọng cần nêu rõ, vì bỏ sót ca tái phát (false negative) thường nguy hiểm hơn báo động giả.

### 🖥️ Demo tương tác

App Streamlit cung cấp:
- Form nhập thông tin lâm sàng (tuổi, giới tính, tiền sử hút thuốc, giai đoạn TNM, giải phẫu bệnh...)
- Kết quả dự đoán (Tái phát / Không tái phát) kèm xác suất
- Biểu đồ cột xác suất
- Biểu đồ Top-10 đặc trưng quan trọng nhất

> ⚠️ **Lưu ý:** App này được xây dựng cho mục đích học tập/portfolio, **không phải** công cụ chẩn đoán y khoa được cấp phép. Luôn tham khảo ý kiến bác sĩ chuyên khoa cho các quyết định lâm sàng thực tế.

### ⚙️ Cài đặt và sử dụng

#### 1. Clone repository
```bash
git clone https://github.com/MinhTran110/thyroid-cancer-recurrence-prediction.git
cd thyroid-cancer-recurrence-prediction
```

#### 2. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

#### 3. (Tùy chọn) Chạy lại preprocessing
```bash
python src/preprocessing.py
```

#### 4. Chạy app Streamlit
```bash
cd app
streamlit run app.py
```
Sau đó mở link hiện trong terminal (mặc định: `http://localhost:8501`).

### 🛠️ Thư viện sử dụng

| Nhóm | Thư viện |
|---|---|
| Xử lý dữ liệu | pandas, numpy |
| Trực quan hóa | matplotlib, seaborn |
| Machine Learning | scikit-learn |
| Xử lý mất cân bằng | imbalanced-learn (SMOTE) |
| Lưu model | joblib |
| Web app | streamlit |
