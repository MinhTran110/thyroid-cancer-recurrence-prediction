#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
app.py
------
Streamlit app dự đoán khả năng tái phát ung thư tuyến giáp
(Differentiated Thyroid Cancer Recurrence).

Chạy: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ============================================================
# Cấu hình trang
# ============================================================
st.set_page_config(
    page_title="Dự đoán Tái phát Ung thư Tuyến giáp",
    page_icon="🩺",
    layout="wide"
)

# ============================================================
# Load model và metadata (cache để không load lại mỗi lần tương tác)
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("../models/best_model.pkl")
    feature_columns = joblib.load("../models/feature_columns.pkl")
    encoding_info = joblib.load("../models/encoding_info.pkl")
    return model, feature_columns, encoding_info

model, feature_columns, encoding_info = load_artifacts()

# ============================================================
# Tiêu đề
# ============================================================
st.title("🩺 Dự đoán Khả năng Tái phát Ung thư Tuyến giáp")
st.markdown(
    "Ứng dụng demo sử dụng **Random Forest** để dự đoán khả năng tái phát "
    "dựa trên đặc điểm lâm sàng tại thời điểm chẩn đoán ban đầu "
    "(không sử dụng thông tin phản ứng điều trị về sau, nhằm mô phỏng "
    "tình huống dự đoán sớm thực tế)."
)
st.divider()

# ============================================================
# Form input — chia 2 cột cho gọn
# ============================================================
st.subheader("📋 Nhập thông tin bệnh nhân")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Tuổi (Age)", min_value=15, max_value=85, value=40)
    gender = st.selectbox("Giới tính (Gender)", ["F", "M"])
    smoking = st.selectbox("Đang hút thuốc (Smoking)", ["No", "Yes"])
    hx_smoking = st.selectbox("Tiền sử hút thuốc (Hx Smoking)", ["No", "Yes"])
    hx_radiotherapy = st.selectbox("Tiền sử xạ trị (Hx Radiothreapy)", ["No", "Yes"])
    focality = st.selectbox("Tính đơn/đa ổ (Focality)", ["Uni-Focal", "Multi-Focal"])
    thyroid_function = st.selectbox(
        "Chức năng tuyến giáp (Thyroid Function)",
        ["Euthyroid", "Clinical Hyperthyroidism", "Clinical Hypothyroidism",
         "Subclinical Hyperthyroidism", "Subclinical Hypothyroidism"]
    )
    physical_exam = st.selectbox(
        "Khám thực thể (Physical Examination)",
        ["Normal", "Diffuse goiter", "Multinodular goiter",
         "Single nodular goiter-left", "Single nodular goiter-right"]
    )

with col2:
    risk = st.selectbox("Mức độ nguy cơ (Risk)", encoding_info['risk_order'])
    t_stage = st.selectbox("Giai đoạn khối u (T)", encoding_info['t_order'])
    n_stage = st.selectbox("Giai đoạn hạch (N)", encoding_info['n_order'])
    m_stage = st.selectbox("Di căn xa (M)", ["M0", "M1"])
    stage = st.selectbox("Giai đoạn tổng quát (Stage)", encoding_info['stage_order'])
    adenopathy = st.selectbox(
        "Hạch bạch huyết (Adenopathy)",
        ["No", "Right", "Left", "Bilateral", "Extensive", "Posterior"]
    )
    pathology = st.selectbox(
        "Loại giải phẫu bệnh (Pathology)",
        ["Papillary", "Micropapillary", "Follicular", "Hurthel cell"]
    )

st.divider()
predict_button = st.button("🔍 Dự đoán", type="primary", use_container_width=True)

# ============================================================
# Hàm xử lý input giống pipeline preprocessing đã dùng lúc train
# ============================================================
def build_input_row(age, gender, smoking, hx_smoking, hx_radiotherapy, focality,
                     risk, t_stage, n_stage, m_stage, stage, adenopathy,
                     thyroid_function, physical_exam, pathology,
                     feature_columns, encoding_info):

    row = {col: 0 for col in feature_columns}

    row['Age'] = age
    row['Gender'] = 1 if gender == 'M' else 0
    row['Smoking'] = 1 if smoking == 'Yes' else 0
    row['Hx Smoking'] = 1 if hx_smoking == 'Yes' else 0
    row['Hx Radiothreapy'] = 1 if hx_radiotherapy == 'Yes' else 0
    row['Focality'] = 1 if focality == 'Multi-Focal' else 0
    row['M'] = 1 if m_stage == 'M1' else 0

    row['Risk'] = encoding_info['risk_order'].index(risk)
    row['T'] = encoding_info['t_order'].index(t_stage)
    row['N'] = encoding_info['n_order'].index(n_stage)
    row['Stage'] = encoding_info['stage_order'].index(stage)

    adenopathy_grouped = encoding_info['adenopathy_map'][adenopathy]
    adenopathy_col = f"Adenopathy_{adenopathy_grouped}"
    if adenopathy_col in row:
        row[adenopathy_col] = 1

    tf_col = f"Thyroid Function_{thyroid_function}"
    if tf_col in row:
        row[tf_col] = 1

    pe_col = f"Physical Examination_{physical_exam}"
    if pe_col in row:
        row[pe_col] = 1

    path_col = f"Pathology_{pathology}"
    if path_col in row:
        row[path_col] = 1

    return pd.DataFrame([row], columns=feature_columns)


# ============================================================
# Khi bấm nút Dự đoán
# ============================================================
if predict_button:
    input_df = build_input_row(
        age, gender, smoking, hx_smoking, hx_radiotherapy, focality,
        risk, t_stage, n_stage, m_stage, stage, adenopathy,
        thyroid_function, physical_exam, pathology,
        feature_columns, encoding_info
    )

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    prob_no = probabilities[0]
    prob_yes = probabilities[1]

    st.divider()
    st.subheader("📊 Kết quả dự đoán")

    result_col1, result_col2 = st.columns([1, 1.5])

    with result_col1:
        if prediction == 1:
            st.error(f"⚠️ **Dự đoán: CÓ khả năng tái phát**")
        else:
            st.success(f"✅ **Dự đoán: KHÔNG tái phát**")

        st.metric("Xác suất tái phát (Yes)", f"{prob_yes*100:.1f}%")
        st.metric("Xác suất không tái phát (No)", f"{prob_no*100:.1f}%")

        st.caption(
            "⚠️ Đây là kết quả từ model demo cho mục đích học tập, "
            "không phải công cụ chẩn đoán y khoa. Vui lòng tham khảo "
            "bác sĩ chuyên khoa cho quyết định điều trị thực tế."
        )

    with result_col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(['Không tái phát', 'Tái phát'], [prob_no, prob_yes],
                       color=['#2ecc71', '#e74c3c'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Xác suất')
        ax.set_title('Phân bố xác suất dự đoán')
        for bar, prob in zip(bars, [prob_no, prob_yes]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{prob*100:.1f}%', ha='center', fontweight='bold')
        st.pyplot(fig)

    # ============================================================
    # Feature importance
    # ============================================================
    st.divider()
    st.subheader("🔬 Mức độ ảnh hưởng của các đặc trưng (Feature Importance)")
    st.caption(
        "Biểu đồ thể hiện mức độ quan trọng của từng đặc trưng đối với "
        "quyết định của model Random Forest (tính trên toàn bộ dữ liệu huấn luyện, "
        "không phụ thuộc vào input cụ thể bạn vừa nhập)."
    )

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1],
             color='#3498db')
    ax2.set_xlabel('Mức độ quan trọng')
    ax2.set_title('Top 10 đặc trưng quan trọng nhất')
    plt.tight_layout()
    st.pyplot(fig2)

else:
    st.info("👈 Điền thông tin bệnh nhân ở trên rồi bấm **Dự đoán** để xem kết quả.")


# In[ ]:




