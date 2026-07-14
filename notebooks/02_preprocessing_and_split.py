# -*- coding: utf-8 -*-
"""
Auto-generated Python script from 02_preprocessing_and_split.ipynb
"""

# %% [code] Cell 1
# !pip install imbalanced-learn


# %% [code] Cell 2
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# %% [code] Cell 3
df = pd.read_csv('../data/raw/Thyroid_Diff.csv')
df.shape


# %% [code] Cell 4
df['Recurred'] = df['Recurred'].map({'No': 0, 'Yes': 1})
df['Recurred'].value_counts()


# %% [code] Cell 5
df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})

for col in ['Smoking', 'Hx Smoking', 'Hx Radiothreapy']:
    df[col] = df[col].map({'No': 0, 'Yes': 1})

df['Focality'] = df['Focality'].map({'Uni-Focal': 0, 'Multi-Focal': 1})
df['M'] = df['M'].map({'M0': 0, 'M1': 1})

df[['Gender','Smoking','Hx Smoking','Hx Radiothreapy','Focality','M']].isnull().sum()


# %% [code] Cell 6
risk_order = [['Low', 'Intermediate', 'High']]
t_order = [['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b']]
n_order = [['N0', 'N1a', 'N1b']]
stage_order = [['I', 'II', 'III', 'IVA', 'IVB']]

df['Risk'] = OrdinalEncoder(categories=risk_order).fit_transform(df[['Risk']])
df['T'] = OrdinalEncoder(categories=t_order).fit_transform(df[['T']])
df['N'] = OrdinalEncoder(categories=n_order).fit_transform(df[['N']])
df['Stage'] = OrdinalEncoder(categories=stage_order).fit_transform(df[['Stage']])

df[['Risk', 'T', 'N', 'Stage']].head(10)


# %% [code] Cell 7
df['Adenopathy'].value_counts()


# %% [code] Cell 8
adenopathy_map = {
    'No': 'No',
    'Right': 'Unilateral',
    'Left': 'Unilateral',
    'Bilateral': 'Bilateral_Extensive',
    'Extensive': 'Bilateral_Extensive',
    'Posterior': 'Bilateral_Extensive'
}
df['Adenopathy'] = df['Adenopathy'].map(adenopathy_map)
df['Adenopathy'].value_counts()


# %% [code] Cell 9
nominal_cols = ['Thyroid Function', 'Physical Examination', 'Adenopathy', 'Pathology', 'Response']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=False)
df.shape


# %% [code] Cell 10
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

df.info()


# %% [code] Cell 11
df.isnull().sum().sum()   # phải ra 0


# %% [code] Cell 12
df.to_csv('../data/processed/thyroid_processed.csv', index=False)



#======================================================================
# ## Train test split ##
#======================================================================

# %% [code] Cell 14
X = df.drop('Recurred', axis=1)
y = df['Recurred']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# %% [code] Cell 15
print("Tỷ lệ trong train:")
print(y_train.value_counts(normalize=True))

print("\nTỷ lệ trong test:")
print(y_test.value_counts(normalize=True))


# %% [code] Cell 16
X_train.to_csv('../data/processed/X_train.csv', index=False)
X_test.to_csv('../data/processed/X_test.csv', index=False)
y_train.to_csv('../data/processed/y_train.csv', index=False)
y_test.to_csv('../data/processed/y_test.csv', index=False)


# %% [code] Cell 17
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Trước SMOTE:", y_train.value_counts().to_dict())
print("Sau SMOTE:", y_train_smote.value_counts().to_dict())



#======================================================================
# ## Model dùng class_weight, train trên X_train/y_train ##
#======================================================================

# %% [code] Cell 19
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

model_weighted = LogisticRegression(class_weight='balanced', random_state=42)
model_weighted.fit(X_train, y_train)


# %% [code] Cell 20
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_weighted = LogisticRegression(class_weight='balanced', random_state=42)
model_weighted.fit(X_train_scaled, y_train)


# %% [code] Cell 21
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import matplotlib.pyplot as plt



#======================================================================
# ## Train Logistic Regression ##
#======================================================================

# %% [code] Cell 23
X_train_smote_lr, y_train_smote_lr = smote.fit_resample(X_train_scaled, y_train)

X_train_smote_rf, y_train_smote_rf = smote.fit_resample(X_train, y_train)


# %% [code] Cell 24
log_reg_weighted = LogisticRegression(class_weight='balanced', random_state=42)
log_reg_weighted.fit(X_train_scaled, y_train)

y_pred_lr_weighted = log_reg_weighted.predict(X_test_scaled)
print("=== Logistic Regression (class_weight) ===")
print(classification_report(y_test, y_pred_lr_weighted))


# %% [code] Cell 25
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

log_reg_smote = LogisticRegression(random_state=42)
log_reg_smote.fit(X_train_smote, y_train_smote)

# Predict trên X_test_scaled — khớp đúng loại dữ liệu
y_pred_lr_smote = log_reg_smote.predict(X_test_scaled)
print("=== Logistic Regression (SMOTE) ===")
print(classification_report(y_test, y_pred_lr_smote))


# %% [code] Cell 26
from sklearn.ensemble import RandomForestClassifier

rf_weighted = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_weighted.fit(X_train, y_train)

y_pred_rf_weighted = rf_weighted.predict(X_test)
print("=== Random Forest (class_weight) ===")
print(classification_report(y_test, y_pred_rf_weighted))


# %% [code] Cell 27
from imblearn.over_sampling import SMOTE

smote_rf = SMOTE(random_state=42)
X_train_smote_rf, y_train_smote_rf = smote_rf.fit_resample(X_train, y_train)

rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_smote_rf, y_train_smote_rf)

y_pred_rf_smote = rf_smote.predict(X_test)
print("=== Random Forest (SMOTE) ===")
print(classification_report(y_test, y_pred_rf_smote))


# %% [code] Cell 28
# Tạo bộ X không có Response
response_cols = [c for c in X_train.columns if c.startswith('Response_')]
X_train_no_response = X_train.drop(columns=response_cols)
X_test_no_response = X_test.drop(columns=response_cols)

rf_no_response = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_no_response.fit(X_train_no_response, y_train)

y_pred_no_response = rf_no_response.predict(X_test_no_response)
print("=== Random Forest (KHÔNG có Response) ===")
print(classification_report(y_test, y_pred_no_response))


# %% [code] Cell 29
import joblib


# %% [code] Cell 30
joblib.dump(rf_no_response, '../models/best_model.pkl')
print("Đã lưu model vào models/best_model.pkl")


# %% [code] Cell 31
joblib.dump(X_train_no_response.columns.tolist(), '../models/feature_columns.pkl')
print("Đã lưu danh sách cột vào models/feature_columns.pkl")


# %% [code] Cell 32
encoding_info = {
    'risk_order': ['Low', 'Intermediate', 'High'],
    't_order': ['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b'],
    'n_order': ['N0', 'N1a', 'N1b'],
    'stage_order': ['I', 'II', 'III', 'IVA', 'IVB'],
    'adenopathy_map': {
        'No': 'No', 'Right': 'Unilateral', 'Left': 'Unilateral',
        'Bilateral': 'Bilateral_Extensive', 'Extensive': 'Bilateral_Extensive',
        'Posterior': 'Bilateral_Extensive'
    }
}
joblib.dump(encoding_info, '../models/encoding_info.pkl')


# %% [code] Cell 33
loaded_model = joblib.load('../models/best_model.pkl')
loaded_columns = joblib.load('../models/feature_columns.pkl')

print("Số features model mong đợi:", len(loaded_columns))
print(loaded_columns)

# Thử predict lại để chắc chắn load đúng, kết quả phải giống y_pred_no_response
test_pred = loaded_model.predict(X_test_no_response)
print("Khớp với kết quả gốc:", (test_pred == y_pred_no_response).all())

