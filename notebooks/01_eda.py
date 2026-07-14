# -*- coding: utf-8 -*-
"""
Auto-generated Python script from 01_eda.ipynb
"""


#======================================================================
# # Predicting Differentiated Thyroid Cancer Recurrence using Machine Learning #
#======================================================================


#======================================================================
# ### Load and exploring dataset ###
#======================================================================

# %% [code] Cell 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline


# %% [code] Cell 4
df = pd.read_csv(r'../data/raw/Thyroid_Diff.csv')


# %% [code] Cell 5
print(df.shape)
df.info()



#======================================================================
# ##  EDA  ##
#======================================================================

# %% [code] Cell 7
df.head()


# %% [code] Cell 8
df.describe()


# %% [code] Cell 9
df.dtypes



#======================================================================
# ### Target Distribution ###
#======================================================================

# %% [code] Cell 11
df['Recurred'].value_counts()
df['Recurred'].value_counts(normalize=True)

sns.countplot(data=df, x = 'Recurred')
plt.title('Phân bố Recurred')
plt.show()



#======================================================================
# ### Age Distribution ###
#======================================================================

# %% [code] Cell 13
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(data=df, x='Age', hue='Recurred', kde=True, ax = axes[0])
axes[0].set_title('Phân bố Age theo Recurred')

sns.boxplot(data=df, x='Recurred', y='Age', ax = axes[1])
axes[1].set_title('Boxplot Age theo Recurred')

plt.tight_layout()
plt.show()



#======================================================================
# ### Countplot categorical ###
#======================================================================

# %% [code] Cell 15
categorical_cols = ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy',
                     'Thyroid Function', 'Physical Examination', 'Adenopathy',
                     'Pathology', 'Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response']
fig, axes = plt.subplots(5, 3, figsize=(18, 22))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.countplot(data=df, x=col, hue='Recurred', ax=axes[i])
    axes[i].set_title(col)
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# %% [code] Cell 16
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())


# %% [code] Cell 17
pd.crosstab(df['T'], df['Stage'])
pd.crosstab(df['N'], df['Stage'])
pd.crosstab(df['M'], df['Stage'])



#======================================================================
# ### Check outlier ###
#======================================================================

# %% [code] Cell 19
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Age'] < Q1 - 1.5*IQR) | (df['Age'] > Q3 - 1.5*IQR)]
print(f'Số outlier: {len(outliers)}')



#======================================================================
# ### Percentage ###
#======================================================================

# %% [code] Cell 21
pd.crosstab(df['Risk'], df['Recurred'], normalize='index')


# %% [code] Cell 22
from ydata_profiling import ProfileReport

df = pd.read_csv(r"../data/raw/Thyroid_Diff.csv")

profile = ProfileReport(
    df,
    title="Thyroid Cancer Report",
    explorative=True
)

profile.to_file('Thyroid_Cancer_Profile_Report.html')



#======================================================================
# ## Kết luận từ EDA
# 
# - Risk = High có tỷ lệ tái phát 85% (so với 12% ở Risk = Low) → biến phân biệt mạnh nhất
# - T, N, M, Stage có tương quan cao (Stage IVB luôn đi kèm M1) → cân nhắc giữ Stage, bỏ M để tránh đa cộng tuyến
# - Adenopathy = Posterior chỉ có 3 mẫu → gộp vào nhóm "Other"
# - Age không khác biệt rõ giữa 2 nhóm Recurred → có thể không phải biến quan trọng
# - Target lệch 72%/28% → áp dụng SMOTE hoặc class_weight khi train
#======================================================================
