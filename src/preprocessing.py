# -*- coding: utf-8 -*-
"""
Auto-generated Python script from preprocessing.ipynb
"""

# %% [code] Cell 1
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(current_dir, "../data/raw/Thyroid_Diff.csv"))
df.head()



#======================================================================
# ## Encode Target ##
#======================================================================

# %% [code] Cell 3
df['Recurred'] = df['Recurred'].map({'No': 0, 'Yes': 1})
df['Recurred'].value_counts()


# %% [code] Cell 4
for col in ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 'Focality', 'M']:
    print(col, df[col].unique())



#======================================================================
# ## Encode cột nhị phân ##
#======================================================================

# %% [code] Cell 6
df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})

for col in ['Smoking', 'Hx Smoking', 'Hx Radiothreapy']:
    df[col] = df[col].map({'No': 0, 'Yes': 1})

df['Focality'] = df['Focality'].map({'Uni-Focal': 0, 'Multi-Focal': 1})
df['M'] = df['M'].map({'M0': 0, 'M1': 1})

df[['Gender','Smoking','Hx Smoking','Hx Radiothreapy','Focality','M']].isnull().sum()



#======================================================================
# ## Encode các cột thứ tự ##
#======================================================================

# %% [code] Cell 8
for col in ['Risk', 'T', 'N', 'Stage']:
    print(col, sorted(df[col].unique()))


# %% [code] Cell 9
risk_order = [['Low', 'Intermediate', 'High']]
t_order = [['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b']]
n_order = [['N0', 'N1a', 'N1b']]
stage_order = [['I', 'II', 'III', 'IVA', 'IVB']]

df['Risk'] = OrdinalEncoder(categories=risk_order).fit_transform(df[['Risk']])
df['T'] = OrdinalEncoder(categories=t_order).fit_transform(df[['T']])
df['N'] = OrdinalEncoder(categories=n_order).fit_transform(df[['N']])
df['Stage'] = OrdinalEncoder(categories=stage_order).fit_transform(df[['Stage']])


# %% [code] Cell 10
df[['Risk', 'T', 'N', 'Stage']].head(10)



#======================================================================
# ## Merge Adenopathy ##
#======================================================================

# %% [code] Cell 12
df['Adenopathy'].value_counts()


# %% [code] Cell 13
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



#======================================================================
# ## One hot encode các cột nominal ##
#======================================================================

# %% [code] Cell 15
nominal_cols = ['Thyroid Function', 'Physical Examination', 'Adenopathy', 'Pathology', 'Response']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=False)
df.shape


# %% [code] Cell 16
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)


# %% [code] Cell 17
df.info()
df.isnull().sum().sum()   # phải ra 0


# %% [code] Cell 18
df.to_csv(os.path.join(current_dir, '../data/processed/thyroid_processed.csv'), index=False)

