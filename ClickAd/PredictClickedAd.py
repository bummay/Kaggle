# %%
import pandas as pd
import xgboost as xgb

# 日付の処理で使う
import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# %%
# trainデータにaddデータをappend
#
inputDir = 'input/'
tr_tsv = 'train.tsv'
ts_tsv = 'test.tsv'
click = 'click'

train_df = pd.read_table(inputDir + tr_tsv)
test_df = pd.read_table(inputDir + ts_tsv)
 # %%
# データの欠損値を調査
# train_df：I11～I14で欠損あり
train_df.isna().any()
test_df.isna().any()

# %%
sns.countplot(x=click, data=train_df)

# %%
# ===== I1×clickの件数 → I1の値にかかわらず一定数のクリックはある。
sns.countplot(x='I1', hue=click, data=train_df)

# %%
# ===== I2×clickの件数 → 0～5で増加。15まで横ばい。16で急増して20あたりから漸減。
sns.countplot(x='I2', hue=click, data=train_df)

# %%
# ===== I3×clickの件数 → 0の場合にクリックされることは少ない。1と2の場合は一定割合クリックされている。
sns.countplot(x='I3', hue=click, data=train_df)

# %%
# ===== I4×clickの件数 → 0の場合にクリックされることが多い。1と2の場合はほとんどクリックされない。
sns.countplot(x='I4', hue=click, data=train_df)

# %%
# ===== I5×clickの件数 → 0がほとんど
train_df.groupby(['I5', click]).count()

# %%
# ===== I6×clickの件数 → 0がほとんど。クリック割合はそんなに変わらない？
sns.countplot(x='I6', hue=click, data=train_df)

# %%
# ===== I7×clickの件数 → 0がほとんど。1は絶対数も少ないしクリック数も少なそう。
sns.countplot(x='I7', hue=click, data=train_df)

# %%
# ===== I8×clickの件数 → I7とほぼ同じ。
sns.countplot(x='I8', hue=click, data=train_df)

# %%
# ===== I9×clickの件数 → I7、I8に比べると1の件数もクリック数も多い。
sns.countplot(x='I9', hue=click, data=train_df)

# %%
# ===== I10×clickの件数 → I9とほぼ同じ。
sns.countplot(x='I10', hue=click, data=train_df)

# %%
# ===== I11×clickの件数 →
train_df.groupby(['I11', click]).count()

# %%
