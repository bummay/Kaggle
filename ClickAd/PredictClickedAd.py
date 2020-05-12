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
inputDir = 'input/'
tr_tsv = 'train.tsv'
ts_tsv = 'test.tsv'
click = 'click'

train_df = pd.read_table(inputDir + tr_tsv)
test_df = pd.read_table(inputDir + ts_tsv)
df = train_df

# %%
# ユニークな値の個数を調査
train_df.nunique(dropna=False)
 # %%
# データの欠損値を調査
# train_df：I11～I14で欠損あり
train_df.isna().any()
# %%
#
train_df.isnull().sum()
# %%
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
# C1の内容とClickedの内訳
train_df[['id','click', 'C1']].groupby(['click', 'C1']).count()
# →C1の値が「421256035」or「2068315619」の場合にClicked割合が高い

def processC1(df):
    list_C1 = [421256035,2068315619]

    df['isC1'] = 0

    for item in list_C1:
        df.loc[(df['C1'] == item), 'isC1'] = 1

    df.drop('C1', axis=1, inplace=True)
    return df

train_df = processC1(train_df)
test_df = processC1(test_df)

# %%
# C2は上位5件以外は1%台もしくはそれ以下の割合だから一括にする。
df['C2'] = df['C2'].map(
    {
        3874378935:1,
        1862037199:2,
        2589684548:3,
        1537671376:4,
        1088910726:5,
    }).astype(int)


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
# ===== I11×clickの件数 → 0がほとんど。
train_df.groupby(['I11', click]).count()

# %%
train_df.groupby(['I12', click]).count()
# %%
train_df.groupby(['I13', click]).count()

# %%
