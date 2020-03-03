# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing, linear_model

# %%
# データを読み込み
#
# 欠損している(値が「?」)horsepower列をNaNに変換

inputDir = 'input/'

train_df = pd.read_table(inputDir + 'train.tsv')
test_df = pd.read_table(inputDir + 'test.tsv')

def replaceNa(df, valNa):
    df = df.replace(valNa, np.nan)
    return df

def getCountry(df):
    # まず車名からメーカー名を取得する
    for index, row in df.iterrows():
        name = row['car name']
        spl = name.split(' ')
        df.at[index, 'manufacturer'] = spl[0]

    # メーカー名の表記ゆれを修正する
    df.loc[(
            (df['manufacturer'] == 'chevroelt') |
            (df['manufacturer'] == 'chevy')
            ), 'manufacturer'] = 'chevrolet'
    df.loc[(df['manufacturer'] == 'maxda'), 'manufacturer'] = 'mazda'
    df.loc[(df['manufacturer'] == 'toyouta'), 'manufacturer'] = 'toyota'
    df.loc[(
            (df['manufacturer'] == 'vw') |
            (df['manufacturer'] == 'vokswagen')
            ), 'manufacturer'] = 'volkswagen'
    df.loc[(df['manufacturer'] == 'mercedes-benz'), 'manufacturer'] = 'mercedes'

    # メーカー名から国名を取得する。
    # 全部で7カ国あるが、件数が少ない「フランス、イタリア、スウェーデン、イギリス」はその他扱い
    # 日独米の3カ国+その他の4種類に分ける
    df['country'] = 'USA'
    df.loc[(
            (df['manufacturer'] == 'audi') |
            (df['manufacturer'] == 'bmw') |
            (df['manufacturer'] == 'opel') |
            (df['manufacturer'] == 'mercedes') |
            (df['manufacturer'] == 'volkswagen')
            ), 'country'] = 'GER'
    df.loc[(
            (df['manufacturer'] == 'datsun') |
            (df['manufacturer'] == 'honda') |
            (df['manufacturer'] == 'mazda') |
            (df['manufacturer'] == 'nissan') |
            (df['manufacturer'] == 'subaru') |
            (df['manufacturer'] == 'toyota'))
            , 'country'] = 'JPN'
    df.loc[(
            (df['manufacturer'] == 'peugeot') |
            (df['manufacturer'] == 'renault') |
            (df['manufacturer'] == 'fiat') |
            (df['manufacturer'] == 'volvo') |
            (df['manufacturer'] == 'saab') |
            (df['manufacturer'] == 'triumph')
            )
        , 'country'] = 'EUR'
    df.drop(['car name'], axis=1, inplace=True)

    return df

def cnvToStr(df, cols):
    for col in cols:
        df[col] = df[col].astype(str)

    return df


def dropName(df):
    df.drop([
            'origin'
            ], axis=1, inplace=True)
    return df


def preprocess(df):
    colToStr = ['cylinders']

    df = replaceNa(df, '?')
    df['horsepower'] = df['horsepower'].astype(float)
    df = getCountry(df)
    df['horsepower'] = df.groupby(['manufacturer', 'cylinders'])['horsepower'].transform(lambda x:x.fillna(x.mean()))
    df = cnvToStr(df, colToStr)
    df = dropName(df)

    return df


# %%
# 学習用と評価用のデータを個別に標準化しては意味がないので両者を結合する。
concat_df = pd.concat([train_df, test_df], sort=True)
concat_df = concat_df.drop(['mpg'], axis=1)
concat_df = preprocess(concat_df)


# %%
tr_df = train_df[['id', 'mpg']]
ts_df = test_df['id']

train_df = pd.merge(tr_df, concat_df, on='id')
test_df = pd.merge(ts_df, concat_df, on='id')


# %%
train_df.to_csv(inputDir + 'train_df.csv', index=False)
test_df.to_csv(inputDir + 'test_df.csv', index=False)


# %%
# 気筒数と燃費の散布：4>6>8 の順によさげ
sns.scatterplot(data = train_df, x='cylinders', y = 'mpg')

# %%
# 排気量と燃費の散布
# 排気量が大きいほど
#   ・燃費が低そう
#   ・燃費のばらつきが小さい
sns.scatterplot(data = train_df, x='displacement', y = 'mpg')

# %%
# 馬力と燃費の散布
# 馬力が大きいほど
#   ・燃費が低そう
#   ・燃費のばらつきが小さい
sns.scatterplot(data=train_df, x='horsepower', y='mpg')

# %%
# 車重と燃費の散布
# 車重が重いほど
#   ・燃費が低そう
#   ・燃費のばらつきが少ない
sns.scatterplot(data = train_df, x='weight', y = 'mpg')

# %%
# 加速度と燃費の散布：加速度が低いと燃費も悪そう
sns.scatterplot(data = train_df, x='acceleration', y = 'mpg')

# %%
# MYと燃費の散布：新しいほど燃費はよさそう
sns.scatterplot(data = train_df, x='model year', y = 'mpg')

# %%
sns.scatterplot(data=train_df, x='FRA', y='mpg')
# %%
sns.scatterplot(data=train_df, x='GBR', y='mpg')
# %%
sns.scatterplot(data=train_df, x='GER', y='mpg')
# %%
sns.scatterplot(data=train_df, x='ITA', y='mpg')
# %%
sns.scatterplot(data=train_df, x='JPN', y='mpg')
# %%
sns.scatterplot(data=train_df, x='EUR', y='mpg')
# %%
sns.scatterplot(data=train_df, x='USA', y='mpg')


