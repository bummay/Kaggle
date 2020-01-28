# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing, linear_model
from scipy.stats import skew

# %%
# データを読み込み

inputDir = 'input/'

train_df = pd.read_table(inputDir + 'train.tsv')
test_df = pd.read_table(inputDir + 'test.tsv')
test_df['mpg'] = np.nan

def replaceNa(df, valNa):
    df = df.replace(valNa, np.nan)
    return df

def getManufacturer(df):
    # まず車名からメーカー名を取得する
    for index, row in df.iterrows():
        name = row['car name']
        spl = name.split(' ')
        df.at[index, 'manufacturer'] = spl[0]

    # メーカー名の表記ゆれを修正する
    df.loc[((df['manufacturer'] == 'chevroelt') | (df['manufacturer'] == 'chevy')), 'manufacturer'] = 'chevrolet'
    df.loc[(df['manufacturer'] == 'maxda'), 'manufacturer'] = 'mazda'
    df.loc[(df['manufacturer'] == 'toyouta'), 'manufacturer'] = 'toyota'
    df.loc[((df['manufacturer'] == 'vw') | (df['manufacturer'] == 'vokswagen')), 'manufacturer'] = 'volkswagen'
    df.loc[(df['manufacturer'] == 'mercedes-benz'), 'manufacturer'] = 'mercedes'

    df.drop('car name', axis=1, inplace=True)
    # # メーカー名から国名を取得する。
    # # 全部で7カ国あるが、件数が少ない「フランス、イタリア、スウェーデン、イギリス」はその他扱い
    # # 日独米の3カ国+その他の4種類に分ける
    # df['USA'] = 1
    # df['JPN'] = 0
    # df['GER'] = 0
    # df.loc[((df['manufacturer'] == 'audi') | (df['manufacturer'] == 'bmw') | (df['manufacturer'] == 'opel') | (
    #     df['manufacturer'] == 'mercedes') | (df['manufacturer'] == 'volkswagen')), 'GER'] = 1
    # df.loc[((df['manufacturer'] == 'datsun') | (df['manufacturer'] == 'honda') | (df['manufacturer'] == 'mazda') | (df['manufacturer'] == 'nissan') | (df['manufacturer'] == 'subaru') | (df['manufacturer'] == 'toyota')), 'JPN'] = 1

    # df.loc[(
    #         (df['manufacturer'] == 'peugeot') |
    #         (df['manufacturer'] == 'renault') |
    #         (df['manufacturer'] == 'fiat') |
    #         (df['manufacturer'] == 'volvo') |
    #         (df['manufacturer'] == 'saab') |
    #         (df['manufacturer'] == 'triumph') |
    #         (df['JPN'] == 1) |
    #         (df['GER'] == 1)
    #         )
    #     , 'USA'] = 0

    return df

def regularization(df, colName):
    df[colName] = preprocessing.scale(df[colName])

    return df

def convertToStr(df, colName):
    df[colName] = df[colName].astype(str)

    return df

def fillHorsepower(df):
    df.replace('?', np.nan, inplace=True)
    df['horsepower'] = df['horsepower'].astype(float)
    df['horsepower'] = df.groupby(['manufacturer', 'cylinders'])[
        'horsepower'].transform(lambda x: x.fillna(x.mean()))

    return df

def preprocess(df):
    # メーカー名を取得
    df = getManufacturer(df)

    # 順序尺度の説明変数を文字列に変換
    lstToStr = ['cylinders', 'origin']
    for colName in lstToStr:
        convertToStr(df, colName)

    # 馬力の欠損値を「メーカー/気筒数」ごとの平均値で穴埋め
    df = fillHorsepower(df)

    # 数値型説明変数の歪度を測定し、歪度が大きいものは対数をとる。
    numeric_features = df.dtypes[df.dtypes != 'object'].index
    skewed_features = df[numeric_features].apply(lambda x:skew(x.dropna()))
    skewed_features = skewed_features[skewed_features > 0.75].index

    df = pd.get_dummies(data = df,dummy_na=True)
    df = df.fillna(df.mean())
    df[skewed_features] = np.log1p(df[skewed_features])

    return df


# %%
# 学習用と評価用のデータを個別に標準化しては意味がないので両者を結合する。
concat_df = pd.concat([train_df, test_df], sort=True)
mpg = concat_df['mpg'].values
concat_df = preprocess(concat_df)
concat_df['mpg'] = mpg


# %%
# lstColName = ['displacement', 'horsepower', 'weight',
#                 'model year', 'USA', 'JPN', 'GER'
#                 , 'acceleration'
#                 # , 'GER', 'FRA', 'GBR', 'ITA', 'SWE'
#                 ]
# for colName in lstColName:
#     regularization(concat_df, colName)
# %%
train_df = concat_df[!concat_df.mpg.isna() == False]
test_df = concat_df[concat_df.mpg.isna()]
test_df.drop(['mpg'], axis=1, inplace=True)

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


