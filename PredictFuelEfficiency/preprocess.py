# %%
import pandas as pd
import numpy as np
import seaborn as sns

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
    df.loc[((df['manufacturer'] == 'chevroelt') | (df['manufacturer'] == 'chevy')), 'manufacturer'] = 'chevrolet'
    df.loc[(df['manufacturer'] == 'maxda'), 'manufacturer'] = 'mazda'
    df.loc[(df['manufacturer'] == 'toyouta'), 'manufacturer'] = 'toyota'
    df.loc[((df['manufacturer'] == 'vw') | (df['manufacturer'] == 'vokswagen')), 'manufacturer'] = 'volkswagen'
    df.loc[(df['manufacturer'] == 'mercedes-benz'), 'manufacturer'] = 'mercedes'

    # メーカー名から国名を取得する
    df['FRA'] = 0
    df['GBR'] = 0
    df['GER'] = 0
    df['ITA'] = 0
    df['JPN'] = 0
    df['SWE'] = 0
    df['USA'] = 1
    df.loc[((df['manufacturer'] == 'peugeot') | (df['manufacturer'] == 'renault')), 'FRA'] = 1
    df.loc[((df['manufacturer'] == 'audi') | (df['manufacturer'] == 'bmw') | (df['manufacturer'] == 'opel') | (
        df['manufacturer'] == 'mercedes') | (df['manufacturer'] == 'volkswagen')), 'GER'] = 1
    df.loc[(df['manufacturer'] == 'fiat'), 'ITA'] = 1
    df.loc[((df['manufacturer'] == 'datsun') | (df['manufacturer'] == 'honda') | (df['manufacturer'] == 'mazda') | (df['manufacturer'] == 'nissan') | (df['manufacturer'] == 'subaru') | (df['manufacturer'] == 'toyota')), 'JPN'] = 1
    df.loc[((df['manufacturer'] == 'volvo') | (df['manufacturer'] == 'saab')), 'SWE'] = 1
    df.loc[(df['manufacturer'] == 'triumph'), 'GBR'] = 1
    df.loc[(
        (df['FRA'] == 1) | (df['GER'] == 1) |
        (df['ITA'] == 1) | (df['JPN'] == 1) |
        (df['SWE'] == 1) | (df['GBR'] == 1)),'USA'
    ] = 0
    df = df.drop(['car name', 'manufacturer'], axis=1)
    df = pd.get_dummies(df)

    return df

def dropName(df):
    df = df.drop(['origin'], axis=1)
    return df

def fillHp(df, filled):
    df = df.fillna({'horsepower': filled})
    return df


def regularization(df, colName):
    Xmax = df[colName].max()
    Xmin = df[colName].min()

    df[colName] = (df[colName] - df[colName].mean()) / df[colName].std()
    return df

def preprocess(df):
    vNa = '?'
    df = replaceNa(df, vNa)
    df['horsepower'] = df['horsepower'].astype(float)
    df = getCountry(df)
    df = dropName(df)


    lstColName = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year']
    for colName in lstColName:
        regularization(df, colName)

    return df


# %%
train_df = preprocess(train_df)
test_df = preprocess(test_df)

train_sum = train_df['horsepower'].sum()
test_sum = test_df['horsepower'].sum()
train_count = train_df['horsepower'].count()
test_count = test_df['horsepower'].count()

fhp = round((train_sum + test_sum) / (train_count + test_count) , 0)

train_df = fillHp(train_df, fhp)
test_df = fillHp(test_df, fhp)


# %%
train_df.to_csv(inputDir + 'train_df.csv', index=False)
test_df.to_csv(inputDir + 'test_df.csv', index=False)


# # %%
# # 気筒数と燃費の散布：4>6>8 の順によさげ
# sns.scatterplot(data = train_df, x='cylinders', y = 'mpg')

# # %%
# # 排気量と燃費の散布
# # 排気量が大きいほど
# #   ・燃費が低そう
# #   ・燃費のばらつきが小さい
# sns.scatterplot(data = train_df, x='displacement', y = 'mpg')

# # %%
# # 馬力と燃費の散布
# # 馬力が大きいほど
# #   ・燃費が低そう
# #   ・燃費のばらつきが小さい
# sns.scatterplot(data=train_df, x='horsepower', y='mpg')

# # %%
# # 車重と燃費の散布
# # 車重が重いほど
# #   ・燃費が低そう
# #   ・燃費のばらつきが少ない
# sns.scatterplot(data = train_df, x='weight', y = 'mpg')

# # %%
# # 加速度と燃費の散布：加速度が低いと燃費も悪そう
# sns.scatterplot(data = train_df, x='acceleration', y = 'mpg')

# # %%
# # MYと燃費の散布：新しいほど燃費はよさそう
# sns.scatterplot(data = train_df, x='model year', y = 'mpg')



# %%
