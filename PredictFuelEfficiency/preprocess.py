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

#
def replaceNa(df, valNa):
    df = df.replace(valNa, np.nan)
    return df


def dropName(df):
    df = df.drop(['car name'], axis=1)
    return df

def fillHp(df, filled):
    df = df.fillna({'horsepower': filled})
    return df

def preprocess(df):
    vNa = '?'
    df = replaceNa(df, vNa)
    df = dropName(df)
    df['horsepower'] = df['horsepower'].astype(float)

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





# %%
# pwレシオと燃費の散布：あまり関係なさそう
sns.scatterplot(data = train_df, x='pwRatio', y = 'mpg')

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
# 馬力と燃費の散布：きれいにバラバラ
sns.scatterplot(data = train_df, x='horsepower', y = 'mpg')

# %%
# 車重と燃費の散布：軽いほど燃費がよさそう
sns.scatterplot(data = train_df, x='weight', y = 'mpg')

# %%
# 加速度と燃費の散布：低いと燃費も悪そう
sns.scatterplot(data = train_df, x='acceleration', y = 'mpg')

# %%
# MYと燃費の散布：新しいほど燃費はよさそう
sns.scatterplot(data = train_df, x='model year', y = 'mpg')
