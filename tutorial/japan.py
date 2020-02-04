# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%
# データの読み取り
df = pd.read_csv("input/japan.csv", index_col='prefecture', encoding="shift-jis")
df.rename(columns={'longtitude': 'longitude'}, inplace=True)

# データのとり方(インデックス指定)
# %%
df.loc['神奈川県']
# %%
df.loc[:,['longitude', 'latitude']]
# %%
df.loc['神奈川県', ['longitude', 'latitude']]
# %%
df.loc['神奈川県', 'latitude']
# %%
df.at['神奈川県', 'latitude']


# データのとり方(位置指定)
# %%
df.iloc[13]
# %%
df.iloc[3:5, 0:2]
# %%
df.iat[1,1]

# %%
df2 = df[(df.index != '沖縄県') & (df.index != '北海道') ]
df2

# 新しい列の追加
# 60度単位(度:分:秒)文字列の緯度と経度を度単位の数値に変換
# %%
def calc_number(d_num):
    # convert Degrees(str) to float
    d = d_num.split(':')
    n = int(d[0]) + float(d[1]) / 60 + float(d[2]) / 3600
    return n

df['latitude_num'] = df['latitude'].apply(calc_number)
df['longitude_num'] = df['longitude'].apply(calc_number)
print(df[df['latitude_num'] > 135].index)

# %%
def conv_altitude(str_alt):
    lenstr = len(str_alt)
    n = int(str_alt[:lenstr - 1])

    return n

df['altitude_num'] = df['altitude'].apply(conv_altitude)

# %%
longitude_mean = df['longitude_num'].mean()
latitude_mean = df['latitude_num'].mean()
print('latitude:{:.2f}, longitude:{:.2f}'.format(latitude_mean, longitude_mean))

# %%
# イテレーションの練習
plt.figure(figsize=(8,8))
for key , row in df.loc[:,['longitude_num', 'latitude_num']].iterrows():
    plt.scatter(row['latitude_num'], row['longitude_num'], marker = 'x', c='blue')
plt.scatter(latitude_mean, longitude_mean, c='red')


