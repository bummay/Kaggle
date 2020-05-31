# %%
import pandas as pd
import numpy as np
import datetime
import pickle
with open('list_clubinfo.pkl', mode='rb') as f:
    list_clubinfo = pickle.load(f)

# train,test,conditionの諸データをすべて結合して
# 年、ディビジョン、日付、ホームチーム、アウェイチーム、スコアのみを抽出する。
inputDir = 'input/'

train = pd.read_csv(inputDir + 'train.csv')
add = pd.read_csv(inputDir + 'train_add.csv')
test = pd.read_csv(inputDir + 'test.csv')
add_2014 = pd.read_csv(inputDir + '2014_add.csv')

train = train.append(add)
train = train.append(test)
train = train.append(add_2014)


train = train.sort_values('id')
del add, test

cond = pd.read_csv(inputDir + 'condition.csv')
cond_add = pd.read_csv(inputDir + 'condition_add.csv')
cond = cond.append(cond_add).sort_values('id')
cond = cond[['id', 'home_score', 'away_score']]
del cond_add
# %%
pt_win = 3
pt_draw = 1
pt_lose = 0


df_tmp = pd.merge(train, cond, on='id', how='left')
df = df_tmp[['year', 'stage', 'gameday', 'home', 'away', 'home_score', 'away_score']]

df.loc[df['home'] == 'ザスパ草津', 'home'] = 'ザスパクサツ群馬'
df.loc[df['away'] == 'ザスパ草津', 'away'] = 'ザスパクサツ群馬'
# %%
# 0. dataframeのチームごとの「得点、失点、獲得勝ち点」列を追加し、
# 試合結果から対応する列に値をセット。
for item in list_clubinfo:
    team = item[0]
    # ホーム/アウェイチームのコードを列[home][away]にセットする。
    df.loc[(df['home'] == item[1]), 'home'] = team
    df.loc[(df['home'] == team), team + '_for'] = df['home_score']
    df.loc[(df['home'] == team), team + '_against'] = df['away_score']
    df.loc[(df['home'] == team) & (df['home_score'] > df['away_score']), team + '_point'] = pt_win
    df.loc[(df['home'] == team) & (df['home_score'] == df['away_score']), team + '_point'] = pt_draw
    df.loc[(df['home'] == team) & (df['home_score'] < df['away_score']), team + '_point'] = pt_lose

    df.loc[(df['away'] == item[1]), 'away'] = team
    df.loc[(df['away'] == team), team + '_for'] = df['away_score']
    df.loc[(df['away'] == team), team + '_against'] = df['home_score']
    df.loc[(df['away'] == team) & (df['away_score'] > df['home_score']), team + '_point'] = pt_win
    df.loc[(df['away'] == team) & (df['away_score'] == df['home_score']), team + '_point'] = pt_draw
    df.loc[(df['away'] == team) & (df['away_score'] < df['home_score']), team + '_point'] = pt_lose

df['gamedate'] = pd.to_datetime((df['year'].astype(str) + '/' + df['gameday']).str[:10])
df.drop(['gameday', 'home_score', 'away_score'], axis=1, inplace=True)
df.fillna(0)

# 1. ディビジョン/年/試合日/チームごとの勝ち点、得失点差、得点計、失点計を集計
# %%
# %%
df_stas = pd.DataFrame(index=[], columns=['stage', 'date', 'team', 'points', 'goals_for', 'goals_against'])
# %%
list_stage = df['stage'].unique().tolist()
for stg in list_stage:
    list_date = df[df['stage'] == stg]['gamedate'].unique().tolist()



# 2. ディビジョン/年/試合日/ごとの順位表を作成
# 3. 各試合のレコードに対して、ホームチーム/アウェイチームの各順位を追加

