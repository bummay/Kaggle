# %%
import pandas as pd
import numpy as np
import xgboost as xgb

import pickle

# 日付の処理で使う
import datetime
import jpholiday
import pandas.tseries.offsets as offsets

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from pyproj import Geod
from sklearn.model_selection import train_test_split

# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import fbeta_score
# from sklearn.metrics import make_scorer


# %%
# 諸情報の読み込み
with open('list_clubinfo.pkl', mode='rb') as f:
    list_clubinfo = pickle.load(f)

with open('list_rival.pkl', mode='rb') as f:
    list_rival = pickle.load(f)

# %%
# trainデータにaddデータをappend
#
inputDir = 'input/'

train = pd.read_csv(inputDir + 'train.csv')
train_add = pd.read_csv(inputDir + 'train_add.csv')
train = train.append(train_add)
train = train.sort_values('id')
del train_add

test = pd.read_csv(inputDir + 'test.csv')
# %%
# 2014年の無観客試合(浦和vs清水)を除外
train_df = train
train_df = train_df[train_df['y'] > 0]
test_df = test

# cond_dfは以下の項目のみ残す。
#   ホームチームスコア
#   アウェイチームスコア
#   天候
#   気温
#   湿度
cond_df = pd.read_csv(inputDir + 'condition.csv')
cond_add_df = pd.read_csv(inputDir + 'condition_add.csv')
cond_df = cond_df.append(cond_add_df).sort_values('id')
cond_df.drop([
                'referee',
                'home_team', 'home_01', 'home_02', 'home_03', 'home_04', 'home_05',
                'home_06', 'home_07', 'home_08', 'home_09', 'home_10', 'home_11',
                'away_team', 'away_01', 'away_02', 'away_03', 'away_04', 'away_05',
                'away_06', 'away_07', 'away_08', 'away_09', 'away_10', 'away_11'
            ],axis=1, inplace=True)
del cond_add_df

# stadium_dfは以下のとおり加工
# 住所を削除
# 屋根のカバー状況を整数値に変換
# 略称をダミー変数に変換
stadium_df = pd.read_csv(inputDir + 'stadium.csv')
stadium_df = stadium_df.rename(columns={'name':'stadium'})
stadium_df.drop(['address'],axis=1, inplace=True)
# stadium_df['name'] = stadium_df['abbr']
# stadium_df = pd.get_dummies(stadium_df, columns=['abbr'], prefix='held')

# train/testのそれぞれに、cond_dfとstadium_dfを結合
# 結合後、スタジアム名を略称に置き換えて略称列は削除
def mergeStadiumDf(df):
    df = pd.merge(df, cond_df, on='id', how='left')
    df = pd.merge(df, stadium_df, on='stadium')
    df.drop(['stadium'], axis=1, inplace=True)
    df = df.rename(columns={'abbr':'stadium'})
    return df

train_df = mergeStadiumDf(train_df)
test_df = mergeStadiumDf(test_df)

# 来場者数とスタジアムの収容人数から収容率を取得
train_df['yratio'] = (train_df['y'] / train_df['capa'])

# %%
def renameKusatsu(df):
    df.loc[df['home'] == 'ザスパ草津', 'home'] = 'ザスパクサツ群馬'
    df.loc[df['away'] == 'ザスパ草津', 'away'] = 'ザスパクサツ群馬'
    return df

train_df = renameKusatsu(train_df)
test_df = renameKusatsu(test_df)

# %%
# stageをコードに変換
def processStage(df):
    df['stage'] = df['stage'].map(
        {'Ｊ１': 1, 'Ｊ２': 0}).astype(int)

    return df

train_df = processStage(train_df)
test_df = processStage(test_df)

# %%
# 節数→とりあえず削除
def processMatch(df):
    df.drop(['match'], axis=1, inplace=True)
    return df

train_df = processMatch(train_df)
test_df = processMatch(test_df)


# %%
# gamedayから「月」と「当日が休日か」と「翌日が休日か」を取得
# gamedayとyearは削除
def processGameday(df):
    df['month'] = df['gameday'].str[:2].astype(int)
    df['gamedate'] = pd.to_datetime((df['year'].astype(str) + '/' + df['gameday']).str[:10])

    df['weekday'] = df['gamedate'].dt.weekday

    df['isHoliday'] = ((df['gamedate'].map(
        jpholiday.is_holiday).astype(int) == 1) | (df['weekday'] > 4)).astype(int)

    df['nextday'] = df['gamedate'] + offsets.Day()
    df['nextIsHoliday'] = ((df['nextday'].map(
        jpholiday.is_holiday).astype(int) == 1) | (df['nextday'].dt.weekday > 4)).astype(int)

    df.drop(['gameday','gamedate', 'nextday', 'weekday', 'year'], axis=1, inplace=True)
    return df

train_df = processGameday(train_df)
test_df = processGameday(test_df)

# %%
# timeの分以降を削除。
def processTime(df):
    df['hour'] = df['time'].str[:2].astype(int)
    df.drop(['time'], axis=1, inplace=True)

    return df

train_df = processTime(train_df)
test_df = processTime(test_df)

# %%
# チーム名のダミー変数を作成。
def processTeam(df):
    lst_homestadium = []

    for item in list_clubinfo:
        team = item[0]
        # ホーム/アウェイチームのコードを列[home][away]にセットする。
        df.loc[(df['home'] == item[1]), 'home'] = team
        df.loc[(df['away'] == item[1]), 'away'] = team

        # アウェイチーム列の作成を作成し、フラグを立てる。
        a_code = 'a_' + team
        df[a_code] = 0
        df.loc[(df['away'] == team), a_code] = 1

        # ホームチーム×ホームスタジアムの列を作成
        for stadium in item[2]:
            lst_homestadium.append([team, stadium])

    df['tmpflg'] = 0
    for item in lst_homestadium:
        df[item[0] + '_' + item[1]] = 0
        df.loc[(df['home'] == item[0]) & (df['stadium'] == item[1]), item[0] + '_' + item[1]] = 1
        df.loc[(df['home'] == item[0]) & (df['stadium'] == item[1]), 'tmpflg'] = 1

    lst_heldAtOtherStadium = df[df['tmpflg'] == 0]['home'].to_list()
    for team in lst_heldAtOtherStadium:
        df[team + '_other'] = 0
        df.loc[(df['home'] == team) & (df['tmpflg'] == 0), team + '_other'] = 1

    df.drop(['tmpflg'], axis=1, inplace=True)

    return df

cc_df = pd.concat([train_df, test_df])
cc_df = processTeam(cc_df)

train_df = cc_df[cc_df['y'] > 0]
test_df = cc_df[cc_df['y'].isna()]
# %%
# ホームタウン間の距離を計測する。
# これは2点間の距離を計測する関数
def get_distance(startLon, startLat, toLon, toLat):
    q = Geod(ellps='WGS84')
    fa, ba, d = q.inv(startLon, startLat, toLon, toLat)

    return round(d * 0.001, 1)

# # ホーム/アウェイの両チームのホームタウン座標間の距離を測定して100km刻みで区分け
def processHometownDistance(df, listClub):
    df['homeLon'] = 0.000000
    df['homeLat'] = 0.000000
    df['awayLon'] = 0.000000
    df['awayLat'] = 0.000000
    for item in listClub:
        team = item[0]
        Lat = item[3]
        Lon = item[4]
        df.loc[(df['home'] == team), 'homeLat'] = Lat
        df.loc[(df['home'] == team), 'homeLon'] = Lon
        df.loc[(df['away'] == team), 'awayLat'] = Lat
        df.loc[(df['away'] == team), 'awayLon'] = Lon

    for index, row in df.iterrows():
        df.at[index, 'hometownDistance'] = int(get_distance(row['homeLon'], row['homeLat'], row['awayLon'], row['awayLat']) // 100)

    df.drop(['homeLat', 'homeLon', 'awayLat', 'awayLon'], axis=1, inplace=True)

    return df

train_df = processHometownDistance(train_df, list_clubinfo)
test_df = processHometownDistance(test_df, list_clubinfo)

# %%
# スタジアム名はもういらないので削除
def processStadium(df):
    df.drop(['stadium'], axis=1, inplace=True)

    return df

train_df = processStadium(train_df)
test_df = processStadium(test_df)

# %%
# いわゆる「ダービーマッチ」の対戦相手かを判断する
# ここでチーム名(ホーム/アウェイ)はいらなくなるので削除
def processRival(df, list_rival):
    df['isDerby'] = 0
    for index, row in df.iterrows():
        home = row['home']
        away = row['away']
        for item in list_rival:
            if (home in item) & (away in item):
                df.at[index, 'isDerby'] = int(1)

    df.drop(['home', 'away'], axis=1, inplace=True)
    return df

train_df = processRival(train_df, list_rival)
test_df = processRival(test_df, list_rival)

# %%
# 天気から「雨/雪が含まれているか」を取得
# weatherを削除
def processWeather(df):
    df['RainOrSnow'] = 0

    df.loc[((df['weather'].str.startswith('雨')) | df['weather'].str.startswith('雪')), 'RainOrSnow'] = int(1)
    df.drop(['weather'], axis=1, inplace=True)
    return df

train_df = processWeather(train_df)
test_df = processWeather(test_df)

# %%
# 気温は5度刻みの値に変換
def processTemperature(df):
    df['temperature_index'] = (df['temperature'] // 5).astype(int)
    df.drop(['temperature'], axis=1, inplace=True)

    return df

train_df = processTemperature(train_df)
test_df = processTemperature(test_df)

# %%
# 湿度は10%刻みの値に変換
def processHumidity(df):
    df['humidity_index'] = (df['humidity'].str[:2].astype(int) // 10).astype(int)
    df.drop(['humidity'], axis=1, inplace=True)
    return df

train_df = processHumidity(train_df)
test_df = processHumidity(test_df)

# %%
# テレビ放送は以下のとおり処理
# スカパー各種とｅ２は削除
# NHK総合とBS放送があれば、「全国放送あり」とする。
# NHK総合とBS放送以外の放送があれば、「地方局放送あり」とする。
# 当然「全国放送も地方局放送もあり」というケースもある。

def processTv(df):
    df['isNationalWide'] = 0
    df['isLocal'] = 0
    for index, row in df.iterrows():
        tv = row['tv']
        stations = tv.split('／')
        filtered = [station for station in stations if ('スカパー' not in station) and ('ｅ２' not in station)]
        nationalwide = [station for station in filtered if (station == 'ＮＨＫ総合') or ('ＢＳ' in station)]
        df.at[index, 'isNationalWide'] = (len(nationalwide) > 0) * 1
        local = [station for station in filtered if (station != 'ＮＨＫ総合') and ('ＢＳ' not in station)]
        df.at[index, 'isLocal'] = (len(local) > 0) * 1

    df.drop(['tv'],axis=1, inplace=True)
    return df

train_df = processTv(train_df)
test_df = processTv(test_df)

# %%
# 不要な列を削除
def deleteColumns(df):
    df.drop(['home_score', 'away_score'], axis=1, inplace=True)
    # df.drop(['isDerby'], axis=1, inplace=True)
    return df

train_df = deleteColumns(train_df)
test_df = deleteColumns(test_df)

# %%
# クラブによってはキャパシティの異なる複数のスタジアムで試合をすることもある。
# そのため、収容率(yratio)を目的変数に変更する。
X = train_df.drop(['id', 'y', 'yratio'],axis=1)
y = train_df['yratio']

X_test = test_df.drop(['id', 'y', 'yratio'],axis=1)

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(X))[0]

tr_x, va_x = X.iloc[tr_idx], X.iloc[va_idx]
tr_y, va_y = y.iloc[tr_idx], y.iloc[va_idx]

dtrain = xgb.DMatrix(tr_x, label=tr_y)
dvalid = xgb.DMatrix(va_x, label=va_y)
dtest = xgb.DMatrix(X_test)

params = {'max_depth':5, 'random_state':71}
num_round = 50

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(params, dtrain, num_round, evals=watchlist)

va_pred = model.predict(dvalid)
score = np.sqrt(mean_squared_error(va_y, va_pred))
print(f'rmse: {score:4f}')

pred = model.predict(dtest)

# %%
# reg = xgb.XGBRegressor()
# scores = []
# for tr_idx, va_idx in kf.split(X):
#     tr_x, va_x = X.iloc[tr_idx], X.iloc[va_idx]
#     tr_y, va_y = y.iloc[tr_idx], y.iloc[va_idx]
#     reg_cv = GridSearchCV(reg, {'eval_metric': ['rmse'], 'max_depth': [8], 'n_estimators': [100]}, verbose=1)
#     reg_cv.fit(tr_x, tr_y)
#     va_pred = reg_cv.predict(va_x)

#     score = np.sqrt(mean_squared_error(va_y, va_pred))
#     scores.append(score)

# print(f'rmse:{np.mean(scores):4f}')

# # %%

# reg.fit(X, y)

# # pred_train = reg.predict(X_train)
# pred_test = reg.predict(X_test)


# # %%
# # mean_squared_error(y_train, pred_train)
# # %%
# mean_squared_error(y_test, pred_test)



# %%
submission = pd.DataFrame({
    "id": test_df['id'],
    "capa": test_df['capa'],
    'yratio': pred
})

# %%
submission['y'] = submission['capa'] * submission['yratio']
submission.drop(['capa', 'yratio'], axis=1, inplace=True)
now = datetime.datetime.now()
submission.to_csv('output/' + 'attendanceOfJleague_' +
                now.strftime('%Y%m%d_%H%M%S') + '.csv', index=False, header=False)


# %%
