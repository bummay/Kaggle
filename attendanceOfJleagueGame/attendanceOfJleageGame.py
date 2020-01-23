# %%
import pandas as pd
import xgboost as xgb

import pickle

# 日付の処理で使う
import datetime
import jpholiday
import pandas.tseries.offsets as offsets

import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from pyproj import Geod

# %%
# 諸情報の読み込み
with open('list_clubinfo.pkl', mode='rb') as f:
    list_clubinfo = pickle.load(f)

# %%
# trainデータにaddデータをappend
#
inputDir = 'input/'

train_df = pd.read_csv(inputDir + 'train.csv')
train_add_df = pd.read_csv(inputDir + 'train_add.csv')
train_df = train_df.append(train_add_df)
train_df = train_df.sort_values('id')
del train_add_df

test_df = pd.read_csv(inputDir + 'test.csv')

# cond_dfは以下の項目のみ残す。
#   ホームチームスコア
#   アウェイチームスコア
#   天候
#   気温
#   湿度
cond_df = pd.read_csv(inputDir + 'condition.csv')
cond_add_df = pd.read_csv(inputDir + 'condition_add.csv')
cond_df = cond_df.append(cond_add_df).sort_values('id')
cond_df = cond_df.drop([
                    'referee',
                    'home_team', 'home_01', 'home_02', 'home_03', 'home_04', 'home_05',
                    'home_06', 'home_07', 'home_08', 'home_09', 'home_10', 'home_11',
                    'away_team', 'away_01', 'away_02', 'away_03', 'away_04', 'away_05',
                    'away_06', 'away_07', 'away_08', 'away_09', 'away_10', 'away_11'
                    ],axis=1)
del cond_add_df

# stadium_dfは住所を削除
stadium_df = pd.read_csv(inputDir + 'stadium.csv')
stadium_df = stadium_df.rename(columns={'name':'stadium'})
stadium_df = stadium_df.drop(['address'],axis=1)
stadium_df['coveredwithroof'] = stadium_df['coveredwithroof'].astype(int)

# train/testのそれぞれに、cond_dfとstadium_dfを結合
def mergeDf(df):
    df = pd.merge(df, cond_df, on='id', how='left')
    df = pd.merge(df, stadium_df, on='stadium')

    return df

train_df = mergeDf(train_df)
test_df = mergeDf(test_df)

# 来場者数とスタジアムの収容人数から収容率を取得
train_df['yratio'] = ((train_df['y'] / train_df['capa']) ).round(2)

# %%
# stageをコードに変換
def processStage(df):
    df['stage'] = df['stage'].map(
        {'Ｊ１': 1, 'Ｊ２': 0}).astype(int)

    return df

train_df = processStage(train_df)
test_df = processStage(test_df)


# %%
# gamedayから「当日が休日か」「翌日が休日か」を取得
def processGameday(df):
    df['month'] = df['gameday'].str[:2].astype(int)
    df['gamedate'] = pd.to_datetime((df['year'].astype(str) + '/' + df['gameday']).str[:10])

    df['weekday'] = df['gamedate'].dt.weekday

    df['isHoliday'] = ((df['gamedate'].map(
        jpholiday.is_holiday).astype(int) == 1) | (df['weekday'] > 4)).astype(int)

    df['nextday'] = df['gamedate'] + offsets.Day()
    df['nextIsHoliday'] = ((df['nextday'].map(
        jpholiday.is_holiday).astype(int) == 1) | (df['nextday'].dt.weekday > 4)).astype(int)

    df = df.drop(['gamedate', 'nextday', 'weekday'], axis=1)
    return df

train_df = processGameday(train_df)
test_df = processGameday(test_df)

# %%
# timeの分以降を削除。18時以降の開始はナイトゲーム扱いとして列を追加
# （翌日が休日ではない日のナイトゲームは客足が鈍るかもしれない）
def processTime(df):
    df['hour'] = df['time'].str[:2].astype(int)

    df['isNightGame'] = (df['hour'] > 17) * 1

    return df

train_df = processTime(train_df)
test_df = processTime(test_df)

# %%
# チーム名のダミー変数を作成。
def processTeam(df):
    for item in list_clubinfo:
        h_code = 'h_' + item[0]
        a_code = 'a_' + item[0]
        df[h_code] = 0
        df[a_code] = 0

        for team in item[1]:
            df.loc[
                (df['home'] == team), h_code] = 1
            df.loc[
                (df['away'] == team), a_code] = 1

    return df

train_df = processTeam(train_df)
test_df = processTeam(test_df)

# %%
# スタジアムの情報から「ホームスタジアム開催か」と「ホームスタジアムではないが都道府県内のスタジアムで開催か」を取得
def processHomeClass(df, listClub):
    df['heldInHomeStadium'] = 0
    df['heldInHomeTown'] = 0
    for item in listClub:
        team = 'h_' + item[0]
        for stadium in item[2]:
            df.loc[
                ((df[team] == 1) & (df['stadium'] == stadium)), 'heldInHomeStadium'] = 1
        for subStadium in item[3]:
            df.loc[
                ((df[team] == 1) & (df['stadium'] == subStadium)), 'heldInHomeTown'] = 1

def processStadium(df):

    processHomeClass(df, list_clubinfo)
    return df

train_df = processStadium(train_df)
test_df = processStadium(test_df)

# %%
# ホームタウン間の距離を計測する。
# これは2点間の距離を計測する関数
def get_distance(startLon, startLat, toLon, toLat):
    q = Geod(ellps='WGS84')
    fa, ba, d = q.inv(startLon, startLat, toLon, toLat)

    return round(d * 0.001, 1)

# ホーム/アウェイの両チームのホームタウン座標間の距離を測定
def processHometownDistance(df, listClub):
    df['homeLon'] = 0.000000
    df['homeLat'] = 0.000000
    df['awayLon'] = 0.000000
    df['awayLat'] = 0.000000
    for item in listClub:
        team = item[0]
        Lat = item[4]
        Lon = item[5]
        df.loc[(df['h_' + team] == 1), 'homeLat'] = Lat
        df.loc[(df['h_' + team] == 1), 'homeLon'] = Lon
        df.loc[(df['a_' + team] == 1), 'awayLat'] = Lat
        df.loc[(df['a_' + team] == 1), 'awayLon'] = Lon

    for index, row in df.iterrows():
        df.at[index, 'hometownDistance'] = int(get_distance(row['homeLon'], row['homeLat'], row['awayLon'], row['awayLat']) // 50)

    df = df.drop(['homeLat', 'homeLon', 'awayLat', 'awayLon'], axis=1)
    return df

train_df = processHometownDistance(train_df, list_clubinfo)
test_df = processHometownDistance(test_df, list_clubinfo)


# %%
# 天気から「雨/雪が含まれているか」と「屋内か」を取得
def processWeather(df):
    df['rainOrSnow'] = 0
    df['isIndoor'] = 0

    df.loc[((df['weather'].str.startswith('雨')) | df['weather'].str.startswith('雪')), 'rainOrSnow'] = int(2)
    df.loc[(
            (df['rainOrSnow'] == 0) &
            (
                (df['weather'].str.contains('雨')) |
                (df['weather'].str.contains('雪'))
            )
        ), 'rainOrSnow'] = 1

    df.loc[(df['weather'] == '屋内'), 'isIndoor'] = int(1)

    return df

train_df = processWeather(train_df)
test_df = processWeather(test_df)

# %%
# 気温は5度刻みの値に変換
def processTemperature(df):
    df['temperature'] = (df['temperature'] / 5).round().astype(int)

    return df

train_df = processTemperature(train_df)
test_df = processTemperature(test_df)

# %%
# 湿度は10%刻みの値に変換
def processHumidity(df):
    df['humidity'] = (df['humidity'].str[:2].astype(int) / 10).round().astype(int)

    return df

train_df = processHumidity(train_df)
test_df = processHumidity(test_df)

# %%
# 年/チームごとにホーム開幕戦のフラグを追加
# def processOpeningHomegame(df, listClub):
#     df['isOpeningHomegame'] = 0
#     for item in listClub:
#         team = 'h_' + item[0]
#         tmp_df = df.loc[(df[team] == 1)].groupby('year').min().reset_index()
#         tmp_df = tmp_df[['year', 'match']]
#         for index, row in tmp_df.iterrows():
#             year, match = row[0], row[1]
#             df.loc[
#                 ((df[team] == 1) & (df['year'] == year) & (df['match'] == match)), 'isOpeningHomegame'] = 1
#     return df

# train_df = processOpeningHomegame(train_df, list_clubinfo)
# # テスト用データは2014年後半戦の情報のみ→ホーム開幕戦の情報は存在しない。
# test_df['isOpeningHomegame'] = 0

# # %%
# # 年/チームごとにホーム最終戦のフラグを追加
# def processFinalHomegame(df, listClub):
#     df['isFinalHomegame'] = 0
#     for item in listClub:
#         team = 'h_' + item[0]
#         tmp_df = df.loc[(df[team] == 1)].groupby('year').max().reset_index()
#         tmp_df = tmp_df[['year', 'match']]
#         for index, row in tmp_df.iterrows():
#             year, match = row[0], row[1]
#             # print(team)
#             # print(year)
#             # print(match)
#             # print('---')
#             df.loc[
#                 ((df[team] == 1) & (df['year'] == year) & (df['match'] == match)), 'isFinalHomegame'] = 1
#     return df

# train_df = processFinalHomegame(train_df, list_clubinfo)
# test_df = processFinalHomegame(test_df, list_clubinfo)
# # 学習用データの2014年データは前半戦の情報のみ→ホーム最終戦の情報は存在しない。
# train_df.loc[(train_df['year'] == 2014), 'isFinalHomegame'] = 0

# %%
# 2014年の無観客試合(浦和vs清水)を除外
train_df = train_df.drop(train_df.index[422])

# %%
# 不要な列を削除
def dropColumns(df, listClub):
    df = df.drop(
        [
            'match',
            'home',
            'away',
            'weather',
            'match',
            'tv',
            'stadium',
            'time',
            'home_score',
            'away_score',
            'gameday'
        ], axis = 1
    )

    for item in listClub:
        team = 'a_' + item[0]
        df = df.drop([team], axis=1)
    return df

train_df = dropColumns(train_df, list_clubinfo)
test_df = dropColumns(test_df, list_clubinfo)

# %%
# クラブによってはキャパシティの異なる複数のスタジアムで試合をすることもある。
# そのため、収容率(yratio)を目的変数に変更する。
X = train_df.drop(['id', 'y', 'yratio'],axis=1)
y = train_df['yratio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

reg = xgb.XGBRegressor()

reg_cv = GridSearchCV(reg, {'eval_metric': ['rmse'], 'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]}, verbose=1)
reg_cv.fit(X_train, y_train)

reg = xgb.XGBRFRegressor(**reg_cv.best_params_)
reg.fit(X_train, y_train)

pred_train = reg.predict(X_train)
pred_test = reg.predict(X_test)
# %%
mean_squared_error(y_train, pred_train)
# %%
mean_squared_error(y_test, pred_test)

# %%
importances = pd.Series(reg.feature_importances_, index=X.columns)
importances = importances.sort_values()
importances.plot(kind="barh", figsize=(9,45))
# plt.title("imporance in the xgboost Model")
# plt.show()
# %%
pred_X = test_df.drop(['id'], axis=1)
pred_y = reg.predict(pred_X)
submission = pd.DataFrame({
    "id": test_df['id'],
    "capa": test_df['capa'],
    'yratio': pred_y
})

# %%
submission['y'] = submission['capa'] * submission['yratio']
submission = submission.drop(['capa', 'yratio'], axis=1)
now = datetime.datetime.now()
submission.to_csv('output/' + 'attendanceOfJleague_' +
                now.strftime('%Y%m%d_%H%M%S') + '.csv', index=False, header=False)


# %%
