# %%
import pandas as pd
import numpy as np
from datetime import datetime
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

rank_champion = 1
rank_stay = 15 # J1の15位以上はJ1残留
rank_demote = rank_stay + 1
rank_promote = 2 # J2の2位以上は自動でJ1に昇格(J2の3位～6位の中から1チーム昇格できるけどここでは割愛)

num_matches = [42, 34] # 年間試合数(J1は34試合、J2は42試合。num_matches[stage]で参照する)

df_tmp = pd.merge(train, cond, on='id', how='left')
df = df_tmp[['year', 'stage', 'gameday', 'home', 'away', 'home_score', 'away_score']]

df.loc[df['home'] == 'ザスパ草津', 'home'] = 'ザスパクサツ群馬'
df.loc[df['away'] == 'ザスパ草津', 'away'] = 'ザスパクサツ群馬'

df['stage'] = df['stage'].map({'Ｊ１': 1, 'Ｊ２': 0}).astype(int)

# %%
# 0. dataframeのチームごとの「得点、失点、獲得勝ち点」列を追加し、
# 試合結果から対応する列に値をセット。
for item in list_clubinfo:
    team = item[0]
    # ホーム/アウェイチームのコードを列[home][away]にセットする。
    df.loc[(df['home'] == item[1]), 'home'] = team
    df.loc[(df['home'] == team), team + '_for'] = df['home_score'].astype(int)
    df.loc[(df['home'] == team), team + '_against'] = df['away_score'].astype(int)
    df.loc[(df['home'] == team) & (df['home_score'] > df['away_score']), team + '_point'] = pt_win
    df.loc[(df['home'] == team) & (df['home_score'] == df['away_score']), team + '_point'] = pt_draw
    df.loc[(df['home'] == team) & (df['home_score'] < df['away_score']), team + '_point'] = pt_lose


    df.loc[(df['away'] == item[1]), 'away'] = team
    df.loc[(df['away'] == team), team + '_for'] = df['away_score'].astype(int)
    df.loc[(df['away'] == team), team + '_against'] = df['home_score'].astype(int)
    df.loc[(df['away'] == team) & (df['away_score'] > df['home_score']), team + '_point'] = pt_win
    df.loc[(df['away'] == team) & (df['away_score'] == df['home_score']), team + '_point'] = pt_draw
    df.loc[(df['away'] == team) & (df['away_score'] < df['home_score']), team + '_point'] = pt_lose

df['gamedate'] = pd.to_datetime((df['year'].astype(str) + '/' + df['gameday']).str[:10])
df.drop(['gameday', 'home_score', 'away_score'], axis=1, inplace=True)
df = df.fillna(0)

# %%
# 1. 年/ディビジョン/週番号/チームごとの勝ち点、得失点差、得点計、失点計を集計
df_stas = pd.DataFrame(index=[], columns=[
    'year',
    'stage',
    'team',
    'week',
    'rank',
    'rank_value',       # 順位を決める要素(勝ち点、得失点差、総得点数)から算出
    'games',
    'points',
    'goal_difference',
    'goals_for',
    'goals_against',
    'maybeChampion',    # 次の試合で優勝が決まるかもしれない状態
    'mayPromote',       # 次の試合で昇格が決まるかもしれない状態
    'mayDemote'         # 次の試合で降格が決まるかもしれない状態
    ])
df_week = df_stas   # 週毎のStatsを格納するためのDataFrame

list_year = df['year'].unique().tolist()
list_stage = df['stage'].unique().tolist()
i = 1
for year in list_year:
    for stg in list_stage:
        list_club = df[(df['year'] == year) & (df['stage'] == stg)]['home'].unique().tolist()
        # シーズンの初日と最終日の週番号を取得する。
        # 例外はあるが基本的に試合は週1で行われるので、この期間の週毎に成績を集計する。
        min_date = df[(df['year'] == year) & (df['stage'] == stg)].min()['gamedate']
        min_week = min_date.isocalendar()[1]
        max_date = df[(df['year'] == year) & (df['stage'] == stg)].max()['gamedate']
        max_week = max_date.isocalendar()[1]

        # 週番号は、その年の最初の木曜日が含まれる週からカウントする。
        # 1月1日が金～日曜日の年で元日の週を1とカウントするためには、週番号を-1する必要がある。
        if datetime(year, 1, 1).isocalendar()[1] > 1:
            min_week += 1
            max_week += 1

        for week in range(min_week, max_week):
            df_week.drop(df_week.index, inplace=True)
            for team in list_club:
                # 各週の日曜日をlastSundayとして取得
                lastSunday = datetime.strptime("{} {} {}".format(year, week-1,0), "%Y %W %w")
                gameCnt = df[(df['year'] == year) & (df['stage'] == stg) & ((df['home'] == team) | (df['away'] == team)) & (
                    df['gamedate'] <= lastSunday)]['home'].count()
                sum_df = df[(df['year'] == year) & (df['stage'] == stg) & ((df['home'] == team) | (df['away'] == team)) & (
                    df['gamedate'] <= lastSunday)].sum()
                sum_point = sum_df[team + '_point']
                sum_for = sum_df[team + '_for']
                sum_against = sum_df[team + '_against']
                goal_difference = sum_for - sum_against

                # print('{0}:{1}:{2}:{3}:{4}:{5}:{6}:{7}'.format(i, year, stg, team, lastSunday, sum_point, goal_difference, sum_for, sum_against))
                # 順位は「勝点、得失点差、総得点」の順に決まる
                df_week = df_week.append({
                    'year': year,
                    'stage': stg,
                    'team': team,
                    'week': week + 1,
                    'rank': 0,
                    'rank_value': (sum_point + 1) * 10000 + goal_difference * 100 + sum_for,
                    'games': gameCnt,
                    'points': sum_point,
                    'goal_difference': goal_difference,
                    'goals_for': sum_for,
                    'goals_against': sum_against
                }, ignore_index=True)
                i += 1

            df_week['rank'] = df_week['rank_value'].rank(method='first', ascending=False)
            pt_1st = df_week[df_week['rank'] == rank_champion].sum()['points']
            pt_2nd = df_week[df_week['rank'] == rank_champion + 1].sum()['points']
            pt_3rd = df_week[df_week['rank'] == rank_promote + 1].sum()['points']
            pt_stay = df_week[df_week['rank'] == rank_stay].sum()['points']
            pt_demote = df_week[df_week['rank'] == rank_demote].sum()['points']
            # print('{0}:{1}:{2}:{3}'.format(week, pt_1st, pt_2nd, rest_matches))

            # 【優勝がかかった状態の条件】J1、J2共通
            rest_matches = num_matches[stg] - df_week[df_week['rank'] == rank_champion + 1].sum()['games']
            # 優勝が確定する条件は「1位のチームが残り試合を全敗した場合の勝ち点(=今の勝ち点) > 2位のチームが残り試合を全勝した場合の勝ち点」
            isFixChampion = (pt_1st > pt_2nd + rest_matches * pt_win)
            if not isFixChampion:
                # 1. 今節の結果次第では、2位のチームが次節以降全勝しても1位のチームの勝ち点に届かない場合
                df_week.loc[
                    (df_week['rank'] == rank_champion) &
                    (
                        (pt_1st - pt_2nd - rest_matches * pt_win > -6) |
                        (pt_1st - pt_2nd - rest_matches * pt_win > -5) |
                        (pt_1st - pt_2nd - rest_matches * pt_win > -4) |
                        (pt_1st - pt_2nd - rest_matches * pt_win > -3)
                    )
                    ,'maybeChampion'] = 1
                # 2. 残り1試合の時点で優勝が確定していない場合は、1位、もしくは1位との勝ち点差が3以下のチームに優勝のチャンスあり
                if(rest_matches == 1):
                    df_week.loc[df_week['rank'] == rank_champion, 'maybeChampion'] = 1
                    df_week.loc[(df_week['rank'] != rank_champion) & (df_week['points'] >= pt_1st - pt_win), 'maybeChampion'] = 1

            if stg == 1:
                rest_matches = num_matches[stg] - df_week[df_week['rank'] == rank_demote].sum()['games']
                # 【降格するかもしれない状態の条件】J1のみ
                # 1. 前週時点で残留も降格も確定していないが、次節の結果次第では残り全勝しても15位の勝ち点に届かない
                df_week.loc[
                    (df_week['points'] <= pt_demote + rest_matches * pt_win) &
                    (df_week['points'] + (num_matches[stg] - df_week['games']) * pt_win >= pt_stay) &
                    (
                        (pt_stay - df_week['points'] - (num_matches[stg] - df_week['games']) * pt_win > -6) |
                        (pt_stay - df_week['points'] - (num_matches[stg] - df_week['games']) * pt_win > -5) |
                        (pt_stay - df_week['points'] - (num_matches[stg] - df_week['games']) * pt_win > -4) |
                        (pt_stay - df_week['points'] - (num_matches[stg] - df_week['games']) * pt_win > -3)
                    )
                    , 'mayDemote'] = 1
            else:
                # 【自動昇格がかかった状態の条件】J2のみ
                # 1. 前週時点で2位以上が確定していないが、次節の結果次第では3位のチームが全勝しても逆転できない
                rest_matches = num_matches[stg] - df_week[df_week['rank'] == rank_promote + 1].sum()['games']
                df_week.loc[
                    (df_week['points'] <= pt_3rd + rest_matches * pt_win) &
                    (
                        (df_week['points'] - pt_3rd - rest_matches * pt_win > -6) |
                        (df_week['points'] - pt_3rd - rest_matches * pt_win > -5) |
                        (df_week['points'] - pt_3rd - rest_matches * pt_win > -4) |
                        (df_week['points'] - pt_3rd - rest_matches * pt_win > -3)
                    )
                    , 'mayPromote'] = 1
            df_stas = df_stas.append(df_week)

df_stas = df_stas.fillna(0)
df_stas.drop(['rank_value'], axis=1, inplace=True)
df_stas['rank'] = df_stas['rank'].astype(int)
df_stas['points'] = df_stas['points'].astype(int)
df_stas['goal_difference'] = df_stas['goal_difference'].astype(int)
df_stas['goals_for'] = df_stas['goals_for'].astype(int)
df_stas['goals_against'] = df_stas['goals_against'].astype(int)
df_stas['maybeChampion'] = df_stas['maybeChampion'].astype(int)
df_stas['mayPromote'] = df_stas['mayPromote'].astype(int)
df_stas['mayDemote'] = df_stas['mayDemote'].astype(int)


# %%
df_stas.to_csv('input/standings.csv', header=True, index=False)




# %%
