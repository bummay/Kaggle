# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import fbeta_score
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import GridSearchCV

# %%
inputDir = 'PredictFuturePrice/input/'
items = pd.read_csv(inputDir + 'items.csv')
item_categories = pd.read_csv(inputDir + 'item_categories.csv')
shops = pd.read_csv(inputDir + 'shops.csv')
sales_train = pd.read_csv(inputDir + 'sales_train.csv')
test = pd.read_csv(inputDir + 'test.csv')

# # %%
# items.info()
# items.head()

# # %%
# item_categories.info()
# item_categories.head()

# # %%
# shops.info()
# shops.head()

# %%
# item_categoriesの名前から大分類を作成
item_categories['major_name'] = item_categories['item_category_name'].map(lambda x: x.split(' - ')[0])
item_categories['major_name'].value_counts()

item_categories.loc[
    (item_categories['major_name'] == 'Чистые носители (шпиль)') |
    (item_categories['major_name'] == 'Чистые носители (штучные)'),
    'major_name'] = 'Чистые носители'


item_categories.loc[
    (item_categories['major_name'] == 'Игры') |
    (item_categories['major_name'] == 'Игры MAC') |
    (item_categories['major_name'] == 'Игры PC') |
    (item_categories['major_name'] == 'Игры Android'),
    'major_name'] = 'Игры'


item_categories.loc[
    (item_categories['major_name'] == 'Карты оплаты') |
    (item_categories['major_name'] == 'Карты оплаты (Кино, Музыка, Игры)'),
    'major_name'] = 'Карты оплаты'

# item_categories['major_name'].value_counts()

# %%
# shopsの名前から都市名を作成
shops['city_name'] = shops['shop_name'].map(lambda x: x.split(' ')[0])
shops['city_name'].value_counts()

shops.loc[
    (shops['city_name'] == '!Якутск'),
    'city_name'] = 'Якутск'

# shops['city_name'].value_counts()

# %%
# 都市名から時間帯を作成
shops['time_zone'] = 3

shops.loc[
    (shops['city_name'] == 'Самара') |
    (shops['city_name'] == 'Волжский'),
    'time_zone'] = 4

shops.loc[
    (shops['city_name'] == 'Тюмень') |
    (shops['city_name'] == 'Сургут') |
    (shops['city_name'] == 'Уфа'),
    'time_zone'] = 5

shops.loc[
    (shops['city_name'] == 'Омск'),
    'time_zone'] = 6

shops.loc[
    (shops['city_name'] == 'Красноярск') |
    (shops['city_name'] == 'Новосибирск') |
    (shops['city_name'] == 'Томск'),
    'time_zone'] = 7

shops.loc[
    (shops['city_name'] == 'Якутск'),
    'time_zone'] = 9

shops.loc[
    (shops['city_name'] == 'Выездная') |
    (shops['city_name'] == 'Интернет-магазин') |
    (shops['city_name'] == 'Цифровой'),
    'time_zone'] = 0

# shops['time_zone'].value_counts()

# %%
# sales_trainの商品単価と売上数から売上金額を作成
sales_train['item_sales_day'] = sales_train['item_price'] * sales_train['item_cnt_day']

# %%
# 商品数と売上金額を月ごとに集計
month_shop_item_cnt = sales_train[
    ['date_block_num', 'shop_id', 'item_id', 'item_cnt_day']
].groupby(['date_block_num', 'shop_id', 'item_id'],
    as_index=False).sum().rename(columns={'item_cnt_day':'month_shop_item_cnt'})

month_shop_item_sales = sales_train[
    ['date_block_num', 'shop_id', 'item_id', 'item_sales_day']
].groupby(['date_block_num', 'shop_id', 'item_id'],
          as_index=False).sum().rename(columns={'item_sales_day': 'month_shop_item_sales'})

# %%
# 35ヶ月*shop_id*item_idの組み合わせのDataframeを作成
# (全てのアイテムが毎月売れているわけではないため)
train_full_comb = pd.DataFrame()
for i in range(35):
    mid = test[['shop_id', 'item_id']]
    mid['date_block_num'] = i
    train_full_comb = pd.concat([train_full_comb, mid], axis=0)

# train_full_combに各データを結合
train = pd.merge(
    train_full_comb,
    month_shop_item_cnt,
    on=['date_block_num', 'shop_id', 'item_id'],
    how='left'
)

train = pd.merge(
    train,
    month_shop_item_sales,
    on=['date_block_num', 'shop_id', 'item_id'],
    how='left'
)

train = pd.merge(
    train,
    items[['item_id', 'item_category_id']],
    on='item_id',
    how='left'
)

train = pd.merge(
    train,
    item_categories[['item_category_id', 'major_name']],
    on='item_category_id',
    how='left'
)

train = pd.merge(
    train,
    shops[['shop_id', 'city_name', 'time_zone']],
    on='shop_id',
    how='left'
)
# %%
# データの可視化
plt_df = train.groupby(['date_block_num'], as_index=False).sum()
plt.figure(figsize=(20,10))
sns.lineplot(x = 'date_block_num', y = 'month_shop_item_cnt', data = plt_df)
plt.title('Monthly item counts')


# %%
# 可視化その2
plt_df = train.groupby(['date_block_num', 'city_name'], as_index=False).sum()
plt.figure(figsize=(20, 10))
sns.lineplot(x='date_block_num', y='month_shop_item_cnt', hue='city_name', data=plt_df)
plt.title('Monthly item counts by city_name')

# %%
# 可視化その3
plt_df = train.groupby(['date_block_num', 'time_zone'], as_index=False).sum()
plt.figure(figsize=(20, 10))
sns.lineplot(x='date_block_num', y='month_shop_item_cnt',
             hue='time_zone', data=plt_df)
plt.title('Monthly item counts by time_zone')

# %%
# 月次売上数をClip
train['month_shop_item_cnt'] = train['month_shop_item_cnt'].clip(0, 20)

# ラグ生成対象の列とラグの間隔を定義
lag_col_list = ['month_shop_item_cnt', 'month_shop_item_sales']
lag_num_list = [1, 2, 3]

# shop_id、item_id、date_block_numの順にソート
train = train.sort_values(
    ['shop_id', 'item_id', 'date_block_num'],
    ascending = [True, True, True]
).reset_index(drop=True)

# ラグ特徴量の生成
for lag_col in lag_col_list:
    for lag in lag_num_list:
        set_col_name = lag_col + '_' + str(lag)
        df_lag = train[['shop_id', 'item_id', 'date_block_num', lag_col]].sort_values(
            ['shop_id', 'item_id', 'date_block_num'],
            ascending = [True, True, True]
        ).reset_index(drop=True).shift(lag).rename(columns={lag_col: set_col_name})
        train = pd.concat([train, df_lag[set_col_name]], axis=1)

# 欠損を0埋め
train = train.fillna(0)

# %%
train_ = train[(train['date_block_num'] <= 33) & (train['date_block_num'] >= 12)].reset_index(drop=True)
test_ = train[train['date_block_num'] == 34].reset_index(drop=True)

# %%
# モデルに入力する特徴量とターゲット変数に分割
train_X = train_.drop(columns=['date_block_num','month_shop_item_cnt', 'month_shop_item_sales'])
train_y = train_['month_shop_item_cnt']
test_X = test_.drop(columns=['date_block_num','month_shop_item_cnt', 'month_shop_item_sales'])

# %%
obj_col_list = ['major_name', 'city_name']
for obj_col in obj_col_list:
    le = LabelEncoder()
    train_X[obj_col] = pd.DataFrame({obj_col:le.fit_transform(train_X[obj_col])})
    test_X[obj_col] = pd.DataFrame({obj_col:le.fit_transform(test_X[obj_col])})


# # %%
# # パフォーマンス・チューニング
# params = {
#     'n_estimators':[80,100,120],
#     'max_depth':[None, 5]
# }

# scorer = make_scorer(fbeta_score, beta=0.5)
# clf = GridSearchCV(RandomForestRegressor(), params, cv=5, n_jobs=-1, verbose=5)
# clf_fit = clf.fit(train_X, train_y)
# predictor = clf_fit.best_estimator_

# %%
rfr = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                            max_depth=5, max_features='auto', max_leaf_nodes=None,
                            max_samples=None, min_impurity_decrease=0.0,
                            min_impurity_split=None, min_samples_leaf=1,
                            min_samples_split=2, min_weight_fraction_leaf=0.0,
                            n_estimators=100, n_jobs=None, oob_score=False,
                            random_state=None, verbose=5, warm_start=False)
rfr.fit(train_X,train_y)


# %%
rmse = np.sqrt(
        np.mean(
            np.square(
                np.array(
                    np.array(train_y) - rfr.predict(train_X)
                )
            )
        )
    )
rmse

# %%
# predict
print('predict started : ' + datetime.datetime.now().strftime('%H:%M:%S'))
test_y = rfr.predict(test_X)
#test_y = predictor.predict(test_X)
print('predict finished : ' + datetime.datetime.now().strftime('%H:%M:%S'))
test_X['item_cnt_month'] = test_y
submission = pd.merge(
    test,
    test_X[['shop_id', 'item_id', 'item_cnt_month']],
    on=['shop_id', 'item_id'],
    how='left'
)

# %%
outputDir = 'PredictFuturePrice/output/'
now = datetime.datetime.now()
submission[['ID', 'item_cnt_month']].to_csv(outputDir + 'predictFutureSales' +
                   now.strftime('%Y%m%d_%H%M%S') + '.csv', index=False, header=True)

