# %%
# もろもろ準備
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from category_encoders import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_validate
from IPython.core.interactiveshell import InteractiveShell
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
sns.set_style(style='dark')
InteractiveShell.ast_node_interactivity = "all"


# %%
# データの読み込み
inputDir = 'input/'
users = pd.read_csv(inputDir + 'user_table.csv')
historical_transaction = pd.read_csv(
    inputDir + 'historical_transactions_AtoI.csv')
X_transaction = pd.read_csv(inputDir + 'historical_transactions_X.csv')


# %%
# データの中身確認
users.head(3)
historical_transaction.head(3)
X_transaction.head(3)


# %%
# 重複データを削除
names = ['users', 'historical_transaction', 'X_transaction']
dfs = [users, historical_transaction, X_transaction]

for name, df in zip(names, dfs):
    print(f"{name} - shape before: {df.shape}")
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"{name} - shape after: {df.shape}")

# %%
# ITEM A~Iの合計購入数量を算出してusersに結合
pv_ht = pd.pivot_table(historical_transaction, index='user_id', columns='item', values='num_purchase', aggfunc='sum').reset_index()
users_item_num = pd.merge(users, pv_ht, how='left', on='user_id')
# item A~I を列に変換しただけだと、未購入アイテムの購入数量がNaN。
# そこで、NaNを0に変換する処理が必要になる。
users_item_num[historical_transaction.item.unique()] = users_item_num[historical_transaction.item.unique()].fillna(0)

# %%
# item X の合計数量を算出
# item X を列に変換しただけだと以下略
pv_pt = pd.pivot_table(X_transaction, index='user_id', columns='item', values='num_purchase', aggfunc='sum').reset_index()
users_item_num = pd.merge(users_item_num, pv_pt, how='left', on='user_id')
users_item_num[X_transaction.item.unique()] = users_item_num[X_transaction.item.unique()].fillna(0)

# %%
# item Xの値を購入有無(0/1)に変換
users_item_num['X'] = users_item_num['X'].apply(lambda x: 1 if x>= 1 else 0).values
users_item_num.head(3)


# %%
# dataframe[df]を列[col]でgroupbyして[agg_dict]を取得する
def grouping(df, cols, agg_dict, prefix=''):
    group_df = df.groupby(cols).agg(agg_dict)
    group_df.columns = [prefix + c[0] + '_' + c[1] for c in list(group_df.columns)]
    group_df.reset_index(inplace = True)

    return group_df

# 関数[grouping]で取得したプロットデータをグラフに描画
def plot_target_stats(df, col, agg_dict, plot_config):
    plt_data = grouping(df, col, agg_dict, prefix='')

    fig, ax1 = plt.subplots(figsize = (15, 7))

    # グラフは2軸
    ax2 = ax1.twinx()
    # 軸1は件数を棒グラフで表示。軸2は平均をプロット
    ax1.bar(plt_data[col], plt_data['X_count'], label='X_count', color='skyblue', **plot_config['bar'])
    ax2.plot(plt_data[col], plt_data['X_mean'], label='X_mean',color='red', marker='.', markersize=10)

    # 凡例の設定
    h1, label1 = ax1.get_legend_handles_labels()
    h2, label2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, label1 + label2, loc=2, borderaxespad=0.)

    # X軸の目盛り設定
    ax1.set_xticks(plt_data[col])
    ax1.set_xticklabels(plt_data[col],rotation=-90, fontsize=10)

    # グラフタイトルの設定
    ax1.set_title(f"Relationship between {col}, X_count, and X_mean", fontsize=14)

    # x軸のラベル設定
    ax1.set_xlabel(f"{col}")
    # x軸と軸1のラベルサイズ
    ax1.tick_params(labelsize=12)

    # y軸1/2のラベル設定
    ax1.set_ylabel('X_count')
    ax2.set_ylabel('X_mean')

    # y軸1/2の上限値設定
    ax1.set_ylim([0, plt_data['X_count'].max() * 1.2])
    ax2.set_ylim([0, plt_data['X_mean'].max() * 1.1])


for col in ['age', 'country', 'num_family', 'married', 'job', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
    agg_dict = {
        'X': ['count', 'mean']
    }
    plot_config = {'bar': {'width': 0.8}}
    plot_target_stats(users_item_num, col, agg_dict, plot_config)


# %%
# incomeは100刻みの列に変換して件数と平均をグラフに表示。
tmp_df = users_item_num.copy()
tmp_df['income_100'] = (tmp_df['income'] // 100 * 100)

plot_config = {'bar': {'width': 60}}
plot_target_stats(tmp_df, 'income_100', agg_dict, plot_config)

# %%
# 欠損レコード数を調査
names = ['users', 'historical_transaction', 'X_transaciton']
dfs = [users, historical_transaction, X_transaction]

for name, df in zip(names, dfs):
    print('============================')
    print(f'{name} : null value - counts')
    print('============================')
    print(df.isnull().sum())

# → age, country, num_family, married, income に欠損値あり

# %%
# 欠損値を持つ列の分布(数値編)
missing_num_columns = [
    'age',
    'num_family',
    'income'
]

for col in missing_num_columns:
    plt.figure(figsize=(12, 5))
    plt.hist(users[col])
    plt.xlabel(f"{col}")
    plt.ylabel("X_count")
    plt.title(f'{col} - distribution')
    plt.show()

# %%
# 欠損値を持つ列の分布(カテゴリ編)
missing_cat_columns = [
    'country',
    'married'
]

for col in missing_cat_columns:
    plt.figure(figsize=(12, 5))
    sns.countplot(x=col, data=users, color='salmon')
    plt.title(f'{col} - distribution')
    plt.show()

# %%
# 数値変数の欠損を平均値で補完
for col in missing_num_columns:
    column_mean = users_item_num[col].mean()
    users_item_num[col].fillna(column_mean, inplace=True)


# カテゴリ変数の欠損を最頻値で補完
for col in missing_cat_columns:
    column_mode = users_item_num[col].mode()[0]
    users_item_num[col].fillna(column_mode, inplace=True)


# %%
# country と job について、incomeとageの統計量を特徴に加える。
plt.figure(figsize=(15, 7))
sns.violinplot(x='job', y='income', data=users_item_num, split=True)

plt.title('Distribution of income (job)', fontsize=14)
plt.xticks(rotation=-90, fontsize=12)

value_agg = {
    'age' : ['max', 'min', 'mean', 'std'],
    'income': ['max', 'min', 'mean', 'std'],
}

for key in ['country', 'job']:
    feature_df = grouping(users_item_num, key, value_agg, prefix=key + '_')
    users_item_num = pd.merge(users_item_num, feature_df, how='left', on=key)

# %%
for col in ['country', 'job']:
    for value in ['age', 'income']:
        users_item_num[f'{value}_diff_mean_{col}'] = users_item_num[col + '_' + value + '_mean'] - users_item_num[value]
        users_item_num[f'{value}_diff_max_min_{col}'] = users_item_num[col +"_" + value + "_max"] - users_item_num[col +"_" + value + "_min"]

users_item_num['income_per_num_family'] = users_item_num['income'] / (users_item_num['num_family'] + 1)


# %%
# users_item_numの列から説明変数の列を取得する。条件は以下のとおり
# 1. 「user_id」でも「X」でもない。
# 2. データ型がObjectではない。
# 3. feature_cat_columnsにも含まれていない。
def choose_feature_column(df, feature_cat_columns):
    feature_num_columns = [col for col in df.columns
                                if col not in ['user_id', 'X']
                                and df[col].dtype != 'O'
                                and col not in feature_cat_columns
    ]
    return feature_num_columns

feature_cat_columns = ['country', 'job', 'married']
feature_num_columns = choose_feature_column(users_item_num, feature_cat_columns)
target_column = 'X'

# %%
users_item_num.drop(['user_id', 'name', 'nickname', 'profile'], axis=1, inplace=True)
feature_cat_columns = ['country', 'job', 'married']
target_column = 'X'


# %%
# データセットの作成
X = users_item_num[feature_num_columns]
y = users_item_num[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)


# %%
# Hold-out
X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)


# %%
# K fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for train_index, val_index in kf.split(X_train):
    X_train_f, X_val_f = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_f, y_val_f = y_train.iloc[train_index], y_train.iloc[val_index]

    scaler = StandardScaler()
    X_train_f_scale = scaler.fit_transform(X_train_f)
    X_val_f_scale = scaler.fit_transform(X_val_f)

    model = LogisticRegression(class_weight='balanced')
    _ = model.fit(X_train_f_scale, y_train_f)

    y_pred = model.predict(X_val_f_scale)

    acc_fold = accuracy_score(y_true=y_val_f, y_pred=y_pred)
    accuracies.append(acc_fold)

acc_mean = np.mean(accuracies)
print(f'Mean accuracy: {acc_mean:.1%}')

# %%
# stratified K-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# skf = KFold(n_splits=5, shuffle=True, random_state=42)

for iteration, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
    X_train_f, X_val_f = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_f, y_val_f = y_train.iloc[train_index], y_train.iloc[val_index]

    y_train_positive_ratio = y_train_f.mean()
    y_val_positive_ratio = y_val_f.mean()

    print(f'Iteration {iteration}: train {y_train_positive_ratio:.1%} val {y_val_positive_ratio:.1%}')


# %%
# モデルの評価指標(AUC)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_new, y_train_new)

y_val_pred = clf.predict_proba(X_val)[:, 1]

fpr, tpr, thresholds = roc_curve(y_true=y_val, y_score= y_val_pred)
y_auc = auc(fpr, tpr)
print(f'AUC={y_auc:.4}')

# %%
X = users_item_num[feature_num_columns + feature_cat_columns]
y = users_item_num[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)


# %%
# モデルの訓練
# ロジスティック回帰のパイプライン
lr_model = make_pipeline(
    OneHotEncoder(cols=feature_cat_columns, use_cat_names=True),
    StandardScaler(),
    LogisticRegression(),
)
# ランダムフォレストのパイプライン
rf_model = make_pipeline(
    OneHotEncoder(cols=feature_cat_columns, use_cat_names=True),
    RandomForestClassifier(),
)

# %%
# モデルの評価
lr_scores = cross_validate(lr_model, X_train, y_train, cv=kf, scoring='roc_auc', return_estimator=True)
print('LR', lr_scores['test_score'].mean())

rf_scores = cross_validate(rf_model, X_train, y_train, cv=kf, scoring='roc_auc', return_estimator=True)
print('RF', rf_scores['test_score'].mean())


# %%
# ハイパーパラメータ調整

# make_pipelineを使った場合、クラス名とパラメータ名をアンダーバー2つでつなぐ
lr_params = {
    'logisticregression__penalty': ['elasticnet'],  # penaltyはelasticnetのみ
    'logisticregression__solver': ['saga'],  # elasticnetを使う時はsagaを指定する
    'logisticregression__l1_ratio': [i*0.2 for i in range(6)],  # 0から1の範囲で0.1刻み
    'logisticregression__C': [10**i for i in range(-3, 4)]  # 10^(-3)から2^3の範囲で
}

rf_params = {
    # 100から500まで100刻み
    'randomforestclassifier__n_estimators': [100*i for i in range(1, 6)]
}

# scoring(評価指標)とcv(バリデーションの分割方法)の指定を忘れない
lr_gscv = GridSearchCV(lr_model, lr_params, scoring='roc_auc', cv=kf)
rf_gscv = GridSearchCV(rf_model, rf_params, scoring='roc_auc', cv=kf)

# グリッドサーチを実行
lr_gscv.fit(X_train, y_train)

print('LogisticRegression')
print(f'best_score: {lr_gscv.best_score_}')
print(f'best_params: {lr_gscv.best_params_}')

rf_gscv.fit(X_train, y_train)

print('RandomForest')
print(f'best_score: {rf_gscv.best_score_}')
print(f'best_params: {rf_gscv.best_params_}')


# %%
# predict_probaを使用することで各クラスに対する確率を得る
# 今回ほしいのはtarget=1の確率なので、[:, 1]でアクセスすればよい
lr_pred = lr_gscv.predict_proba(X_test)[:, 1]
rf_pred = rf_gscv.predict_proba(X_test)[:, 1]

print('LR', roc_auc_score(y_test, lr_pred))
print('RF', roc_auc_score(y_test, rf_pred))


# %%
