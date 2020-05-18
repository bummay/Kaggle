# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold

import itertools

from sklearn.linear_model import LogisticRegression

inDir = 'input/'
plt.figure(figsize=(20,10))


train = pd.read_csv(inDir + 'train.csv')
test = pd.read_csv(inDir + 'test.csv')

train_x = train.drop(['Survived'], axis=1)
train_y = train['Survived']

test_x = test.copy()

# %%
# 行数×列数
train_x.shape

# %%
# 平均、標準偏差、最大、最小、分位点
train_x.describe()

# %%
# 項目、データ型、Nullの件数
train_x.info()
# 項目ごとの欠損値の有無
train_x.isna().any()

# %%
# カテゴリ変数の種類と種類ごとの件数
train_x['Sex'].value_counts()
# %%
train_x['Embarked'].value_counts()

# %%
# 項目間の相関係数(数値、bool型のみ対象なので、カテゴリ変数を数値に変換)
train_x['Sex2'] = (train_x['Sex'] == 'female').astype(int)
# 列[Embarked]にはNaNデータがあるので、最出頻度の「S」で補完してから数値に変換
train_x['Embarked2'] = train_x['Embarked'].fillna('S').map({'S':0, 'C':1, 'Q':2}).astype(int)

train_x_corr = train_x.corr()
print(train_x_corr)
train_x.drop(['Sex2', 'Embarked2'],axis=1, inplace=True)

# %%
g = sns.catplot(x='Pclass', y='Survived', hue='Sex', data=train, kind='bar')
# %%
g = sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train)
# %%
g = sns.scatterplot(x='Fare', y='Survived', hue='Sex', data=train)
sns.heatmap(train_x_corr)

# %%
train_x.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],axis=1, inplace=True)
test_x.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],axis=1, inplace=True)

# %%
# カテゴリ変数をLabelEncoderを使って数値に変換する方法
for c in ['Sex', 'Embarked'] :
    le = LabelEncoder()
    le.fit(train_x[c].fillna('NA'))
    le.classes_ # ←これで変換する内容のListを作る。

    train_x[c] = le.transform(train_x[c].fillna('NA')).astype(int)
    test_x[c] = le.transform(test_x[c].fillna('NA')).astype(int)

# %%
model = XGBClassifier(n_estimater=20, random_state=71)
model.fit(train_x, train_y)

# %%
pred = model.predict_proba(test_x)[:, 1] 

# %%
pred_label = np.where(pred > 0.5, 1, 0)


# %%
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('output/submisson_first.csv', index=False)

# %%
# 各Foldのスコアを保存するリスト
scores_accuracy = []
scores_logloss = []

# %%
# クロスバリデーションを行う。
# trainデータを4つに分割する。1つずつをvalidateデータとして4回繰り返す。
kf = KFold(n_splits=4 , shuffle=True, random_state=71)

for tr_idx, va_idx, in kf.split(train_x) :
    # trainデータをtrainとvalidateに分ける。
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # 分けた後のtrainを使って学習
    model = XGBClassifier(n_estimators=20, random_state=71)
    model.fit(tr_x, tr_y)

    # validateの予測値を確率で出力する。
    va_pred = model.predict_proba(va_x)[:, 1]

    # validateの予測値と実際の値を突き合わせてスコアを計算する。
    logloss = log_loss(va_y, va_pred)
    accuracy = accuracy_score(va_y, va_pred > 0.5)

    # 計算されたスコアをListに保存する。
    scores_logloss.append(logloss)
    scores_accuracy.append(accuracy)

    # 各Foldのスコアの平均を出力する。
    logloss = np.mean(scores_logloss)
    accuracy = np.mean(scores_accuracy)
    print(f'logloss: {logloss:.4f}, accuracy: {accuracy:.4f}')

# %%
# パラメータチューニングを加える
# パラメータチューニングの候補
param_space = {
    'max_depth' : [3, 5, 7],
    'min_child_weight' : [1.0, 2.0, 4.0]
}

# 探索するパラメータの組み合わせ
param_combinations = itertools.product(param_space['max_depth'], param_space['min_child_weight'])

# 各パラメータの組み合わせとそれに対するスコアを保存するリスト
params = []
scores = []

# 各パラメータの組み合わせにおいてクロスバリデーションで評価を行う。
for max_depth, min_child_weight in param_combinations :
    score_folds = []
    kf = KFold(n_splits=4 , shuffle=True, random_state=123456)
    print(f'max_depth: {max_depth}, min_child_weight: {min_child_weight}')
    for tr_idx, va_idx, in kf.split(train_x) :
        print(f'tr_idx: {tr_idx}, va_idx: {va_idx}')
        # trainデータをtrainとvalidateに分ける。
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        # 分けた後のtrainを使って学習
        model = XGBClassifier(n_estimators=20, random_state=71, max_depth=max_depth, min_child_weight=min_child_weight)
        model.fit(tr_x, tr_y)

        # validateの予測値を確率で出力する。
        va_pred = model.predict_proba(va_x)[:, 1]
        logloss = log_loss(va_y, va_pred)
        score_folds.append(logloss)
    # 各フォルドのスコアを平均する。
    score_mean = np.mean(score_folds)

    # パラメータの組み合わせおよびそのスコアをListに保存する。
    params.append((max_depth, min_child_weight))
    scores.append(score_mean)

best_idx = np.argsort(scores)[0]
best_param = params[best_idx]
print(f'max_depth: {best_param[0]}, min_child_weight: {best_param[1]}')

# %%
