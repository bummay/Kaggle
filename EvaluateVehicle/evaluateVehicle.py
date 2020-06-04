# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.model_selection import KFold

# %%
inputDir = 'input/'
train_df = pd.read_table(inputDir + 'train.tsv')
test_df = pd.read_table(inputDir + 'test.tsv')

# %%
# 欠損値の確認→全項目OK
train_df.isna().any()

# %%
# ===== buying×classの件数 → buyingが高いとClassが低そう
sns.countplot(x='class', hue='buying', data=train_df,
                order=['unacc', 'acc', 'good', 'vgood'],
                hue_order=['low', 'med', 'high', 'vhigh'])
# %%
sns.countplot(x='buying', hue='class', data=train_df,
                order=['low', 'med', 'high', 'vhigh'],
                hue_order = ['unacc', 'acc', 'good', 'vgood'])

# %%
# ===== maint×classの件数 → maintが高いとclassが低そう
sns.countplot(x='class', hue='maint', data=train_df,
                order=['unacc', 'acc', 'good', 'vgood'],
                hue_order=['low', 'med', 'high', 'vhigh'])

# %%
sns.countplot(x='maint', hue='class', data=train_df,
                order=['low', 'med', 'high', 'vhigh'],
                hue_order=['unacc', 'acc', 'good', 'vgood'])

# %%
# ===== doors×classの件数 → あまり影響なさそう
sns.countplot(x='class', hue='doors', data=train_df,
                order=['unacc', 'acc', 'good', 'vgood'],
                hue_order=['2', '3', '4', '5more'])

# %%
sns.countplot(x='doors', hue='class', data=train_df,
                order=['2', '3', '4', '5more'],
                hue_order=['unacc', 'acc', 'good', 'vgood'])


# %%
# ===== persons×classの件数 → 2人乗りはclassが低い
sns.countplot(x='class', hue='persons', data=train_df,
                order=['unacc', 'acc', 'good', 'vgood'],
                hue_order=['2', '4', 'more'])

# %%
sns.countplot(x='persons', hue='class', data=train_df,
                order=['2', '4', 'more'],
                hue_order=['unacc', 'acc', 'good', 'vgood'])


# %%
# ===== lug_boot×classの件数 → smallほどclassが低い割合が高い
sns.countplot(x='class', hue='lug_boot', data=train_df,
                order=['unacc', 'acc', 'good', 'vgood'],
                hue_order=['small', 'med', 'big'])
# %%
sns.countplot(x='lug_boot', hue='class', data=train_df,
                order=['small', 'med', 'big'],
                hue_order=['unacc', 'acc', 'good', 'vgood'])


# %%
# ===== safety×classの件数 → lowだとclassは低い
sns.countplot(x='class', hue='safety', data=train_df,
                order=['unacc', 'acc', 'good', 'vgood'],
                hue_order=['low', 'med', 'high'])
# %%
sns.countplot(x='safety', hue='class', data=train_df,
                order=['low', 'med', 'high'],
                hue_order=['unacc', 'acc', 'good', 'vgood'])

# %%
# one-hot encodingする。
cc_df = pd.concat([train_df, test_df])
cc_df = pd.get_dummies(cc_df, columns=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

train_df = cc_df[~cc_df['class'].isna()]
test_df = cc_df[cc_df['class'].isna()]

train_df['class'] = train_df['class'].map(
    {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}).astype(int)

# %%
X = train_df.drop(['id', 'class'],axis=1)
y = train_df['class']
X_test = test_df.drop(['id', 'class'], axis=1)

kf = KFold(n_splits=4, shuffle=True, random_state=71)

tr_idx, va_idx = list(kf.split(X))[0]

tr_x, va_x = X.iloc[tr_idx], X.iloc[va_idx]
tr_y, va_y = y.iloc[tr_idx], y.iloc[va_idx]

dtrain = xgb.DMatrix(tr_x, label=tr_y)
dvalid = xgb.DMatrix(va_x, label=va_y)
dtest = xgb.DMatrix(X_test)

params = {
    'objective': 'multi:softprob',
    'eval_metric':'logloss',
    'max_depth': 5,
    'random_state': 71,
    'num_class':4}
# num_round = 50

# watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
# model = xgb.train(params, dtrain, num_round, evals=watchlist)
model = xgb.train(params, dtrain)

va_pred = model.predict(dvalid)
score = log_loss(va_y, va_pred)

print(f'multi-class logloss: {score:4f}')

pred = model.predict(dtest)


# %%
def outputCSV(pred, csvName):
    pred_class = []
    for item in pred:
        pred_class.append(np.argmax(item))

    submission = pd.DataFrame({
        "id": test_df['id'],
        'class': pred_class
    })
    submission['class'] = submission['class'].map(
        {0: 'unacc', 1: 'acc', 2: 'good', 3: 'vgood'})
    now = datetime.datetime.now()
    submission.to_csv('output/' + csvName +
                    now.strftime('%Y%m%d_%H%M%S') + '.csv', index=False, header=False)

outputCSV(pred, 'EvaluateVehicle')


# %%
