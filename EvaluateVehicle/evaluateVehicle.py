# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import GridSearchCV

# %%
inputDir = 'evaluateVehicle/input/'
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
# ===== classをコード化
def processClass(df):
    df['class'] = df['class'].map(
        {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}).astype(int)
    return df

train_df = processClass(train_df)

# %%
# ===== buyingをコード化
def processBuying(df):
    df['buying'] = df['buying'].map(
        {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}).astype(int)
    return df

train_df = processBuying(train_df)
test_df = processBuying(test_df)

# %%
# ===== maintをコード化
def processMaint(df):
    df['maint'] = df['maint'].map(
        {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}).astype(int)
    return df

train_df = processMaint(train_df)
test_df = processMaint(test_df)

# %%
# ===== doorsをコード化
def processDoors(df):
    df.loc[
    (df['doors'] == '5more'),
    'doors'] = 9

    df['doors'] = df['doors'].astype(int)
    return df

train_df = processDoors(train_df)
test_df = processDoors(test_df)

# %%
# ===== personsをコード化
def processPersons(df):
    df.loc[
        (df['persons'] == 'more'),
        'persons'] = 9

    df['persons'] = df['persons'].astype(int)
    return df

train_df = processPersons(train_df)
test_df = processPersons(test_df)

# %%
# ===== lug_bootをコード化
def processLugboot(df):
    df['lug_boot'] = df['lug_boot'].map(
    {'small': 0, 'med': 1, 'big': 2}).astype(int)
    return df

train_df = processLugboot(train_df)
test_df = processLugboot(test_df)

# %%
# ===== safetyをコード化

def processSafety(df):
    df['safety'] = df['safety'].map(
        {'low': 0, 'med': 1, 'high': 2}).astype(int)
    return df

train_df = processSafety(train_df)
test_df = processSafety(test_df)

# %%
# ===== 不要な列を削除
def dropColumns(df):
    df = df.drop(['doors'], axis=1)
    return df
train_df = dropColumns(train_df)
test_df = dropColumns(test_df)

# %%
train_X = train_df.drop(['id', 'class'],axis=1)
train_y = train_df['class']
test_x = test_df.drop('id', axis=1)

# %%
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(train_X, train_y)

rfc.score(train_X, train_y)
Y_pred = rfc.predict(test_x)


# %%
def outputCSV(pred, csvName):
    submission = pd.DataFrame({
        "id": test_df['id'],
        'class': pred
    })
    submission['class'] = submission['class'].map(
        {0: 'unacc', 1: 'acc', 2: 'good', 3: 'vgood'})
    now = datetime.datetime.now()
    submission.to_csv('evaluateVehicle/output/' + csvName +
                    now.strftime('%Y%m%d_%H%M%S') + '.csv', index=False, header=False)

outputCSV(Y_pred, 'EvaluateVehicle')


# %%
