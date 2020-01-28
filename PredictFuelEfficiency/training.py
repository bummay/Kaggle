# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import model_selection , preprocessing, linear_model
import sklearn
from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV

# 日付の処理で使う
import datetime

# %%
inputDir = 'input/'

train_df = pd.read_csv(inputDir + 'train_df.csv')
test_df = pd.read_csv(inputDir + 'test_df.csv')


X = train_df.drop(['id', 'mpg'], axis=1)
y = train_df['mpg']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)



# %%
reg = LinearRegression()
reg.fit(X_train, y_train)
print(pd.DataFrame({'Name': X_train.columns, 'Coefficients': reg.coef_}).sort_values(by='Coefficients'))
print(reg.intercept_)

pred_train = reg.predict(X_train)
pred_test = reg.predict(X_test)

print('MSE(train): %f' % mean_squared_error(y_train, pred_train))
print('MSE(test): %f' % mean_squared_error(y_test, pred_test))


# %%
pred_X = test_df.drop(['id'], axis=1)
pred_y = reg.predict(pred_X)
submission = pd.DataFrame({
    "id": test_df['id'],
    'y': pred_y
})
submission['y'] = round(submission['y'], 1)


# %%
now = datetime.datetime.now()
submission.to_csv('output/' + 'predictFuelEfficiency_' +
                  now.strftime('%Y%m%d_%H%M%S') + '.csv', index=False, header=False)


# %%
