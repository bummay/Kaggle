# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import sklearn
from category_encoders import OneHotEncoder
from sklearn import model_selection , preprocessing, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 日付の処理で使う
import datetime

# %%
inputDir = 'input/'

train_df = pd.read_csv(inputDir + 'train_df.csv')
test_df = pd.read_csv(inputDir + 'test_df.csv')
cat_columns = [
                'cylinders',
                'manufacturer',
                'country'
            ]


# %%
X = train_df.drop(['id', 'mpg'], axis=1)
y = train_df['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

# %%

rf_model = make_pipeline(
    OneHotEncoder(cols=cat_columns, use_cat_names=True),
    RandomForestRegressor()
)

kf = KFold(n_splits=5, shuffle=True, random_state=1)
scoring = 'neg_mean_squared_error'
rf_scores = cross_validate(rf_model, X_train, y_train, cv=kf,
                           scoring= scoring, return_estimator=True)

print('RF', rf_scores['test_score'].mean())


# %%
reg = LinearRegression()
reg.fit(X_train, y_train)
print(pd.DataFrame({'Name': X_train.columns, 'Coefficients': reg.coef_}).sort_values(by='Coefficients'))
print(reg.intercept_)

pred_train = reg.predict(X_train)
pred_test = reg.predict(X_test)

print('MSE(train) : %f' % mse(y_train, pred_train))
print('RMSE(train): %f' % np.sqrt(mse(y_train, pred_train)))
print('MSE(test) : %f' % mse(y_test, pred_test))
print('RMSE(test): %f' % np.sqrt(mse(y_test, pred_test)))

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
