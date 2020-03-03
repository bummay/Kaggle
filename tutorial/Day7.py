# %%
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# %%
data = datasets.load_boston()
print("data shape:", data.data.shape)
print("feature name:", data.feature_names)
print("target value:", data.target)


# %%
data = datasets.load_iris()
print("data shape:", data.data.shape)
print("feature name:", data.feature_names)
print("target class", data.target)
print("class name:", data.target_names)


# %%
X, y = shuffle(data.data, data.target, random_state=0)
print("target class", y)


# %%
iris = datasets.load_iris()
data, target = iris.data, iris.target

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=23)

model = RandomForestClassifier()
model.fit(data_train, target_train)

target_pred = model.predict(data_test)
print(accuracy_score(target_test, target_pred))

