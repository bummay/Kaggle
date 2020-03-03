# %%
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# %%
# %%
breast_cancer = load_breast_cancer()
clf = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target,train_size=0.8)
clf.fit(X_train,y_train)
# %%
# 結果を返す
pred_y = clf.predict(X_test)
# スコアを出すための関数。
# ref : [decision_functionとの違いは？](https://stackoverflow.com/questions/36543137/whats-the-difference-between-predict-proba-and-decision-function-in-sklearn-py)
pred_y_score = clf.predict_proba(X_test)[:,1]
print("accuracy\n", metrics.accuracy_score(y_test, pred_y))
print("classification report\n",metrics.classification_report(y_test,pred_y))
# %%
# Compute ROC curve and ROC area for each class
fpr, tpr, _ = metrics.roc_curve(y_test, pred_y_score,pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
print("auc area", roc_auc)
