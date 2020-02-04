# https://note.nkmk.me/python-sklearn-roc-curve-auc-score/
# %%
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, recall_score, confusion_matrix
import matplotlib.pyplot as plt

y_true = [0, 0, 0, 0, 1, 1, 1, 1]
y_score = [0.2, 0.3, 0.6, 0.8, 0.4, 0.5, 0.7, 0.9]

roc = roc_curve(y_true, y_score)


# %%
fpr, tpr, thresholds = roc_curve(y_true, y_score)


# %%
print(fpr)
# [0.   0.   0.25 0.25 0.5  0.5  1.  ]
print(tpr)
# [0.   0.25 0.25 0.5  0.5  1.   1.  ]
print(thresholds)
# [1.9 0.9 0.8 0.7 0.6 0.4 0.2]


# %%
plt.plot(fpr, tpr, marker='o')
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()
# %%
fpr_all, tpr_all, thresholds_all = roc_curve(y_true, y_score,
                                             drop_intermediate=False)
# %%
print(fpr_all)
# [0.   0.   0.25 0.25 0.5  0.5  0.5  0.75 1.  ]
print(tpr_all)
# [0.   0.25 0.25 0.5  0.5  0.75 1.   1.   1.  ]
print(thresholds_all)
# [1.9 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2]

plt.plot(fpr_all, tpr_all, marker='o')
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()


# %%

y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
y_score = np.array([0.2, 0.3, 0.6, 0.8, 0.4, 0.5, 0.7, 0.9])


# %%
print(y_score >= 0.5)
# [False False  True  True False  True  True  True]
print((y_score >= 0.5).astype(int))
# [0 0 1 1 0 1 1 1]

# %%
def fpr_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).flatten()
    return fp / (tn + fp)


# %%
print('fpr:%.2f' % fpr_score(y_true, y_score >= 0.5))
print('tpr:%.2f' % recall_score(y_true, y_score >= 0.5))

# %%
# もし閾値を予測スコアの最小値に設定すると、モデルは全てのケースにおいてTrueと予測する。
# このとき、FPRもTPRも最大値の1.00となる。
# (FPR：全てのFalseを正しく予測できなかった)
# (TPR：全てのTrueを正しく予測できた)
th_min = min(y_score)
print(th_min)

print((y_score >= th_min).astype(int))
print('fpr:%.2f' % fpr_score(y_true, y_score >= th_min))
print('tpr:%.2f' % recall_score(y_true, y_score >= th_min))

# %%
# 反対に閾値を予測スコアの最大値に設定すると、モデルは全てのケースにおいてFalseと予測する。
# このとき、FPRもTPRも最小値の0.00となる。
# (FPR：全てのFalseを正しく予測できた)
# (TPR：全てのTrueを正しく予測できなかった)
th_max = max(y_score) + 1
print(th_max)

print((y_score >= th_max).astype(int))
print('fpr:%.2f' % fpr_score(y_true, y_score >= th_max))
print('tpr:%.2f' % recall_score(y_true, y_score >= th_max))


# %%
# 全ての予測スコアを閾値に設定した場合のFPRとTPRを取得する。
df = pd.DataFrame({'true': y_true, 'score': y_score})
df['TPR'] = df.apply(lambda row: recall_score(y_true, y_score >= row['score']), axis=1)
df['FPR'] = df.apply(lambda row: fpr_score(y_true, y_score >= row['score']), axis=1)
df
# %%
df.sort_values('score', ascending=False)

# %%
# このときのROC曲線の元になるデータはこんな感じ。
fpr_all, tpr_all, th_all = roc_curve(y_true, y_score, drop_intermediate=False)
df_roc = pd.DataFrame({'th_all': th_all, 'tpr_all': tpr_all, 'fpr_all': fpr_all})
df_roc


# think about ideal condition
# %%
# 適切な閾値を設定することで完全な予測ができる場合、
# FPRは最小値の0
# TPRは最大値の1 となる。
y_true_perfect = np.array([0, 0, 0, 0, 1, 1, 1, 1])
y_score_perfect = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
print(y_true_perfect)
print((y_score_perfect >= 0.5).astype(int))

print('fpr:%.2f' % fpr_score(y_true_perfect, y_score_perfect >= 0.5))
print('tpr:%.2f' % recall_score(y_true_perfect, y_score_perfect >= 0.5))

roc_p = roc_curve(y_true_perfect, y_score_perfect, drop_intermediate=False)

plt.plot(roc_p[0], roc_p[1], marker='o')
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()

# %%
# どのような閾値を設定しても完全な予測ができない場合と、
# 前述の完全な予測ができる場合を比較してみる。
y_true_1 = np.array([0, 0, 0, 1, 0, 1, 1, 1])
y_score_1 = y_score_perfect

roc_1 = roc_curve(y_true_1, y_score_1, drop_intermediate=False)

y_true_2 = np.array([0, 0, 1, 1, 0, 0, 1, 1])
y_score_2 = y_score_perfect

roc_2 = roc_curve(y_true_2, y_score_2, drop_intermediate=False)

plt.plot(roc_p[0], roc_p[1], marker='s')
plt.plot(roc_1[0], roc_1[1], marker='o')
plt.plot(roc_2[0], roc_2[1], marker='x')
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()


# %%
# ROC曲線に影響するのは予測スコアの順番(順位)のみ
y_true_org = np.array([0, 0, 1, 1, 0, 0, 1, 1])
y_score_org = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

roc_org = roc_curve(y_true_org, y_score_org,drop_intermediate=False)
plt.plot(roc_org[0], roc_org[1], marker='s')

# %%
# 予測スコアの中身を変えてもROC曲線は変わらない。
y_score_scale = y_score_org / 2
print(y_score_scale)

roc_scale = roc_curve(y_true_org, y_score_scale, drop_intermediate=False)
plt.plot(roc_scale[0], roc_scale[1], marker='x')

# %%
# 予測スコアの中身を変えてもROC曲線は変わらない。その2
y_score_interval = np.array([0.01, 0.02, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96])
roc_interval = roc_curve(y_true_org, y_score_interval, drop_intermediate=False)
plt.plot(roc_interval[0], roc_interval[1], marker='o')

# %%
# 3つのROC曲線を重ねると、同一なのがよく分かる。
plt.plot(roc_org[0], roc_org[1], marker='s')
plt.plot(roc_scale[0], roc_scale[1], marker='x')
plt.plot(roc_interval[0], roc_interval[1], marker='o')
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()


# %%
# ランダムに分類した(当てずっぽうに予測した)場合のROC曲線は
# 座標(0, 0)と座標(1, 1)を結ぶ線になる。
# 予測性能が悪いと、ROC曲線はこの直線に近づいていく
np.random.seed(0)
y_true_random = np.array([0] * 5000 + [1] * 5000)
y_score_random = np.random.rand(10000)

roc_random = roc_curve(y_true_random, y_score_random)

plt.plot(roc_random[0], roc_random[1])
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()


# %%
# AUC(Area Under the Curve)：曲線下の面積
# ROC曲線のAUCのことをROC-AUCという。
y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
y_score = np.array([0.2, 0.3, 0.6, 0.8, 0.4, 0.5, 0.7, 0.9])

roc = roc_curve(y_true, y_score, drop_intermediate=False)
plt.plot(roc[0], roc[1], marker='x')

print('ROC-AUC:%.4f' %roc_auc_score(y_true, y_score))


# %%
# 完全な分類ができる閾値が存在する場合のROC-UACは1.0となる。
y_true_perfect = np.array([0, 0, 0, 0, 1, 1, 1, 1])
y_score_perfect = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

roc_perfect = roc_curve(y_true_perfect, y_score_perfect, drop_intermediate=False)
plt.plot(roc_perfect[0], roc_perfect[1], marker='x')

print('ROC-AUC:%.4f' % roc_auc_score(y_true_perfect, y_score_perfect))

# %%
# ランダムに分類した場合のROC-AUCは0.5に近い値を取る。
np.random.seed(0)
y_true_random = np.array([0] * 5000 + [1] * 5000)
y_score_random = np.random.rand(10000)

roc_random = roc_curve(y_true_random, y_score_random, drop_intermediate=False)
plt.plot(roc_random[0], roc_random[1], marker='x')

print('ROC-AUC:%.4f' % roc_auc_score(y_true_random, y_score_random))


# %%
