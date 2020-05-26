# %%
import numpy as np
from sklearn.metrics import mean_squared_error as rmse
from sklearn.metrics import mean_squared_log_error as rmsle
from sklearn.metrics import mean_absolute_error as mae
# %%
y_true = [1.0, 1.5, 2.0, 1.2, 1.8]
y_pred = [0.8, 1.5, 1.8, 1.3, 3.0]

rmse = np.sqrt(rmse(y_true, y_pred))
print(f'rmse:{rmse}')
# rmse:0.5531726674375732

y_true = [100, 0, 400]
y_pred = [200, 10, 200]

rmsle = np.sqrt(rmsle(y_true, y_pred))
print(f'rmsle:{rmsle}')
# rmsle:1.4944905400842203


y_true = [100, 160, 60]
y_pred = [80, 100, 100]

mae = mae(y_true, y_pred)
print(f'mae:{mae}')
# mae:40.0

# %%
from sklearn.metrics import confusion_matrix

y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]

tp = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
tn = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))
fp = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
fn = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))

confusion_matrix1 = np.array([[tp, fp],
                            [fn, tn]])

print(confusion_matrix1)

confusion_matrix2 = confusion_matrix(y_true, y_pred)
print(confusion_matrix2)


# %%
from sklearn.metrics import accuracy_score

y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]

accuracy = accuracy_score(y_true, y_pred)
print(accuracy)
# 0.625

# %%
from sklearn.metrics import f1_score, matthews_corrcoef
# 正例が多いケース
y_true1 = [1, 1, 1, 1, 1, 1, 1, 0, 1, 0]
y_pred1 = [1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
# 負例が多いケース
y_true2 = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
y_pred2 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1]

accuracy1 = accuracy_score(y_true1, y_pred1)
f1_score1 = f1_score(y_true1, y_pred1)
matthews1 = matthews_corrcoef(y_true1, y_pred1)
accuracy2 = accuracy_score(y_true2, y_pred2)
f1_score2 = f1_score(y_true2, y_pred2)
matthews2 = matthews_corrcoef(y_true2, y_pred2)

# 正例と負例の数が逆転した場合、F1-scoreは変化するがMCCは変化しない。
print(f'accuracy:{accuracy1}')
print(f'f1:{f1_score1}')
print(f'mcc:{matthews1}')
print('----------')
print(f'accuracy:{accuracy2}')
print(f'f1:{f1_score2}')
print(f'mcc:{matthews2}')

# %%
from sklearn.metrics import log_loss

y_true = [1, 0, 1, 1, 0, 1]
y_prob = [0.1, 0.2, 0.8, 0.8, 0.1, 0.3]

logloss = log_loss(y_true, y_prob)
print(logloss)
# 0.7135581778200728

# %%
from sklearn.metrics import log_loss
y_true = np.array([0, 2, 1, 2, 2])
y_pred = np.array([
                    [0.68, 0.32, 0.00],
                    [0.00, 0.00, 1.00],
                    [0.60, 0.40, 0.00],
                    [0.00, 0.00, 1.00],
                    [0.28, 0.12, 0.60]
])
logloss = log_loss(y_true, y_pred)
print(logloss)
# 0.3625557672904274

# %%
from sklearn.metrics import f1_score

# マルチラベル分類の真の値、予測値は、評価指数の計算上はレコード×クラスの二値の行列としたほうが扱いやすい。
# ここでは真の値を[[1,2], [1], [1,2,3], [2,3], [3]]とする。
y_true = np.array([
                    [1, 1, 0],
                    [1, 0, 0],
                    [1, 1, 1],
                    [0, 1, 1],
                    [0, 0, 1]
])
# 予測値は[[1,3], [2], [1,3], [3], [3]]とする。
y_pred = np.array([
                    [1, 0, 1],
                    [0, 1, 0],
                    [1, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1]
])

# mean-f1ではレコードごとにF1-scoreを計算して平均を取る。
mean_f1 = np.mean([f1_score(y_true[i, :], y_pred[i, :]) for i in range(len(y_true))])

# macro-f1ではクラスごとにF1-scoreを計算して平均を取る。
n_class = 3
macro_f1 = np.mean([f1_score(y_true[:, c], y_pred[:, c]) for c in range(n_class)])

# micro-f1ではレコード×クラスの組み合わせごとにTP/TN/FP/FNを計算し、F1-scoreを求める。
micro_f1 = f1_score(y_true.reshape(-1), y_pred.reshape(-1))

print(mean_f1, macro_f1, micro_f1)
# 0.5933333333333334 0.5523809523809523 0.6250000000000001

#scikit-learnのメソッドを使っても計算できる。
mean_f1 = f1_score(y_true, y_pred, average='samples')
macro_f1 = f1_score(y_true, y_pred, average='macro')
micro_f1 = f1_score(y_true, y_pred, average='micro')
print(mean_f1, macro_f1, micro_f1)
# 0.5933333333333334 0.5523809523809523 0.6250000000000001

# %%
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# quadratic_weighted_kappaを計算する関数
def quadratic_weighted_kappa(c_matrix):
    numer = 0.0
    denom = 0.0

    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            n = c_matrix.shape[0]
            wij = ((i - j ) ** 2.0)
            oij = c_matrix[i, j]
            eij = c_matrix[i, :].sum() * c_matrix[:, j].sum() / c_matrix.sum()
            numer += wij * oij
            denom += wij * eij
        
    return 1.0 - numer / denom

y_true = [1, 2, 3, 4, 3]
y_pred = [2, 2, 4, 4, 5]

c_matrix = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])

kappa = quadratic_weighted_kappa(c_matrix)
print(kappa)
# 0.6153846153846154

# scikit-learnのメソッドでも計算できるけどね
kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
print(kappa)
# 0.6153846153846154

# %%
K = 3
# レコード数は5、クラスは4種類とする。

y_true = [[1,2], [1,2], [4], [1,2,3,4], [3,4]]
# K=3なので、予測値はレコードごとに順位をつけて3個まで
y_pred = [
            [1,2,4],
            [4,1,2],
            [1,4,3],
            [1,2,3],
            [1,2,4]
]

def average_precision_K(y_i_true, y_i_pred):
    # y_predがK以下の長さで、要素がすべて異なることが必要
    assert (len(y_i_pred) <= K)
    assert (len(np.unique(y_i_pred)) == len(y_i_pred))

    sum_precision = 0.0
    num_hits = 0.0

    for i, p in enumerate(y_i_pred):
        if p in y_i_true:
            num_hits += 1
            precision = num_hits / (i + 1)
            sum_precision += precision
    
    return sum_precision / min(len(y_i_true), K)

# MAP@K
def mapk(y_true, y_pred):
    return np.mean([average_precision_K(y_i_true, y_i_pred) for y_i_true, y_i_pred in zip(y_true, y_pred)])

print(mapk(y_true, y_pred))
# 0.6499999999999999


# %%
from sklearn.metrics import f1_score
from scipy.optimize import minimize

# サンプルデータ生成の準備
rand = np.random.RandomState(seed=71)
train_y_prob = np.linspace(0, 1.0, 10000) # 0 ~ 1.0 を10000分割

# 真の値
train_y = pd.Series(rand.uniform(0.0, 1.0, train_y_prob.size) < train_y_prob)
# 予測値
train_pred_prob = np.clip(train_y_prob * np.exp(rand.standard_normal(train_y_prob.shape) * 0.3), 0.0, 1.0)

# 閾値を0.5とすると、F1-Scoreは0.7224831529507862
init_threshold = 0.5
init_score = f1_score(train_y, train_pred_prob >= init_threshold)
print(init_score)
# 0.7224831529507862

# 最適化の目的関数を設定
def f1_opt(x):
    return -f1_score(train_y, train_pred_prob >= x)

# scipy.optimizeのminimizeメソッドで最適な閾値を求める。
result = minimize(f1_opt, x0=np.array([0.5]), method='Nelder-Mead')
best_threshold = result['x'].item()
best_score = f1_score(train_y, train_pred_prob >= best_threshold)
print(best_score)
# 0.7557317703844165

# %%
# from sklearn.preprocessing import StandardScaler

# # 標準化の定義
# scaler = StandardScaler()
# scaler.fit(train_x[num_cols])

# # 変換後のデータで各列を置き換え
# train_x[num_cols] = scaler.transform(train_x[num_cols])
# test_x[num_cols] = scaler.transform(test_x[num_cols])

# # %%
# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# scaler.fit(train_x[num_cols])

# # 変換後のデータで各列を置き換え
# train_x[num_cols] = scaler.transform(train_x[num_cols])
# test_x[num_cols] = scaler.transform(test_x[num_cols])

# %%
x = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])

x1 = np.log(x)
print(x1)
# [0.         2.30258509 4.60517019 6.90775528 9.21034037]
# %%
x = np.array([0, 1.0, 10.0, 100.0, 1000.0, 10000.0])
x1 = np.log(x)
print(x)
x2 = np.log1p(x)
print(x2)

# %%
x = np.array([-100.0, -10.0, -1.0, 0, 1.0, 10.0, 100.0, 1000.0, 10000.0])
x1 = np.log(x)
print(x1)
#  [       nan        nan        nan       -inf 0.         2.30258509 4.60517019 6.90775528 9.21034037]

x2 = np.log1p(x)
print(x2)
#[       nan        nan       -inf 0.         0.69314718 2.39789527 4.61512052 6.90875478 9.21044037]

x3 = np.sign(x) * np.log1p(np.abs(x))
print(x3)
#[-4.61512052 -2.39789527 -0.69314718  0.          0.69314718  2.39789527 4.61512052  6.90875478  9.21044037]
# %%
pos_cols = [c for c in num_cols if (train_x[c] > 0.0).all() and (test_x[c] > 0.0).all()]

from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='box-cox')
pt.fit(train_x[pos_cols])

train_x[pos_cols] = pt.transform(train_x[pos_cols])
test_x[pos_cols] = pt.transform(test_x[pos_cols])

pt = PowerTransformer(method='yeo-johnson')
pt.fit(train_x[pos_cols])

train_x[pos_cols] = pt.transform(train_x[pos_cols])
test_x[pos_cols] = pt.transform(test_x[pos_cols])

# %%
# 学習データの1%点、99%点を算出
p01 = train_x[num_cols].quantile(0.01)
p99 = train_x[num_cols].quantile(0.99)

train_x[num_cols] = train_x[num_cols].clip(p01, p99, axis=1)
test_x[num_cols] = test_x[num_cols].clip(p01, p99, axis=1)

# %%
x = [1, 7, 5, 4, 6, 3]

# とりあえず3等分する場合
binned = pd.cut(x, 3, labels=False)
print(binned)
# [0 2 1 1 2 0]

# binの範囲を指定する場合(3.0以下、3.0超〜5.0以下、5.0超)
binn_edges = [-float('inf'), 3.0, 5.0, float('inf')]
binned = pd.cut(x, binn_edges, labels=False)
print(binned)
# [0 2 1 1 2 0]

# %%
x = [10, 20, 30, 0, 40, 40]

# pandasのrank関数を使う方法
rank = pd.Series(x).rank()
print(rank.values)
# [2.  3.  4.  1.  5.5 5.5]

# numpyのargsort関数を2回適用する方法
order = np.argsort(x)
rank = np.argsort(order)
print(order)
print(rank)
# [3 0 1 2 4 5]
# [1 2 3 0 4 5]
# %%
from sklearn.preprocessing import QuantileTransformer

transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
transformer.fit(train_x[num_cols])

train_x[num_cols] = transformer.transform(train_x[num_cols])
test_x[num_cols] = transformer.transform(test_x[num_cols])

# %%
# 学習データとテストデータを結合してone-hot encodingを行う。
all_x = pd.concat([train_x, tets_x])
all_x = pd.get_dummies(all_x, columns=cat_cols)

# %%
# one-hot encoding実行後に、学習データとテストデータを再分割。
train_x = all_x.iloc[:train_x.shape[0], :].reset_index(drop=True)
test_x = all_x.iloc[train_x.shape[0]:, :].reset_index(drop=True)

from sklearn.feature_extraction import FeatureHasher

for c in cat_cols:
    fh = FeatureHasher(n_features=5, input_type='string')

    hash_train = fh.transform(train_x[[c]].astype(str)values)
    hash_test = fh.transform(test_x[[c]].astype(str)values)

    # 変換後は行列になってしまうので、これをデータフレームに変換
    hash_train = pd.DataFrame(hash_train.todense(), columns=[f' {c}_{i}' for i in range(5)])
    hash_test = pd.DataFrame(hash_test.todense(), columns=[f' {c}_{i}' for i in range(5)])

    # 元のデータフレームと結合
    train_x = pd.concat([train_x, hash_train], axis=1)
    test_x = pd.concat([test_x, hash_test], axis=1)

# 本のカテゴリ変数を削除
train_x.drop(cal_cols, axis=1, inplace=True)
test_x.drop(cal_cols, axis=1, inplace=True)

# %%
cat_x = pd.concat(train_x, test_x)

for c in cat_cols:
    freq = cat_x[c].value_counts()
    train_x[c] = train_x[c].map(freq)
    test_x[c] = test_x[c].map(freq)

# %%
from sklearn.model_selection import KFold

for c in cat_cols:
    # 学習データ全体で各カテゴリにおけるtargetの平均を計算
    data_tmp = pd.DataFrame({c: train_x[c], 'target': train_y})
    target_mean = data_tmp.groupby(c)['target'].mean()
    # テストデータのカテゴリを置換
    test_x[c] = test_x[c].map(target_mean)

    # 学習データの変換後の値を格納するための配列
    tmp = np.repeat(np.nan, train_x.shape[0])

    # 学習データを分割
    kf = KFold(n_splits=4, shuffle=True, random_state=72)
    for idx_1, idx_2 in kf.split(train_x):
        # out-of-foldで各カテゴリにおける目的変数の平均を計算
        target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
        tmp[idx_2] = train_x[c].iloc[idx_2].map(target_mean)

    train_x[c] = tmp
# %%
