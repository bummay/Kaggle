# タスクの種類
|種類|予測するもの|予測する種類|評価指標|
|:--|:--|:--|:--|
|回帰|物の値段、購入者数など|数値|RMSE、MAE|
|【分類】二値分類|検査結果(陽性/陰性)など|- 0or1のラベル<br> - 1である確率|- F1-score<br> - logloss、AUC|
|【分類】マルチクラス分類|検査結果(陽性/陰性)など|- 0〜nのラベルのいずれか<br> - 各ラベルである確率(合計=1)|multi-class、logloss|
|【分類】マルチラベル分類|検査結果(陽性/陰性)など|- 0〜nのラベル(複数選択もあり)<br> - 各ラベルである確率(合計≠1)|mean-F1、macro-F1|
|レコメンデーション|ユーザーが購入しそうな商品など。順位をつけて商品数を固定する方法と、順位をつけず任意の個数を予測する方法の2種類がある。||(順位あり)MAP@K<br>(順位なし)mean-F1、macro-F1|
|物体検出|画像の中の自動車と歩行者など|画像に含まれる物体のクラスと存在する矩形領域||
|セグメンテーション|画像の中の自動車と歩行者など|画像に含まれる物体のクラスと存在する領域(ピクセル単位)||

# 評価指標
## 回帰
### RMSE(Root Mean Squared Error)：平均平方二条誤差
各レコードの目的変数について予測値と真の値の差の二乗を取り、それらの平均の平方根を取る。

$RMSE = \sqrt{\frac{1}{N}\sum^{N}_{i=1}(y_{i} - \hat{y_{i}})^2}
$
- 統計学的にも大きな意味を持っている。
- 誤差の二乗を取るので、外れ値の影響を受けやすい(事前に外れ値を除外しておくなどの対処が必要)

scikit-learn.metricsモジュールの _mean_squared_error_ で計算可能。
```
import numpy as np
from sklearn.metrics import mean_squared_error

y_true = [1.0, 1.5, 2.0, 1.2, 1.8]
y_pred = [0.8, 1.5, 1.8, 1.3, 3.0]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(rmse)
# 0.5531726674375732
```
$RMSE = \sqrt{\frac{1}{5} \{(0.8 - 1.0)^2 + (1.5 - 1.5)^2 + (1.8 - 2.0)^2 + (1.3 - 1.2)^2 + (3.0 - 1.8)^2\}} \\\\　
= \sqrt{\frac{1}{5} (-0.2^2 + 0^2 -0.2^2 + 0.1^2 + 1.2^2)}　\\\\　
= \sqrt{\frac{1}{5} (0.04 + 0 + 0.04 + 0.01 + 1.44)}　\\\\　
= \sqrt{0.306} \\\\　= 0.5531726674375732
$



### RMSLE(Root Mean Squared Logarithmic Error)：平均平方二条対数誤差
RMSEとの違いは、予測値と真の値のそれぞれの対数を取るところ。両対数の誤差を二乗し、それらの平均の平方根を取る。

$RMSLE = \sqrt{\frac{1}{N}\sum^{N}_{i=1}(\log(1 + y_{i}) - \log(1 + \hat{y_{i}}))^2}
$

$log(1+x)=x-\frac{1}{2}x^2+\frac{1}{3}x^3-\frac{1}{4}x^4 ... $という難しい計算をする。マクローリン展開というらしい。
- 目的変数が「裾の重い分布」を持ち、実の値を使用すると大きな値の影響を受けやすいときや、真の値と予測値の比率に着目したい場合に用いられる。


scikit-learn.metricsモジュールの _mean_squared_log_error_ で計算可能。

### MAE(Mean Absolute Error)
真の値と予測値の差の絶対値の平均を計算する。
$MAE=\frac{1}{N}\sum^{N}_{i=1}|y_i-\hat{y_i}| $
- 外れ値の影響を低減した形での評価に適した関数
- 仮に１つの代表値で予測を行う場合、MAEを最小化する予測値は中央値。

scikit-learn.metricsモジュールの _mean_absolute_error_ で計算可能。
```
import numpy as np
from sklearn.metrics import mean_absolute_error as mae

y_true = [100, 160, 60]
y_pred = [80, 100, 100]

mae = mae(y_true, y_pred)
print(f'mae:{mae}')
# mae:40.0
```
$MAE = \frac{1}{3} (|100-80| + |160-100| + |60-100|) \\\\　
= \frac{1}{3} (20 + 60 + 40) \\\\　
= \frac{1}{3} * 120 \\\\　
= 40 \\\\　
$

### 決定係数($R^2$)
回帰分析の当てはまりの良さを表す。最大で1となり、1に近づくほど精度の高い予測ができていることを指す。
$R^2 = 1- \frac{\sum^{N}_{i=1}(y_i-\hat y_i)^2}{\sum^{N}_{i=1}(y_i- \bar{y})^2}
$
$ 
\bar{y} = \frac{1}{N}\sum^{N}_{i=1}y_i 
$
要は、真の値の平均値。

scilit-learn.metricsモジュールの _r2_score_ で計算可能。
## 二値における評価指標
### 混同行列
- TP(True Positive、真陽性)：予測「Positive」→結果「正しい」
- TN(True Negative、真陰性)：予測「Negative」→結果「正しい」
- FP(False Positive、偽陽性)：予測「Positve」→結果「誤り」
- FN(False Negative、偽陰性)：予測「Negative」→結果「誤り」

```
from sklearn.metrics import confusion_matrix

y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]

tp = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
tn = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))
fp = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
fn = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))

confusion_matrix1 = np.array([[tp, fp], [fn, tn]])

print(confusion_matrix1)
#
from sklearn.metrics import confusion_matrix...
[[3 1]
 [2 2]]

# scikit-learnのconfusion_metrixは、要素の配置が違うので注意
confusion_matrix2 = confusion_matrix(y_true, y_pred)
print(confusion_matrix2)
# [[2 1]
 [2 3]]
```
### 正答率(accuracy)と誤答率(error rate)
正答件数を全件数で割ったものが正答率。
accuracy = $ \frac{TP + TN}{TP+TN+FP+FN} 
$
error rate = 1 - accuracy

scikit-learn.metricsの _accuracy_score_ で計算可能。
```
from sklearn.metrics import accuracy_score

y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]

accuracy = accuracy_score(y_true, y_pred)
print(accuracy)
# 0.625
```

### 適合率(precision)と再現率(recall)
適合率(precision)は、予測が「Positive」の件数のうち、実際にPositiveだった割合。
再現率(recall)は、結果が「Positive」の件数のうち、正しく予測できた割合。
それぞれの値は0〜1の値を取り、1に近づくほど好成績。

presicion = $\frac{TP}{TP + FP}$
recall = $\frac{TP}{TP+FN} $

precisionとrecallはトレードオフの関係にあるので、ケースによってどちらを重視するかを考えること。
- 誤検知を少なくしたい → precisionを重視
- 正例を見逃したくない → recallを重視
recall重視は「オオカミ少年もやむ無し」という考え方。

### F1-scoreとF$\beta$-score
precisionとrecallの調和平均で計算されるのがF1-score。両者のバランスを取った指標。
F$\beta$-scoreは、F1-scoreからrecallをどれだけ重視するかを表す係数$\beta$によって調整した指標。
$F_1=\frac{2}{\frac{1}{recall} + \frac{1}{precision}}=\frac{2\cdot recall\cdot precison}{recall+precision}=\frac{2TP}{2TP+FP+FN} \\\\
f_\beta=\frac{(1+\beta^2)}{\frac{\beta^2}{recall}+\frac{1}{precision}}=\frac{(1+\beta^2)\cdot recall\cdot precision}{recall+\beta^2precision}
$
scikit-learn.metricsモジュールの _f1_score_ 、_fbeta_score_で計算可能。

### MCC(Matthews Correlation Coefficient)
-1から+1の範囲の値を取る。+1で完璧な予測、0でランダムな予測、-1で完全に反対の予測をしていることを表す。
$MCC=\frac{TP\times TN-FP\times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$

```
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

# accuracy:0.8
# f1:0.875
# mcc:0.375
# ----------
# accuracy:0.8
# f1:0.5
# mcc:0.375
```

### logloss
各レコードがTrueである確率を予測し、その確率を評価する。loglossの値が低いほうが良い。予測が正しくない(確率が高いと予測した例と実際の例が異なる)場合に、ペナルティが大きく与えられる。
$logloss=-\frac{1}{N}\sum^{N}_{i=1}(y_i\log p_i+(1-y_i)\log(1-p_i)) \\\\ 
=-\frac{1}{N}\sum^{N}_{i=1}\log p'_i \\\\
P'_i$は真の値を予測している確率。正例の場合は$p_i$、負例の場合は1-$p_i$。

```
from sklearn.metrics import log_loss

y_true = [1, 0, 1, 1, 0, 1]
y_prob = [0.1, 0.2, 0.8, 0.8, 0.1, 0.3]

logloss = log_loss(y_true, y_prob)
print(logloss)
# 0.7135581778200728
```
### AUC(Area Under the ROC Curve)
x軸を偽陽性率、y軸を真陽性率としてプロットして書かれた曲線(ROC曲線)の下部の面積。
- 偽陽性率：$\frac{FP}{(FP+TN)}$ 全体のNegativeケースのうち、誤ってPositiveと予測した割合。 
- 真陽性率：$\frac{TP}{(TP+FN)}$ 全体のPositiveケースのうち、正しくPositiveと予測できた割合。

完全な予測を行った場合、ROC曲線は(0.0, 0.0)から(0.0, 1.0)を経由して(1.0, 1.0)に至り、AUC=1.0となる。
ランダムな予測を行った場合、ROC曲線は(0.0, 0.0)と(1.0, 1.0)を結ぶ対角線に近くなり、AUC=0.5程度となる。

## 他クラス分類における評価
### multi-class accuracy
### multi-class logloss
```
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
```
### mean-F1、macro-F1、micro-F1
F1-scoreを多クラス分類に拡張したもの。
mean-F1：___レコード単位___ でF1-Scoreを算出し、それらの平均を取る。
macro-F1：___クラス単位___ でF1-Scoreを算出し、それらの平均を取る。
micro-F1：レコード×クラスの各組み合わせを(TP,TN,FP,FN)に当てはめ、その混同行列に基づいてF1-Scoreを算出する。
```
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
```
### quadratic weighted kappa
マルチクラス分類のうち、クラス間に順序があるようなケースに使用される(例：レーティング)。
$\kappa=1-\frac{\sum^{}_{i,j}w_{i,j}O_{i,j}}{\sum^{}_{i,j} w_{i,j} {E_{i,j}}}
$
$O_{i,j}$：真の値のクラスが$i$、予測値のクラスが$j$のレコード数。
$E_{i,j}$：$(i,j)$に属するレコード数の期待値。「真の値が$i$である割合$\times$ 予測値が$j$である割合 $\times$ 全体のレコード数」で計算される。
$E_{i,j}$：$(i-j)^2$。真の値と予測値が大きく離れたときに値が大きくなるので、予測を大きく外したときに大きなペナルティが課せられることになる。

表1.$O_{(i,j)}$
|真の値$i$／<br>予測値$j$|1|2|3|合計|
|:--:|--:|--:|--:|--:|
|1|10|5|5|20|
|2|5|35|0|40|
|3|15|0|25|40|
|合計|30|40|30|100|

表2.表1を元に算出した$E_{(i,j)}$
|真の値$i$／<br>予測値$j$|1|2|3|合計|
|:--:|--:|--:|--:|--:|
|1|6|8|6|20|
|2|12|16|12|40|
|3|12|16|12|40|
|合計|30|40|30|100|
$E_{(1,1)}=20 \times 30 \div 100 = 6 \\\\
E_{(2,2)}=40 \times 40 \div 100 = 16$

表3.$w_{(i,j)}$
|真の値$i$／<br>予測値$j$|1|2|3|
|:--:|--:|--:|--:|
|1|0|1|4|
|2|1|0|1|
|3|4|1|0|

$\sum w_{(i,j)}O_{(i,j)}=90 \\\\
\sum w_{(i,j)}E_{(i,j)}=120 \\\\$
から$\kappa=1- 90/120 = 0.25$となる。
```
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
```

## レコメンデーション
### MAP@K(Mean Average Precision @ K)
MAP@K$=\frac{1}{N}\sum ^{N}_{i=1}(\frac{1}{\min(m_{i},K)}\sum^{K}_{k=1}P_i(k)) $
$m_i$：レコード$i$の属しているクラスの数
$P_i(k)$：レコード$i$について$k(1\le k\le K)$番目までの予測値で計算されるPrecision。

- K=5
- 真の値のクラス：B,E,F
- 予測値のクラス：E,D,C,B,A (E→Aの順で高順位)

|予測順位$k$|予測値|正解/不正解|$P_i(k)$|
|:--:|:--:|:--:|:--:|
|1|E|○|1/1=1|
|2|D|×|ー|
|3|C|×|ー|
|4|B|○|2/4=0.5|
|5|A|×|ー|

$\sum P_i(k)=1.5$
$\min (m_{i},K)=3$ ＊真の値のクラス数$m_i$=3、予測可能な個数$K$=5
$MAP@5=1.5/3=0.5$
```
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
``` 

# 欠損値の扱い
## 欠損値のまま取り扱う
## 欠損値を代表値で埋める
平均値、中央値など。
平均値も全体の平均ではなく、カテゴリ変数ごとの平均を取る方法もある。
## 欠損値を他の変数から予測する
例）データ項目「年齢」に欠損が見られる場合
1. 学習データとテストデータを結合する。
1. 結合したデータを、年齢が入力されているデータ(A)と年齢が欠損しているデータ(B)に分ける。
1. (A)を学習データとして目的変数「年齢」のモデルを作成する。
1. 上記のモデルを使用して(B)の年齢を予測する。
1. 予測結果を元の学習データとテストデータに反映する。
## 欠損値から新たな特徴量を作成する。
- 欠損しているかの二値変数を作成する。こうすると、欠損を補完しても「欠損していた」という情報を残すことができる。
- レコードごとに欠損している変数をカウントする。
- 欠損している変数の組み合わせを調べる。もしその組み合わせにパターンがあれば、それを新たな特徴量とする。

## 何が欠損かを把握すること。
NaNだけが欠損じゃない。数値データでは「−1」や「9999」といった極端な値によって欠損を表すこともある。

複数の欠損がある場合は、事前に欠損値を指定することができる。
```
train = pd.read_csv('train.csv', na_values=['', 'NA', 1, 9999])
```
ただしこの場合は、すべての変数に対して欠損値の指定がされてしまう。特定の変数に対して欠損値の指定をしたい場合は、readした後に変換を行う。
```
# 列col1の値「−1」を欠損値に置き換える。
train['col1'] = train['col1'].replace(-1, np.nan)
```
# 数値変数の変換
## 標準化(standardization)
$x' = \frac{x-\mu}{\sigma}$
$\mu$：平均、$\sigma$：標準偏差
```
from sklearn.preprocessing import StandardScaler

# 標準化の定義
scaler = StandardScaler()
scaler.fit(train_x[num_cols])

# 変換後のデータで各列を置き換え
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])
```
標準化をするときに、学習データだけでなくテストデータも結合するのもあり。ただし、学習データとテストデータのそれぞれで平均と分散を計算し、それぞれのデータに対して置き換えることはダメ。
```
scaler.fit(pd.concat(train_x[num_cols], test_x[num_cols])
```

## Min-Maxスケーリング
変数の取る範囲を特定の区間(通常は0〜1)に収める方法。
$x'=\frac{x-x_{\min}}{x_{\max}-x_{\min}}
$
以下のようなデメリットがあるので、標準化が用いられることが多い。
- 変換後の平均がちょうど0にならない
- 外れ値の悪影響が大きく出る。

ただし、画像データの画素は値の範囲が0〜255と固定されているのでこちらを使うことも多い。

scikit-learnのMinMaxScalerを用いて計算が可能。
```
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(train_x[num_cols])

# 変換後のデータで各列を置き換え
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])
```

## 非線形変換
標準化、Min-Maxスケーリングは線形変換 __(変数分布の形状は変わらない)__
変数分布の裾が長いような場合、非線形変換を行って分布の形状を変えることが望ましい。
### 対数を取る
```
x = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])

x1 = np.log(x)
print(x1)
# [0.         2.30258509 4.60517019 6.90775528 9.21034037]
```
### log(x+1)による変換
値0が含まれるとそのまま対数を取ることができない。その場合はlog(x+1)とする。
numpyモジュールの _log1p_ 関数を使って変換できる。
```
x = np.array([0, 1.0, 10.0, 100.0, 1000.0, 10000.0])
# 0が含まれているので、普通に対数変換するとうまういかない。
x1 = np.log(x)
print(x)
# [0.e+00 1.e+00 1.e+01 1.e+02 1.e+03 1.e+04]

# そこで1を加えたあとに対数を取る。
x2 = np.log1p(x)
print(x2)
# [0.         0.69314718 2.39789527 4.61512052 6.90875478 9.21044037]
```
### 絶対値の対数を取る
負の値が含まれるとそのまま対数変換ができない。その場合は絶対値に対数変換をかけ、変換後の値に元の符号を付加することで対処する。
```
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
```

### Box-Cox変換とYeo-Johonson変換
#### Box-Cox変換
$x^{\lambda} = \begin{cases}
   \frac{x^{\lambda}-1}{\lambda} &\text{if } \lambda \not =0 \\
   \log x &\text{if } \lambda=0
\end{cases}
$
#### Yeo-Johnson変換
負の値が含まれていても使える。
$x^{\lambda} = \begin{cases}
   \frac{x^{\lambda}-1}{\lambda} &\text{if } \lambda \not =0,x_i \ge0 \\
   \log (x+1) &\text{if } \lambda=0, x_i \ge0 \\
   \frac{-|(-x+1)^{2-\lambda}-1|}{2-\lambda} &\text{if } \lambda \not =2, x_i <0 \\
   -\log(-x+1) &\text{if } \lambda=2, x_i<0
\end{cases}
$
```
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
```
### generalized log transformation
あまり使われないらしい。
$x^{(\lambda)}=\log(x+\sqrt{x^2+\lambda})
$

### その他
- 絶対値を取る
- 平方根を取る
- 二乗を取る、n乗を取る
- 二値変数を取る(正の値かどうか、ゼロかどうかなど)
- 数値の端数を取り除く
- 四捨五入、切り上げ、切り捨てなど

## clipping
外れ値を排除するのに有効な方法。
上限や下限を設定し、それを外れた値を上限/下限の値に置き換える。
pandasモジュールの _clip_ 関数で行うことができる。
```
# 学習データの1%点、99%点を算出
p01 = train_x[num_cols].quantile(0.01)
p99 = train_x[num_cols].quantile(0.99)

train_x[num_cols] = train_x[num_cols].clip(p01, p99, axis=1)
test_x[num_cols] = test_x[num_cols].clip(p01, p99, axis=1)
```

## binning
数値変数を区間でグループ分けをしてカテゴリ変数に変換する方法。
区間の設け方はいろいろ(等間隔、分位点、任意)。
binningをかけた直後は順序のあるカテゴリ変数となるので、そのまま使うもよし、one-hot encodingするもよし。
pandasモジュールの _cut_ 関数や、numpyモジュールの _digitize_ 関数で行うことができる。
```
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
```

## 順位への変換
数値の大小を排除し、大小関係のみを抽出する。
(例)店舗の日次来場者数は休日が突出してしまう。
pandasモジュールの _rank_ 関数で行うことができるし、numpyモジュールの _argsort_ 関数を2回適用することでもOK。
(argsort関数はソートされた値ではなくインデックスのndarrayを返す)
```
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
```

## RankGauss
数値変数を順位に変換した後、(半ば無理矢理に)正規分布となるように変換する手法。
```
from sklearn.preprocessing import QuantileTransformer

transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
transformer.fit(train_x[num_cols])

train_x[num_cols] = transformer.transform(train_x[num_cols])
test_x[num_cols] = transformer.transform(test_x[num_cols])
```

# カテゴリ変数の変換
どのように変換するかの前に、テストデータにのみ存在する水準(カテゴリ)の有無を確認すること。
仮にあった場合は、以下のいずれかの対応を取ること。
- そのカテゴリを持つレコード件数が少なく影響が微小である場合は無視しても良い。
- 最頻値で補完する。
- 欠損値とみなして、他の変数から予測して保管する。
- 平均といえるカテゴリを入れる。
## one-hot encoding
もっともオーソドックスな手法。
各水準を表す二値変数を作成する。

【変換前】
|カテゴリ|
|:--:|
|A|
|B|
|C|
|B|
|C|
|A|

【変換後】
|A|B|C|
|:--:|:--:|:--:|
|1|0|0|
|0|1|0|
|0|0|1|
|0|1|0|
|0|0|1|
|1|0|0|

pandasモジュールの _get_dummies_ 関数を使うと、引数に指定した列全てに対して one-hot encodingができる。
```
# 学習データとテストデータを結合してone-hot encodingを行う。
all_x = pd.concat([train_x, tets_x])
all_x = pd.get_dummies(all_x, columns=cat_cols)

# one-hot encoding実行後に、学習データとテストデータを再分割。
train_x = all_x.iloc[:train_x.shape[0], :].reset_index(drop=True)
test_x = all_x.iloc[train_x.shape[0]:, :].reset_index(drop=True)
```

## label encoding
各水準を単純に数値に置き換える。one-hot encodingのように新しいカテゴリ変数を作成しない。 __ordinal encoding__ とも呼ばれる。
決定木をベースにした手法以外ではあまり適切ではない。
【変換前】
|カテゴリ|
|:--:|
|A|
|B|
|C|
|B|
|C|
|A|

【変換後】
|カテゴリ|
|:--:|
|0|
|1|
|2|
|1|
|2|
|0|

## feature hashing
変換後の特徴量の数を予め指定し、ハッシュ関数を用いて水準ごとにフラグを立てる場所を決定する。
```
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

# 元のカテゴリ変数を削除
train_x.drop(cal_cols, axis=1, inplace=True)
test_x.drop(cal_cols, axis=1, inplace=True)
```

## frequency encoding
各水準の出現回数/頻度でカテゴリ変数を置き換える方法。
置き換える際は、事前に学習データとテストデータを結合しておくこと。
```
cat_x = pd.concat(train_x, test_x)

for c in cat_cols:
    freq = cat_x[c].value_counts()
    train_x[c] = train_x[c].map(freq)
    test_x[c] = test_x[c].map(freq)
```

## target encoding
目的変数を用いてカテゴリ変数を数値に変換する方法。
学習データを変換する際は、out-of-foldの方法で自身のレコードのカテゴリ変数を目的変数に含まないようにすること。また、テストデータの変換は学習データ全体の平均値を用いる。
```
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
```

## embedding
自然言語処理における単語やカテゴリ変数を実数ベクトルに変換する方法。
自然言語処理ではいくつかのモデルが公開されている __(Word2Vec, GloVe, fastTextなど)__ 
【変換前】
|カテゴリ|
|:--:|
|A1|
|A2|
|A3|
|B1|
|B2|
|A1|
|A2|
|A1|

【変換後】
||||
|:--:|:--:|:--:|
|0.200|0.469|0.019|
|0.115|0.343|0.711|
|0.240|0.514|0.991|
|0.760|0.002|0.444|
|0.603|0.128|0.949|
|0.200|0.469|0.019|
|0.115|0.343|0.711|
|0.200|0.469|0.019|

## 順序変数の扱い
順序変数とは、値の順序に意味があるが値同士の間隔に意味がない変数。[1位、2位、3位]など。
この場合、数値に置き換えてもいいしそのままカテゴリ変数として扱ってもいい。

## カテゴリ変数の値の意味を抽出する。
- 「ABC-9999」といった型番の場合、前半の英字3文字と後半の数字4文字に分割する。
- 数字と英字が混在している場合、数字が否かを特徴量にする。
- 水準間で文字数が異なる場合、文字数を特徴量にする。

# 日付・時刻を表す変数の変換
