"""
重回帰
相関係数を用いたフィルタ法による特徴量選択
R2 train: 0.744
R2 test: 0.734
Cross-validation scores: [0.745 0.721 0.573 0.624 0.584]
average score: 0.649
number of features: 22
"""
from kaggel_phonepricing.features import DataSet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


c = DataSet()

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(c.data))
y = c.target

# インデックスを揃える
X.index = c.data.index
y.index = c.data.index


#columns name
feature_names = X.columns
len(feature_names)

# 各特徴量とターゲット変数の相関係数を計算
corrs = X.apply(lambda col: col.corr(y))

# 標準偏差がゼロでない特徴量のみを対象にする
non_zero_std_features = X.loc[:, X.std() != 0].columns.tolist()
corrs = corrs[non_zero_std_features]
# NaNを除外する
corrs = corrs.dropna()
threshold = 0.1
#フィルタ法を実行
selected_features = corrs[abs(corrs) >= threshold].index.tolist()


print("Selected features:", selected_features)

#split data
X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)

# 選択された特徴量でモデルを訓練
model = LinearRegression()
model.fit(X_train, y_train)


# 交差検証
cv_scores = cross_val_score(model, X[selected_features], y, cv=5)

print("SUMMARY: ")
print("R2 train:", np.round(model.score(X_train,y_train), 3))
print("R2 test:", np.round(model.score(X_test,y_test), 3))
print("Cross-validation scores:", np.round(cv_scores, 3))
print("average score:", np.round(np.mean(cv_scores), 3))
print("number of features:", len(selected_features))