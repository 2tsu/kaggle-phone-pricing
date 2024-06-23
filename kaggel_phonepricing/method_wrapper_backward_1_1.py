"""
重回帰
p-value > 0.05 の特徴量を削除するラッパー法
予測価格帯を10,0000以下にする
"""
from kaggel_phonepricing.features import DataSet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

c = DataSet()

# 元のデータフレーム
df = pd.concat([c.data, c.target.rename('Price')], axis=1)

# 価格が10,000以上の行を除外
df_filtered = df[df['Price'] < 100000]

# 説明変数と目的変数に再分割
X = df_filtered.drop('Price', axis=1)
y = df_filtered['Price']

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X))


# インデックスを揃える
X.index = X.index
y.index = X.index

#p-value selection
# 後退除去法の定義
def backward_selection(X_train, y_train, c):
    cols = X_train.columns.tolist()
    while len(cols) > 0:
        X_ = sm.add_constant(X_train[cols])
        lr = sm.OLS(y_train, X_).fit()
        # p-valuesの取得
        p = lr.pvalues[1:]  # 定数項を除外
        # p-value > significance_level の特徴量を削除
        if p.max() > c:
            cols.remove(p.idxmax())
        else:
            break
    return cols

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

selected_cols = backward_selection(X_train, y_train, c=0.05)
print(selected_cols, len(selected_cols))

# モデルの作成
model = LinearRegression()

# 学習
model.fit(X_train, y_train)

# 交差検証
cv_scores = cross_val_score(model, X[selected_cols], y, cv=5)

print("SUMMARY")
print("R2 train: ", np.round(model.score(X_train, y_train), 3))
print("R2 test: ", np.round(model.score(X_test, y_test), 3))
print("Cross-validation scores: ", np.round(cv_scores, 3))
print("average score: ", np.round(np.mean(cv_scores), 3))
print("number of features: ", len(selected_cols))
# 予測
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 残差プロットの作成
plt.figure(figsize=(10, 6))
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=min(y_train_pred), xmax=max(y_train_pred), color='red')
plt.title('Residuals Plot')
plt.show()

# 実際の値 vs 予測値プロットの作成
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, c='blue', marker='o', label='Training data')
plt.scatter(y_test, y_test_pred, c='lightgreen', marker='s', label='Test data')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.legend(loc='upper left')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', lw=2)  # 45度の直線
plt.title('Actual vs Predicted Values')
plt.show()


# 選択された特徴量でヒートマップの作成
plt.figure(figsize=(12, 8))
sns.heatmap(X_train[selected_cols].corr(), cmap="RdBu", annot=True, fmt=".2f")
plt.title("Heatmap of Selected Features Correlations")
plt.show()

# ヒストグラムの作成
X_train[selected_cols].hist(bins=30, figsize=(15, 10))
plt.suptitle("Histogram of Selected Features", y=1.02)
plt.show()

# 相関行列ヒートマップの作成
corr_matrix = X_train[selected_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix Heatmap of Selected Features")
plt.show()
