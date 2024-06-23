""""
WARNING UNDER DEVL
重回帰
ランダムフォレストを用いて特徴量を削除するラッパー法
開発中
"""

from kaggel_phonepricing.features import DataSet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np


c = DataSet()

X = c.data
y = c.target


#columns name
feature_names = X.columns
len(feature_names)


# 後退除去法の定義
def backward_selection(X_train, y_train, min_features=1):
    cols = X_train.columns.tolist()
    while len(cols) > min_features:
        # ランダムフォレストモデルの訓練
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train[cols], y_train)
        
        # 特徴量の重要度を取得
        importances = pd.Series(rf.feature_importances_, index=cols)
        
        # 重要度が最も低い特徴量を削除
        least_important = importances.idxmin()
        cols.remove(least_important)
        
        # 交差検証スコアの計算
        scores = cross_val_score(rf, X_train[cols], y_train, cv=5)
        print(f"Removed {least_important}, CV Score: {scores.mean()}")
    
    return cols

# 後退選択を実行
selected_features = backward_selection(X, y)

print("Selected features:", selected_features)

#split data
X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)

# モデルの作成
model = LinearRegression()
# 学習
model.fit(X_train[selected_features], y_train)

# 交差検証
cv_scores = cross_val_score(model, X[selected_features], y, cv=5)

print("SUMMARY: ")
print("R2 train:", np.round(model.score(X_train[selected_features], y_train), 3))
print("R2 test:", np.round(model.score(X_test[selected_features], y_test), 3))
print("Cross-validation scores:", np.round(cv_scores, 3))