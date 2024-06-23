from kaggel_phonepricing.features import DataSet, eval_score, train
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import cross_val_score



c = DataSet()

X = c.data
y = c.target

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#columns name
feature_names = X.columns
len(feature_names)

#p-value selection

# 後退除去法の定義
def backward_selection(cols, c):
    while len(cols) > 0:
        X_ = sm.add_constant(X_train[cols])
        lr = sm.OLS(y_train, X_).fit()
        # p-valuesの取得
        p = lr.pvalues[1:]
        # p-value > c の特徴量を削除
        if max(p) > c:
            cols = cols.drop(p.idxmax())
        else:
            break
    return cols

selected_cols = backward_selection(feature_names, c=0.05)
print(selected_cols, len(selected_cols))

model = LinearRegression()
# 学習（学習データ利用）
model.fit(X_train[selected_cols], y_train)
# 精度検証（決定係数R2）
print('決定係数R2（学習データ）:', 
      model.score(X_train[selected_cols],y_train))
print('決定係数R2（テストデータ）:', 
      model.score(X_test[selected_cols],y_test))

# Cross validation
# 選択された特徴量を用いたデータフレームの作成
X_selected = X[selected_cols]
cv_scores = cross_val_score(model, X_selected, y, cv=5)
print('Cross-validation scores:', cv_scores)

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

X_shuffled = X_selected.sample(frac=1, random_state=42)
y_shuffled = y.sample(frac=1, random_state=42)

scores = cross_val_score(model, X_shuffled, y_shuffled, cv=kf)
print('Cross-validation scores:', scores)



