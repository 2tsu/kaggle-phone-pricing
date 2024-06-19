from kaggel_phonepricing.features import DataSet, eval_score, train
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import pandas as pd
c = DataSet()

c.feature_vector.info()

c.drop_columns = ['Unnamed: 0', 'Name', 'Android_version', 'Processor_name']

c.feature_vector = c.feature_vector.drop(columns=c.drop_columns)

#drop nan
c.feature_vector = c.feature_vector.dropna()

bool_columns = c.feature_vector.select_dtypes(include=['bool']).columns
c.feature_vector[bool_columns] = c.feature_vector[bool_columns].astype(int)

c.feature_vector.info()
#Wrapper method 
#split data
X = c.feature_vector.drop(columns=['Price'])
y = c.feature_vector['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Isolation Forestを用いて外れ値を検出
iso = IsolationForest(contamination=0.1, random_state=42)
yhat = iso.fit_predict(X)

# 正常なデータと異常なデータを分離
mask = yhat != -1
X_inliers, y_inliers = X[mask], y[mask]
X_outliers, y_outliers = X[~mask], y[~mask]

# 外れ値除去後のデータセットを表示
print("Inliers shape:", X_inliers.shape)
print("Outliers shape:", X_outliers.shape)

# データを再分割
X_train, X_test, y_train, y_test = train_test_split(X_inliers, y_inliers, test_size=0.2, random_state=42)

#columns name
feature_names = X_train.columns
len(feature_names)

#p-value selection

# 後退除去法の定義
def backward_selection(cols, c):
    while len(cols) > 0:
        X_ = sm.add_constant(X_train[cols])
        lr = sm.OLS(y_train, X_).fit()
        # p-valuesの取得
        p = pd.Series(lr.pvalues.values[:], index=cols)  # 定数項を除外
        # p-value > c の特徴量を削除
        if max(p) > c:
            cols = cols.drop(p.idxmax())
        else:
            break
    return cols

X_selected = backward_selection(feature_names, c=0.05)
print(X_selected)

model = LinearRegression()
# 学習（学習データ利用）
model.fit(X_train[X_selected], y_train)
# 精度検証（決定係数R2）
print('決定係数R2（学習データ）:', 
      model.score(X_train[X_selected],y_train))
print('決定係数R2（テストデータ）:', 
      model.score(X_test[X_selected],y_test))

# Cross validation
from sklearn.model_selection import cross_val_score
# 選択された特徴量を用いたデータフレームの作成
X_selected = X[X_selected]
cv_scores = cross_val_score(model, X_selected, y, cv=5)
print('Cross-validation scores:', cv_scores)