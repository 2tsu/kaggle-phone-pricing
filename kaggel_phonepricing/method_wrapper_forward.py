"""
p-value > 0.05 の特徴量を削除する前進選択ラッパー法
"""


from kaggel_phonepricing.features import DataSet
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import pandas as pd
import numpy as np  
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
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


# 前進選択の定義
def forward_selection(X, y, significance_level=0.05):
    initial_features = X.columns.tolist()
    best_features = []
    while len(initial_features) > 0:
        remaining_features = list(set(initial_features) - set(best_features))
        p_values = pd.Series(index=remaining_features)
        for feature in remaining_features:
            model = sm.OLS(y, sm.add_constant(X[best_features + [feature]])).fit()
            p_values[feature] = model.pvalues[feature]
        min_p_value = p_values.min()
        if min_p_value < significance_level:
            best_features.append(p_values.idxmin())
        else:
            break
    return best_features

# 前進選択を実行
selected_features = forward_selection(X, y)

print("Selected features:", selected_features)

#split data
X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)

# モデルの作成
model = LinearRegression()
# 学習
model.fit(X_train, y_train)

# 交差検証
cv_scores = cross_val_score(model, X[selected_features], y, cv=5)

print("SUMMARY: ")
print("R2 train:", np.round(model.score(X_train,y_train), 3))
print("R2 test:", np.round(model.score(X_test,y_test), 3))
print("Cross-validation scores:", np.round(cv_scores, 3))
print("average score:", np.round(np.mean(cv_scores), 3))
print("number of features:", len(X_train.columns))
