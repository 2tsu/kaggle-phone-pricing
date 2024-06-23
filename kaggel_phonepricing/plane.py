"""
plane estimation
R2 train: 0.772
R2 test: 0.752
Cross-validation scores: [0.764 0.616 0.542 0.645 0.606]
average score: 0.649
number of features: 46
"""

from kaggel_phonepricing.features import DataSet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


c = DataSet()

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(c.data))
y = c.target

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# モデルの作成
model = LinearRegression()
# 学習
model.fit(X_train, y_train)


# 交差検証
cv_scores = cross_val_score(model, X, y, cv=5)

print("SUMMARY: ")
print("R2 train:", np.round(model.score(X_train,y_train), 3))
print("R2 test:", np.round(model.score(X_test,y_test), 3))
print("Cross-validation scores:", np.round(cv_scores, 3))
print("average score:", np.round(np.mean(cv_scores), 3))
print("number of features:", len(X.columns))

#write summary
with open("./out/plane/summary.txt", "w") as f:
    f.write("R2 train: " + str(np.round(model.score(X_train,y_train), 3)) + "\n")
    f.write("R2 test: " + str(np.round(model.score(X_test,y_test), 3)) + "\n")
    f.write("Cross-validation scores: " + str(np.round(cv_scores, 3)) + "\n")
    f.write("average score: " + str(np.round(np.mean(cv_scores), 3)) + "\n")
    f.write("number of features: " + str(len(X.columns)) + "\n")

"""
# 散布図行列
sns.pairplot(X_train)
plt.suptitle("Scatterplot Matrix", y=1.02)
plt.show()
"""

# ヒストグラム
X_train.hist(bins=30, figsize=(15, 10))
plt.suptitle("Histogram of Features", y=1.02)
plt.savefig("./out/plane/histogram.png")
plt.show()


# 相関行列ヒートマップ
corr_matrix = X_train.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix Heatmap")
plt.savefig("./out/plane/heatmap.png")
plt.show()


# ボックスプロット
plt.figure(figsize=(15, 10))
X_train[X.columns.tolist()].boxplot()
plt.title("Boxplot of Selected Features")
plt.xticks(ticks=range(1, len(X.columns.tolist()) + 1), rotation=90)
plt.savefig("./out/plane/boxplot.png")
plt.show()

