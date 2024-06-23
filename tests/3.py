from kaggel_phonepricing.features import DataSet, eval_score, train


c = DataSet()

c.feature_vector.info()

c.drop_columns = ['Unnamed: 0', 'Name', 'Android_version', 'Processor_name']

c.feature_vector = c.feature_vector.drop(columns=c.drop_columns)

c.feature_vector.info()

#drop nan
c.feature_vector = c.feature_vector.dropna()

#Wrapper method 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

#split data
X = c.feature_vector.drop(columns=['Price'])
y = c.feature_vector['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#columns name
feature_names = X_train.columns

#forward selection

def forward_selection(X_train, y_train, X_test, y_test):
    initial_features = []
    best_features = initial_features.copy()
    best_score = float('inf')

    while True:
        scores_with_candidates = []
        for feature in X_train.columns:
            if feature not in initial_features:
                candidate_features = initial_features + [feature]
                X_train_sm = sm.add_constant(X_train[candidate_features])
                model = sm.OLS(y_train, X_train_sm).fit()
                y_pred = model.predict(sm.add_constant(X_test[candidate_features]))
                score = mean_squared_error(y_test, y_pred)
                scores_with_candidates.append((score, feature))

        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates[0]

        if best_new_score < best_score:
            best_score = best_new_score
            best_features.append(best_candidate)
            initial_features.append(best_candidate)
        else:
            break

    return best_features
# 最適な特徴量の選択
selected_features = forward_selection(X_train, y_train, X_test, y_test)
print("Selected features:", selected_features)