import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.cluster.hierarchy as sch
import seaborn as sns
import matplotlib.pyplot as plt
def extract_camera_info(camera_str):
    rear_cameras = re.findall(r'(\d+) MP', camera_str.split('&')[0])
    front_camera = re.search(r'(\d+) MP Front', camera_str)
    
    max_rear_mp = max([int(mp) for mp in rear_cameras], default=0)
    front_mp = int(front_camera.group(1)) if front_camera else 0
    
    return max_rear_mp, front_mp


def extract_external_memory_info(memory_str):
    if 'Memory Card Supported' in memory_str:
        supported = 1
        max_capacity_match = re.search(r'(\d+)\s*(TB|GB)', memory_str)
        if max_capacity_match:
            max_capacity = int(max_capacity_match.group(1))
            if max_capacity_match.group(2) == 'TB':
                max_capacity *= 1024  # Convert TB to GB
        else:
            max_capacity = 0
    else:
        supported = 0
        max_capacity = 0
    return supported, max_capacity


class DataSet:
    def __init__(self):
        
        self.df = pd.read_csv('./data/data.csv')
        self.feature_vector = self.setup_features(self.df)

    #欠損値の処理
    def setup_features(self, df):
        # 'No_of_sim' カラムから各技術のサポート情報を抽出
        df['Dual_Sim'] = df['No_of_sim'].apply(lambda x: 1 if 'Dual Sim' in x else 0)
        df['3G'] = df['No_of_sim'].apply(lambda x: 1 if '3G' in x else 0)
        df['4G'] = df['No_of_sim'].apply(lambda x: 1 if '4G' in x else 0)
        df['5G'] = df['No_of_sim'].apply(lambda x: 1 if '5G' in x else 0)
        df['VoLTE'] = df['No_of_sim'].apply(lambda x: 1 if 'VoLTE' in x else 0)
        # 元の 'No_of_sim' カラムを削除
        df = df.drop(columns=['No_of_sim'])
        # 'Ram' カラムから数値を抽出
        df['Ram'] = df['Ram'].str.extract('(\d+)').astype(int)
        
        # 'Battery' カラムから数値を抽出
        df['Battery'] = df['Battery'].str.extract('(\d+)').astype(int)
        
        # 'Display' カラムから数値を抽出
        df['Display'] = df['Display'].str.extract('(\d+\.\d+)').astype(float)
        
        # 'Camera' カラムから背面と前面カメラのMPを抽出
        df[['Max_Rear_Camera_MP', 'Front_Camera_MP']] = df['Camera'].apply(lambda x: pd.Series(extract_camera_info(x)))
        #元データ消去
        df = df.drop(columns=['Camera'])
        
        # 'External_Memory' カラムから外部メモリのサポート情報と最大容量を抽出
        df[['External_Memory_Supported', 'External_Memory_Max_Capacity']] = df['External_Memory'].apply(lambda x: pd.Series(extract_external_memory_info(x)))
        #元データ消去
        df = df.drop(columns=['External_Memory'])
        
        # 'company' カラムのワンホットエンコーディング
        df = pd.get_dummies(df, columns=['company'])
        
        # 'Inbuilt_memory' カラムから数値を抽出
        df['Inbuilt_memory'] = df['Inbuilt_memory'].str.extract('(\d+)').fillna(0).astype(int)
        
        # 'fast_charging' をバイナリ変換
        df['fast_charging'] = df['fast_charging'].str.extract('(\d+)').fillna(0).astype(int)
        
        # 'Screen_resolution' から幅と高さを抽出し、
        #　新しいcolumnとして幅と高さの積を画面面積として追加
        # 'Screen_resolution' 列から幅と高さを抽出する関数を定義
        def extract_dimensions(resolution):
            match = re.search(r'(\d+)\s*x\s*(\d+)', resolution)
            if match:
                width, height = match.groups()
                return int(width), int(height)
            else:
                return None,  None      # 幅と高さを新しい列に追加
        df[['Screen_width', 'Screen_height']] = df['Screen_resolution'].fillna("").apply(lambda x: pd.Series(extract_dimensions(x)))

        # 幅と高さの積を計算して 'Screen_area' 列を追加
        df['Screen_area'] = df['Screen_width'] * df['Screen_height'] 
        # 元の 'Screen_resolution' 列を削除
        df = df.drop(columns=['Screen_resolution'])
        
        # 'Processor' からクロック速度を抽出し、NaN を 0 で埋める
        df['Processor_speed_GHz'] = df['Processor'].str.extract('(\d+\.?\d*) GHz').astype(float)
        
        # 'Processor' カラムからクロック速度がない場合の処理
        # 'Octa Core' の場合は任意でクロック速度を設定 (例えば 2.0 GHz)
        def set_default_speed(row):
            if pd.isna(row['Processor_speed_GHz']):
                if isinstance(row['Processor'], str) and 'Octa Core' in row['Processor']:
                    return 2.0
                else:
                    return 0
            else:
                return row['Processor_speed_GHz']
        
        df['Processor_speed_GHz'] = df.apply(set_default_speed, axis=1)
        # 元の 'Processor' カラムを削除
        df = df.drop(columns=['Processor'])
        
        # 'Price' カラムを数値に変換
        df['Price'] = df['Price'].str.replace(',', '').str.extract('(\d+\.?\d*)').astype(float)
        
        return df

def eval_score(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def train(df):
    # 特徴量とターゲットの分割
    X = df.drop(columns=['Price'])
    y = df['Price']

    # データのトレーニングセットとテストセットへの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # モデルのインスタンス作成とトレーニング
    model = LinearRegression()
    model.fit(X_train, y_train)

    # テストセットで予測
    y_pred = model.predict(X_test)
    
    return y_test, y_pred, model


a = DataSet()
a.feature_vector.info()
fv = a.feature_vector
fv.drop(columns=['Unnamed: 0', 'Name', 'Android_version', 'Processor_name'], inplace=True)
fv.columns

# 各列の欠損値の数と割合をサマライズ
nan_summary = fv.isna().sum().to_frame('NaN Count')
nan_summary['NaN Percentage'] = (fv.isna().mean() * 100).to_frame('NaN Percentage')

# NaNが含まれる列のみ表示
nan_summary = nan_summary[nan_summary['NaN Count'] > 0]

print(nan_summary)
# 欠損値を含む行を削除
fv = fv.dropna()

fv = fv.drop(columns=["4G"])
drop_company = ["company_Coolpad","company_IQOO","company_Itel","company_Gionee" , "company_LG"]
fv =fv.drop(columns=drop_company)

X = fv.drop(columns=['Price'])
y = fv['Price']
y_test, y_pred, model = train(fv)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
eval_score( y_test, y_pred)

if __name__ == "__main__":
    #0.805061556872748
    data = DataSet()
    df = data.feature_vector
    
    # 欠損値を持つ行を削除
    df = df.dropna()
    # 分位数の計算（PRICE列の95%分位数を取得）
    price_threshold = fv['Price'].quantile(0.95)

    # 外れ値の除外（PRICE列が95%分位数を超えるデータを除外）
    fv = fv[fv['Price'] <= price_threshold]

    #意味のない列を削除
    df = df.drop(columns=["4G"])
    drop_company = ["company_Coolpad","company_IQOO","company_Itel","company_Gionee" , "company_LG"]
    df = df.drop(columns=drop_company)
    # 特徴量とターゲットの分割
    X = df.drop(columns=['Price', 'Unnamed: 0', 'Name', 'Android_version', 'Processor_name'])
    y = df['Price']

    # データのトレーニングセットとテストセットへの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # モデルのインスタンス作成とトレーニング
    model = LinearRegression()
    model.fit(X_train, y_train)

    # テストセットで予測
    y_pred = model.predict(X_test)
    
    # モデルの評価
    mse, r2 = eval_score(y_test, y_pred)

    # 訓練データのスコア
    train_score = model.score(X_train, y_train)

    # テストデータのスコア
    test_score = model.score(X_test, y_test)

    print(f'Train R2 score: {train_score}')
    print(f'Test R2 score: {test_score}')


    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f'Cross-validation scores: {cv_scores}')
    print(f'Mean cross-validation score: {cv_scores.mean()}')

    # 特徴量の重要性を棒グラフで表示
    # 特徴量の重要性を確認

    # Create DataFrame
    coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    # Create a barplot with seaborn
    sns.barplot(x=coefficients.index, y=coefficients['Coefficient'])

    # Rotate x-labels for better readability
    plt.xticks(rotation=90, fontsize=12)

    # Set the title and labels
    plt.title('Feature Importance (Coefficient Values)', fontsize=20)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Coefficient', fontsize=15)

    # Save the figure in high quality
    plt.savefig('./out/feature_importance.png', dpi=300)

    # Show the plot
    #plt.show()

    # 特徴量の分散を箱ひげ図で表示
    plt.figure(figsize=(12, 10))
    X.boxplot(rot=90)
    plt.title('Feature Variance')
    plt.ylabel('Value')
    plt.savefig("./out/feature_variance.png")
    #plt.show()
    
    residuals = y_test - y_pred
    # 残差プロットの作成
    plt.figure(figsize=(10, 6))
    sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 2})
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig("./out/residual_plot.png")
    plt.show()
    
    # 予測値と実測値の散布図を作成
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # y=x の直線
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.savefig("./out/actual_vs_predicted_prices.png")
    plt.show()

    # 相関行列の計算
    corr = fv.corr()

    # 相関行列の距離行列を計算
    dist = 1 - corr

    # 階層型クラスタリングを実行
    linkage = sch.linkage(dist, method='ward')

    # デンドログラムの表示
    plt.figure(figsize=(10, 7))
    sch.dendrogram(linkage, labels=corr.columns, leaf_rotation=90)
    plt.title('Dendrogram')
    plt.xlabel('Features')
    plt.ylabel('Euclidean distances')
    plt.savefig("./out/dendrogram.png")
    plt.show()

    # クラスタリング結果に基づいて特徴量を並び替える
    dendro = sch.dendrogram(linkage, labels=corr.columns, no_plot=True)
    ordered_columns = dendro['ivl']

    # 並び替えた相関行列のヒートマップを表示
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr.loc[ordered_columns, ordered_columns], annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f", annot_kws={"size": 7}, cbar_kws={"shrink": .8})
    plt.title('Clustered Correlation Matrix')
    plt.savefig("./out/clustered_correlation_matrix.png")
    plt.show()

    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
