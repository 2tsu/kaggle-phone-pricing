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

        #drop na
        self.feature_vector = self.feature_vector.dropna()
        
        #delte outlier with isolatuon forest
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(contamination=0.1, random_state=42)
        yhat = iso.fit_predict(self.feature_vector)
        mask = yhat != -1
        self.feature_vector_inliers = self.feature_vector[mask]
        
        self.data = self.feature_vector_inliers.drop(columns=['Price'])
        self.target = self.feature_vector_inliers['Price']

    #欠損値の処理
    def setup_features(self, df):
        """特徴量エンジニアリングを行う."""
        
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
        
        #他の不要カラムを削除
        df = df.drop(columns=['Unnamed: 0', 'Name', 'Android_version', 'Processor_name'])
        
        #型変換
        bool_columns = df.select_dtypes(include=['bool']).columns
        df[bool_columns] = df[bool_columns].astype(int)
        
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
