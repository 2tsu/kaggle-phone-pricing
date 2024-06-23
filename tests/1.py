import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_csv('./data/data.csv')

df.head()

df.columns

for col in df.columns:
    print(df[col].head())
    
    
    

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

# 変換後のデータを確認
print(df[['Ram']].head())

# 'Battery' カラムから数値を抽出
df['Battery'] = df['Battery'].str.extract('(\d+)').astype(int)

# 'Display' カラムから数値を抽出
df['Display'] = df['Display'].str.extract('(\d+\.\d+)').astype(float)


def extract_camera_info(camera_str):
    rear_cameras = re.findall(r'(\d+) MP', camera_str.split('&')[0])
    front_camera = re.search(r'(\d+) MP Front', camera_str)
    
    max_rear_mp = max([int(mp) for mp in rear_cameras], default=0)
    front_mp = int(front_camera.group(1)) if front_camera else 0
    
    return max_rear_mp, front_mp

# 'Camera' カラムから背面と前面カメラのMPを抽出
df[['Max_Rear_Camera_MP', 'Front_Camera_MP']] = df['Camera'].apply(lambda x: pd.Series(extract_camera_info(x)))

#元データ消去
df = df.drop(columns=['Camera'])


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

# 'External_Memory' カラムから外部メモリのサポート情報と最大容量を抽出
df[['External_Memory_Supported', 'External_Memory_Max_Capacity']] = df['External_Memory'].apply(lambda x: pd.Series(extract_external_memory_info(x)))

# 変換後のデータを確認
print(df[['External_Memory_Supported', 'External_Memory_Max_Capacity']].head())

#元データ消去
df = df.drop(columns=['External_Memory'])

# 'company' カラムのワンホットエンコーディング
df = pd.get_dummies(df, columns=['company'])

df['Inbuilt_memory'] = df['Inbuilt_memory'].str.extract('(\d+)').fillna(0).astype(int)

# 'fast_charging' をバイナリ変換
df['fast_charging'] = df['fast_charging'].str.extract('(\d+)').fillna(0).astype(int)

# 'Screen_resolution' から幅と高さを抽出し、NaN を 0 で埋める
df['Screen_width'] = df['Screen_resolution'].str.extract('(\d+) x \d+').fillna(0).astype(int)
df['Screen_height'] = df['Screen_resolution'].str.extract('\d+ x (\d+)').fillna(0).astype(int)

# 元の 'Screen_resolution' カラムを削除
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
df['Price'] = df['Price'].str.replace('[^\d.]', '').astype(float)



df.info()


# 'Price' カラムを数値に変換
df['Price'] = df['Price'].str.replace(',', '').str.extract('(\d+\.?\d*)').astype(float)

# 欠損値を持つ行を削除
df = df.dropna()

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
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# 特徴量の重要性を確認
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])

# 特徴量の重要性を棒グラフで表示
plt.figure(figsize=(10, 8))
coefficients['Coefficient'].plot(kind='bar')
plt.title('Feature Importance (Coefficient Values)')
plt.xlabel('Features')
plt.ylabel('Coefficient')
plt.savefig('out/feature_importance.png')
plt.show()

# 特徴量の分散を箱ひげ図で表示
plt.figure(figsize=(12, 10))
X.boxplot(rot=90)
plt.title('Feature Variance')
plt.ylabel('Value')
plt.savefig("./out/feature_variance.png")
plt.show()