import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 데이터 불러오기
file_path = 'final_data_nized.csv'  # 파일 경로를 여기에 입력
data = pd.read_csv(file_path)

# 1. PCA 가중치 계산
X_with_top5 = data.drop(columns=['latitude', 'longitude'])  # latitude와 longitude 제외한 모든 변수 사용
scaler = StandardScaler()
scaled_data_with_top5 = scaler.fit_transform(X_with_top5)

# PCA 모델 생성 및 학습
pca_with_top5 = PCA(n_components=2)
pca_with_top5.fit(scaled_data_with_top5)

# PCA 가중치 계산
pca_components_with_top5 = pca_with_top5.components_
explained_variance_with_top5 = pca_with_top5.explained_variance_ratio_
pca_weights_with_top5 = abs(pca_components_with_top5.T @ explained_variance_with_top5)

# 가중치 결과를 데이터프레임으로 정리
pca_weight_df_with_top5 = pd.DataFrame({
    'Feature': X_with_top5.columns,
    'PCA Weight': pca_weights_with_top5
}).sort_values(by='PCA Weight', ascending=False)

print(pca_weight_df_with_top5)

data['pred_score'] = 0

# pca_weight_df_with_top5에 있는 Feature와 PCA Weight에 따라 가중치를 적용하여 pred_score 계산
for i, row in pca_weight_df_with_top5.iterrows():
    feature = row['Feature']  # 변수 이름
    weight = row['PCA Weight']  # 가중치
    data['pred_score'] += data[feature] * weight  # 각 변수에 가중치를 곱해 합산

data.to_csv("final_data_pred.csv", index=False)
print(min(data['pred_score']), max(data['pred_score']))