import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 데이터 로드
file_path = 'final_data.csv'
data = pd.read_csv(file_path)

# 범죄 위험 점수 및 안전 시설 관련 변수 선택
features = data[['final_score_top_5', 'final_score_violence', 'final_score_theft',
                 'final_score_robbery', 'final_score_sexual_violence',
                 'final_score_women_night', 'nearest_police', 'nearest_hospital', 
                 'nearest_fire_station', 'nearest_convenience_store']]

# 'final_score_violence'의 결측치를 중앙값으로 대체
features['final_score_violence'].fillna(features['final_score_violence'].median(), inplace=True)

# 데이터 표준화
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 엘보우 방법을 위한 WCSS 값 계산
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)  # WCSS 값을 저장

# 엘보우 그래프 그리기
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()
