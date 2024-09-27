import folium
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

# 파일 경로 지정
file_path = 'freeze_data.csv'

# 데이터 불러오기
data = pd.read_csv(file_path)

# 중심 좌표 설정 (데이터의 평균 위도와 경도로 설정)
map_center = [data['latitude'].mean(), data['longitude'].mean()]

# 위험도에 따른 색상 맵핑 설정 (낮은 위험도: 녹색, 중간: 노란색, 높은 위험도: 붉은색)
colormap = mcolors.LinearSegmentedColormap.from_list("risk_color_map", ["blue", "green", "yellow", "orange", "red"])

# 지도 생성
mymap2 = folium.Map(location=map_center, zoom_start=12)

# 각 좌표에 대해 0.001의 범위를 원형으로 나타내며, 위험도에 따라 색상을 설정
for index, row in data.iterrows():
    # 각 지점의 위험도
    risk = row['normalized_est_score']
    
    # 위험도에 따라 색상 결정 (초록-노랑-빨강 그라데이션 적용)
    norm_risk = risk / data['normalized_est_score'].max()  # 정규화
    color = mcolors.to_hex(colormap(norm_risk))  # 색상 결정
    
    # 절대적인 반지름(미터 단위)을 사용하여 원 추가 (Circle 사용)
    folium.Circle(
        location=[row['latitude']+0.0005, row['longitude']+0.0005],
        radius=60,  # 절대적인 반경 크기 (예: 100미터)
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.1,  # 투명도 설정
        weight=0
    ).add_to(mymap2)

# 지도를 HTML 파일로 저장
map2_path = "circle_abs_map.html"
mymap2.save(map2_path)
