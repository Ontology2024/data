import matplotlib.colors as mcolors
import folium
from folium.plugins import HeatMap
import pandas as pd

# 파일 경로 지정
file_path = 'freeze_data.csv'

# 데이터 불러오기
data = pd.read_csv(file_path)

# 중심 좌표 설정 (데이터의 평균 위도와 경도로 설정)
map_center = [data['latitude'].mean(), data['longitude'].mean()]

# 지도 생성
mymap = folium.Map(location=map_center, zoom_start=12)

# 데이터에서 필요한 값들 추출 (위도, 경도, normalized_est_score)
heat_data = [
    [row['latitude'] + 0.005, row['longitude'] + 0.005, row['normalized_est_score']] for index, row in data.iterrows()
]

# HeatMap 레이어 추가
HeatMap(heat_data, radius=15,blur=10, max_zoom=13).add_to(mymap)

map1_path = "heat_map.html"
#mymap.save(map1_path)


# 위험도에 따른 색상 맵핑 설정 (낮은 위험도: 녹색, 높은 위험도: 붉은색)
colormap = mcolors.LinearSegmentedColormap.from_list("risk_color_map", ["green", "yellow", "red"])

# 지도 생성
mymap2 = folium.Map(location=map_center, zoom_start=12)

# 각 좌표에 대해 0.001의 범위를 사각형으로 나타내며, 위험도에 따라 색상을 설정
for index, row in data.iterrows():
    # 각 지점의 위험도
    risk = row['normalized_est_score']
    
    # 사각형 범위 설정 (위도와 경도에서 0.001만큼 확장)
    bounds = [
        [row['latitude'], row['longitude']], 
        [row['latitude'] + 0.001, row['longitude'] + 0.001]
    ]
    
    # 위험도에 따라 색상 결정
    color = mcolors.to_hex(colormap(risk / data['normalized_est_score'].max()))
    
    # 사각형 추가
    folium.Rectangle(
        bounds,
        color=color,
        fill=True,
        fill_opacity=0.1,
        weight=0
    ).add_to(mymap2)

# 지도를 HTML 파일로 저장
map2_path = "sq_map.html"
mymap2.save(map2_path)
