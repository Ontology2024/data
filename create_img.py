import os
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# 폴더 경로 설정
folder_path = 'bandal_map'

# 이미지 크기
image_size = 1024  # WMS 요청 시 사용한 이미지 크기와 일치해야 함

# 파일명을 기반으로 이미지 좌표를 추출하는 함수
def extract_coordinates_from_filename(filename):
    match = re.search(r'image_(\d+\.\d+)_(\d+\.\d+)\.png', filename)
    if match:
        lon = float(match.group(1))
        lat = float(match.group(2))
        return lon, lat
    return None, None

# 폴더에서 좌표 추출 및 범위 결정
def determine_bounds_from_images(folder_path):
    lons = []
    lats = []
    
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    for image_file in image_files:
        lon, lat = extract_coordinates_from_filename(image_file)
        if lon is not None and lat is not None:
            lons.append(lon)
            lats.append(lat)
    
    if lons and lats:
        lon_min = min(lons)
        lon_max = max(lons)
        lat_min = min(lats)
        lat_max = max(lats)
        return lon_min, lon_max, lat_min, lat_max
    else:
        raise ValueError("No valid images found in the folder.")

# 이미지 시각화
def plot_images_on_map(folder_path):
    # 범위 결정
    lon_min, lon_max, lat_min, lat_max = determine_bounds_from_images(folder_path)
    
    # 좌표에 따른 이미지 크기 및 범위 설정
    width = lon_max - lon_min
    height = lat_max - lat_min

    # 플롯 설정
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 폴더 내 이미지 파일 리스트
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    for image_file in image_files:
        lon, lat = extract_coordinates_from_filename(image_file)
        if lon is not None and lat is not None:
            # 이미지 위치 설정
            x = (lon - lon_min) / width
            y = (lat - lat_min) / height
            
            # 이미지 열기
            img = Image.open(os.path.join(folder_path, image_file))
            
            # 이미지를 플롯에 추가
            ax.imshow(img, extent=[lon, lon + 0.001, lat, lat + 0.001])
    
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Map with Overlaid Images')
    
    plt.show()

# 이미지 시각화 호출
plot_images_on_map(folder_path)
