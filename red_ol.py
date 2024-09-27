import os
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, FuncFormatter
from PIL import Image
import numpy as np
import matplotlib.font_manager as fm

# 한글 또는 UTF-8 텍스트를 표시할 수 있는 폰트 설정
plt.rc('font', family='Malgun Gothic') 

# 폴더 경로 설정
folder__index = ["범죄주의_전체", "범죄주의_강도", "범죄주의_성폭력", "범죄주의_절도", "범죄주의_폭력", "노후건물정보", "어린이대상범죄주의구간 ", "여성밤길치안안전"]
folder_path = folder__index[3]
step = 0.0005

def format_func(value, tick_number):
    return f'{value:.4f}'

def rate(score):
    score_arr = [300, 200, 100, 70, 50, 0]
    if score == 0:
        return 0
    for i in score_arr:
        if score > i:
            return 5 - score_arr.index(i)

# 파일명을 기반으로 이미지 좌표를 추출하는 함수
def extract_coordinates_from_filename(filename):
    match = re.search(folder_path + r'_(\d+\.\d+)_(\d+\.\d+)\.png', filename)
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

# 붉은 픽셀 점수 계산 함수
def calculate_redness_score(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    width, height = img.size
    
    total_score = 0
    red_pixel_count = 0
    red_pixel_arr = []

    for y in range(height):
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            if r > g and r > b:
                score = (r ** 2 + (255*2 - b - g) ** 2) / 1000
                total_score += score
                red_pixel_count += 1
                red_pixel_arr.append(r)
    average_score = total_score / red_pixel_count if red_pixel_count > 0 else 0
    return red_pixel_count, average_score, np.mean(red_pixel_arr)

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
            x = lon
            y = lat
            extent = [x, x + step, y, y + step]
            
            # 이미지 경로
            image_path = os.path.join(folder_path, image_file)
            
            # 붉은 픽셀 수와 붉은 정도 점수 계산
            red_pixel_count, score, red_pixel_avg = calculate_redness_score(image_path)
            final_score = round(score * red_pixel_count / 100, 0)
            danger_rate = rate(final_score)
            
            # 이미지를 플롯에 추가
            img = Image.open(image_path)
            ax.imshow(img, extent=extent, alpha=0.5)
            
            if (1):
                # 점수 오버레이 추가
                text = f'RP: {red_pixel_count}\nES: {score:.2f}\n FS: {final_score: .2f}'
                text = f'{danger_rate}'
                ax.text(x + step/2, y + step/2, text, fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # x축 및 y축 보조선 설정
    ax.set_xlim(lon_min, lon_max + step)
    ax.set_ylim(lat_min, lat_max + step)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Map with Safety Scores of {folder_path}')

    # 보조선 추가
    x_ticks = np.arange(lon_min, lon_max + 0.001, step)
    y_ticks = np.arange(lat_min, lat_max + 0.001, step)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    ax.xaxis.set_major_formatter(FuncFormatter(format_func))
    ax.yaxis.set_major_formatter(FuncFormatter(format_func))

    plt.show()

# 이미지 시각화 호출
plot_images_on_map(folder_path)
