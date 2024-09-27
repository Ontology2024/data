import os
import re
import csv
from PIL import Image
import numpy as np

# 폴더 경로 목록
folder__index = ["범죄주의_전체", "범죄주의_강도", "범죄주의_성폭력", "범죄주의_절도", "범죄주의_폭력", "노후건물정보", "어린이대상범죄주의구간 ", "여성밤길치안안전"]
step = 0.0005  # 위도, 경도 스텝

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

# 위험도를 계산하는 함수
def rate(score):
    score_arr = [300, 200, 100, 70, 50, 0]
    if score == 0:
        return 0
    for i in score_arr:
        if score > i:
            return 5 - score_arr.index(i)

# 파일명을 기반으로 이미지 좌표를 추출하는 함수
def extract_coordinates_from_filename(filename, folder_path):
    # 정규표현식을 수정하여 소수점 자릿수에 상관없이 매칭되도록 변경
    match = re.search(r'_(\d+\.\d+?)_(\d+\.\d+?)\.png', filename)
    if match:
        lon = float(match.group(1))
        lat = float(match.group(2))
        return lon, lat
    print(f"Filename pattern did not match: {filename}")  # 디버깅용 메시지
    return None, None

# 폴더 내 이미지 처리 및 CSV 파일 생성 함수 (디버깅 추가)
def create_csv_from_images(folder_path, csv_filename):
    # CSV 파일 열기
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        # CSV 헤더 작성
        csv_writer.writerow(['folder', 'latitude', 'longitude', 'danger_rate', 'red_pixel_count', 'score', 'final_score'])

        # 폴더 내 이미지 파일 리스트
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

        if not image_files:
            print(f"No PNG images found in folder: {folder_path}")  # 디버깅용 메시지
            return

        for image_file in image_files:
            lon, lat = extract_coordinates_from_filename(image_file, folder_path)
            if lon is not None and lat is not None:
                # 이미지 경로
                image_path = os.path.join(folder_path, image_file)
                
                # 붉은 픽셀 수와 붉은 정도 점수 계산
                red_pixel_count, score, red_pixel_avg = calculate_redness_score(image_path)
                final_score = round(score * red_pixel_count / 100, 0)
                danger_rate = rate(final_score)
                
                # CSV에 정보 저장
                csv_writer.writerow([folder_path, lat, lon, danger_rate, red_pixel_count, score, final_score])
            else:
                print(f"Could not extract coordinates from filename: {image_file}")  # 디버깅용 메시지

    print(f"CSV 파일 '{csv_filename}' 생성 완료.")

# 모든 폴더에 대해 CSV 파일 생성
def create_csv_for_all_folders(base_path, folder__index):
    for folder in folder__index:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            csv_filename = f'image_data_{folder}.csv'
            create_csv_from_images(folder_path, csv_filename)
        else:
            print(f"폴더 {folder}이 존재하지 않습니다.")

# 호출 예시
base_path = './'  # 실제 이미지 폴더가 저장된 경로로 변경
create_csv_for_all_folders(base_path, folder__index)
