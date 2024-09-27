import requests
import numpy as np
import os
from PIL import Image, ImageChops
from io import BytesIO

api_index = {"범죄주의_전체": ["A2SM_CRMNLHSPOT_TOT", "A2SM_CrmnlHspot_Tot_Tot"],
             "범죄주의_강도": ["A2SM_CRMNLHSPOT_TOT", "A2SM_CrmnlHspot_Tot_Brglr"],
             "범죄주의_성폭력": ["A2SM_CRMNLHSPOT_TOT", "A2SM_CrmnlHspot_Tot_Rape"],
             "범죄주의_절도": ["A2SM_CRMNLHSPOT_TOT", "A2SM_CrmnlHspot_Tot_Theft"],
             "범죄주의_폭력": ["A2SM_CRMNLHSPOT_TOT", "A2SM_CrmnlHspot_Tot_Violn"],
             "여성밤길치안안전": ["A2SM_CRMNLHSPOT_F1_TOT", "A2SM_OdblrCrmnlHspot_Tot_20_24"],
             }
download_n = "all"

# WMS 요청 URL 템플릿
wms_url_template = "https://geo.safemap.go.kr/geoserver/safemap/wms?&request=GetMap&bbox={bbox}&format=image/png&version=1.1.1&exceptions=text/xml&transparent=TRUE&srs=EPSG:4326&service=WMS&layers={layers}&width={res}&styles={styles}&height={res}"

# 범위와 해상도 설정
lon_min, lon_max = 127.073, 127.080
lat_min, lat_max = 37.247, 37.251
step = 0.0005
res = 16

for i in api_index:

    # 폴더 경로 설정
    folder_path = i
    download_n = i

    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 좌표를 소수점 셋째 자리까지만 반올림하는 함수
    def format_coordinate(coord):
        return round(coord, 4)
    
    empty_image = Image.new('RGB', (1, 1), (255, 255, 255))  # 1x1 크기의 흰색 이미지

    # BBox 생성 및 이미지 다운로드
    for lon in np.arange(lon_min, lon_max, step):
        for lat in np.arange(lat_min, lat_max, step):
            lon = format_coordinate(lon)
            lat = format_coordinate(lat)
            bbox = f"{lon},{lat},{lon+step},{lat+step}"
            url = wms_url_template.format(bbox=bbox, res=res, layers=api_index[download_n][0], styles=api_index[download_n][1])

            # 요청 및 응답 처리
            response = requests.get(url)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))

                white_image = Image.new('RGB', image.size, (255, 255, 255))

                # 이미지가 흰색이 아닐 때만 저장
                if not ImageChops.difference(image, white_image).getbbox():
                    # print(f"Image at {lon}, {lat} is completely white, skipping save.")
                    empty_image.save(file_path, 'PNG')
                else:
                    # 이미지 저장
                    file_path = os.path.join(folder_path, f"{i}_{lon}_{lat}.png")
                    image.save(file_path, 'PNG')
                    # print(f"이미지 저장 완료: {file_path}")
                
            else:
                print(f"Failed to retrieve image for BBox: {bbox}, Status Code: {response.status_code}")

    print(f"{i} 전체 이미지 다운로드 완료")
