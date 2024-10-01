import pandas as pd
import requests
import json

with open('config.json') as f:
    config = json.load(f)

# API 키 입력
api_key = config['google_api_key']

# 검색할 장소 유형과 추가 가능한 서브 유형을 지정
place_types = {
    '파출서': ['지구대', '경찰서', '112', '파출소', 'police'],
    '편의점': ['편의점','convenience_store', 'grocery_or_supermarket', 'food'],
    '병원': ['병원','hospital', 'doctor', 'clinic', 'health'],
    '소방서': ['소방서', '안전센터','119', 'fire_station', 'fire_department'],
    # CCTV는 구글 맵스에서 직접적인 타입이 없으므로 대체 데이터 사용 필요
}

# 검색할 범위 지정 (위도 및 경도)
lon_min, lat_min = 126.964, 37.169
lon_max, lat_max = 127.103, 37.290

# 검색할 위치 중심점과 반경 설정
location = '37.229,127.033'  # 중심점 (평균값 또는 임의의 중심점 설정)
radius = 20000  # 15km 반경 설정

# 각 장소 유형별로 데이터를 수집하는 함수
def fetch_places(place_types):
    places = []
    for place_type in place_types:
        url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location}&radius={radius}&name={place_type}&key={api_key}"
        response = requests.get(url)
        data = response.json()

        if 'results' in data:
            for result in data['results']:
                lat = result['geometry']['location']['lat']
                lon = result['geometry']['location']['lng']
                
                # 지정된 범위 안에 있는지 확인
                if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                    places.append({
                        'name': result['name'],
                        'latitude': lat,
                        'longitude': lon,
                        'type': place_type  # 장소 유형 기록
                    })
    return places

# 장소별로 데이터 수집 후 엑셀 파일로 저장
for place_name, place_type_list in place_types.items():
    all_places = []
    
    # 각 장소별 서브 유형까지 검색
    for place_type in place_type_list:
        places_data = fetch_places([place_type])
        all_places.extend(places_data)
    
    # 데이터프레임 생성
    df = pd.DataFrame(all_places)
    
    # 엑셀 파일로 저장
    file_name = f"{place_name}.csv"
    df.to_csv(file_name, index=False)
    ## df.('convenience_stores.csv', index=False, encoding='utf-8')
    print(f"{place_name} 데이터가 {file_name}로 저장되었습니다.")