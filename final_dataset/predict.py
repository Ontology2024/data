import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

# 데이터 불러오기
file_path = 'freeze_data.csv'
data = pd.read_csv(file_path)

import json

def is_safe(latitude, longitude):
    # 위치에 가까운 데이터를 찾기 위해 특정 위도, 경도와의 차이를 계산하여 가장 가까운 점을 찾습니다.
    data['distance'] = ((data['latitude'] - latitude)**2 + (data['longitude'] - longitude)**2)**0.5
    nearest_location = data.loc[data['distance'].idxmin()]
    
    # 범죄 위험도가 높은 범죄 목록과 해당 위험 점수 추출
    crime_risk = {
        "theft": nearest_location['final_score_theft'],
        "robbery": nearest_location['final_score_robbery'],
        "sexual_violence": nearest_location['final_score_sexual_violence'],
        "violence": nearest_location['final_score_violence'],
        "women_night": nearest_location['final_score_women_night']
    }
    
    # 위험한 범죄가 있는지 여부를 판단 (점수가 0보다 큰 경우 위험하다고 간주)
    dangerous_crimes = [{"crime_type": crime, "risk_score": score} for crime, score in crime_risk.items() if score > 0]
    notable_crimes = [{"crime_type": crime, "risk_score": score} for crime, score in crime_risk.items() if score > 2]

    # crime_kor = ["강도", "절도", "성폭행", "폭력", "여성밤치안"]
 
    # for name, i in enumerate(notable_crimes):
    #     comment += crime_kor
    # if notable_crimes is not None:
    #     comment = ""
    
    # 안전 여부를 판단 (가장 가까운 안전 시설이 0.5 km 이내인 경우 안전)
    safety_facilities = {
        "police": nearest_location['nearest_police'],
        "hospital": nearest_location['nearest_hospital'],
        "fire_station": nearest_location['nearest_fire_station'],
        "convenience_store": nearest_location['nearest_convenience_store']
    }
    
    safe = all([distance <= 0.5 for distance in safety_facilities.values()])
    
    # JSON 형식의 결과 반환
    result = {
        "latitude": latitude,
        "longitude": longitude,
        "safety_level": nearest_location['normalized_est_score'],
        "dangerous_crimes": dangerous_crimes if dangerous_crimes else "None",
        "near_safety": "safe" if safe else "No safety fac.",
        "safety_facilities": safety_facilities
    }
    
    return json.dumps(result, indent=4)

# 예시 함수 호출:
print(is_safe(37.169, 126.964))

## {
##     "latitude": float,   # 위도
##     "longitude": float,  # 경도
##     "normalized_est_score": int,  # 정규화된 안전 점수 (예: 0 - 매우 안전, 5 - 매우 위험)
##     "dangerous_crimes": list,  # 위험한 범죄와 해당 위험 점수 목록
##     "safety_reason": str,  # 안전한 이유 또는 안전하지 않은 이유
##     "safety_facilities": {  # 근처의 안전 시설 정보 (경찰서, 병원, 소방서, 편의점과의 거리 km)
##         "police": float,
##         "hospital": float,
##         "fire_station": float,
##         "convenience_store": float
##     }
## }