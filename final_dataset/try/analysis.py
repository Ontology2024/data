import pandas as pd
from geopy.distance import geodesic
from scipy.spatial import distance
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
import os
from tqdm import tqdm

# 모델 및 거리 계산 결과 저장 경로
MULTI_OUTPUT_MODEL_PATH = 'xgb_multioutput_model_with_safety.pkl'
CALCULATED_DISTANCES_PATH = 'calculated_distances.csv'

# 데이터 파일 경로
DATA_ROBBERY_PATH = 'data_robbery.csv'
DATA_SEXUAL_VIOLENCE_PATH = 'data_sexual_violence.csv'
DATA_THEFT_PATH = 'data_theft.csv'
DATA_TOP_5_CRIMES_PATH = 'data_top_5_crimes.csv'
DATA_VIOLENCE_PATH = 'data_violence.csv'
POLICE_PATH = 'police.csv'
HOSPITAL_PATH = 'hospital.csv'
FIRE_STATION_PATH = 'fire_station.csv'
CONVENIENCE_STORE_PATH = 'convenience_store.csv'
WOMEN_NIGHT_SAFETY_PATH = 'data_women_night_safety.csv'

# 데이터 로딩
print("Loading datasets...")
df_robbery = pd.read_csv(DATA_ROBBERY_PATH)
df_sexual_violence = pd.read_csv(DATA_SEXUAL_VIOLENCE_PATH)
df_theft = pd.read_csv(DATA_THEFT_PATH)
df_top_5_crimes = pd.read_csv(DATA_TOP_5_CRIMES_PATH)
df_violence = pd.read_csv(DATA_VIOLENCE_PATH)
df_women_night_safty = pd.read_csv(WOMEN_NIGHT_SAFETY_PATH)

# 안전 시설 데이터 로딩
df_police = pd.read_csv(POLICE_PATH)
df_hospital = pd.read_csv(HOSPITAL_PATH)
df_fire_station = pd.read_csv(FIRE_STATION_PATH)
df_convenience_store = pd.read_csv(CONVENIENCE_STORE_PATH)

# 데이터 병합 (latitude와 longitude를 기준으로 inner join)
print("Merging crime datasets...")
# 데이터를 손실하지 않으려면 outer join 사용
df_merged = pd.merge(df_top_5_crimes[['latitude','longitude','final_score']], df_violence[['latitude','longitude','final_score']], 
                     on=['latitude', 'longitude'], how='outer', suffixes=('_top_5', '_violence'))
df_merged = pd.merge(df_merged, df_theft[['latitude','longitude','final_score']], 
                     on=['latitude', 'longitude'], how='outer', suffixes=('', '_theft'))
df_merged = pd.merge(df_merged, df_robbery[['latitude','longitude','final_score']], 
                     on=['latitude', 'longitude'], how='outer', suffixes=('', '_robbery'))
df_merged = pd.merge(df_merged, df_sexual_violence[['latitude','longitude','final_score']], 
                     on=['latitude', 'longitude'], how='outer', suffixes=('', '_sexual_violence'))
df_merged = pd.merge(df_merged, df_women_night_safty[['latitude','longitude','final_score']], 
                     on=['latitude', 'longitude'], how='outer', suffixes=('', '_women_night'))


# 거리 계산 함수 (벡터화)
def calculate_nearest_facility_vectorized(lat_lons, facilities):
    return distance.cdist(lat_lons, facilities, metric='euclidean').min(axis=1)

# 거리 계산 및 저장
if not os.path.exists(CALCULATED_DISTANCES_PATH):
    print("Calculating nearest facilities distances...")
    
    crime_lat_lons = df_merged[['latitude', 'longitude']].values
    police_lat_lons = df_police[['latitude', 'longitude']].values
    hospital_lat_lons = df_hospital[['latitude', 'longitude']].values
    fire_station_lat_lons = df_fire_station[['latitude', 'longitude']].values
    convenience_store_lat_lons = df_convenience_store[['latitude', 'longitude']].values
    
    # 경찰서 거리 계산
    print("Calculating distances to police stations...")
    df_merged['nearest_police'] = calculate_nearest_facility_vectorized(crime_lat_lons, police_lat_lons)
    
    # 병원 거리 계산
    print("Calculating distances to hospitals...")
    df_merged['nearest_hospital'] = calculate_nearest_facility_vectorized(crime_lat_lons, hospital_lat_lons)
    
    # 소방서 거리 계산
    print("Calculating distances to fire stations...")
    df_merged['nearest_fire_station'] = calculate_nearest_facility_vectorized(crime_lat_lons, fire_station_lat_lons)

    # 편의점 거리 계산
    print("Calculating distances to convenience stores...")
    df_merged['nearest_convenience_store'] = calculate_nearest_facility_vectorized(crime_lat_lons, convenience_store_lat_lons)
    
    # 계산된 거리 데이터를 저장
    df_merged.to_csv(CALCULATED_DISTANCES_PATH, index=False)
    print("Distance calculations complete. Saved to file.")
else:
    print("Loading pre-calculated distances...")
    df_merged = pd.read_csv(CALCULATED_DISTANCES_PATH)

# 결측치 처리 (적절한 값으로 채우기, 예: 0)
print("Handling missing data...")
df_merged.fillna(0, inplace=True)
