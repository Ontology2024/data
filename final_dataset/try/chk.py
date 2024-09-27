import json
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# 데이터 로드
data = pd.read_csv('final_data.csv')

class MultiOutputCrimeModel(nn.Module):
    def __init__(self, input_size):
        super(MultiOutputCrimeModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)

        # 범죄별로 6개의 범주(0~5)를 예측하는 출력층 정의
        self.output = nn.Linear(64, 6)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        output = self.output(x)  # 다중 클래스 출력을 위한 최종 출력
        return output

# 저장된 모델을 불러오기 위한 함수 정의
def load_model(model_path, input_size, device):
    # 모델 구조 정의
    model = MultiOutputCrimeModel(input_size=input_size)
    
    # 저장된 가중치 로드 (weights_only=True 설정)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # 모델을 GPU/CPU로 이동
    model.to(device)
    model.eval()  # 평가 모드로 전환
    
    return model

# 예측 및 결과를 JSON으로 반환하는 함수 정의
def predict_danger_level_and_reason(latitude, longitude, model, scaler, device):
    # 주어진 위도, 경도에 대한 정보를 데이터에서 찾기
    location_data = data[
        (abs(data['latitude'] - latitude) < 0.0001) & 
        (abs(data['longitude'] - longitude) < 0.0001)
    ]
    
    if location_data.empty:
        return json.dumps({"error": f"Data not found for latitude {latitude} and longitude {longitude}."})
    
    # 입력 데이터 준비
    X_input = location_data.drop(columns=['final_score_top_5'])  # 예측에 사용될 입력 변수들

    # 결측치 처리
    X_input = np.nan_to_num(X_input)

    # 입력 데이터를 스케일링 (훈련 데이터에서 사용된 스케일러로 스케일링)
    X_input_scaled = scaler.transform(X_input)

    # 텐서로 변환
    X_input_tensor = torch.tensor(X_input_scaled, dtype=torch.float32).to(device)

    # 모델 예측
    with torch.no_grad():
        output = model(X_input_tensor)

    # 위험도를 각각의 범죄 유형별로 예측된 결과로 변환
    crime_types = ['robbery', 'sexual_violence', 'theft', 'violence', 'other_crime']  # 예시 범죄 유형
    danger_levels = torch.softmax(output, dim=1).cpu().numpy()

    # 범죄 유형별 위험도 계산 (각 범죄에 대해 위험도가 1 이상일 경우 위험하다고 판단)
    dangerous_crimes = []
    for i, crime_type in enumerate(crime_types):
        if danger_levels[0][i] >= 1:  # 위험도 임계값을 1로 설정
            dangerous_crimes.append(crime_type)

    # 전체 위험도를 기반으로 위치의 위험 여부 판단
    total_danger_level = float(np.max(danger_levels))  # 가장 높은 위험도 선택하고 float으로 변환
    if total_danger_level >= 1:
        crime_status = "dangerous"
    else:
        crime_status = "safe"

    # 안전 이유 계산 (근처 안전 시설이 있는지 확인)
    safety_reasons = {}
    safety_facilities = ['nearest_police', 'nearest_hospital', 'nearest_fire_station']
    for facility in safety_facilities:
        distance = float(location_data[facility].values[0])  # float으로 변환
        if distance <= 0.5:  # 0.5km 이내에 있으면 안전하다고 판단
            safety_reasons[facility] = f"{facility.replace('nearest_', '').capitalize()} is within {distance} km"

    # JSON 결과 반환
    result = {
        "latitude": latitude,
        "longitude": longitude,
        "total_danger_level": total_danger_level,
        "crime_status": crime_status,
        "dangerous_crimes": dangerous_crimes if crime_status == "dangerous" else "None",
        "safety_reasons": safety_reasons if crime_status == "safe" else "Not applicable",
    }

    return json.dumps(result, indent=4)

# GPU/CPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 스케일러 로드 (훈련할 때 사용한 스케일러로 스케일링해야 함)
scaler = StandardScaler()
X = data.drop(columns=['final_score_top_5'])  # 스케일링할 원본 데이터
scaler.fit(X)  # 스케일러에 맞게 학습

# 저장된 모델 불러오기
model_path = "multi_output_crime_model.pth"
input_size = X.shape[1]  # 입력 크기
model = load_model(model_path, input_size, device)


# 예시 좌표를 사용해 위험도 예측하기
latitude = 37.242  # 예시 위도
longitude = 126.9685  # 예시 경도

# 예측 수행 및 JSON 결과 출력
result_json = predict_danger_level_and_reason(latitude, longitude, model, scaler, device)
print(result_json)