from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

file_path = 'final_data.csv'  # 파일 경로를 여기에 입력
data = pd.read_csv(file_path)

# 정규화할 컬럼들 선택
columns_to_normalize = [
    'final_score_top_5', 'final_score_violence', 'final_score_theft', 
    'final_score_robbery', 'final_score_sexual_violence', 'final_score_women_night',
]

columns_to_normalize2 = [
    'nearest_police', 'nearest_hospital', 'nearest_fire_station', 'nearest_convenience_store'
]

# MinMaxScaler를 사용해 0~5로 정규화
scaler = MinMaxScaler(feature_range=(0, 5))

data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
data[columns_to_normalize2] = np.round(scaler.fit_transform(data[columns_to_normalize2]), 3)

# 정규화된 데이터의 일부 확인
print(data.head())
data.to_csv("final_data_n.csv", index=False)

score_columns = ['final_score_top_5', 'final_score_violence', 'final_score_theft', 
                 'final_score_robbery', 'final_score_sexual_violence', 'final_score_women_night']

data.fillna(0, inplace=True)

# Step 1: Define the algorithm to calculate est_score based on the provided rules
def calculate_est_score(row):
    # Sort scores in descending order
    sorted_scores = sorted([row[col] for col in score_columns], reverse=True)
    
    # Apply weights: 1, 0.3, 0.2, 0.1, 0, 0
    weights = [1, 0.3, 0.2, 0.1, 0, 0]
    weighted_score = sum([s * w for s, w in zip(sorted_scores, weights)])
    
    # Calculate the distance-related score
    total_distance = (row['nearest_police'] + row['nearest_hospital'] +
                      row['nearest_fire_station'] + row['nearest_convenience_store'])
    distance_score = (total_distance / 20) * 0.5
    
    # Final estimated score
    est_score = round(weighted_score + distance_score, 3)
    return est_score

# Apply the est_score calculation function to the dataset
data['est_score'] = data.apply(calculate_est_score, axis=1)

# Step 2: Calculate the quantile thresholds for normalization
updated_thresholds = data['est_score'].quantile([0.97, 0.92, 0.85, 0.80, 0.75])

# Step 3: Define the normalization function
def threshold_normalization(score):
    if score >= updated_thresholds[0.97]:
        return 5
    elif score >= updated_thresholds[0.92]:
        return 4
    elif score >= updated_thresholds[0.85]:
        return 3
    elif score >= updated_thresholds[0.80]:
        return 2
    elif score >= updated_thresholds[0.75]:
        return 1
    else:
        return 0

# Apply the normalization function to the est_score column
data['normalized_est_score'] = data['est_score'].apply(threshold_normalization)

import matplotlib.pyplot as plt

# Step 4: Visualize the normalized_est_score on a scatter plot based on latitude and longitude
plt.figure(figsize=(10, 6))

# Create scatter plot with color representing the normalized_est_score
scatter = plt.scatter(data['longitude'], data['latitude'], c=data['normalized_est_score'], cmap='viridis', alpha=0.7)

# Add color bar for reference
plt.colorbar(scatter, label='Normalized Estimated Score')

# Label the plot
plt.title('Normalized Estimated Score by Latitude and Longitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Display the plot
plt.show()

data[columns_to_normalize] = np.round(scaler.fit_transform(data[columns_to_normalize]),0)

data.to_csv("freeze_data.csv", index=False)