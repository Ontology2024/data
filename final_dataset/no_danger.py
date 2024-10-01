import numpy as np
import heapq
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# 움직일 수 있는 방향 케이스 (상하좌우, 대각선)
diagonal_directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

# 다익스트라 기반 경로 탐색 함수
def is_valid(coord, grid):
    x, y = coord
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

def find_path_avoid_danger_with_diagonal(df, start, end):
    try:
        pq = [(df[(df['Latitude'] == start[0]) & (df['Longitude'] == start[1])]['Danger Level'].values[0], start[0], start[1])]
    except:
        return [], -1
    
    visited = set()
    risks = np.full((df['Latitude'].max() + 1, df['Longitude'].max() + 1), np.inf)
    risks[start] = df[(df['Latitude'] == start[0]) & (df['Longitude'] == start[1])]['Danger Level'].values[0]
    parents = {start: None}

    while pq:
        current_risk, x, y = heapq.heappop(pq)
        if (x, y) == end:
            break
        
        if (x, y) in visited:
            continue
        
        visited.add((x, y))

        # 주변 탐색
        for dx, dy in diagonal_directions:
            nx, ny = x + dx, y + dy
            if is_valid((nx, ny), risks) and (nx, ny) not in visited:
                new_risk = current_risk + df[(df['Latitude'] == nx) & (df['Longitude'] == ny)]['Danger Level'].values[0]
                if new_risk < risks[nx, ny]:
                    risks[nx, ny] = new_risk
                    parents[(nx, ny)] = (x, y)
                    heapq.heappush(pq, (new_risk, nx, ny))

    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parents.get(current)
    
    path.reverse()
    return path, risks[end]/len(path)


# 체크포인트 생성 함수
def create_checkpoints_on_direction_change(path):
    checkpoints = [path[0]]  # 첫번째 경로부터 시작
    prev_direction = None

    for i in range(1, len(path) - 1):
        # 현재 방향 확인
        current_direction = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])

        # 방향이 바뀌면 체크포인트 추가
        if prev_direction is None or current_direction != prev_direction:
            checkpoints.append(path[i])
        
        prev_direction = current_direction

    checkpoints.append(path[-1])

    return checkpoints

# 위험한 경로를 지나는지 확인하는 함수
def is_safe_path(df, point1, point2, danger_threshold, debug=False):
    x0, y0 = point1
    x1, y1 = point2
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    start_time = time.time()  # 성능 디버깅을 위한 시간 측정 시작

    while (x0, y0) != (x1, y1):
        # 현재 좌표 및 위험 레벨을 출력 (디버깅)
        if debug:
            print(f"Checking point: ({x0}, {y0})")
        
        try:
            danger_value = df[(df['Latitude'] == x0) & (df['Longitude'] == y0)]['Danger Level'].values[0]
        except IndexError:
            print(f"Point ({x0}, {y0}) not found in the dataset.")
            return False  # 좌표가 데이터프레임에 없으면 경로가 안전하지 않다고 판단
        
        if danger_value >= danger_threshold:
            if debug:
                print(f"Dangerous point: ({x0}, {y0}) with danger level {danger_value}")
            return False  # 경로가 위험할 경우
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    # 마지막 지점 확인
    try:
        danger_value = df[(df['Latitude'] == x1) & (df['Longitude'] == y1)]['Danger Level'].values[0]
    except IndexError:
        print(f"Final point ({x1}, {y1}) not found in the dataset.")
        return False
    
    if debug:
        print(f"Final point: ({x1}, {y1}), danger level: {danger_value}")
    
    end_time = time.time()  # 성능 디버깅을 위한 시간 측정 종료
    if debug:
        print(f"is_safe_path took {end_time - start_time:.2f} seconds.")

    return danger_value < danger_threshold

def optimize_checkpoints(grid, checkpoints, danger_threshold, debug=False):
    optimized_checkpoints = [checkpoints[0]]  # 첫 번째 체크포인트는 무조건 포함
    
    if debug:
        print(f"Starting optimization with {len(checkpoints)} checkpoints")

    i = 0
    start_time = time.time()  # 전체 함수 성능 측정을 위한 시작 시간

    while i < len(checkpoints) - 1:
        j = i + 1
        while j < len(checkpoints) and is_safe_path(grid, checkpoints[i], checkpoints[j], danger_threshold, debug=False):
            j += 1
        
        # 안전한 경로가 아닌 첫 번째 체크포인트 바로 전까지 포함
        optimized_checkpoints.append(checkpoints[j - 1])
        if debug:
            print(f"Added checkpoint: {checkpoints[j - 1]}")

        i = j + 1
    
    end_time = time.time()  # 전체 함수 성능 측정을 위한 종료 시간
    if debug:
        print(f"optimize_checkpoints took {end_time - start_time:.2f} seconds.")
    
    return optimized_checkpoints


# 경로 시각화 함수
def visualize_path_with_checkpoints(df, path, checkpoints):
    fig, ax = plt.subplots(figsize=(10, 10))

    danger_grid = df.pivot(index='Latitude', columns='Longitude', values='Danger Level').values
    im = ax.imshow(danger_grid, cmap='YlOrRd')

    # 경로 표시
    path_coords = list(zip(*path))
    ax.plot(path_coords[1], path_coords[0], color='blue', marker='o', linestyle='-', linewidth=2, markersize=5)

    # 체크포인트 표시
    checkpoints_coords = list(zip(*checkpoints))
    ax.scatter(checkpoints_coords[1], checkpoints_coords[0], color='red', marker='x', s=100, label='Checkpoints')

    # 위험도 레벨 표시
    for i, row in df.iterrows():
        ax.text(row['Longitude'], row['Latitude'], int(row['Danger Level']), ha='center', va='center', color='black')

    ax.set_title('Optimized Path with Checkpoints')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Danger Level')

    plt.legend()
    plt.show()

# 체크포인트 시각화 함수
def visualize_checkpoints_with_labels_and_lines(df, checkpoints):
    fig, ax = plt.subplots(figsize=(10, 10))

    danger_grid = df.pivot(index='Latitude', columns='Longitude', values='Danger Level').values
    im = ax.imshow(danger_grid, cmap='YlOrRd')

    checkpoints_coords = [(row['Latitude'], row['Longitude']) for idx, row in df.iterrows() if (row['Latitude'], row['Longitude']) in checkpoints]

    # 체크포인트 표시
    ax.scatter([y for _, y in checkpoints_coords], [x for x, _ in checkpoints_coords], color='green', marker='o', s=100, label='Checkpoints')

    for idx, (x, y) in enumerate(checkpoints_coords):
        ax.text(y, x, str(idx + 1), ha='center', va='center', color='black', fontsize=12)

    # 체크포인트 연결 라인 그리기
    ax.plot([y for _, y in checkpoints_coords], [x for x, _ in checkpoints_coords], color='blue', marker='', linestyle='-', linewidth=2, label='Path between Checkpoints')

    ax.set_title('Checkpoints with Labels and Lines')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Danger Level')

    plt.legend()
    plt.show()

# 가상 케이스 제작
np.random.seed(random.randrange(0,10000))
large_danger_map = np.random.rand(100, 100)

for _ in range(20):
    x, y = np.random.randint(0, 100, 2)
    large_danger_map[x-5:x+5, y-5:y+5] += np.random.randint(3, 6)

large_danger_map = gaussian_filter(large_danger_map, sigma=3)
large_danger_map = np.clip((large_danger_map / large_danger_map.max()) * 5, 0, 5).astype(int)

# 가상 케이스 데이타프레임으로 변환
def danger_map_to_dataframe(grid):
    data = []
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            data.append({"Latitude": i, "Longitude": j, "Danger Level": grid[i, j]})
    
    return pd.DataFrame(data)

# 가상 케이스 시험
# danger_map_df = danger_map_to_dataframe(large_danger_map)

file_path = 'final_dataset/freeze_data.csv'
data = pd.read_csv(file_path)
data = data[["latitude", "longitude", "normalized_est_score"]]

def scaling_and_filtering(df, start, end):
    df['scaled_latitude'] = np.round((df['latitude'] - start[0]) * 2000).astype(int)
    df['scaled_longitude'] = np.round((df['longitude'] - start[1]) * 2000).astype(int)
    
    max_latitude = np.round((end[0] - start[0]) * 2000).astype(int)
    max_longitude = np.round((end[1] - start[1]) * 2000).astype(int)
    
    df_filtered = df[(df['scaled_latitude'] >= 0) & (df['scaled_latitude'] <= max_latitude) &
                     (df['scaled_longitude'] >= 0) & (df['scaled_longitude'] <= max_longitude)]
    
    return df_filtered

def convert_relative_to_absolute(df, relative_coords):
    absolute_coords = []
    
    for rel_lat, rel_lon in relative_coords:
        # 데이터프레임에서 주어진 상대좌표에 해당하는 행을 찾기
        matching_row = df[(df['Latitude'] == rel_lat) & (df['Longitude'] == rel_lon)]
        
        if not matching_row.empty:
            # 해당하는 절대 좌표를 찾으면 리스트에 추가
            abs_latitude = matching_row.iloc[0]['latitude']
            abs_longitude = matching_row.iloc[0]['longitude']
            absolute_coords.append((abs_latitude, abs_longitude))
        else:
            # 해당 좌표가 없으면 None
            absolute_coords.append((None, None))
    
    return absolute_coords

def find_checkpoint(start, end, danger_threshold = 4, log=True, visualize=False):

    # Defining start and end points
    # start = (37.247, 127.073)
    # end = (37.251, 127.080)
    # start = (0, 0)
    # end = (99, 99)

    danger_map_df = scaling_and_filtering(data, start, end)
    danger_map_df = danger_map_df.rename(columns={
        "scaled_latitude": "Latitude",
        "scaled_longitude": "Longitude",
        "normalized_est_score": "Danger Level"
    })

    # 다익스트라
    safest_path_diagonal, avg_risk_diagonal = find_path_avoid_danger_with_diagonal(danger_map_df, (0,0), (np.round((end[0] - start[0]) * 2000).astype(int), np.round((end[1] - start[1]) * 2000).astype(int)))

    if avg_risk_diagonal < 0:
        print("No info")
        return [start, end]
    
    # 방향 기준 체크포인트 생성
    checkpoints = create_checkpoints_on_direction_change(safest_path_diagonal)

    # 체크포인트 최소화 (체크포인트를 조합하어 일자로 이어봄 -> 기준치 이상 위험지역 없으면 -> 사이 체크포인트 삭제)
    optimized_checkpoints = optimize_checkpoints(danger_map_df, checkpoints, danger_threshold)
    abs_checkpoints = convert_relative_to_absolute(danger_map_df, optimized_checkpoints)

    if log:
        print(f"Avg risk of the safest path with diagonal movement: {avg_risk_diagonal}")
        print(f"Relative Checkpoints: {optimized_checkpoints}")
        print(f"Absolute Checkpoints: {abs_checkpoints}")
        return abs_checkpoints

    if visualize:
        # 경로 시각화
        visualize_path_with_checkpoints(danger_map_df, safest_path_diagonal, optimized_checkpoints)

        # 체크포인트 시각화
        visualize_checkpoints_with_labels_and_lines(danger_map_df, optimized_checkpoints)


'''
시작과 끝경로를 포함한 체크포인트 튜플을 가지는 리스트 반환
'''
print(find_checkpoint((37.247, 127.073), (37.251, 127.080)))
print(find_checkpoint((37.247, 213.073), (37.251, 1244.080)))