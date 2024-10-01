import heapq
import numpy as np
import pandas as pd

# 샘플 데이터
data = pd.DataFrame({
    'Latitude': [37.1690, 37.1695, 37.1700, 37.1705, 37.1710],
    'Longitude': [126.964, 126.964, 126.964, 126.964, 126.964],
    'Danger Level': [1, 1, 1, 1, 1]
})

# 위도와 경도에 1000을 곱하고 정수로 변환
data["Latitude"] = (data["Latitude"] * 1000).astype(int)
data["Longitude"] = (data["Longitude"] * 1000).astype(int)

# 열 이름 변경
data = data.rename(columns={
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "Danger Level": "Danger Level"
})

# 대각선 방향을 포함한 모든 방향
diagonal_directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

# 좌표를 인덱싱하는 방식 유지 (이미 정수로 변환)
large_start = (int(37.169 * 1000), int(126.964 * 1000))  # (37247, 127073)
large_end = (int(37.171 * 1000), int(127.964 * 1000))    # (37251, 127080)

# 경로 찾기 알고리즘
def find_path_avoid_danger_with_diagonal(df, start, end):
    # 시작점과 종료점 정의
    start_lat, start_lon = start
    end_lat, end_lon = end

    # 시작점과 종료점의 위험도 검색
    start_point = df[(df['Latitude'] == start_lat) & (df['Longitude'] == start_lon)].iloc[0]
    end_point = df[(df['Latitude'] == end_lat) & (df['Longitude'] == end_lon)].iloc[0]

    # 우선순위 큐, 위험도 및 경로 관리 딕셔너리 초기화
    pq = [(start_point['Danger Level'], start_point['Latitude'], start_point['Longitude'])]
    visited = set()
    risks = {(start_point['Latitude'], start_point['Longitude']): start_point['Danger Level']}
    parents = {(start_point['Latitude'], start_point['Longitude']): None}

    while pq:
        current_risk, x, y = heapq.heappop(pq)
        if (x, y) == (end_point['Latitude'], end_point['Longitude']):
            break

        if (x, y) in visited:
            continue
        visited.add((x, y))

        # 이웃 탐색 (대각선 포함)
        for dx, dy in diagonal_directions:
            nx, ny = x + dx, y + dy
            neighbor = df[(df['Latitude'] == nx) & (df['Longitude'] == ny)]
            if neighbor.empty:
                continue
            neighbor_point = neighbor.iloc[0]

            # 위험도를 업데이트하고 새로운 경로 찾기
            new_risk = current_risk + neighbor_point['Danger Level']
            if (nx, ny) not in risks or new_risk < risks[(nx, ny)]:
                risks[(nx, ny)] = new_risk
                parents[(nx, ny)] = (x, y)
                heapq.heappush(pq, (new_risk, nx, ny))

    # 경로 재구성
    path = []
    current = (end_point['Latitude'], end_point['Longitude'])
    while current is not None:
        path.append(current)
        current = parents.get(current)

    path.reverse()
    return path, risks[(end_point['Latitude'], end_point['Longitude'])]

# 경로 찾기 실행
safest_large_path_diagonal, total_large_risk_diagonal = find_path_avoid_danger_with_diagonal(data, large_start, large_end)

print("경로:", safest_large_path_diagonal)
print("총 위험도:", total_large_risk_diagonal)
