
import cv2

def calculate_redness_score(r, g, b):
    return (r - g) ** 2 + (r - b) ** 2

# 이미지 읽기
img = cv2.imread('bandal_map\image_127.08_37.25.png')

height, width, _ = img.shape
total_score = 0
red_pixel_count = 0

for y in range(height):
    for x in range(width):
        b, g, r = img[y, x]
        # 빨간색 성분이 우세한 경우
        if r > g and r > b:
            score = calculate_redness_score(r, g, b)
            total_score += score
            red_pixel_count += 1

# 평균 점수 계산 (붉은 픽셀이 없는 경우 0으로 설정)
average_score = total_score / red_pixel_count if red_pixel_count > 0 else 0

print(f'총 붉은 픽셀 수: {red_pixel_count}')
print(f'평균 붉은 정도 점수: {average_score}')
