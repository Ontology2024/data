import requests
import os
import re
import csv
import numpy as np
import logging
from PIL import Image
from io import BytesIO
import json

folder_index = ["범죄주의_전체", "범죄주의_강도", "범죄주의_성폭력", "범죄주의_절도", "범죄주의_폭력", "여성밤길치안안전"]

# Configure logging
logging.basicConfig(filename='process.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# WMS 요청 URL 템플릿
wms_url_template = "https://geo.safemap.go.kr/geoserver/safemap/wms?&request=GetMap&bbox={bbox}&format=image/png&version=1.1.1&exceptions=text/xml&transparent=TRUE&srs=EPSG:4326&service=WMS&layers={layers}&width={res}&styles={styles}&height={res}"

api_index = {
    "범죄주의_전체": ["A2SM_CRMNLHSPOT_TOT", "A2SM_CrmnlHspot_Tot_Tot"],
    "범죄주의_강도": ["A2SM_CRMNLHSPOT_TOT", "A2SM_CrmnlHspot_Tot_Brglr"],
    "범죄주의_성폭력": ["A2SM_CRMNLHSPOT_TOT", "A2SM_CrmnlHspot_Tot_Rape"],
    "범죄주의_절도": ["A2SM_CRMNLHSPOT_TOT", "A2SM_CrmnlHspot_Tot_Theft"],
    "범죄주의_폭력": ["A2SM_CRMNLHSPOT_TOT", "A2SM_CrmnlHspot_Tot_Violn"],
    "여성밤길치안안전": ["A2SM_CRMNLHSPOT_F1_TOT", "A2SM_OdblrCrmnlHspot_Tot_20_24"],
}

# 설정
#lon_min, lat_min, lon_max, lat_max = 127.073, 37.247, 127.080, 37.251  # 반달공원
lon_min, lat_min, lon_max, lat_max = 126.964, 37.169, 127.103, 37.290 # 분석범위
step = 0.0005
res = 16
chunk_size = 10000 # 1024 * 1024 * 100  # 100 MB


###################################################################################################

def format_coordinate(coord):
    return round(coord, 4)

def calculate_redness_score(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    width, height = img.size
    
    total_score = 0
    red_pixel_count = 0

    for y in range(height):
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            if r > g and r > b:
                score = (r ** 2 + (255*2 - b - g) ** 2) / 1000
                total_score += score
                red_pixel_count += 1
    
    average_score = total_score / red_pixel_count if red_pixel_count > 0 else 0
    return red_pixel_count, average_score

def rate(score):
    score_arr = [300, 200, 100, 70, 50, 0]
    for i in score_arr:
        if score > i:
            return 5 - score_arr.index(i)
    return 0

def extract_coordinates_from_filename(filename):
    match = re.search(r'(\d+\.\d+),(\d+\.\d+),(\d+\.\d+),(\d+\.\d+)\.png', filename)
    if match:
        lon_min = float(match.group(1))
        lat_min = float(match.group(2))
        lon_max = float(match.group(3))
        lat_max = float(match.group(4))
        # return (lon_min + lon_max) / 2, (lat_min + lat_max) / 2
        return lon_min, lat_min
    logging.warning(f"Filename pattern did not match: {filename}")
    return None, None

def create_csv_from_images(folder_path, csv_filename, data_rows):
    csv_file_path = os.path.join(folder_path, csv_filename)

    # Write data rows to CSV
    write_header = not os.path.exists(csv_file_path)  # Check if file exists to decide whether to write header

    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        if write_header:
            # Write the header only if the file is newly created
            csv_writer.writerow(['folder', 'latitude', 'longitude', 'danger_rate', 'red_pixel_count', 'score', 'final_score'])
        csv_writer.writerows(data_rows)
    
    logging.info(f"Data appended to CSV file '{csv_filename}'.")

# Helper function to load the last processed bbox
def load_last_processed_bbox(folder_name):
    state_file = os.path.join(base_path, folder_name, 'state.json')
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            return state.get('last_bbox', None)
    return None

# Helper function to save the last processed bbox
def save_last_processed_bbox(folder_name, bbox):
    state_file = os.path.join(base_path, folder_name, 'state.json')
    state = {'last_bbox': bbox}
    with open(state_file, 'w') as f:
        json.dump(state, f)
    
def download_images_and_create_csv(folder_name, base_path, bbox_list):
    folder_path = os.path.join(base_path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    total_size = 0
    image_paths = []
    data_rows = []  # Store data to be written to CSV
    total_bboxes = len(bbox_list)
    last_bbox = load_last_processed_bbox(folder_name)
    start_index = bbox_list.index(last_bbox) + 1 if last_bbox else 0

    for file in os.listdir(folder_path):
            if file.endswith('.png'):
                file_path = os.path.join(folder_path, file)
                os.remove(file_path)
                print(f"Removed file: {file_path}")

    for i, bbox in enumerate(bbox_list[start_index:], start=start_index):
        file_name = f"{bbox}.png"
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            logging.info(f"Image already exists: {file_path}")
            continue

        url = wms_url_template.format(bbox=bbox, res=res, layers=api_index[folder_name][0], styles=api_index[folder_name][1])
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            image_size = os.path.getsize(file_path)
            total_size += image_size
            image_paths.append(file_path)

            if total_size > chunk_size:
                logging.info(f"Processing chunk: {len(image_paths)} images downloaded")
                print(f"\rProcessing chunk: {len(image_paths)} images downloaded", end='')
                
                # Calculate and prepare data to be written to CSV
                csv_data_rows = []
                for image_file in image_paths:
                    lon, lat = extract_coordinates_from_filename(os.path.basename(image_file))
                    if lon is not None and lat is not None:
                        red_pixel_count, score = calculate_redness_score(image_file)
                        final_score = round(score * red_pixel_count / 100, 0)
                        danger_rate = rate(final_score)
                        csv_data_rows.append([folder_path, lat, lon, danger_rate, red_pixel_count, score, final_score])
                
                create_csv_from_images(folder_path, f'data_{folder_name}.csv', csv_data_rows)
                save_last_processed_bbox(folder_name, bbox)
                
                for path in image_paths:
                    os.remove(path)
                image_paths = []
                total_size = 0
        else:
            logging.error(f"Failed to retrieve image for BBox: {bbox}, Status Code: {response.status_code}")

        # Update progress on the same line
        progress = (i + 1) / total_bboxes * 100
        progress_message = f"Progress: {progress:.2f}% - {i + 1}/{total_bboxes} images processed"
        print(f"\r{progress_message}", end='')

    if image_paths:
        logging.info(f"\nProcessing final chunk: {len(image_paths)} images downloaded")
        print(f"\nProcessing final chunk: {len(image_paths)} images downloaded")
        
        # Calculate and prepare data to be written to CSV
        csv_data_rows = []
        for image_file in image_paths:
            lon, lat = extract_coordinates_from_filename(os.path.basename(image_file))
            if lon is not None and lat is not None:
                red_pixel_count, score = calculate_redness_score(image_file)
                final_score = round(score * red_pixel_count / 100, 0)
                danger_rate = rate(final_score)
                csv_data_rows.append([folder_path, lat, lon, danger_rate, red_pixel_count, score, final_score])
        
        create_csv_from_images(folder_path, f'data_{folder_name}.csv', csv_data_rows)
        save_last_processed_bbox(folder_name, bbox)
        for path in image_paths:
            os.remove(path)

    # Progress finished
    print()

def generate_bbox_list(lon_min, lon_max, lat_min, lat_max, step):
    bbox_list = []
    for lon in np.arange(lon_min, lon_max, step):
        for lat in np.arange(lat_min, lat_max, step):
            lon = format_coordinate(lon)
            lat = format_coordinate(lat)
            bbox = f"{lon},{lat},{format_coordinate(lon+step)},{format_coordinate(lat+step)}"
            bbox_list.append(bbox)
    return bbox_list

def process_all_folders(base_path, folder_index):
    for folder in folder_index:
        logging.info(f"Starting processing for folder: {folder}")
        print(f"Starting processing for folder: {folder}")
        bbox_list = generate_bbox_list(lon_min, lon_max, lat_min, lat_max, step)
        download_images_and_create_csv(folder, base_path, bbox_list)
        logging.info(f"Completed processing for folder: {folder}")
        print(f"Completed processing for folder: {folder}")

# Example call
base_path = './'  # Update with the actual path to image folders
process_all_folders(base_path, folder_index)
