import folium
import pandas as pd
import json
import branca.colormap as cm

# 파일 경로 지정
file_path = 'freeze_data.csv'

# 데이터 불러오기
data = pd.read_csv(file_path)

# 중심 좌표 설정 (데이터의 평균 위도와 경도로 설정)
map_center = [data['latitude'].mean(), data['longitude'].mean()]

# 지도 생성 (ID 명시적으로 지정)
mymap = folium.Map(location=map_center, zoom_start=12, control_scale=True, id="mymap")

# 컬러맵 설정 (위험도에 따라 색상 결정)
colormap = cm.LinearColormap(colors=['green', 'yellow', 'red'], vmin=data['normalized_est_score'].min(), vmax=data['normalized_est_score'].max())
colormap.caption = 'Risk Level'

# grid_data 준비 (Python 데이터를 JavaScript로 전달할 수 있도록 JSON 변환)
grid_data = json.dumps([{
    'lat': row['latitude'], 
    'lon': row['longitude'], 
    'risk': int(row['normalized_est_score'])  
} for index, row in data.iterrows()])

# HTML에 추가될 JavaScript 코드 및 버튼
corrected_script = '''
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script src="https://cdnjs.cloudflare.com/ajax/libs/chroma-js/2.1.0/chroma.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chroma-js/2.1.0/chroma.min.js"></script>

<style>
    /* 기본 버튼 스타일 */
    button {
        padding: 10px 20px;
        margin: 5px;
        font-size: 16px;
        cursor: pointer;
        border: 1px solid #000;
    }
    /* 선택된 버튼 스타일 */
    .selected {
        background-color: gray;
    }
</style>

<script>
    document.addEventListener("DOMContentLoaded", function() {
        // Folium에서 생성된 지도 객체를 명확하게 가져옴
        var mapInstance = map_40133eff4d047f97487959fc57107ac9;

        let selectedRisks = [];

        // 지도에서 사각형 레이어를 업데이트하는 함수
        function updateMap() {
            if (mapInstance && typeof mapInstance.eachLayer === 'function') {
                // 기존에 추가된 모든 사각형 레이어 제거
                mapInstance.eachLayer(function(layer) {
                    if (layer instanceof L.Rectangle) {
                        mapInstance.removeLayer(layer);
                    }
                });
            } else {
                console.error("Map instance is not properly initialized or 'eachLayer' is not a function.");
                return;
            }

            // 선택된 위험도에 해당하는 사각형을 다시 그리기
            gridData.forEach(function (item) {
                if (selectedRisks.includes(item.risk)) {
                    let bounds = [
                        [item.lat, item.lon],
                        [item.lat + 0.0005, item.lon + 0.0005]
                    ];
                    L.rectangle(bounds, {
                        color: null,
                        fillColor: getColor(item.risk),
                        fillOpacity: 0.3
                    }).addTo(mapInstance);
                }
            });
        }

        // 위험도 토글 함수
        function toggleRisk(riskLevel) {
            let index = selectedRisks.indexOf(riskLevel);
            if (index > -1) {
                selectedRisks.splice(index, 1);  // 이미 선택된 경우 제거
            } else {
                selectedRisks.push(riskLevel);   // 선택되지 않은 경우 추가
            }
            updateMap();  // 지도 업데이트
        }

        // 위험도에 따른 색상 결정 함수
        function getColor(risk) {
            // Linear interpolation between colors
            const colormap = chroma.scale(['green', 'yellow', 'orange', 'red']).domain([0, 5]);

            // Get color for the risk value
            return colormap(risk).hex();
        }

        // Python에서 전달된 데이터를 JavaScript로 사용
        var gridData = %s;

        // 버튼 클릭 이벤트 리스너 등록
        document.querySelectorAll('button').forEach(function(button) {
            button.addEventListener('click', function() {
                const riskLevel = parseInt(button.innerText.split(' ')[1]);
                toggleRisk(riskLevel);
            });
        });
    });
</script>

<!-- 위험도를 위한 버튼 추가 -->
<div>
        <button onclick="toggleColor(this)">Risk 0</button>
        <button onclick="toggleColor(this)">Risk 1</button>
        <button onclick="toggleColor(this)">Risk 2</button>
        <button onclick="toggleColor(this)">Risk 3</button>
        <button onclick="toggleColor(this)">Risk 4</button>
        <button onclick="toggleColor(this)">Risk 5</button>
</div>

<script>
    // 클릭된 버튼의 색깔을 토글하는 함수
    function toggleColor(button) {
        // 버튼에 'selected' 클래스가 있으면 제거하고, 없으면 추가
        button.classList.toggle('selected');
    }
</script>
''' % grid_data

# 지도에 기본 사각형 추가 (초기에는 모든 데이터가 나타나지 않도록 설정)
for grid in json.loads(grid_data):
    bounds = [
        [grid['lat'], grid['lon']],
        [grid['lat'] + 0.0005, grid['lon'] + 0.0005]
    ]
    # 기본적으로 비어있는 사각형을 추가, 초기에는 아무것도 표시되지 않음
    folium.Rectangle(
        bounds=bounds,
        color=None,
        fill=True,
        fill_color=colormap(grid['risk'] / 5),
        fill_opacity=0.0  # 투명하게 설정
    ).add_to(mymap)

# 컬러맵을 지도에 추가
mymap.add_child(colormap)

# HTML에 버튼 및 스크립트 추가
mymap.get_root().html.add_child(folium.Element(corrected_script))

# 지도 저장
mymap.save('cus_grid_map.html')
print('Map saved as corrected_interactive_toggle_button_grid_map.html')
