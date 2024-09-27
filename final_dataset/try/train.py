import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 데이터 로드
data = pd.read_csv('final_data.csv')

# 데이터셋 준비 (위도, 경도, 범죄 점수 등 입력, 'final_score_top_5'를 목표로 사용)
X = data.drop(columns=['final_score_top_5'])  
y = pd.cut(data['final_score_top_5'], bins=[-1, 0, 50, 100, 150, 200, 518], labels=[0, 1, 2, 3, 4, 5])

# 결측치 처리
X = np.nan_to_num(X)
y = np.nan_to_num(y)

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 특성 스케일링 (표준화)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 클래스 가중치 계산 (데이터 불균형 해결을 위한 가중치 설정)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

print("클래스 가중치:", class_weights_dict)

# 데이터 준비 (PyTorch 텐서로 변환)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# 라벨도 텐서로 변환
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 다중 출력 모델 정의 (각 범죄 유형별 위험도 예측)
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


# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델 초기화 및 GPU로 이동
model = MultiOutputCrimeModel(input_size=X_train_scaled.shape[1])
model.to(device)

# 손실 함수와 옵티마이저 설정 (L2 정규화 포함)
criterion = nn.CrossEntropyLoss()  # 다중 클래스 교차 엔트로피 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 학습률 스케줄러 정의
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 손실 및 정확도 기록을 위한 리스트 초기화
train_losses = []
train_accuracies = []
test_accuracies = []

# 학습 루프
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        # 손실 계산
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    scheduler.step()

# 테스트 최종 정확도 평가
model.eval()
with torch.no_grad():
    total = 0
    correct = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        # 예측 및 정확도 계산
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 모델 저장
torch.save(model.state_dict(), "multi_output_crime_model.pth")
print("모델 저장 완료: multi_output_crime_model.pth")
