import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "iTransformer"))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from iTransformer.model.iTransformer import Model

from sklearn.metrics import mean_squared_error, mean_absolute_error

w_2013 = pd.read_csv("weather_108_2013.csv",encoding="cp949")
solar = pd.read_excel("solar_seoul.xlsx")

#열 추출
solar['date']=pd.to_datetime(solar['date'])
s_2013 = solar[solar['date'].dt.year == 2013]
s_2013['solar generation'] = s_2013['solar generation'].astype(float)
df_2013 = pd.concat([w_2013,s_2013[['solar generation']]],axis=1)
df_2013 = df_2013.select_dtypes(include=['number'])
print(s_2013.dtypes)
features = df_2013.iloc[:,3:28].values.astype(np.float32)
target = df_2013['solar generation'].values.astype(np.float32)


# 3. 정규화
feature_scaler = StandardScaler()
features_scaled = feature_scaler.fit_transform(features)

target_scaler = StandardScaler()
target_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).squeeze()

# 4. 슬라이딩 윈도우 생성
def create_sequences(features, target, input_len=168, pred_len=24):
    X, Y = [], []
    for i in range(len(features) - input_len - pred_len):
        X.append(features[i:i+input_len])  
        Y.append(target[i+input_len:i+input_len+pred_len])  
    return np.array(X), np.array(Y)

X, Y = create_sequences(features_scaled, target_scaled)

print(np.isnan(X).sum())
print(np.isnan(Y).sum())

# NaN 값을 각 특성(feature)의 평균값으로 대체
nan_indices = np.isnan(X)  # NaN 값의 인덱스를 찾음
col_mean = np.nanmean(X, axis=0)  # 각 열의 평균값 구하기
X[nan_indices] = np.take(col_mean, np.nonzero(nan_indices)[1])  # NaN 값을 평균값으로 대체

print("대체 후")
print(np.isnan(X).sum())
print(np.isnan(Y).sum())

# 5. Y의 차원 변경: (batch, 24, 1)
Y = Y[..., np.newaxis]

print("입력 X shape:", X.shape)
print("출력 Y shape:", Y.shape)  

class SolarDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# 데이터셋 및 데이터로더 생성
dataset = SolarDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 모델 파라미터 설정
input_len = 168
pred_len = 24
num_features = 25

class Config:
    def __init__(self):
        self.seq_len = input_len         # = 168
        self.pred_len = pred_len         # = 24
        self.output_attention = False
        self.use_norm = True
        self.d_model = 512
        self.embed = 'timeF'
        self.freq = 'h'
        self.dropout = 0.1
        self.class_strategy = None       # 혹은 다른 값으로 설정
        self.factor = 5                  # FullAttention에서 쓰는 값
        self.n_heads = 8
        self.e_layers = 2
        self.d_ff = 2048
        self.activation = 'gelu'

configs = Config()
model = Model(configs)

# 1. DataLoader 준비 (이미 있는 dataset을 사용)
dataset = SolarDataset(X, Y)  # X와 Y는 이미 준비된 feature와 타겟 데이터
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. DataLoader에서 배치 단위로 데이터 받기
for X_batch, Y_batch in dataloader:
    # X_batch는 모델 입력 데이터 (features)
    # Y_batch는 모델 출력 데이터 (타겟)

    # x_enc는 X_batch와 동일하게, x_mark_enc는 시간 정보 등을 넣어줘야 함
    x_enc = X_batch  # 여기서는 X_batch 그대로 사용
    x_mark_enc = torch.ones(x_enc.shape[0], x_enc.shape[1], 1)  # 시간 정보는 예시로 1로 채움
    x_dec = torch.zeros_like(x_enc)  # 예측 부분은 0으로 초기화
    x_mark_dec = torch.ones(x_dec.shape[0], 24, 1)  # 예측 길이인 24로 설정

    # 3. 모델에 배치 데이터 넣고 실행
    outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    # 4. 이후에 outputs로 원하는 작업을 할 수 있음 (예: loss 계산, 예측 결과 출력 등)
    break  # 한 배치만 테스트할 때는 break로 한번만 실행

# 손실 함수 및 옵티마이저 설정
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, Y_batch in dataloader:
        optimizer.zero_grad()

        # 입력 준비
        x_enc = X_batch
        x_mark_enc = torch.ones(x_enc.shape[0], x_enc.shape[1], 1)
        x_dec = torch.zeros(x_enc.shape[0], pred_len, x_enc.shape[2])  # 예측용 입력 (0으로 채움)
        x_mark_dec = torch.ones(x_enc.shape[0], pred_len, 1)

        outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        outputs = outputs[:, :, -1:]

        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
    
# 모델을 평가 모드로 전환
model.eval()

# 예측 수행
with torch.no_grad():
    for X_batch, Y_batch in dataloader:
        x_enc = X_batch
        x_mark_enc = torch.ones(x_enc.shape[0], x_enc.shape[1], 1)
        x_dec = torch.zeros(x_enc.shape[0], pred_len, x_enc.shape[2])
        x_mark_dec = torch.ones(x_enc.shape[0], pred_len, 1)

        # 모델 출력
        outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # 마지막 feature만 사용 (태양광 발전량 예측)
        outputs = outputs[:, :, -1:]  # 여기서 수정!

        break  # 첫 번째 배치에 대한 예측만 수행

# 예측 결과를 numpy 배열로 변환
predictions = outputs.numpy()

# 정규화된 값을 원래 스케일로 복원
predictions_original = target_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1, pred_len, 1)

# 결과 확인
#print("예측된 태양광 발전량 (첫 번째 배치):")
#print(predictions_original)

# 1. 예측값을 평탄화해서 하나의 1D 배열로 만들어줌
flattened_preds = predictions_original.reshape(-1)

# 2. 예측 결과가 붙을 시점: input_len + pred_len 이후부터
start_idx = input_len + pred_len

# 3. df_2013에 맞춰 빈 값(NaN)으로 채운 전체 prediction column 생성
full_preds = np.full(len(df_2013), np.nan)
full_preds[start_idx:start_idx + len(flattened_preds)] = flattened_preds

# 4. df_2013에 'prediction' 컬럼으로 추가
df_2013['prediction'] = full_preds

# 결과 확인
print(df_2013[['solar generation', 'prediction']].head(30))

df_2013.to_csv("solar_prediction_2013.csv", index=False, encoding='utf-8-sig')

# NaN이 아닌 부분만 비교
valid_idx = ~df_2013['prediction'].isna()
y_true = df_2013.loc[valid_idx, 'solar generation'].values
y_pred = df_2013.loc[valid_idx, 'prediction'].values

# 정확도 지표 계산
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)

# 결과 출력
print(f"\n📊 예측 정확도 평가:")
print(f"MSE  (Mean Squared Error):      {mse:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"MAE  (Mean Absolute Error):     {mae:.4f}")