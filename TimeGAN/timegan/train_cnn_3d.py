import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from models.cnn_3d import CNN3D
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import imageio

# 저장 경로 설정
SAVE_DIR = "/home/hyeonmin/ISF/3DCNN/Plot"
STEP1_DIR = os.path.join(SAVE_DIR, "step1")  # 1칸씩 이동
STEP24_DIR = os.path.join(SAVE_DIR, "step24")  # 24칸씩 이동
os.makedirs(STEP1_DIR, exist_ok=True)
os.makedirs(STEP24_DIR, exist_ok=True)

class TimeSeriesDataset(Dataset):
    def __init__(self, data, target_channel=0, target_column=0, window_size=24, step=1):
        self.data = data.astype(np.float32)  # 데이터를 float32로 변환
        self.target_channel = target_channel
        self.target_column = target_column
        self.window_size = window_size
        self.step = step
        
        # 전체 데이터를 윈도우 크기로 나누어 처리
        self.num_windows = (len(data) - window_size) // step + 1
        self.remainder = (len(data) - window_size) % step
        
    def __len__(self):
        return self.num_windows
        
    def __getitem__(self, idx):
        # 시작 인덱스 계산
        start_idx = idx * self.step
        
        # 입력 데이터 준비 (24개의 연속된 시계열 데이터)
        x = self.data[start_idx:start_idx+self.window_size, :, :]
        
        # 타겟 데이터 준비 (첫 번째 채널의 첫 번째 열의 window_size에 해당하는 값)
        y = self.data[start_idx:start_idx+self.window_size, self.target_channel, self.target_column]
        
        # 데이터 형태 변환
        x = torch.FloatTensor(x).unsqueeze(0)  # 채널 차원 추가
        y = torch.FloatTensor(y)  # 타겟도 float32로 변환
        
        return x, y

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_dir):
    model.train()
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # tqdm으로 진행 상황 표시
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, targets in train_loop:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
        
        # 검증
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc='Validation')
            for inputs, targets in val_loop:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                val_loop.set_postfix(loss=val_loss/len(val_loader))
        
        # 학습률 조정
        scheduler.step()
        
        # 평균 손실 계산
        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 예측값과 실제값 시각화
        plt.figure(figsize=(12, 6))
        # 첫 번째 배치의 첫 번째 샘플만 시각화
        plt.plot(all_targets[0], label='Actual', alpha=0.7)
        plt.plot(all_predictions[0], label='Predicted', alpha=0.7)
        plt.title(f'Epoch {epoch+1}')
        plt.xlabel('Time Steps')
        plt.ylabel('Target Value')
        plt.legend()
        plt.grid(True)
        
        # 이미지 저장
        image_path = os.path.join(save_dir, f'epoch_{epoch+1}.png')
        plt.savefig(image_path)
        plt.close()
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')
    
    # 모델 저장
    torch.save(model.state_dict(), os.path.join(save_dir, 'cnn_3d_model.pth'))
    
    return train_losses, val_losses

def main():
    # 데이터셋 경로
    data_path = "/home/hyeonmin/ISF/timegan-pytorch-main/combined_output/stacked_channel_data.pickle"
    
    # 데이터셋 로드
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # 1칸씩 이동하는 경우
    print("\n=== Training with step=1 ===")
    dataset_step1 = TimeSeriesDataset(data, step=1)
    train_size = int(0.8 * len(dataset_step1))
    val_size = len(dataset_step1) - train_size
    train_dataset_step1, val_dataset_step1 = torch.utils.data.random_split(dataset_step1, [train_size, val_size])
    
    train_loader_step1 = DataLoader(train_dataset_step1, batch_size=32, shuffle=True)
    val_loader_step1 = DataLoader(val_dataset_step1, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_step1 = CNN3D().to(device)
    
    criterion = nn.MSELoss()
    optimizer_step1 = optim.AdamW(model_step1.parameters(), lr=0.001, weight_decay=0.01)
    scheduler_step1 = CosineAnnealingLR(optimizer_step1, T_max=100, eta_min=0.0001)
    
    train_losses_step1, val_losses_step1 = train_model(
        model_step1, train_loader_step1, val_loader_step1, criterion, 
        optimizer_step1, scheduler_step1, num_epochs=100, device=device,
        save_dir=STEP1_DIR
    )
    
    # 24칸씩 이동하는 경우
    print("\n=== Training with step=24 ===")
    dataset_step24 = TimeSeriesDataset(data, step=24)
    train_size = int(0.8 * len(dataset_step24))
    val_size = len(dataset_step24) - train_size
    train_dataset_step24, val_dataset_step24 = torch.utils.data.random_split(dataset_step24, [train_size, val_size])
    
    train_loader_step24 = DataLoader(train_dataset_step24, batch_size=32, shuffle=True)
    val_loader_step24 = DataLoader(val_dataset_step24, batch_size=32, shuffle=False)
    
    model_step24 = CNN3D().to(device)
    optimizer_step24 = optim.AdamW(model_step24.parameters(), lr=0.001, weight_decay=0.01)
    scheduler_step24 = CosineAnnealingLR(optimizer_step24, T_max=100, eta_min=0.0001)
    
    train_losses_step24, val_losses_step24 = train_model(
        model_step24, train_loader_step24, val_loader_step24, criterion,
        optimizer_step24, scheduler_step24, num_epochs=100, device=device,
        save_dir=STEP24_DIR
    )

if __name__ == "__main__":
    main() 