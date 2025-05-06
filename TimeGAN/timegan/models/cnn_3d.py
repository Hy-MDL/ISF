import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super(ResidualBlock, self).__init__()
        self.expansion = expansion
        
        # 첫 번째 컨볼루션 블록
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        # 두 번째 컨볼루션 블록 (3x3)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 1), 
                              stride=stride, padding=(1, 1, 0))
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # 세 번째 컨볼루션 블록 (1x1)
        self.conv3 = nn.Conv3d(out_channels, out_channels * expansion, kernel_size=(1, 1, 1), 
                              stride=1, padding=0)
        self.bn3 = nn.BatchNorm3d(out_channels * expansion)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * expansion:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels * expansion, kernel_size=(1, 1, 1),
                         stride=stride, padding=0),
                nn.BatchNorm3d(out_channels * expansion)
            )
            
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        channel_weights = self.channel_attention(x)
        spatial_weights = self.spatial_attention(x)
        
        out = x * channel_weights * spatial_weights
        return out

class CNN3D(nn.Module):
    def __init__(self, input_channels=101, output_size=24):
        super(CNN3D, self).__init__()
        
        # 초기 컨볼루션 레이어
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 1), stride=(2, 2, 1), padding=(3, 3, 0))
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))
        
        # Residual Blocks with Attention
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.attention1 = AttentionBlock(256)
        
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.attention2 = AttentionBlock(512)
        
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.attention3 = AttentionBlock(1024)
        
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        self.attention4 = AttentionBlock(2048)
        
        # Global Context Block
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(2048, 2048, kernel_size=1),
            nn.BatchNorm3d(2048),
            nn.ReLU()
        )
        
        # 완전 연결 레이어
        self.fc_layers = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_size)
        )
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels * 4, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        # 초기 컨볼루션
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Residual Blocks with Attention
        x = self.layer1(x)
        x = self.attention1(x)
        
        x = self.layer2(x)
        x = self.attention2(x)
        
        x = self.layer3(x)
        x = self.attention3(x)
        
        x = self.layer4(x)
        x = self.attention4(x)
        
        # Global Context
        x = self.global_context(x)
        x = x.view(x.size(0), -1)
        
        # 완전 연결 레이어
        x = self.fc_layers(x)
        
        return x

# 모델 테스트를 위한 함수
def test_model():
    # 테스트 데이터 생성
    batch_size = 32
    height = 24
    width = 18
    channels = 101
    
    # 랜덤 입력 데이터 생성
    x = torch.randn(batch_size, 1, height, width, channels)
    
    # 모델 초기화
    model = CNN3D(input_channels=channels)
    
    # 모델 실행
    output = model(x)
    
    print(f"입력 데이터 형태: {x.shape}")
    print(f"출력 데이터 형태: {output.shape}")

if __name__ == "__main__":
    test_model() 