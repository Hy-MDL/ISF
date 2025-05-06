import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

def inverse_transform(data, scaler_params, scaling_method="minmax"):
    """스케일링된 데이터를 원래 스케일로 복구"""
    if scaling_method == "minmax":
        scaler = MinMaxScaler()
        scaler.min_, scaler.scale_ = scaler_params
        scaler.data_min_ = scaler_params[0]
        scaler.data_max_ = scaler_params[1]
    else:  # standard
        scaler = StandardScaler()
        scaler.mean_ = scaler_params[0]
        scaler.scale_ = np.sqrt(scaler_params[1])
    
    # 3D 데이터를 2D로 변환하여 역변환 수행
    original_shape = data.shape
    data_2d = data.reshape(-1, data.shape[-1])
    data_inverse = scaler.inverse_transform(data_2d)
    return data_inverse.reshape(original_shape)

def plot_distribution(original_data, generated_data, feature_idx=0, title="Feature Distribution Comparison"):
    """원본 데이터와 생성된 데이터의 분포를 시각화"""
    # 특정 feature만 선택
    original_feature = original_data[:, :, feature_idx].flatten()
    generated_feature = generated_data[:, :, feature_idx].flatten()
    
    # 패딩값(-1) 제거
    original_feature = original_feature[original_feature != -1]
    generated_feature = generated_feature[generated_feature != -1]
    
    # 데이터 통계 정보 출력
    print(f"\nFeature {feature_idx} 통계:")
    print(f"Original - Min: {np.min(original_feature):.4f}, Max: {np.max(original_feature):.4f}, Mean: {np.mean(original_feature):.4f}")
    print(f"Generated - Min: {np.min(generated_feature):.4f}, Max: {np.max(generated_feature):.4f}, Mean: {np.mean(generated_feature):.4f}")
    
    plt.figure(figsize=(15, 5))
    
    # 히스토그램
    plt.subplot(1, 3, 1)
    sns.histplot(original_feature, color='blue', alpha=0.5, label='Original', bins=50)
    sns.histplot(generated_feature, color='red', alpha=0.5, label='Generated', bins=50)
    plt.title('Histogram')
    plt.legend()
    
    # KDE 플롯
    plt.subplot(1, 3, 2)
    sns.kdeplot(original_feature, color='blue', label='Original')
    sns.kdeplot(generated_feature, color='red', label='Generated')
    plt.title('Kernel Density Estimation')
    plt.legend()
    
    # QQ 플롯
    plt.subplot(1, 3, 3)
    original_sorted = np.sort(original_feature)
    generated_sorted = np.sort(generated_feature)
    plt.scatter(original_sorted, generated_sorted, alpha=0.5)
    plt.plot([np.min(original_sorted), np.max(original_sorted)], 
             [np.min(original_sorted), np.max(original_sorted)], 
             'k--', alpha=0.5)
    plt.title('QQ Plot')
    plt.xlabel('Original Data Quantiles')
    plt.ylabel('Generated Data Quantiles')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'feature_{feature_idx}_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 데이터 로드
    print("데이터 로드 중...")
    with open('output/test/test_data.pickle', 'rb') as f:
        test_data = pickle.load(f)
    with open('output/test/fake_data.pickle', 'rb') as f:
        fake_data = pickle.load(f)
    
    try:
        # 스케일링 파라미터 로드 시도
        with open('output/test/scaling_params.pickle', 'rb') as f:
            scaling_params = pickle.load(f)
            scaling_method = "minmax"  # 또는 "standard"
            print("스케일링 파라미터를 성공적으로 로드했습니다.")
    except:
        print("스케일링 파라미터를 찾을 수 없어 원본 스케일링된 데이터를 사용합니다.")
        scaling_params = None
        scaling_method = None
    
    print(f"\n데이터 구조:")
    print(f"Test data shape: {test_data.shape}")
    print(f"Fake data shape: {fake_data.shape}")
    
    # 데이터 통계 정보 출력
    print("\n전체 데이터 통계:")
    print(f"Test data - Min: {np.min(test_data):.4f}, Max: {np.max(test_data):.4f}, Mean: {np.mean(test_data):.4f}")
    print(f"Fake data - Min: {np.min(fake_data):.4f}, Max: {np.max(fake_data):.4f}, Mean: {np.mean(fake_data):.4f}")
    
    # 스케일링 파라미터가 있는 경우 역변환
    if scaling_params is not None:
        print("\n데이터 역변환 중...")
        test_data = inverse_transform(test_data, scaling_params, scaling_method)
        fake_data = inverse_transform(fake_data, scaling_params, scaling_method)
    
    # 각 feature에 대해 분포 시각화
    for feature_idx in range(test_data.shape[-1]):
        plot_distribution(
            test_data, 
            fake_data, 
            feature_idx,
            f"Feature {feature_idx} Distribution Comparison"
        )
        print(f"Feature {feature_idx} 분포 시각화 완료")

if __name__ == "__main__":
    main() 