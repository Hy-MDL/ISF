import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os
import re
import argparse

# Korean feature names to English mapping
korean_to_english = {
    '습도(%)': 'Humidity(%)',
    '기온(°C)': 'Temperature(°C)',
    '강수량(mm)': 'Precipitation(mm)',
    '풍속(m/s)': 'Wind_Speed(m/s)',
    '일조(hr)': 'Sunshine(hr)',
    '일사(MJ/m2)': 'Solar_Radiation(MJ/m2)',
    '현지기압(hPa)': 'Pressure(hPa)'
}

def load_data(args):
    """Load data"""
    print("Loading data...")
    # Load generated data
    with open(f'output/{args.exp}/fake_data.pickle', 'rb') as f:
        fake_data = pickle.load(f)
    
    # Load train and test data
    with open(f'output/{args.exp}/train_data.pickle', 'rb') as f:
        train_data = pickle.load(f)
    
    with open(f'output/{args.exp}/test_data.pickle', 'rb') as f:
        test_data = pickle.load(f)
    
    # Load original CSV data
    original_data = pd.read_csv('data/energy_numeric.csv')
    sliding_data = pd.read_csv('data/sliding_windows_indexed.csv')
    
    print(f"Fake data shape: {fake_data.shape}")
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Original data shape: {original_data.shape}")
    print(f"Sliding window data shape: {sliding_data.shape}")
    
    # Check column names
    print("Original data columns:", original_data.columns.tolist())
    print("Generated data features:", fake_data.shape[2])
    
    return fake_data, train_data, test_data, original_data, sliding_data

def get_scaling_info(data):
    """Check scaling information of generated data"""
    # Calculate min and max values for each feature
    min_vals = np.min(data, axis=(0, 1))
    max_vals = np.max(data, axis=(0, 1))
    
    print("\nGenerated data scaling information:")
    for i in range(len(min_vals)):
        print(f"  Feature {i}: Min = {min_vals[i]:.4f}, Max = {max_vals[i]:.4f}")
    
    return min_vals, max_vals

def inverse_minmax_scaling(data, original_data):
    """원본 데이터의 중앙값에 맞춰 생성 데이터를 shift"""
    print("생성 데이터를 원본 데이터의 중앙값에 맞게 조정하는 중...")
    
    # 생성 데이터에서 마지막 열 제거
    data_without_last = data
    print(f"마지막 열 제거 후 생성 데이터 형태: {data_without_last.shape}")
    
    # 원본 데이터에서 특성 컬럼만 선택
    numeric_cols = [col for col in original_data.columns if col not in ['Unnamed: 0', 'Idx']]
    
    # 원본 데이터의 중앙값, 최소값, 최대값 계산
    original_medians = original_data[numeric_cols].median().values
    original_mins = original_data[numeric_cols].min().values
    original_maxs = original_data[numeric_cols].max().values
    
    # 각 특성에 대해 생성 데이터의 현재 범위 확인 및 중앙값 계산
    # 먼저 생성 데이터를 2D로 변환 (샘플*시퀀스, 특성)
    data_2d = data_without_last.reshape(-1, data_without_last.shape[2])
    # 패딩 값(-1) 제거
    valid_indices = data_2d[:, 0] != -1
    valid_data = data_2d[valid_indices]
    
    # 생성 데이터의 중앙값, 최소값, 최대값 계산
    gen_medians = np.median(valid_data, axis=0)
    gen_mins = np.min(valid_data, axis=0)
    gen_maxs = np.max(valid_data, axis=0)
    
    # print("\n생성 데이터 통계 정보:")
    # for i in range(len(gen_medians)):
    #     print(f"  특성 {i}: 최소값 = {gen_mins[i]:.4f}, 중앙값 = {gen_medians[i]:.4f}, 최대값 = {gen_maxs[i]:.4f}")
    #     print(f"  원본 데이터: 최소값 = {original_mins[i]:.4f}, 중앙값 = {original_medians[i]:.4f}, 최대값 = {original_maxs[i]:.4f}")
    
    # MinMax 방식으로 원본 스케일로 변환 후 중앙값을 원본 중앙값으로 shift
    # 초기화
    shifted_data = np.zeros_like(data_without_last)
    
    # 각 특성에 대해 변환 수행
    for j in range(min(data_without_last.shape[2], len(original_medians))):
        # 먼저 MinMax 방식으로 원본 스케일로 변환
        # 원본 데이터 범위
        orig_min = original_mins[j]
        orig_max = original_maxs[j]
        
        # 생성 데이터의 범위
        gen_min = gen_mins[j]
        gen_max = gen_maxs[j]
        
        print(f"특성 {j} 스케일 조정: 생성 데이터 범위 [{gen_min:.4f}, {gen_max:.4f}] -> 원본 범위 [{orig_min:.4f}, {orig_max:.4f}]")
        
        # MinMax 스케일링
        normalized = (data_without_last[:, :, j] - gen_min) / (gen_max - gen_min)
        scaled_data = normalized * (orig_max - orig_min) + orig_min
        
        # 중앙값을 계산하기 위해 2D로 변환 및 패딩 제거
        scaled_2d = scaled_data.reshape(-1)
        valid_scaled = scaled_2d[scaled_2d != -1]
        
        # 스케일링된 데이터의 중앙값
        scaled_median = np.median(valid_scaled)
        
        # 원본 중앙값과의 차이 계산
        median_shift = original_medians[j] - scaled_median
        
        # print(f"  특성 {j} 중앙값 조정: 생성 데이터 중앙값 {scaled_median:.4f} -> 원본 중앙값 {original_medians[j]:.4f}, 차이: {median_shift:.4f}")
        
        # 중앙값 shift 적용 (패딩 값(-1)은 그대로 유지)
        shifted_data[:, :, j] = scaled_data
        mask = shifted_data[:, :, j] != -1
        shifted_data[:, :, j][mask] += median_shift
    
    # 최종 결과 통계 정보 출력
    # 2D로 변환 및 패딩 제거
    final_2d = shifted_data.reshape(-1, shifted_data.shape[2])
    valid_final = final_2d[final_2d[:, 0] != -1]
    
    final_medians = np.median(valid_final, axis=0)
    final_mins = np.min(valid_final, axis=0)
    final_maxs = np.max(valid_final, axis=0)
    
    # print("\n최종 결과 통계 정보:")
    # for i in range(len(final_medians)):
    #     print(f"  특성 {i}: 최소값 = {final_mins[i]:.4f}, 중앙값 = {final_medians[i]:.4f}, 최대값 = {final_maxs[i]:.4f}")
    #     print(f"  원본 데이터: 최소값 = {original_mins[i]:.4f}, 중앙값 = {original_medians[i]:.4f}, 최대값 = {original_maxs[i]:.4f}")
    
    return shifted_data

def reconstruct_original_format(reversed_data, original_data):
    """Reconstruct to original data format"""
    print("Reconstructing to original data format...")
    
    # Use actual size of reversed data
    original_rows = len(reversed_data)
    print(f"Number of sequences in generated data: {original_rows}")
    
    # Convert 3D data to 2D (removing sliding window effect)
    reconstructed_data = np.zeros((original_rows, reversed_data.shape[2]))
    
    for i in range(original_rows):
        # Exclude padding values (-1) and calculate mean
            valid_indices = reversed_data[i, :, 0] != -1
            if np.any(valid_indices):
                for j in range(reversed_data.shape[2]):
                    reconstructed_data[i, j] = np.mean(reversed_data[i, valid_indices, j])
    
    # Convert to DataFrame
    reconstructed_df = pd.DataFrame(reconstructed_data)
    
    # Select only feature columns from original data, excluding identifiers
    numeric_cols = [col for col in original_data.columns if col not in ['Unnamed: 0', 'Idx']]
    
    # Replace Korean column names with English ones
    english_cols = []
    for col in numeric_cols:
        if col in korean_to_english:
            english_cols.append(korean_to_english[col])
        else:
            english_cols.append(col)
    
    # Match column count
    if len(english_cols) == reconstructed_data.shape[1]:
        reconstructed_df.columns = english_cols
    else:
        # Create new column names if counts don't match
        print(f"Warning: Column count mismatch. Original: {len(numeric_cols)}, Generated: {reconstructed_data.shape[1]}")
        reconstructed_df.columns = [f'feature_{i}' for i in range(reconstructed_data.shape[1])]
    
    # Also change column names in original data for comparison
    original_renamed = original_data.copy()
    rename_dict = {old: new for old, new in zip(numeric_cols, english_cols)}
    original_renamed.rename(columns=rename_dict, inplace=True)
    
    return reconstructed_df, original_renamed

def main():
    # Parse arguments to match with main.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='test', type=str)
    parser.add_argument('--train_rate', default=0.1, type=float)
    args = parser.parse_args()
    
    # Load data
    fake_data, train_data, _, original_data, sliding_data = load_data(args)
    
    # Check padding value
    padding_value = -1.0
    print(f"Padding value: {padding_value}")
    
    # Inverse transform to original scale (생성 데이터만 변환)
    reversed_data = inverse_minmax_scaling(fake_data, original_data)
    
    # Reconstruct to original data format with English column names
    reconstructed_df, original_renamed = reconstruct_original_format(reversed_data, original_data)
    
    # Save results
    reconstructed_df.to_csv('reconstructed_data.csv', index=False)
    print(f"Reconstructed data shape: {reconstructed_df.shape}")
    print("Reconstructed data saved to reconstructed_data.csv")
    
    # 원본 데이터도 저장 (비교용)
    numeric_cols = [col for col in original_renamed.columns if col not in ['Unnamed: 0', 'Idx']]
    original_df = original_renamed[numeric_cols]
    original_df.to_csv('original_data.csv', index=False)
    print(f"Original data shape: {original_df.shape}")
    print("Original data saved to original_data.csv")
    
    # Visualize comparison for each feature
    for i in range(min(10, min(len(original_df.columns), len(reconstructed_df.columns)))):
        print(f"\nFeature {i} ({original_df.columns[i]}) Statistics:")
        orig_data = original_df.iloc[:, i].values
        recon_data = reconstructed_df.iloc[:, i].values
        
        print(f"  Original data: Min={np.min(orig_data):.4f}, Max={np.max(orig_data):.4f}, Mean={np.mean(orig_data):.4f}, Std={np.std(orig_data):.4f}")
        print(f"  Generated data: Min={np.min(recon_data):.4f}, Max={np.max(recon_data):.4f}, Mean={np.mean(recon_data):.4f}, Std={np.std(recon_data):.4f}")
        
        # 시각화
        plt.figure(figsize=(15, 12))
        
        # 시계열 비교
        plt.subplot(3, 1, 1)
        plt.plot(original_df.iloc[:100, i], label='Original', color='blue')
        plt.title(f'Time Series Comparison - {original_df.columns[i]}')
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(reconstructed_df.iloc[:100, i], label='Generated', color='red')
        plt.title(f'Generated Data - {reconstructed_df.columns[i]}')
        plt.legend()
        
        # 분포 비교
        plt.subplot(3, 1, 3)
        sns.histplot(original_df.iloc[:, i], color='blue', label='Original', alpha=0.5, bins=50)
        sns.histplot(reconstructed_df.iloc[:, i], color='red', label='Generated', alpha=0.5, bins=50)
        plt.title(f'Distribution Comparison - {original_df.columns[i]}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'./visualization/image/comparison_feature_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 박스플롯 비교
        plt.figure(figsize=(10, 6))
        box_data = [original_df.iloc[:, i], reconstructed_df.iloc[:, i]]
        plt.boxplot(box_data, labels=['Original', 'Generated'])
        plt.title(f'Box Plot Comparison - {original_df.columns[i]}')
        plt.savefig(f'./visualization/image/boxplot_feature_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature {i} ({original_df.columns[i]}) comparison visualization completed")

if __name__ == "__main__":
    main() 