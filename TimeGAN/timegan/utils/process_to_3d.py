#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 필요한 라이브러리 임포트
import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import torch

# 생성된 TimeGAN 데이터를 원본 데이터 형식으로 변환하는 함수
def inverse_transform_data(generated_data, params, scaling_method="minmax"):
    """
    TimeGAN에서 생성된 정규화된 데이터를 원래 스케일로 되돌립니다.
    
    Args:
        generated_data (np.ndarray): TimeGAN에서 생성된 데이터 [샘플 수, 시퀀스 길이, 특성 수]
        params (list): 스케일링 파라미터 [min_vals, max_vals] 또는 [mean_vals, var_vals]
        scaling_method (str): 적용된 스케일링 방법 "minmax" 또는 "standard"
        
    Returns:
        np.ndarray: 원래 스케일로 변환된 데이터
    """
    # 데이터 형태 및 차원 확인
    no, seq_len, dim = generated_data.shape
    
    # 데이터를 2D로 변환 (스케일링 변환을 위해)
    reshaped_data = generated_data.reshape(-1, dim)
    
    # 역변환 적용
    if scaling_method == "minmax":
        min_vals, max_vals = params
        
        # 파라미터 형태 확인 및 조정
        if len(min_vals) != dim:
            print(f"스케일링 파라미터 크기 조정: min_vals 크기={len(min_vals)}, 필요한 크기={dim}")
            # 크기가 맞지 않으면 확장하거나 자름
            if len(min_vals) > dim:
                min_vals = min_vals[:dim]
                max_vals = max_vals[:dim]
            else:
                # 부족한 경우 0과 1로 채움
                min_vals = np.pad(min_vals, (0, dim - len(min_vals)), 'constant')
                max_vals = np.pad(max_vals, (0, dim - len(max_vals)), 'constant', constant_values=1)
        
        # 역변환 계산
        scaled_data = reshaped_data * (max_vals - min_vals) + min_vals
    elif scaling_method == "standard":
        mean_vals, var_vals = params
        
        # 파라미터 형태 확인 및 조정
        if len(mean_vals) != dim:
            print(f"스케일링 파라미터 크기 조정: mean_vals 크기={len(mean_vals)}, 필요한 크기={dim}")
            # 크기가 맞지 않으면 확장하거나 자름
            if len(mean_vals) > dim:
                mean_vals = mean_vals[:dim]
                var_vals = var_vals[:dim]
            else:
                # 부족한 경우 0과 1로 채움
                mean_vals = np.pad(mean_vals, (0, dim - len(mean_vals)), 'constant')
                var_vals = np.pad(var_vals, (0, dim - len(var_vals)), 'constant', constant_values=1)
        
        # 역변환 계산
        scaled_data = reshaped_data * np.sqrt(var_vals) + mean_vals
    else:
        raise ValueError("지원되지 않는 스케일링 방법입니다. minmax 또는 standard를 사용하세요.")
    
    # 원래 3D 형태로 복원
    return scaled_data.reshape(no, seq_len, dim)

# 데이터를 3D 구조로 변환하는 함수
def reshape_to_3d_structure(data, time_steps_per_sample=10, features_to_use=[0, 1, 2]):
    """
    시계열 데이터를 3D 시각화용 구조로 변환합니다.
    
    Args:
        data (np.ndarray): 입력 데이터 [샘플 수, 시퀀스 길이, 특성 수]
        time_steps_per_sample (int): 각 3D 샘플에 사용할 시간 단계 수
        features_to_use (list): 사용할 특성 인덱스 (x, y, z 축에 매핑)
        
    Returns:
        list: 3D 샘플 리스트
    """
    if len(features_to_use) != 3:
        raise ValueError("정확히 3개의 특성을 제공해야 합니다 (x, y, z 축)")
    
    no, seq_len, feature_dim = data.shape
    
    # 특성 인덱스 범위 확인 및 조정
    for i in range(3):
        if features_to_use[i] >= feature_dim:
            print(f"경고: 특성 인덱스 {features_to_use[i]}가 범위를 벗어납니다. 사용 가능한 최대 인덱스는 {feature_dim-1}입니다.")
            features_to_use[i] = min(features_to_use[i], feature_dim-1)
    
    print(f"3D 변환에 사용할 특성 인덱스: {features_to_use}")
    samples_3d = []
    
    for i in range(no):
        # 현재 샘플에서 필요한 시간 단계만 사용
        time_steps = min(time_steps_per_sample, seq_len)
        
        # x, y, z 좌표 추출
        x = data[i, :time_steps, features_to_use[0]]
        y = data[i, :time_steps, features_to_use[1]]
        z = data[i, :time_steps, features_to_use[2]]
        
        # 유효하지 않은 값 처리 (NaN, inf)
        invalid_values = np.isnan(x) | np.isnan(y) | np.isnan(z) | np.isinf(x) | np.isinf(y) | np.isinf(z)
        if np.any(invalid_values):
            print(f"경고: 샘플 {i}에서 유효하지 않은 값이 발견되었습니다. 이 값들은 제외됩니다.")
            valid_indices = ~invalid_values
        else:
            valid_indices = np.ones_like(x, dtype=bool)
        
        # 충분한 유효 데이터가 있는지 확인
        if np.sum(valid_indices) > 3:  # 최소 3개 이상의 점이 필요
            samples_3d.append({
                'x': x[valid_indices],
                'y': y[valid_indices],
                'z': z[valid_indices]
            })
    
    return samples_3d

# 3D 데이터 시각화 함수
def visualize_3d_samples(samples_3d, num_samples=5):
    """
    생성된 3D 샘플을 시각화합니다.
    
    Args:
        samples_3d (list): 3D 샘플 리스트
        num_samples (int): 시각화할 샘플 수
    """
    num_samples = min(num_samples, len(samples_3d))
    
    fig = plt.figure(figsize=(15, 3 * num_samples))
    
    for i in range(num_samples):
        sample = samples_3d[i]
        ax = fig.add_subplot(num_samples, 1, i+1, projection='3d')
        
        ax.plot(sample['x'], sample['y'], sample['z'], 'r-')
        ax.scatter(sample['x'][0], sample['y'][0], sample['z'][0], c='g', marker='o', s=100, label='시작')
        ax.scatter(sample['x'][-1], sample['y'][-1], sample['z'][-1], c='b', marker='o', s=100, label='끝')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'샘플 {i+1}')
        ax.legend()
    
    plt.tight_layout()
    return fig

# 메인 함수
def main(args):
    try:
        # 생성된 데이터 로드
        print("생성된 데이터 로딩 중...")
        try:
            with open(f"{args.model_path}/fake_data.pickle", "rb") as fb:
                generated_data = pickle.load(fb)
            
            with open(f"{args.model_path}/train_data.pickle", "rb") as fb:
                train_data = pickle.load(fb)
        except FileNotFoundError as e:
            print(f"오류: 데이터 파일을 찾을 수 없습니다: {e}")
            return False
        except Exception as e:
            print(f"오류: 데이터 로드 중 문제가 발생했습니다: {e}")
            return False
        
        print(f"생성된 데이터 형태: {generated_data.shape}")
        
        # 모델 출력 경로 설정
        output_dir = os.path.join(args.model_path, "3d_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # 스케일링 파라미터 로드 시도
        try:
            model_args = torch.load(f"{args.model_path}/args.pickle", weights_only=False)
            print("모델 아규먼트 로드 성공")
        except Exception as e:
            print(f"모델 아규먼트 로드 실패: {e}")
            model_args = None
        
        # 스케일링 파라미터 처리
        try:
            with open(f"{args.model_path}/scaling_params.pickle", "rb") as fb:
                params = pickle.load(fb)
                print("스케일링 파라미터 로드 성공")
        except FileNotFoundError:
            print("경고: 스케일링 파라미터를 찾을 수 없습니다. 데이터에서 파라미터를 계산합니다.")
            # 데이터의 최소, 최대값을 기반으로 파라미터 설정
            min_vals = np.min(train_data, axis=(0, 1))
            max_vals = np.max(train_data, axis=(0, 1))
            params = [min_vals, max_vals]
            
            # 계산된 파라미터 저장
            with open(os.path.join(args.model_path, "scaling_params.pickle"), "wb") as fb:
                pickle.dump(params, fb)
                print("계산된 스케일링 파라미터가 저장되었습니다.")
        
        # 생성 데이터를 원래 스케일로 역변환
        print("생성된 데이터를 원래 스케일로 변환 중...")
        try:
            inverse_generated_data = inverse_transform_data(
                generated_data, 
                params, 
                scaling_method=args.scaling_method
            )
            print("데이터 역변환 완료")
        except Exception as e:
            print(f"오류: 데이터 역변환 중 문제가 발생했습니다: {e}")
            return False
        
        # 데이터를 3D 구조로 변환
        print("데이터를 3D 구조로 변환 중...")
        try:
            samples_3d = reshape_to_3d_structure(
                inverse_generated_data,
                time_steps_per_sample=args.time_steps,
                features_to_use=args.features
            )
            print(f"총 {len(samples_3d)}개의 3D 샘플 생성됨")
            
            if len(samples_3d) == 0:
                print("경고: 생성된 3D 샘플이 없습니다.")
                return False
        except Exception as e:
            print(f"오류: 3D 구조 변환 중 문제가 발생했습니다: {e}")
            return False
        
        # 3D 샘플 저장
        try:
            with open(os.path.join(output_dir, "3d_samples.pickle"), "wb") as fb:
                pickle.dump(samples_3d, fb)
            print("3D 샘플이 성공적으로 저장되었습니다.")
        except Exception as e:
            print(f"오류: 3D 샘플 저장 중 문제가 발생했습니다: {e}")
            return False
        
        # 일부 샘플 시각화
        try:
            print("샘플 시각화 중...")
            fig = visualize_3d_samples(samples_3d, num_samples=min(5, len(samples_3d)))
            fig.savefig(os.path.join(output_dir, "3d_samples_visualization.png"))
            print("시각화 이미지가 저장되었습니다.")
        except Exception as e:
            print(f"경고: 샘플 시각화 중 문제가 발생했습니다: {e}")
        
        # 3D 샘플을 CSV로 저장
        if args.save_csv:
            try:
                print("3D 샘플을 CSV로 저장 중...")
                for i, sample in enumerate(samples_3d):
                    df = pd.DataFrame({
                        'x': sample['x'],
                        'y': sample['y'],
                        'z': sample['z']
                    })
                    df.to_csv(os.path.join(output_dir, f"sample_{i+1}.csv"), index=False)
                print("CSV 파일로 저장되었습니다.")
            except Exception as e:
                print(f"경고: CSV 파일 저장 중 문제가 발생했습니다: {e}")
        
        print(f"처리 완료! 결과는 {output_dir}에 저장되었습니다.")
        return True
        
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")
        return False

if __name__ == "__main__":
    # 입력 인자 파싱
    parser = argparse.ArgumentParser(description="TimeGAN 생성 데이터를 3D 형태로 처리")
    
    parser.add_argument(
        '--model_path',
        default='./output/test',
        type=str,
        help='생성된 모델과 데이터 경로'
    )
    
    parser.add_argument(
        '--scaling_method',
        default='minmax',
        choices=['minmax', 'standard'],
        type=str,
        help='데이터 스케일링 방법'
    )
    
    parser.add_argument(
        '--time_steps',
        default=20,
        type=int,
        help='각 3D 샘플의 시간 단계 수'
    )
    
    parser.add_argument(
        '--features',
        default=[0, 1, 2],
        nargs='+',
        type=int,
        help='3D 시각화에 사용할 특성 인덱스 (X, Y, Z축에 매핑)'
    )
    
    parser.add_argument(
        '--save_csv',
        action='store_true',
        help='3D 샘플을 CSV로 저장'
    )
    
    args = parser.parse_args()
    
    # 특성 인덱스 확인
    if len(args.features) != 3:
        parser.error("--features 인자는 정확히 3개의 값이 필요합니다 (X, Y, Z 축)")
    
    main(args) 