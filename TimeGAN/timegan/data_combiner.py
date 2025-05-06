#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
import os
import pickle
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

def inverse_minmax_scaling(data, original_data):
    """원본 데이터의 중앙값에 맞춰 생성 데이터를 shift"""
    print("생성 데이터를 원본 데이터의 중앙값에 맞게 조정하는 중...")
    
    # 원본 데이터에서 특성 컬럼만 선택
    numeric_cols = [col for col in original_data.columns if col not in ['Unnamed: 0', 'Idx']]
    
    # 원본 데이터의 중앙값, 최소값, 최대값 계산
    original_medians = original_data[numeric_cols].median().values
    original_mins = original_data[numeric_cols].min().values
    original_maxs = original_data[numeric_cols].max().values
    
    # 각 특성에 대해 생성 데이터의 현재 범위 확인 및 중앙값 계산
    data_2d = data.reshape(-1, data.shape[2])
    valid_indices = data_2d[:, 0] != -1
    valid_data = data_2d[valid_indices]
    
    # 생성 데이터의 중앙값, 최소값, 최대값 계산
    gen_medians = np.median(valid_data, axis=0)
    gen_mins = np.min(valid_data, axis=0)
    gen_maxs = np.max(valid_data, axis=0)
    
    # MinMax 방식으로 원본 스케일로 변환 후 중앙값을 원본 중앙값으로 shift
    shifted_data = np.zeros_like(data)
    
    # 각 특성에 대해 변환 수행
    for j in range(min(data.shape[2], len(original_medians))):
        # 원본 데이터 범위
        orig_min = original_mins[j]
        orig_max = original_maxs[j]
        
        # 생성 데이터의 범위
        gen_min = gen_mins[j]
        gen_max = gen_maxs[j]
        
        print(f"특성 {j} 스케일 조정: 생성 데이터 범위 [{gen_min:.4f}, {gen_max:.4f}] -> 원본 범위 [{orig_min:.4f}, {orig_max:.4f}]")
        
        # MinMax 스케일링
        normalized = (data[:, :, j] - gen_min) / (gen_max - gen_min)
        scaled_data = normalized * (orig_max - orig_min) + orig_min
        
        # 중앙값을 계산하기 위해 2D로 변환 및 패딩 제거
        scaled_2d = scaled_data.reshape(-1)
        valid_scaled = scaled_2d[scaled_2d != -1]
        
        # 스케일링된 데이터의 중앙값
        scaled_median = np.median(valid_scaled)
        
        # 원본 중앙값과의 차이 계산
        median_shift = original_medians[j] - scaled_median
        
        # 중앙값 shift 적용 (패딩 값(-1)은 그대로 유지)
        shifted_data[:, :, j] = scaled_data
        mask = shifted_data[:, :, j] != -1
        shifted_data[:, :, j][mask] += median_shift
    
    return shifted_data

def convert_3d_to_2d(data_3d, padding_value=-1.0):
    """
    3D 데이터를 2D로 변환하는 함수
    """
    samples, seq_len, features = data_3d.shape
    
    # 결과 데이터를 저장할 배열 초기화
    result_2d = np.zeros((samples, features))
    
    print(f"3D 데이터를 2D로 변환 중... (원래 형태: {data_3d.shape})")
    
    # 각 샘플에 대해 처리
    for i in tqdm(range(samples)):
        # 패딩 값이 아닌 유효한 데이터 찾기
        valid_indices = data_3d[i, :, 0] != padding_value
        
        # 유효한 데이터가 있는 경우에만 평균 계산
        if np.any(valid_indices):
            for j in range(features):
                # 각 특성에 대해 유효한 값들의 평균 계산
                result_2d[i, j] = np.mean(data_3d[i, valid_indices, j])
        # 유효한 데이터가 없는 경우, 전체 시퀀스의 평균 사용
        else:
            for j in range(features):
                result_2d[i, j] = np.mean(data_3d[:, :, j][data_3d[:, :, j] != padding_value])
    
    print(f"3D 데이터를 2D로 변환 완료: {data_3d.shape} -> {result_2d.shape}")
    
    return result_2d

def load_original_data():
    """
    원본 데이터를 로드하는 함수
    """
    # 원본 데이터 로드
    original_data_path = '/home/hyeonmin/ISF/timegan-pytorch-main/data/energy_numeric.csv'
    df = pd.read_csv(original_data_path)
    
    # 'Unnamed: 0'와 'Idx' 열 제거
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis=1)
    if 'Idx' in df.columns:
        df = df.drop(['Idx'], axis=1)
    
    # numpy 배열로 변환
    original_values = df.values
    
    # fake 데이터와 동일한 크기로 맞춤 (94707)
    original_values = original_values[:94707]
    
    print(f"원본 데이터 shape: {original_values.shape}")
    return original_values, df.columns.tolist(), df

def load_fake_data(args):
    """
    생성된 fake 데이터를 로드하는 함수
    """
    fake_data_list = []
    
    # fake 데이터 경로 설정
    fake_data_dir = '/home/hyeonmin/ISF/timegan-pytorch-main/output/test/generated'
    
    # fake 데이터 경로 목록 가져오기
    fake_data_paths = glob.glob(os.path.join(fake_data_dir, 'fake_*.pickle'))
    
    if not fake_data_paths:
        print(f"경고: {fake_data_dir} 디렉토리에서 fake_*.pickle 파일을 찾을 수 없습니다.")
        return fake_data_list
    
    print(f"총 {len(fake_data_paths)}개의 fake 데이터 파일을 찾았습니다.")
    
    # 각 fake 데이터 로드
    for data_path in tqdm(fake_data_paths, desc="Fake 데이터 로드 중"):
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                fake_data_list.append(data)
                print(f"로드된 데이터 shape: {data.shape}")
        except Exception as e:
            print(f"경고: {data_path} 로드 중 오류 발생: {e}")
    
    # 모든 fake 데이터의 shape 출력
    print("\n모든 fake 데이터 shape:")
    for i, data in enumerate(fake_data_list):
        print(f"Fake 데이터 {i+1}: {data.shape}")
    
    return fake_data_list

def process_fake_data(fake_data_list, original_df):
    """
    모든 fake 데이터를 처리하는 함수
    """
    processed_data_list = []
    
    print("모든 fake 데이터 처리 중...")
    for i, fake_data in enumerate(fake_data_list):
        print(f"\nFake 데이터 {i+1}/{len(fake_data_list)} 처리 중...")
        print(f"원본 데이터 shape: {fake_data.shape}")
        
        # 차원 확인 및 조정
        if fake_data.shape[2] == 20:
            print("마지막 차원 제거 (20 -> 19)")
            fake_data = fake_data[:, :, :-1]
        
        # 스케일링 복원
        print("스케일링 복원 중...")
        scaled_data = inverse_minmax_scaling(fake_data, original_df)
        
        # 2D로 변환
        print("2D로 변환 중...")
        fake_2d = convert_3d_to_2d(scaled_data)
        processed_data_list.append(fake_2d)
        
        print(f"처리 완료: {fake_data.shape} -> {fake_2d.shape}")
    
    return processed_data_list

def stack_data_as_channels(original_data, fake_data_list, original_df):
    """
    원본 데이터와 fake 데이터들을 채널처럼 쌓는 함수
    """
    # 모든 fake 데이터 처리
    processed_data_list = process_fake_data(fake_data_list, original_df)
    
    # 가능한 최대 샘플 수 계산 (가장 작은 데이터셋에 맞춤)
    max_samples = min(original_data.shape[0], *[data.shape[0] for data in processed_data_list])
    features = original_data.shape[1]
    
    # 결과 배열 초기화 (원본 + fake 데이터 채널)
    total_channels = 1 + len(processed_data_list)  # 원본 + fake 데이터 개수
    stacked_data = np.zeros((max_samples, features, total_channels))
    
    print(f"\n데이터를 채널로 쌓는 중... (샘플 수: {max_samples}, 특성 수: {features}, 채널 수: {total_channels})")
    
    # 원본 데이터를 첫 번째 채널에 저장
    stacked_data[:, :, 0] = original_data[:max_samples]
    
    # 각 fake 데이터를 채널에 추가
    for i, fake_2d in enumerate(processed_data_list):
        stacked_data[:, :, i+1] = fake_2d[:max_samples]
    
    print(f"데이터를 채널로 쌓기 완료. 결과 shape: {stacked_data.shape}")
    
    return stacked_data

def combine_datasets(args):
    """데이터셋을 하나로 합치는 함수"""
    # 원본 데이터 로드
    original_data, columns, original_df = load_original_data()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"출력 디렉토리:\t{args.output_dir}\n")
    
    # 원본 데이터 저장
    output_data_file = os.path.join(args.output_dir, "original_data.pickle")
    with open(output_data_file, "wb") as fb:
        pickle.dump(original_data, fb)
    print(f"원본 데이터 저장 완료: {output_data_file}")
    
    # 칼럼 이름 저장
    output_columns_file = os.path.join(args.output_dir, "columns.pickle")
    with open(output_columns_file, "wb") as fb:
        pickle.dump(columns, fb)
    print(f"칼럼 이름 저장 완료: {output_columns_file}")
    
    # fake 데이터를 포함한 채널 생성 (fake_data_dir이 지정된 경우)
    if args.fake_data_dir:
        # fake 데이터 로드
        fake_data_list = load_fake_data(args)
        
        if fake_data_list:
            # 채널로 쌓기
            stacked_data = stack_data_as_channels(original_data, fake_data_list, original_df)
            
            # 결과 저장
            output_stacked_file = os.path.join(args.output_dir, "stacked_channel_data.pickle")
            with open(output_stacked_file, "wb") as fb:
                pickle.dump(stacked_data, fb)
            print(f"채널로 쌓은 데이터 저장 완료: {output_stacked_file}")
            
            # 채널 정보 저장
            channel_info = ["원본"] + [f"fake_{i+1}" for i in range(len(fake_data_list))]
            output_channel_info_file = os.path.join(args.output_dir, "channel_info.pickle")
            with open(output_channel_info_file, "wb") as fb:
                pickle.dump(channel_info, fb)
            print(f"채널 정보 저장 완료: {output_channel_info_file}")
            
            # 채널 데이터 통계 출력
            print("\n채널 데이터 통계:")
            for c in range(stacked_data.shape[2]):
                channel_name = channel_info[c]
                channel_data = stacked_data[:, :, c]
                print(f"  채널 {c} ({channel_name}): 평균={np.mean(channel_data):.4f}, 표준편차={np.std(channel_data):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="원본 데이터와 fake 데이터 결합")
    parser.add_argument(
        '--output_dir',
        default='./combined_output',
        type=str,
        help='결합된 데이터 저장 경로'
    )
    parser.add_argument(
        '--fake_data_dir',
        default='',
        type=str,
        help='fake 데이터가 있는 디렉토리 경로'
    )
    
    args = parser.parse_args()
    combine_datasets(args)