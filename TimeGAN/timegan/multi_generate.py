#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Local modules
import argparse
import logging
import os
import pickle
import random
import time
import numpy as np
import torch
from tqdm import tqdm

# Self-Written Modules
from models.timegan import TimeGAN
from models.utils import timegan_generator

def multi_generator(args):
    """
    미리 학습된 TimeGAN 모델을 사용하여 여러 시드로 데이터를 생성합니다.
    Args:
        - args: 명령줄 인수
    """
    # 출력 디렉토리 설정
    out_dir = os.path.abspath(args.model_path)
    output_dir = os.path.join(out_dir, "generated")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n입력 모델 경로:\t\t{out_dir}")
    print(f"출력 디렉토리:\t\t{output_dir}")
    print(f"생성할 데이터셋 수:\t{args.num_generations}")
    print(f"시드 범위:\t\t{args.seed_start}~{args.seed_start + args.num_generations - 1}\n")

    # 학습 데이터와 시간 정보 로드
    print("학습 데이터 로드 중...")
    train_data_path = os.path.join(args.model_path, "train_data.pickle")
    train_time_path = os.path.join(args.model_path, "train_time.pickle")
    
    if not os.path.exists(train_data_path) or not os.path.exists(train_time_path):
        raise ValueError(f"학습 데이터 파일을 찾을 수 없습니다: {train_data_path}, {train_time_path}")
    
    with open(train_data_path, "rb") as fb:
        train_data = pickle.load(fb)
    with open(train_time_path, "rb") as fb:
        train_time = pickle.load(fb)

    # 모델 초기화
    print("모델 초기화 중...")
    # args.pickle에서 모델 파라미터 로드
    with open(f"{args.model_path}/args.pickle", "rb") as fb:
        model_args = torch.load(fb, weights_only=False)
    
    # CUDA 설정
    if args.device == "cuda" and torch.cuda.is_available():
        print("CUDA 사용 중\n")
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(args.seed_start)
    else:
        print("CPU 사용 중\n")
        device = torch.device("cpu")

    model_args.device = device
    
    # 여러 시드로 데이터 생성
    for i in tqdm(range(args.num_generations), desc="데이터셋 생성 중"):
        # 현재 시드 설정
        current_seed = args.seed_start + i
        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(current_seed)
        
        # 모델 생성 및 가중치 로드
        model = TimeGAN(model_args)
        model.load_state_dict(torch.load(f"{args.model_path}/model.pt", weights_only=False))
        model.to(device)
        
        # 데이터 생성
        start_time = time.time()
        print(f"\n[{i+1}/{args.num_generations}] 시드 {current_seed}로 데이터 생성 중...")
        
        model.eval()
        with torch.no_grad():
            # 랜덤 노이즈 생성 - 이미 CUDA 텐서로 생성
            Z = torch.rand((len(train_time), model_args.max_seq_len, model_args.Z_dim), device=device)
            
            # timegan.py의 forward 함수에서 Z를 다시 FloatTensor로 변환하는 코드를 우회하기 위해 
            # 직접 inference 로직을 구현합니다
            # generator와 supervisor를 통해 데이터 생성
            E_hat = model.generator(Z, train_time)
            H_hat = model.supervisor(E_hat, train_time)
            
            # 생성된 데이터 복원
            generated_data = model.recovery(H_hat, train_time)
            generated_data = generated_data.cpu().numpy()
        
        # 생성된 데이터 저장
        output_file = os.path.join(output_dir, f"fake_data_seed_{current_seed}.pickle")
        time_file = os.path.join(output_dir, f"fake_time_seed_{current_seed}.pickle")
        
        with open(output_file, "wb") as fb:
            pickle.dump(generated_data, fb)
        with open(time_file, "wb") as fb:
            pickle.dump(train_time, fb)
        
        print(f"생성 완료: {output_file}")
        print(f"소요 시간: {(time.time() - start_time):.2f}초")
    
    print(f"\n총 {args.num_generations}개의 데이터셋 생성 완료")
    print(f"생성된 데이터 저장 경로: {output_dir}")

if __name__ == "__main__":
    # 명령줄 인수 파서 설정
    parser = argparse.ArgumentParser(description="TimeGAN 모델로 여러 시드를 사용해 데이터 생성")
    
    parser.add_argument(
        '--model_path',
        default='./output/test',
        type=str,
        help='학습된 모델 경로'
    )
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        type=str,
        help='계산 장치 (cuda 또는 cpu)'
    )
    parser.add_argument(
        '--num_generations',
        default=5,
        type=int,
        help='생성할 데이터셋 수'
    )
    parser.add_argument(
        '--seed_start',
        default=42,
        type=int,
        help='시작 시드 값'
    )
    
    args = parser.parse_args()
    
    # 메인 함수 호출
    multi_generator(args) 