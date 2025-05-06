#!/bin/bash

# 미리 학습된 TimeGAN 모델을 사용하여 여러 시드로 데이터 생성하는 스크립트

# 기본 경로 설정
MODEL_PATH="./output/test"
DEVICE="cuda"
NUM_GENERATIONS=10
SEED_START=42

# 명령줄 인자 파싱
while getopts ":m:d:n:s:" opt; do
  case $opt in
    m) MODEL_PATH="$OPTARG" ;;
    d) DEVICE="$OPTARG" ;;
    n) NUM_GENERATIONS="$OPTARG" ;;
    s) SEED_START="$OPTARG" ;;
    \?) echo "유효하지 않은 옵션: -$OPTARG" >&2; exit 1 ;;
    :) echo "옵션 -$OPTARG에 인자가 필요합니다." >&2; exit 1 ;;
  esac
done

echo "==== TimeGAN 다중 데이터 생성 시작 ===="
echo "모델 경로: $MODEL_PATH"
echo "장치: $DEVICE"
echo "생성 횟수: $NUM_GENERATIONS"
echo "시작 시드: $SEED_START"
echo "====================================="

# Python 스크립트 실행
python multi_generate.py \
  --model_path $MODEL_PATH \
  --device $DEVICE \
  --num_generations $NUM_GENERATIONS \
  --seed_start $SEED_START

echo "==== 데이터 생성 완료 ====" 