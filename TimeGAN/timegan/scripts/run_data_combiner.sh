#!/bin/bash

# 여러 시드로 생성된 데이터를 합치는 스크립트

# 기본 경로 설정
MODEL_PATH="./output/test"
OUTPUT_DIR=""
SHUFFLE=false

# 명령줄 인자 파싱
while getopts ":m:o:s" opt; do
  case $opt in
    m) MODEL_PATH="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    s) SHUFFLE=true ;;
    \?) echo "유효하지 않은 옵션: -$OPTARG" >&2; exit 1 ;;
    :) echo "옵션 -$OPTARG에 인자가 필요합니다." >&2; exit 1 ;;
  esac
done

echo "==== TimeGAN 생성 데이터 합치기 시작 ===="
echo "모델 경로: $MODEL_PATH"
if [ -n "$OUTPUT_DIR" ]; then
  echo "출력 경로: $OUTPUT_DIR"
else
  echo "출력 경로: $MODEL_PATH/combined (기본값)"
fi
if $SHUFFLE; then
  echo "셔플링: 활성화"
  SHUFFLE_ARG="--shuffle"
else
  echo "셔플링: 비활성화"
  SHUFFLE_ARG=""
fi
echo "====================================="

# Python 스크립트 실행
if [ -n "$OUTPUT_DIR" ]; then
  python data_combiner.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    $SHUFFLE_ARG
else
  python data_combiner.py \
    --model_path $MODEL_PATH \
    $SHUFFLE_ARG
fi

echo "==== 데이터 합치기 완료 ====" 