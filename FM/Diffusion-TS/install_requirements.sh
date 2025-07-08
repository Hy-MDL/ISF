#!/usr/bin/env bash
set -euo pipefail

# 1) requirements.txt에서 mujoco 항목 제외
grep -Ev '^\s*mujoco\b' requirements.txt > reqs_no_mujoco.txt

# 2) PyPI 휠 다운로드 안정성을 위해 pip 옵션 지정
PIP_OPTS=(
  --no-cache-dir
  --timeout 120
  --retries 5
  --resume-retries 5
)

# 3) torch/torchvision은 PyTorch 공식 CUDA 휠 인덱스에서 설치
#    (requirements.txt에 명시돼 있지 않다면 여기서 별도 설치)
pip install "${PIP_OPTS[@]}" \
  torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4) 나머지 패키지 설치
pip install "${PIP_OPTS[@]}" -r reqs_no_mujoco.txt

# 5) mujoco는 필요 시 공식 배포 채널 또는 최신 버전으로 별도 설치
# 예시: pip install mujoco==2.3.7
