#!/bin/bash
#SBATCH --job-name=diffusion_parallel
#SBATCH --partition=gpu4          # GPU 파티션
#SBATCH --nodes=1
#SBATCH --gres=gpu:4              # GPU 4장
#SBATCH --cpus-per-task=56        # DataLoader용
#SBATCH --time=24:00:00
#SBATCH -o logs/diff_%j.out       # STDOUT
#SBATCH -e logs/diff_%j.err       # STDERR

echo "Start: $(date)"
echo "Node : $HOSTNAME"
echo "JobID: $SLURM_JOB_ID"
echo "GPUs : $CUDA_VISIBLE_DEVICES"

# ── GPU 모듈 & conda 환경 ──────────────────────────────
module purge
module load cuda/11.8.0          # 클러스터에 맞는 CUDA
source ~/miniconda3/etc/profile.d/conda.sh
conda activate df                # Diffusion-TS 설치된 env

# 필요하면 스레드 제한
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ── 실행 ──────────────────────────────────────────────
python parallel_diffusion.py     # <─ 앞서 저장한 완전판
echo "Finish: $(date)"
