#!/bin/bash
#SBATCH --nodes=1 # 자동으로 알아서 컴퓨터가 배정해줌, 노드 한개만을 쓰겠다는 뜻 
#SBATCH --partition=gpu6 # 어떤 파티션을 쓸 것인지 선택 (위 파티션을 참조하여 선택)
#SBATCH --cpus-per-task=20 # CPU나 GPU 코어를 몇 개를 쓸 것인지 지정 
#SBATCH --gres=gpu:1 # CPU나 GPU를 몇 개를 쓸 것인지 지정 
#SBATCH --job-name=Check
#SBATCH -o ./output/jupyter.%N.%j.out  # STDOUT 
#SBATCH -e ./output/jupyter.%N.%j.err  # STDERR

echo "start at:" `date` # 접속한 날짜 표기
echo "node: $HOSTNAME" # 접속한 노드 번호 표기 
echo "jobid: $SLURM_JOB_ID" # jobid 표기 

# GPU 환경을 이용하고 싶은 경우에만 해당! 그렇지 않은 경우 해당 명령어들은 지우셔도 무관합니다.
module avail CUDA # CUDA 어떤 버전들이 설치되어있는지 확인하는 방법 
module unload CUDA/11.2.2 # 기본적으로 탑재되어있는 쿠다버전은 unload. 
module load cuda/11.8.0 # GPU를 사용하는 경우 CUDA 버전을 지정해 줄 수 있으며, UBAI 최신 CUDA 버전은 12.2.1입니

nvidia-smi

python -m jupyter lab $HOME \
        --ip=0.0.0.0 \
        --no-browser

