#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
parallel_diffusion.py
* 여러 seq_len(720·2160·4320·8640) 윈도우 길이에 대해
  Diffusion-TS 모델을 병렬 학습·복원하고, 결과를 pkl로 저장.

필수 외부 라이브러리  : PyTorch, PyYAML, sklearn, pandas
필수 프로젝트 레포    : Diffusion-TS (import path가 잡혀 있어야 함)
"""

# ---------------- 표준/외부 모듈 -----------------
import os, copy, pickle, importlib, multiprocessing as mp
from pathlib import Path
import yaml, pandas as pd, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ---------------- Utils -----------------
def normalize_to_neg_one_to_one(arr: np.ndarray) -> np.ndarray:
    """0~1 구간 배열 → -1~1 구간으로 스케일"""
    return arr * 2.0 - 1.0

def unnormalize_to_zero_to_one(arr: np.ndarray) -> np.ndarray:
    """-1~1 구간 배열 → 0~1 구간으로 역스케일"""
    return (arr + 1.0) / 2.0

def load_yaml_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def instantiate_from_config(cfg: dict):
    """
    cfg: {'target': 'package.module.Class', 'params': {...}}
    """
    target = cfg["target"]
    params = cfg.get("params", {})
    module_name, cls_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls(**params)

# ---------------- Dataset -----------------
class SeqDataset(Dataset):
    """
    time-series 시퀀스 → (pred_len) 뒤쪽 마스크 복원 학습용 Dataset
    """
    def __init__(self, data: np.ndarray, pred_length=24, regular=True):
        self.data = data                      # [N, win_size, C]
        self.regular = regular
        mask = np.ones_like(data, dtype=bool) # True=입력, False=타깃
        mask[:, -pred_length:, :] = False
        self.mask = mask

    def __len__(self):  return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx]).float()
        if self.regular:
            return x
        m = torch.from_numpy(self.mask[idx]).bool()
        return x, m                           # (x, mask)

# ---------------- Args stub -----------------
class Args:
    """
    Diffusion-TS engine.solver.Trainer 가 기대하는 최소 인자
    """
    def __init__(self):
        self.config_path = "./Config/solar.yaml"   # YAML 위치
        self.gpu = 0                               # 스크립트 내부에서 수정

# ---------------- Worker -----------------
def run_pipeline(seq_len: int, pred_len: int, stride: int, n_runs: int,
                 numeric_df: np.ndarray, base_cfg: dict,
                 gpu_id: int, out_dir: str):
    """
    하나의 seq_len 실험을 지정 GPU에서 수행
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda")

    feat_num = numeric_df.shape[1]
    win_size = seq_len + pred_len

    # 1) 윈도우 슬라이싱
    seqs = np.stack([numeric_df[i:i+win_size]
                     for i in range(0, len(numeric_df)-win_size+1, stride)])
    # 2) 스케일 → [-1,1]
    n_tr = int(len(seqs)*0.8)
    tr, te = seqs[:n_tr], seqs[n_tr:]
    scaler = MinMaxScaler()
    tr_s = normalize_to_neg_one_to_one(scaler.fit_transform(tr.reshape(-1,feat_num))).reshape(tr.shape)
    te_s = normalize_to_neg_one_to_one(scaler.transform(   te.reshape(-1,feat_num))).reshape(te.shape)

    # 3) DataLoader
    tr_loader = DataLoader(SeqDataset(tr_s, pred_len, True),  batch_size=32, shuffle=True)
    te_loader = DataLoader(SeqDataset(te_s, pred_len, False), batch_size=len(te_s))

    # 4) YAML 복사 + 길이 갱신
    cfg = copy.deepcopy(base_cfg)
    for k in ('seq_length','seq_len','max_seq_len','max_len','context_length'):
        if k in cfg["model"]["params"]:
            cfg["model"]["params"][k] = win_size
    cfg["model"]["params"]["feature_size"] = feat_num
    cfg["solver"]["save_cycle"] = 10**9             # 체크포인트 off

    # 5) 모델·트레이너
    model = instantiate_from_config(cfg["model"]).to(device)

    # PosEnc 길이 강제 교체
    try:
        from Models.interpretable_diffusion.model_utils import LearnablePositionalEncoding
        d_model = cfg["model"]["params"]["d_model"]
        if hasattr(model,'model') and hasattr(model.model,'pos_enc'):
            model.model.pos_enc = LearnablePositionalEncoding(d_model, max_len=win_size).to(device)
    except (ImportError, AttributeError):
        pass  # 경로가 다르면 생략

    from engine.solver import Trainer      # Diffusion-TS solver import
    trainer = Trainer(config=cfg, args=Args(), model=model,
                      dataloader={'dataloader': tr_loader})
    trainer.save = lambda *a,**k: None     # 체크포인트 저장 차단

    # 6) 학습·복원 반복
    merged = []
    for r in range(1, n_runs+1):
        print(f"[GPU{gpu_id}] seq={seq_len}  run {r}/{n_runs}", flush=True)
        trainer.train()
        samp, *_ = trainer.restore(te_loader, shape=[win_size, feat_num],
                                   coef=1e-2, stepsize=5e-2, sampling_steps=200)
        samp = scaler.inverse_transform(
                   unnormalize_to_zero_to_one(samp.reshape(-1,feat_num))
               ).reshape(te_s.shape)
        merged.append(samp)
    merged = np.concatenate(merged)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir)/f"merged_seq{seq_len}.pkl", "wb") as f:
        pickle.dump(merged, f)
    print(f"[GPU{gpu_id}] seq_len={seq_len} 완료  shape={merged.shape}", flush=True)

# ---------------- Main -----------------
def main():
    # (1) CSV → NumPy
    csv_path = "/home1/gkrtod35/ISF/TimeGAN/Origin_data/merged_data_processed_seoul.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[-(8640+24):]   # 최근 8640+24시간
    numeric_df = (df.drop(columns=['Idx','date','time','일시'])
                    .apply(pd.to_numeric, errors='coerce')
                    .to_numpy())

    # (2) 실험 설정
    seq_list = [720, 2160, 4320, 8640]
    pred_len, stride, n_runs = 24, 24, 100
    base_cfg = load_yaml_config(Args().config_path)

    gpus = list(range(torch.cuda.device_count()))
    if not gpus:
        raise RuntimeError("CUDA GPU를 찾을 수 없습니다!")

    # (3) spawn 병렬
    ctx   = mp.get_context("spawn")
    procs = []
    for i, sl in enumerate(seq_list):
        p = ctx.Process(target=run_pipeline,
                        args=(sl, pred_len, stride, n_runs,
                              numeric_df, base_cfg,
                              gpus[i % len(gpus)], "./merged_results"))
        p.start()
        procs.append(p)
    for p in procs: p.join()
    print("★★ 모든 실험 완료 ★★")

# ---------------- Entry -----------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
