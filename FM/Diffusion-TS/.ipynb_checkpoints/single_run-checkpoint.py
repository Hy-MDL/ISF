# ------------------------------------------------------------
# 0) 공통 준비  (CSV → df, 하이퍼파라미터 고정)
# ------------------------------------------------------------
import os, gc, copy, pickle, importlib, yaml
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from engine.solver import Trainer
from Utils.io_utils import load_yaml_config, instantiate_from_config
from Models.interpretable_diffusion.model_utils import (
    normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
)

df_full = pd.read_csv(
    "/home1/gkrtod35/Diffusion-TS/Data/merged_data_processed_seoul.csv",
    low_memory=False
)

SEQ_LEN_FIXED  = 24       # 입력 24
PRED_LEN       = 24       # 예측 24
STRIDE         = 24
WIN_SIZE       = SEQ_LEN_FIXED + PRED_LEN   # 48
BATCH_SIZE     = 32
N_RUNS         = 200
FEAT_NUM       = 19       # 열 개수
CFG_PATH       = "./Config/solar.yaml"

class SeqDataset(Dataset):
    def __init__(self, data, pred_len=PRED_LEN, regular=True):
        self.data, self.regular = data, regular
        mask = np.ones_like(data, bool)
        mask[:, -pred_len:, :] = False
        self.mask = mask
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx]).float()
        if self.regular: return x
        m = torch.from_numpy(self.mask[idx]).bool()
        return x, m

def diffusion_run(seq_len_recent: int, gpu_id: int = 0):
    """
    seq_len_recent : 720 / 2160 / 4320 / 8640
                     (실제 자르는 행수는 +24)
    """
    # ── 1) 최근 구간 잘라오기 ──────────────────────
    cut_len = seq_len_recent + PRED_LEN          # 744, 2184, …
    df = df_full[-cut_len:]
    numeric = (df.drop(columns=['Idx','date','time','일시'])
                 .apply(pd.to_numeric, errors='coerce')
                 .to_numpy())

    # ── 2) 슬라이딩 윈도우 (고정 길이 48) ──────────
    seqs = np.stack([numeric[i:i+WIN_SIZE]
                     for i in range(0, len(numeric)-WIN_SIZE+1, STRIDE)])
    if len(seqs) == 0:
        print(f"[{seq_len_recent}] 창 0개 → 스킵")
        return

    n_train = int(len(seqs)*0.8) or 1            # 최소 1
    train, test = seqs[:n_train], seqs[n_train:]

    scaler = MinMaxScaler()
    train_s = normalize_to_neg_one_to_one(
        scaler.fit_transform(train.reshape(-1, FEAT_NUM))
    ).reshape(train.shape)
    test_s  = normalize_to_neg_one_to_one(
        scaler.transform(test.reshape(-1, FEAT_NUM))
    ).reshape(test.shape)

    tr_loader = DataLoader(SeqDataset(train_s, regular=True),
                           batch_size=BATCH_SIZE, shuffle=True)
    te_loader = DataLoader(SeqDataset(test_s,  regular=False),
                           batch_size=len(test_s))

    # ── 3) 모델 & 트레이너 ─────────────────────────
    cfg = load_yaml_config(CFG_PATH)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model  = instantiate_from_config(cfg["model"]).to(device)
    trainer = Trainer(config=cfg, args=type("A", (), {"gpu":gpu_id})(),
                      model=model, dataloader={'dataloader': tr_loader})
    trainer.save = lambda *a,**k: None    # 체크포인트 끔

    # ── 4) 반복 학습·복원 ──────────────────────────
    merged = []
    for r in range(1, N_RUNS+1):
        print(f"[GPU{gpu_id}] {seq_len_recent} run {r}/{N_RUNS}", flush=True)
        trainer.train()
        samp, *_ = trainer.restore(
            te_loader, shape=[WIN_SIZE, FEAT_NUM],
            coef=1e-2, stepsize=5e-2, sampling_steps=200
        )
        samp = scaler.inverse_transform(
                   unnormalize_to_zero_to_one(samp.reshape(-1, FEAT_NUM))
               ).reshape(test_s.shape)
        merged.append(samp)
        del samp; torch.cuda.empty_cache(); gc.collect()

    merged = np.concatenate(merged)
    out = f"merged_results/merged_seq{seq_len_recent}.pkl"
    os.makedirs("merged_results", exist_ok=True)
    pickle.dump(merged, open(out, "wb"))
    print(f"✔ {seq_len_recent} 종료 → {merged.shape}")

# ------------------------------------------------------------
# 5) 여러 길이를 순차 or 병렬 실행
#    (아래는 순차 예시, 필요하면 subprocess 로 분배 가능)
# ------------------------------------------------------------
for seq_len_recent in [720, 2160, 4320, 8640]:
    diffusion_run(seq_len_recent, gpu_id=0)   # gpu_id 원하는 값으로