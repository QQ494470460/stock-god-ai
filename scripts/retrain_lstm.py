# -*- coding: utf-8 -*-
"""
重训 LSTM 并写入特征元数据（让超级集成的 LSTM 腿可用）
====================================================

旧版 lstm_best.pt 没有保存训练特征列，推理时无法对齐输入，故一直被闲置。
本脚本在与 XGBoost **完全相同的生产特征**上重训 LSTM，并在 checkpoint 里写入：
    meta = {"feature_cols": [...], "classes": [-1, 0, 1], "horizon": 5}
供 src/models/ensemble.py 的 LSTMSubPredictor 直接加载并对齐输入。

标签：逐股未来 horizon 日收益方向（三分类 涨/震荡/跌），阈值取自 config.ml。
训练：逐股构建不跨界序列后汇总（避免股票边界泄漏），CPU 可跑。

用法
----
    py -3.12 scripts/retrain_lstm.py --max-stocks 60 --epochs 15
    py -3.12 scripts/retrain_lstm.py --max-stocks 120 --epochs 30   # 更充分
"""

import sys
import argparse
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from loguru import logger  # noqa: E402
logger.remove()
logger.add(sys.stderr, level="INFO")

from config.settings import MODEL_DIR, config as app_config  # noqa: E402
from scripts.train_v2 import load_pool  # noqa: E402
from src.features.engine import feature_engine  # noqa: E402


def build_sequences(store, seq_len: int, horizon: int, up: float, down: float):
    """逐股构建不跨界序列，返回 (X, y) numpy 数组。

    X: [N, seq_len, F]，y: [N]（三分类 0/1/2，对应 跌/震/涨）。
    """
    X_parts, y_parts = [], []
    ref_cols = None
    kept = 0
    for code, df in store.items():
        try:
            feats = feature_engine.build_features(df)
            if "close" in feats.columns:
                feats = feats.drop(columns=["close"])
        except Exception:
            continue
        if feats.shape[0] < seq_len + horizon + 5 or feats.isna().all().all():
            continue
        if ref_cols is None:
            ref_cols = list(feats.columns)
        feats = feats.reindex(columns=ref_cols).fillna(0.0)

        close = df["close"].reindex(feats.index)
        fwd = close.shift(-horizon) / close - 1.0
        lab = np.where(fwd.values > up, 1, np.where(fwd.values < down, -1, 0))

        vals = feats.values.astype(float)
        n = len(vals)
        for i in range(n - seq_len):
            j = i + seq_len - 1  # 预测日索引（窗口末）
            if j >= len(lab) or np.isnan(fwd.values[j]):
                continue
            X_parts.append(vals[i:i + seq_len])
            y_parts.append(lab[j] + 1)  # -> 0/1/2
        kept += 1
    if not X_parts:
        raise RuntimeError("无可用序列，请检查数据/特征")
    logger.info(f"  构建序列: {len(X_parts)} 条 × seq_len={seq_len} × F={len(ref_cols)} | 股票 {kept} 只")
    return np.array(X_parts, dtype=np.float32), np.array(y_parts, dtype=np.int64), ref_cols


def main() -> int:
    ap = argparse.ArgumentParser(description="重训 LSTM（带特征元数据）")
    ap.add_argument("--max-stocks", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from src.models.lstm import StockLSTM

    horizon = app_config.ml.forecast_days
    up = app_config.ml.threshold_up
    down = app_config.ml.threshold_down

    print("=" * 80)
    print(" 选股神 · 重训 LSTM（写特征元数据，供超级集成 LSTM 腿加载）")
    print("=" * 80)

    print(f"\n[1/3] 加载数据（最多 {args.max_stocks} 只）...")
    store = load_pool(args.max_stocks, pool="legacy")
    if len(store) < 20:
        print("  数据不足，退出"); return 1
    print(f"  股票: {len(store)} 只")

    print("\n[2/3] 构建序列（逐股不跨界）...")
    X, y, ref_cols = build_sequences(store, args.seq_len, horizon, up, down)
    F = len(ref_cols)
    print(f"  X={X.shape} y 分布: " +
          ", ".join(f"{k}={int((y==k).sum())}" for k in (0, 1, 2)))

    # 时间序切分（最后 15% 验证）
    split = int(len(X) * 0.85)
    tr = TensorDataset(torch.FloatTensor(X[:split]), torch.LongTensor(y[:split]))
    va = TensorDataset(torch.FloatTensor(X[split:]), torch.LongTensor(y[split:]))

    print(f"\n[3/3] 训练 LSTM（epochs={args.epochs}, device=cpu）...")
    device = torch.device("cpu")
    params = app_config.ml.lstm_params
    model = StockLSTM(input_size=F, hidden_size=params["hidden_size"],
                      num_layers=params["num_layers"], dropout=params["dropout"]).to(device)
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False)

    counts = np.bincount(y[:split], minlength=3).astype(float)
    cw = 1.0 / (counts + 1e-9); cw = cw / cw.sum() * 3
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(cw).to(device))
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = 0.0
    best_state = None
    for ep in range(args.epochs):
        model.train()
        tl = 0.0
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for bx, by in va_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx).argmax(1)
                correct += (pred == by).sum().item(); total += by.size(0)
        acc = correct / max(total, 1)
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        logger.info(f"  epoch {ep+1}/{args.epochs} loss={tl/len(tr_loader):.4f} val_acc={acc:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    out = MODEL_DIR / "lstm_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": F,
        "sequence_length": args.seq_len,
        "meta": {"feature_cols": ref_cols, "classes": [-1, 0, 1], "horizon": horizon},
    }, out)
    print(f"\n✅ LSTM 已重训并保存: {out}（最佳验证准确率 {best_acc:.4f}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
