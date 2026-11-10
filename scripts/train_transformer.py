# -*- coding: utf-8 -*-
"""
重训 Transformer 时序排序器并写入特征元数据（供超级集成 Transformer 腿加载）
============================================================================

与 retrain_lstm.py 完全对齐的数据契约：复用其 build_sequences，确保
Transformer 腿在**完全相同的生产特征、完全相同的未来 horizon 日收益标签、
完全相同的序列窗**上训练，从而与 XGBoost / Alpha / LSTM 三腿无缝融合。

模型：src/models/transformer_ranker.py 的 TemporalTransformer
（多头自注意力，捕捉长程时序依赖，强于 LSTM 的顺序传播）。

用法
----
    py -3.12 scripts/train_transformer.py --max-stocks 60 --epochs 15
    py -3.12 scripts/train_transformer.py --max-stocks 150 --epochs 40 --model-dir models
    # 冒烟/调试：写到临时目录，不污染生产 models/
    py -3.12 scripts/train_transformer.py --max-stocks 20 --epochs 3 --model-dir /tmp/tf_smoke
"""

import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from loguru import logger  # noqa: E402
logger.remove()
logger.add(sys.stderr, level="INFO")

from config.settings import MODEL_DIR, config as app_config  # noqa: E402
from scripts.retrain_lstm import build_sequences  # noqa: E402
from scripts.train_v2 import load_pool  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="重训 Transformer 时序排序器（带特征元数据）")
    ap.add_argument("--max-stocks", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model-dir", type=str, default=str(MODEL_DIR),
                    help="输出目录（默认 config MODEL_DIR）；冒烟可指临时目录")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from src.models.transformer_ranker import TemporalTransformer

    horizon = app_config.ml.forecast_days
    up = app_config.ml.threshold_up
    down = app_config.ml.threshold_down
    params = app_config.ml.transformer_params

    print("=" * 80)
    print(" 选股神 · 重训 Transformer 时序排序器（供超级集成 Transformer 腿加载）")
    print("=" * 80)

    print(f"\n[1/3] 加载数据（最多 {args.max_stocks} 只）...")
    store = load_pool(args.max_stocks, pool="legacy")
    if len(store) < 20:
        print("  数据不足，退出")
        return 1
    print(f"  股票: {len(store)} 只")

    print("\n[2/3] 构建序列（复用 retrain_lstm.build_sequences，同 LSTM 契约）...")
    X, y, ref_cols = build_sequences(store, args.seq_len, horizon, up, down)
    F = len(ref_cols)
    print(f"  X={X.shape} y 分布: " +
          ", ".join(f"{k}={int((y==k).sum())}" for k in (0, 1, 2)))

    # 时间序切分（最后 15% 验证）
    split = int(len(X) * 0.85)
    tr = TensorDataset(torch.FloatTensor(X[:split]), torch.LongTensor(y[:split]))
    va = TensorDataset(torch.FloatTensor(X[split:]), torch.LongTensor(y[split:]))

    print(f"\n[3/3] 训练 Transformer（epochs={args.epochs}, device=cpu）...")
    device = torch.device("cpu")
    model = TemporalTransformer(
        input_size=F,
        d_model=params["d_model"],
        nhead=params["nhead"],
        num_layers=params["num_layers"],
        dim_feedforward=params["dim_feedforward"],
        dropout=params["dropout"],
    ).to(device)
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False)

    counts = np.bincount(y[:split], minlength=3).astype(float)
    cw = 1.0 / (counts + 1e-9)
    cw = cw / cw.sum() * 3
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
                correct += (pred == by).sum().item()
                total += by.size(0)
        acc = correct / max(total, 1)
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        logger.info(f"  epoch {ep+1}/{args.epochs} loss={tl/len(tr_loader):.4f} val_acc={acc:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    out_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "transformer_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": F,
        "sequence_length": args.seq_len,
        "meta": {"feature_cols": ref_cols, "classes": [-1, 0, 1], "horizon": horizon},
    }, out)
    print(f"\n✅ Transformer 已重训并保存: {out}（最佳验证准确率 {best_acc:.4f}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
