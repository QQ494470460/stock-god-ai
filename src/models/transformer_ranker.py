# -*- coding: utf-8 -*-
"""
Transformer 深度时序排序器
=========================
替代 / 增强现有 LSTM 腿的更强大时序模型：对逐股特征序列做多头自注意力，
捕捉长程时序依赖与跨特征交互，输出 涨/震/跌 三分类概率。

设计严格对齐 scripts/retrain_lstm.py 的契约，保证与 XGBoost / LSTM 腿「同特征、
同标签、同序列窗」，从而可无缝并入超级集成（EnsemblePredictor）：
  - 输入：feature_engine.build_features(df).drop(columns=["close"]) 的 [seq_len, F] 序列；
  - 标签：未来 horizon 日收益方向（与 retrain_lstm 完全一致）；
  - checkpoint 写入 meta={feature_cols, classes, horizon}，供 TransformerSubPredictor 对齐输入。

CPU 友好：d_model=64 / nhead=4 / 2 层；序列长 60 时显存与算力占用远低于大模型。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """正弦位置编码（batch_first）。"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """时序自注意力排序器（TFT-lite）。

    流程：特征投影 -> LayerNorm -> 位置编码 -> TransformerEncoder ->
    时序注意力池化 -> 分类头。多头自注意力让模型在「时间轴」上直接比较任意两日
    的状态，弥补 LSTM 只能顺序传播、长程记忆衰减的短板。
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.3,
        num_classes: int = 3,  # 涨/震/跌
        max_len: int = 512,
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.num_classes = num_classes

        self.input_proj = nn.Linear(input_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 时序注意力池化（与 LSTM 腿一致的注意力机制，把变长时序压成定长上下文）
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, input_size]
        h = self.input_norm(self.input_proj(x))
        h = self.pos_enc(h)
        h = self.encoder(h)                      # [B, seq_len, d_model]
        w = self.attn(h)                         # [B, seq_len, 1]
        w = torch.softmax(w, dim=1)
        ctx = torch.sum(w * h, dim=1)            # [B, d_model]
        out = self.dropout(ctx)
        return self.head(out)
