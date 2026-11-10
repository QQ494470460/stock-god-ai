# -*- coding: utf-8 -*-
"""
Transformer 时序排序器腿的单元测试
=================================
1) TemporalTransformer 能在 CPU 上跑通（小随机输入 → 输出 [B, n_classes]）。
2) TransformerSubPredictor 在无 checkpoint 时优雅降级（available=False，绝不抛错），
   保证超级集成缺这一腿时不会拖垮主模型。
"""

import numpy as np
import pytest
import torch


def test_temporal_transformer_forward_cpu():
    from src.models.transformer_ranker import TemporalTransformer

    F = 24
    m = TemporalTransformer(input_size=F, d_model=64, nhead=4, num_layers=2)
    m.eval()
    x = torch.randn(4, 60, F)  # [B, seq_len, features]
    with torch.no_grad():
        out = m(x)
    assert out.shape == (4, 3), f"输出形状应为 [4,3]，实际 {tuple(out.shape)}"
    # 概率合理（softmax 后每行为正且和为 1 的近似由 head 直接输出，这里只查无 NaN/Inf）
    assert torch.isfinite(out).all(), "模型输出含 NaN/Inf"


def test_transformer_subpredictor_degrades_without_model(tmp_path, monkeypatch):
    import src.models.ensemble as ens

    # 指向空目录，确保没有 transformer_best.pt
    monkeypatch.setattr(ens, "MODEL_DIR", tmp_path)
    # 同时清掉模块内已缓存的候选路径（若有）
    sub = ens.TransformerSubPredictor()
    assert sub.available is False, "无 checkpoint 时必须优雅降级为不可用"
    # score_for 在不可用时返回 None，不抛
    assert sub.score_for(None, "600000") is None


def test_transformer_subpredictor_constructible():
    from src.models.ensemble import TransformerSubPredictor

    # 构造本身不应触发任何昂贵的加载
    sub = TransformerSubPredictor()
    assert sub.name == "transformer"
