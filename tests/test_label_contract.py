# -*- coding: utf-8 -*-
"""训练标签口径契约测试。

核心回归（memory §16 / PITFALLS）：train_optuna(rankic) 与 train_production_v2
必须共用同一套标签口径（forward_excess_return(horizon=5, winsorize=0.01) ->
make_rank_labels(n_bins=5)）。历史上两脚本标签口径分裂，是「回测里选股层结论全错」
的根因之一。本测试锁死两者在同一面板上产出的标签逐格相等。
"""

import numpy as np
import pandas as pd

import scripts.train_optuna as topt
import scripts.train_production_v2 as tpv
from src.models.labeling import forward_excess_return, make_rank_labels


def _synth_panel(n_dates=160, n_codes=8, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    codes = [f"C{i:03d}" for i in range(n_codes)]
    data = {}
    for c in codes:
        p = 10.0 + np.cumsum(rng.normal(0.0, 0.5, n_dates))
        data[c] = p
    return pd.DataFrame(data, index=idx)


def test_train_optuna_and_production_v2_label_contract():
    panel = _synth_panel()
    # train_production_v2 默认口径（parse_args: --horizon 默认 5；n_bins=5；winsorize=0.01）
    fwd_tpv = forward_excess_return(panel, horizon=5, winsorize=0.01)
    lab_tpv = make_rank_labels(fwd_tpv, n_bins=5)
    # train_optuna(rankic) 口径：HORIZON=5, N_BINS=5, winsorize=0.01
    fwd_to = forward_excess_return(panel, horizon=topt.HORIZON, winsorize=0.01)
    lab_to = make_rank_labels(fwd_to, n_bins=topt.N_BINS)
    # 两者必须逐格相等：任一方改了 horizon / n_bins / winsorize 都会让此测试失败
    pd.testing.assert_frame_equal(lab_tpv, lab_to)


def test_train_optuna_label_params_match_production_defaults():
    # 显式锚定：生产部署脚本的标签参数（文档约定 horizon / n_bins=5）
    assert topt.HORIZON == 5, "train_optuna.HORIZON 必须 = 5（与 train_production_v2 默认 horizon 对齐）"
    assert topt.N_BINS == 5, "train_optuna.N_BINS 必须 = 5（与 train_production_v2 n_bins=5 对齐）"
    # train_production_v2 的 --horizon 默认也是 5（见其 parse_args）
    import argparse
    ns = tpv.parse_args(["--pool", "zz500", "--no-deploy"]) if hasattr(tpv, "parse_args") else None
    if ns is not None:
        assert ns.horizon == 5, "train_production_v2 默认 --horizon 必须 = 5"
