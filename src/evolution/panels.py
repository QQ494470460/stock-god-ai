# -*- coding: utf-8 -*-
"""自我进化 · 样本外面板（给闸门吃的 A/B 数据）。

为什么单独一个模块
==================
闸门要在「同一份评分面板」上对比当前参数与候选参数。这份面板的构建很贵
（拉股票池 → 逐股造特征 → 用已部署模型打分，几百只股票要好几分钟），
所以：

* **构建是显式动作**（``scripts/evolve.py --build-panel``），不在 runner 里偷偷跑；
* **结果落盘缓存**到 ``data/evolution/panel/``，之后的 A/B 对比瞬间完成；
* 缓存不存在时 :func:`provider_from_cache` 返回 ``None``，
  闸门据此**判负并给出构建指引**，而不是编一份数据糊过去。

口径一致性
==========
打分口径刻意与 ``scripts/walk_forward_oos.run_deployed`` 保持一致
（scaler 的 ``feature_names_in_`` 决定特征列、``proba @ arange`` 作连续分数），
否则「闸门算出来的基线」与「线上真实表现」不是一回事，A/B 结论就没有意义。
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from config.settings import DATA_DIR, MODEL_DIR

__all__ = [
    "PANEL_DIR", "panel_available", "save_panel", "load_panel",
    "provider_from_cache", "build_panel", "panel_meta",
]

PANEL_DIR = Path(DATA_DIR) / "evolution" / "panel"

_SCORE = "score_panel.parquet"
_CLOSE = "close_panel.parquet"
_BENCH = "benchmark.parquet"
_META = "meta.json"

#: 部署模型的三件套（与 predictor 口径一致）
_MODEL_FILES = ("xgboost_model.pkl", "scaler.pkl", "features_xgboost.json")


# ---------------------------------------------------------------------------
# 缓存读写
# ---------------------------------------------------------------------------

def panel_available() -> bool:
    return (PANEL_DIR / _SCORE).exists() and (PANEL_DIR / _CLOSE).exists()


def panel_meta() -> Dict[str, Any]:
    try:
        p = PANEL_DIR / _META
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"面板元信息读取失败: {e}")
    return {}


def save_panel(score_panel, close_panel, benchmark=None,
               meta: Optional[Dict[str, Any]] = None) -> Path:
    """把面板落盘（parquet）。返回目录路径。"""
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    score_panel.to_parquet(PANEL_DIR / _SCORE)
    close_panel.to_parquet(PANEL_DIR / _CLOSE)
    if benchmark is not None:
        benchmark.to_frame("close").to_parquet(PANEL_DIR / _BENCH)

    info = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "n_days": int(score_panel.shape[0]),
        "n_codes": int(score_panel.shape[1]),
        "start": str(score_panel.index[0]),
        "end": str(score_panel.index[-1]),
    }
    info.update(meta or {})
    (PANEL_DIR / _META).write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"📦 样本外面板已缓存：{info['n_days']} 天 × {info['n_codes']} 只"
                f"（{info['start']} ~ {info['end']}）→ {PANEL_DIR}")
    return PANEL_DIR


def load_panel() -> Optional[Tuple[Any, Any, Optional[Any]]]:
    """读取缓存面板；缺失/损坏返回 ``None``（绝不抛异常）。"""
    try:
        import pandas as pd

        if not panel_available():
            return None
        score = pd.read_parquet(PANEL_DIR / _SCORE)
        close = pd.read_parquet(PANEL_DIR / _CLOSE)
        if score.empty or close.empty:
            return None
        bench = None
        bp = PANEL_DIR / _BENCH
        if bp.exists():
            bench = pd.read_parquet(bp)["close"]
        logger.info(f"📦 已加载样本外面板缓存：{score.shape[0]} 天 × "
                    f"{score.shape[1]} 只")
        return score, close, bench
    except Exception as e:  # noqa: BLE001
        logger.warning(f"面板缓存加载失败（忽略）: {e}")
        return None


def provider_from_cache():
    """给闸门用的面板提供者。缓存缺失时返回 ``None`` → 闸门判负。"""
    return lambda: load_panel()


# ---------------------------------------------------------------------------
# 构建（显式动作，重）
# ---------------------------------------------------------------------------

def build_panel(max_stocks: int = 200, pool: str = "hs300",
                min_cover: float = 0.6) -> Optional[Path]:
    """用**已部署模型**构建样本外评分面板并缓存。

    这是「显式、重、可观察」的动作，由 ``scripts/evolve.py --build-panel`` 触发。

    Returns:
        缓存目录；失败返回 ``None``。
    """
    import numpy as np
    import pandas as pd

    model_path = Path(MODEL_DIR) / _MODEL_FILES[0]
    scaler_path = Path(MODEL_DIR) / _MODEL_FILES[1]
    feat_path = Path(MODEL_DIR) / _MODEL_FILES[2]
    if not (model_path.exists() and scaler_path.exists()):
        logger.error(f"缺少部署模型文件（{model_path.name} / {scaler_path.name}）"
                     "，无法构建面板")
        return None

    try:
        import scripts.walk_forward_oos as wfo
    except Exception as e:  # noqa: BLE001
        logger.error(f"无法复用 walk_forward_oos 的装配逻辑: {e}")
        return None

    logger.info(f"🏗️ 构建样本外面板：pool={pool} max_stocks={max_stocks}")
    store: Dict[str, pd.DataFrame] = wfo.load_pool(max_stocks, pool=pool)
    if len(store) < 20:
        logger.error(f"股票池数据不足（{len(store)} 只），放弃构建")
        return None

    ref_cols: List[str] = wfo.build_feat_store(store)
    feat_store = wfo._FEAT_STORE                       # noqa: SLF001
    if not ref_cols or not feat_store:
        logger.error("特征构建失败，放弃构建面板")
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    deps = _resolve_feature_cols(scaler, feat_path, ref_cols, feat_store)

    # 收盘价面板（与训练口径一致：覆盖率过滤）
    close_panel = pd.DataFrame(
        {c: df["close"] for c, df in store.items() if c in feat_store}
    ).sort_index()
    cover = close_panel.notna().mean(axis=1)
    close_panel = close_panel.loc[cover >= min_cover]

    # 用部署模型逐股打分
    score_by_code: Dict[str, pd.Series] = {}
    for code, feat in feat_store.items():
        if code not in close_panel.columns:
            continue
        f = feat.reindex(columns=deps)
        if f.isna().all().all():
            continue
        try:
            xs = scaler.transform(f.fillna(0.0).values)
            proba = model.predict_proba(xs)
            score_by_code[code] = pd.Series(
                proba @ np.arange(proba.shape[1], dtype=float), index=f.index)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"打分失败 {code}: {e}")
            continue

    if not score_by_code:
        logger.error("没有任何可用分数，放弃构建面板")
        return None

    score_panel = pd.DataFrame(score_by_code).sort_index()
    idx = score_panel.index.intersection(close_panel.index)
    if len(idx) < 30:
        logger.error(f"对齐后仅 {len(idx)} 天，样本太少，放弃构建")
        return None
    score_panel = score_panel.loc[idx]
    close_panel = close_panel.loc[idx]

    return save_panel(score_panel, close_panel, None, {
        "pool": pool,
        "max_stocks": max_stocks,
        "n_features": len(deps),
        "score_source": model_path.name,
        "min_cover": min_cover,
    })


def _resolve_feature_cols(scaler, feat_path: Path, ref_cols: List[str],
                          feat_store: Dict[str, Any]) -> List[str]:
    """特征列口径：scaler 的 ``feature_names_in_`` 优先，其次 SHAP 列表。

    必须与线上推理一致 —— 用错特征列会得到一个「看起来正常」但
    与真实表现无关的分数面板，A/B 结论就全是假的。
    """
    fn = getattr(scaler, "feature_names_in_", None)
    if fn is not None and len(fn) > 0:
        return list(fn)
    if feat_path.exists():
        try:
            cols = json.loads(feat_path.read_text(encoding="utf-8")).get("features")
            if cols:
                return list(cols)
        except Exception:  # noqa: BLE001
            pass
    if ref_cols:
        return list(ref_cols)
    return list(next(iter(feat_store.values())).columns)
