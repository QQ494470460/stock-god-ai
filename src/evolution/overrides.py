# -*- coding: utf-8 -*-
"""自我进化 · 运行期覆盖层。

为什么需要它
============
`config/settings.py` 里的 ``config = AppConfig()`` 是**硬编码默认值**，
没有任何外部加载 —— 也就是说 `trading_loop.max_positions` 这类参数
以前只能改代码。自我进化要落地改动，就必须有一个**运行期可写的覆盖层**。

方案
====
1. 非 ``evolution.*`` 的参数 → 写 ``CONFIG_DIR/evolution_overrides.json``
   （外挂目录，重建 exe 不丢，与 `trading_guard.yaml` 同级）。
   `config/settings.py` 在构造 ``config`` 后调用
   :func:`config.settings.load_evolution_overrides` 应用之。
2. ``evolution.*`` 的参数（选股阈值）→ 写 ``data/evolution_state.json``，
   由已有的 `EvolutionEngine` 管理，避免两套真相。

所有写入前都会经 `space.validate()` 校验，**风控硬线一律拒绝**。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from config.settings import CONFIG_DIR
from src.evolution import space
from src.evolution.types import Change

__all__ = [
    "OVERRIDES_FILE", "load_overrides", "save_overrides", "apply_changes",
    "read_effective", "refresh_app_config", "overrides_summary",
]

OVERRIDES_FILE = Path(CONFIG_DIR) / "evolution_overrides.json"

# 走 evolution_state.json 的命名空间（其余走 overrides 文件）
_EVOLUTION_NS = "evolution."


# ---------------------------------------------------------------------------
# overrides 文件
# ---------------------------------------------------------------------------

def load_overrides() -> Dict[str, Any]:
    """读取覆盖层（文件缺失/损坏 → 空字典，绝不抛异常）。

    文件结构为 ``{"_comment":…, "_updated_at":…, "overrides":{key: value}}``，
    本函数返回**内层** ``overrides`` 字典；为兼容早期写平铺结构的文件，
    若没有 ``overrides`` 键则退回顶层（滤掉 ``_`` 开头的元数据键）。
    """
    try:
        if OVERRIDES_FILE.exists():
            data = json.loads(OVERRIDES_FILE.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}
            inner = data.get("overrides")
            if isinstance(inner, dict):
                return {k: v for k, v in inner.items() if isinstance(k, str)}
            # 兼容平铺写法
            return {k: v for k, v in data.items()
                    if isinstance(k, str) and not k.startswith("_")}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"进化覆盖层读取失败（忽略）: {e}")
    return {}


def save_overrides(d: Dict[str, Any]) -> None:
    OVERRIDES_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": "由「自我进化」自动维护。删除本文件即恢复全部默认参数。",
        "_updated_at": datetime.now().isoformat(timespec="seconds"),
        "overrides": d,
    }
    OVERRIDES_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def overrides_summary() -> Dict[str, Any]:
    """给 UI/CLI 展示：当前覆盖了哪些参数、值是多少、何时写的。"""
    raw = {}
    try:
        if OVERRIDES_FILE.exists():
            raw = json.loads(OVERRIDES_FILE.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        raw = {}
    return {
        "file": str(OVERRIDES_FILE),
        "exists": OVERRIDES_FILE.exists(),
        "updated_at": raw.get("_updated_at"),
        "overrides": raw.get("overrides", {}) or {},
    }


# ---------------------------------------------------------------------------
# 读写生效值
# ---------------------------------------------------------------------------

def _evolution_engine():
    from src.models.evolution import EvolutionEngine
    return EvolutionEngine()


def read_effective(key: str) -> Any:
    """读取某参数的**当前生效值**（含已应用的覆盖）。"""
    space.get_spec(key)                    # 顺带做保护/登记校验
    if key.startswith(_EVOLUTION_NS):
        leaf = key[len(_EVOLUTION_NS):]
        return _evolution_engine().state.get(leaf, space.get_default(key))
    try:
        from config.settings import config as app_config
    except Exception:  # noqa: BLE001
        return space.get_default(key)
    return space.get_value(app_config, key)


def _write_evolution_ns(key: str, value: Any) -> None:
    eng = _evolution_engine()
    leaf = key[len(_EVOLUTION_NS):]
    eng.state[leaf] = value
    eng.save()


def _write_override(key: str, value: Any) -> None:
    ov = load_overrides()
    ov[key] = value
    save_overrides(ov)


def apply_changes(changes: List[Change]) -> List[str]:
    """把一组变更写进运行期（调用方负责先做闸门与批准）。

    Returns:
        实际生效的 key 列表

    Raises:
        space.ParamError: 任一改动非法（越界 / 受保护 / 未登记）。
            抛出前**不会**写入任何内容（先全校验，后统一写）。
    """
    validated: List[Tuple[str, Any]] = []
    for c in changes:
        v = space.validate(c.key, c.new)   # 风控硬线在这里被拒
        validated.append((c.key, v))

    applied: List[str] = []
    evo_updates: Dict[str, Any] = {}
    file_updates: Dict[str, Any] = {}
    for key, value in validated:
        if key.startswith(_EVOLUTION_NS):
            evo_updates[key[len(_EVOLUTION_NS):]] = value
        else:
            file_updates[key] = value

    if evo_updates:
        eng = _evolution_engine()
        eng.state.update(evo_updates)
        eng.save()
        applied += [f"{_EVOLUTION_NS}{k}" for k in evo_updates]
    if file_updates:
        ov = load_overrides()
        ov.update(file_updates)
        save_overrides(ov)
        applied += list(file_updates)

    refresh_app_config()
    return applied


def refresh_app_config() -> List[str]:
    """把覆盖层重新应用到已导入的 ``config`` 单例上（无需重启进程）。

    注意：本函数**只做「加」，不做「减」** —— 覆盖为空时直接返回，
    不会把此前覆盖过的参数退回默认值。撤销覆盖请用 :func:`reset_all`，
    它会在清空文件后显式把每个 key 写回 ``space`` 里的默认值。

    Returns:
        被应用的 key 列表
    """
    ov = load_overrides()
    if not ov:
        return []
    try:
        from config.settings import config as app_config
    except Exception as e:  # noqa: BLE001
        logger.warning(f"刷新覆盖层失败: {e}")
        return []
    applied = []
    for key, value in ov.items():
        try:
            if space.is_protected(key) or key not in space.PARAM_SPACE:
                logger.warning(f"覆盖层含非法参数 {key}，已跳过")
                continue
            v = space.validate(key, value)
            space.set_value(app_config, key, v)
            applied.append(key)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"应用覆盖 {key} 失败: {e}")
    return applied


def _restore_defaults(keys: List[str]) -> List[str]:
    """把一组 key 显式写回参数空间默认值。

    存在的理由：光把 overrides 文件清空，**内存里已生效的值不会自己变回去**。
    这是「点重置但没重置」的典型成因，故撤销必须主动回写。
    """
    restored: List[str] = []
    evo_defaults: Dict[str, Any] = {}

    try:
        from config.settings import config as app_config
    except Exception:  # noqa: BLE001
        app_config = None

    for key in keys:
        try:
            if space.is_protected(key) or key not in space.PARAM_SPACE:
                continue
            default = space.get_default(key)
            if key.startswith(_EVOLUTION_NS):
                evo_defaults[key[len(_EVOLUTION_NS):]] = default
            elif app_config is not None:
                space.set_value(app_config, key, default)
            restored.append(key)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"恢复 {key} 默认值失败: {e}")

    if evo_defaults:
        try:
            eng = _evolution_engine()
            eng.state.update(evo_defaults)
            eng.save()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"恢复 evolution.* 默认值失败: {e}")
    return restored


def reset_all() -> List[str]:
    """清除**全部**覆盖（恢复默认参数）。返回真正被重置的 key。

    两处都要清，否则就是「假重置」：

    * 文件覆盖层 ``CONFIG_DIR/evolution_overrides.json``；
    * ``evolution.*`` —— 它们写在 ``data/evolution_state.json`` 里，
      **不在覆盖层文件中**，故必须按参数空间逐项比对，挑出
      「当前值 ≠ 默认值」的才算被改过（不然返回一堆没动过的 key，
      调用方无法判断到底重置了什么）。
    """
    keys = list(load_overrides().keys())

    for key in space.PARAM_SPACE:
        if not key.startswith(_EVOLUTION_NS):
            continue
        try:
            if read_effective(key) != space.get_default(key):
                keys.append(key)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"检查 {key} 是否被覆盖失败: {e}")

    keys = sorted(set(keys))
    save_overrides({})
    return _restore_defaults(keys)
