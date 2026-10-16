# -*- coding: utf-8 -*-
"""
模型注册中心（Model Registry）
=============================

把 `models/` 目录下所有"带指标"的模型产物登记造册，按**样本外 RankIC**
自动择优，解决两个真实痛点：

1. 当前 predictor 硬编码加载 `xgboost_model.pkl`（OOS RankIC 0.0747），
   而 `leaksafe_xgboost_model.pkl` 只有 0.049（更弱却也躺在目录里）。
   注册中心让你一眼看清「当前跑的是不是最好的那一个」。

2. `walk_forward_alpha.json` 报告了 RankIC 0.123 的 alpha 模型，但那只
   是**实验日志**，并没有存成可加载的 `.pkl`。注册中心会把它标记为
   「最优但不可部署」，明确提示「需先训练生成 pkl 才能上线」。

设计原则
--------
- 只读扫描，绝不动模型文件。
- 「可部署」的严格判定：必须有 model_path 指向已存在的 .pkl，且对应的
  scaler / features 文件齐全，且特征体系与主流 predictor 管线一致。
- 不同特征体系的模型（如 leaksafe）标记为不可直接热替换，避免特征错位崩溃。
"""

from __future__ import annotations

import json
import pickle
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from config.settings import CONFIG_DIR, MODEL_DIR


# 版本化产物根目录：每个「可切换」的模型产物放在 versions/<version_id>/ 下，
# 含 model.pkl / scaler.pkl / label_encoder.pkl / features.json 四件套 + <id>.json 指标。
# 这样训练不再只是原地覆盖 models/xgboost_model.pkl（「根本没有另一个可切换的产物」），
# 而是一个带明确 id 的版本，经人工批准名单（或 ml.active_version 钉子）后由 predictor 真正加载。
VERSION_ROOT = MODEL_DIR / "versions"


@dataclass
class ModelInfo:
    """单个已登记模型的信息"""
    key: str                                    # 唯一键，如 "production_metrics" / "alpha_xgboost"
    display_name: str = ""                      # 展示名（测试可省略；生产登记均显式赋值）
    model_path: Optional[Path] = None           # 可加载的 .pkl（无则 None）
    prefix: str = ""                            # 文件前缀（如 "alpha"），用于定位 scaler/features
    rank_ic: float = float("nan")               # 样本外 RankIC（主排序指标）
    ic_ir: float = float("nan")
    win_rate: float = float("nan")
    top_excess: Optional[float] = None
    n_samples: Optional[int] = None
    # 能否**作为唯一主模型**被 predictor 管线加载。
    # 注意这不是「这个产物有没有用」—— alpha 模型很有用，但它带着自己的一套
    # alpha 特征列（来自 parquet 缓存），塞进主管线会因特征错位直接崩，
    # 所以它必须 loadable=False。历史上这里只检查了主模型三件套是否存在，
    # 于是 alpha 被误判为「可部署最优」，让注册表报出一个**永远不会被加载**的冠军。
    loadable: bool = False
    role: str = "main"                          # main | ensemble_leg | log_only
    note: str = ""


# 主流 predictor 管线使用的特征/预处理文件名（用于判定可部署）
_MAIN_FAMILY_FILES = ("xgboost_model.pkl", "scaler.pkl", "features_xgboost.json")


@dataclass
class VersionSet:
    """一个模型版本对应的「模型 + 预处理 + 特征」四件套路径，及完整性。

    - ``legacy=True`` 表示这是遗留平铺布局（models/xgboost_model.pkl + 平铺
      scaler.pkl / features_xgboost.json），predictor 沿用旧路径加载，行为不变。
    - ``legacy=False`` 表示版本化布局（versions/<id>/model.pkl 等四件套）。
    - ``complete``：四件套是否齐全。predictor 加载前必查此项，不齐则**拒绝**
      该版本并回退到遗留默认——这就是「拒绝三件套不齐的版本集」。
    """
    model_path: Path
    scaler_path: Path
    label_encoder_path: Path
    features_path: Path
    version_id: Optional[str] = None
    legacy: bool = False
    complete: bool = False


def version_set_for(info: "ModelInfo", model_dir: Optional[Path] = None) -> Optional[VersionSet]:
    """由一条 ModelInfo 解析出它应当加载的四件套路径与完整性。

    布局判定规则（只看 model_path 落在哪）：
    - 在 ``versions/<id>/`` 下 → 版本化：四件套都取该目录内的
      model.pkl / scaler.pkl / label_encoder.pkl / features.json。
    - 否则（平铺在 MODEL_DIR 根）→ 遗留：model 即 model_path，
      scaler/label_encoder 取 MODEL_DIR 根的 scaler.pkl / label_encoder.pkl，
      features 取 ``features_{prefix}.json``（prefix 为空时即 features_xgboost.json）。

    返回 None 仅当 info 无 model_path；其余情况都返回 VersionSet，
    ``complete`` 如实反映四件套是否齐全。
    """
    root = Path(model_dir) if model_dir else MODEL_DIR
    mpath = getattr(info, "model_path", None)
    if not mpath:
        return None
    mpath = Path(mpath)
    vroot = root / "versions"

    if mpath.parent.parent == vroot:  # versions/<id>/model.pkl
        d = mpath.parent
        vid = d.name
        model = mpath
        scaler = d / "scaler.pkl"
        le = d / "label_encoder.pkl"
        feats = d / "features.json"
        legacy = False
    else:
        prefix = getattr(info, "prefix", "") or "xgboost"
        d = mpath.parent
        model = mpath
        scaler = d / "scaler.pkl"
        le = d / "label_encoder.pkl"
        feats = d / f"features_{prefix}.json"
        vid = None
        legacy = True

    complete = model.exists() and scaler.exists() and le.exists() and feats.exists()
    return VersionSet(
        model_path=model, scaler_path=scaler, label_encoder_path=le,
        features_path=feats, version_id=vid, legacy=legacy, complete=complete,
    )


def write_version_bundle(
    version_id: str,
    *,
    model: Any = None,
    scaler: Any = None,
    label_encoder: Any = None,
    features: Optional[List[str]] = None,
    metrics: Optional[dict] = None,
) -> Path:
    """把一个训练产物落盘为一个带明确 id 的版本化副本（四件套 + 指标）。

    这是「产物按版本命名」的真正落点：训练完不再只覆盖 xgboost_model.pkl，
    而是额外写一份 ``versions/<version_id>/`` 下的可切换产物。predictor 在
    该版本被批准/钉住后，会经由 version_set_for 真正加载它。

    任何单文件写入失败都不影响其余文件（各自 try），但最终若模型文件缺失则抛错。
    """
    vid = str(version_id).strip()
    if not vid:
        raise ValueError("version_id 不能为空")
    # 用模块级 MODEL_DIR（可被测试 monkeypatch）动态推导 versions 根，
    # 而非导入期冻结的 VERSION_ROOT 常量，确保测试隔离到临时目录。
    vdir = (MODEL_DIR / "versions") / vid
    vdir.mkdir(parents=True, exist_ok=True)

    if model is not None:
        with open(vdir / "model.pkl", "wb") as f:
            pickle.dump(model, f)
    if scaler is not None:
        with open(vdir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
    if label_encoder is not None:
        with open(vdir / "label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)
    if features is not None:
        with open(vdir / "features.json", "w", encoding="utf-8") as f:
            json.dump({"features": list(features)}, f, ensure_ascii=False, indent=2)

    # 指标 JSON（供 scan_models 登记；rank_ic 缺失时仍登记为可加载候选）
    if metrics is not None:
        metrics = dict(metrics)
        metrics.setdefault("model_path", f"versions/{vid}/model.pkl")
        with open(vdir / f"{vid}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    if not (vdir / "model.pkl").exists():
        raise RuntimeError(f"版本 {vid} 的模型文件未能落盘：{vdir / 'model.pkl'}")
    logger.info(f"📦 已写入版本化产物：versions/{vid}/（model+scaler+encoder+features）")
    return vdir



def _safe_num(v):
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _parse_metrics_file(path: Path) -> List[ModelInfo]:
    """解析单个 metrics / walk-forward json，返回一个或多个 ModelInfo"""
    out: List[ModelInfo] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug(f"模型指标解析失败 {path.name}: {e}")
        return out

    # —— 情况 A：production_metrics.json 形式（含 walk_forward 块 + model_path）
    wf = data.get("walk_forward") or {}
    mp = data.get("model_path")
    wf_ric = _safe_num(wf.get("rank_ic"))
    # 必须有数值型 rank_ic 才登记为生产类模型，避免 optuna/diag 等实验产物混入
    if not np.isnan(wf_ric):
        prefix = ""  # 主流模型前缀为空
        model_path = Path(mp) if mp else (MODEL_DIR / "xgboost_model.pkl")
        if not model_path.is_absolute():
            model_path = MODEL_DIR / model_path
        mp_exists = model_path.exists()
        files_ok = mp_exists and all((MODEL_DIR / f).exists() for f in _MAIN_FAMILY_FILES)
        # 用文件名做唯一键，避免多个 production 类文件都叫 "xgboost"
        stem = path.stem
        disp = "XGBoost（主流生产模型）" if stem == "production_metrics" else f"XGBoost（{stem}）"
        out.append(ModelInfo(
            key=stem,
            display_name=disp,
            model_path=model_path if mp_exists else None,
            prefix=prefix,
            rank_ic=wf_ric if not np.isnan(wf_ric) else _safe_num(data.get("oos_rank_ic")),
            ic_ir=_safe_num(wf.get("ic_ir", data.get("oos_ic_ir"))),
            win_rate=_safe_num(wf.get("top_win_rate", data.get("oos_win_rate"))),
            top_excess=wf.get("top_excess", data.get("oos_top_excess")),
            n_samples=wf.get("n_days"),
            loadable=bool(files_ok),
            role="main",
            note="与主流 predictor 管线特征体系一致" if files_ok
                  else "主模型三件套（pkl/scaler/features）不全，暂不可部署",
        ))
        return out

    # —— 情况 B：leaksafe_metrics.json 形式（oos_rank_ic 直接在外层）
    if "oos_rank_ic" in data:
        prefix = "leaksafe"
        model_path = MODEL_DIR / f"{prefix}_xgboost_model.pkl"
        # leaksafe 用独立特征集（leaksafe_features.json），与主流管线不一致 → 不可直接热替换
        files_ok = all(
            (MODEL_DIR / f"{prefix}_{suffix}").exists()
            for suffix in ("xgboost_model.pkl", "scaler.pkl", "features.json")
        )
        out.append(ModelInfo(
            key="leaksafe",
            display_name="XGBoost（leaksafe 防泄漏版）",
            model_path=model_path if model_path.exists() else None,
            prefix=prefix,
            rank_ic=_safe_num(data.get("oos_rank_ic")),
            ic_ir=_safe_num(data.get("oos_ic_ir")),
            win_rate=_safe_num(data.get("oos_win_rate")),
            top_excess=data.get("oos_top_excess"),
            n_samples=data.get("n_oos_samples"),
            loadable=False,  # 独立特征体系，不是主模型候选
            role="log_only",
            note="使用独立 leaksafe 特征集，与主流管线不一致；"
                 "只作对照实验，不能作为主模型加载，也未接入超级集成",
        ))
        return out

    # —— 情况 D：alpha_model_metrics.json 形式（超级集成的一条腿，**不是**主模型候选）
    if data.get("model_type") == "alpha_xgboost":
        ric = _safe_num(data.get("rank_ic"))
        if not np.isnan(ric):
            prefix = "alpha"
            model_path = MODEL_DIR / "alpha_xgboost_model.pkl"
            files_ok = (
                model_path.exists()
                and (MODEL_DIR / "alpha_scaler.pkl").exists()
                and (MODEL_DIR / "alpha_label_encoder.pkl").exists()
            )
            out.append(ModelInfo(
                key="alpha_xgboost",
                display_name="XGBoost（生产+alpha 北向/融资/龙虎榜/行业）",
                model_path=model_path if model_path.exists() else None,
                prefix=prefix,
                rank_ic=ric,
                ic_ir=_safe_num(data.get("ic_ir")),
                win_rate=_safe_num(data.get("top_win_rate")),
                top_excess=data.get("top_excess"),
                n_samples=data.get("n_days"),
                # 关键：它不是主模型候选。
                # AlphaSubPredictor 会给它单独拼 alpha 列（来自 parquet 缓存），
                # 列集合 = feature_cols + alpha_features，与主流管线**不同**。
                # 把它塞进 predictor._load_models 会因特征错位崩在推理上，
                # 所以哪怕它 RankIC 最高，也绝不能出现在「可部署最优」里。
                loadable=False,
                role="ensemble_leg",
                note="超级集成的一条腿：由 EnsemblePredictor 用 alpha_scaler/"
                     "alpha_label_encoder 单独加载，特征列含 alpha_* 列，"
                     "不能作为唯一主模型"
                     if files_ok else "alpha 模型文件缺失，需重跑 scripts/train_alpha_model.py",
            ))
            return out

    # —— 情况 C：walk_forward_alpha.json 形式（实验日志，无 model_path）
    if "model_A" in data or "model_B" in data:
        for tag in ("model_A", "model_B"):
            if tag not in data:
                continue
            s = data[tag].get("summary", {})
            out.append(ModelInfo(
                key=f"alpha_{tag[-1]}",
                display_name=f"Alpha 模型（{tag}，含北向/融资/龙虎榜特征）",
                model_path=None,  # 仅为实验日志，未存 pkl
                prefix="alpha",
                rank_ic=_safe_num(s.get("rank_ic")),
                ic_ir=_safe_num(s.get("ic_ir")),
                win_rate=_safe_num(s.get("top_win_rate")),
                top_excess=s.get("top_excess"),
                n_samples=s.get("n_days"),
                loadable=False,
                role="log_only",
                note="仅 walk-forward 实验日志，未生成可部署 .pkl；"
                     "需运行 alpha 训练脚本（alpha_set=industry/lhb/margin/northbound）后上线",
            ))
        return out

    return out


def scan_models(model_dir: Optional[Path] = None) -> List[ModelInfo]:
    """扫描 models 目录下所有指标文件，返回全部已登记模型（含不可部署的）。

    **按 key 去重**：同一个模型常常有多个指标文件（例如 walk_forward_alpha.json
    与 walk_forward_alpha_ablation.json 都描述了 model_A / model_B）。不去重的话
    注册表会出现两行同名条目、数值还不同 —— 用户根本不知道该信哪一行。
    保留 RankIC 较高者，并在 note 里记下它的来源文件，便于回溯。
    """
    root = Path(model_dir) if model_dir else MODEL_DIR
    picked: Dict[str, ModelInfo] = {}
    for jf in sorted(root.glob("*.json")):
        # 只解析"看起来像指标/实验日志"的文件，跳过纯配置
        try:
            txt = jf.read_text(encoding="utf-8")
            if "rank_ic" not in txt and "oos_rank_ic" not in txt:
                continue
        except Exception:
            continue
        try:
            parsed = _parse_metrics_file(jf)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"跳过 {jf.name}: {e}\n{traceback.format_exc()}")
            continue
        for m in parsed:
            cur = picked.get(m.key)
            if cur is None or _ranks_above(m, cur):
                m.note = f"{m.note} ｜ 来源 {jf.name}"
                picked[m.key] = m

    # —— 版本化产物（versions/<id>/）发现 ——
    # 每个版本目录是一个带明确 id 的、可切换的训练产物；其 model.pkl / scaler.pkl /
    # features.json 三件套齐全即为「可作主模型」候选。登记键即版本 id，与平铺
    # 指标文件的 key（如 production_metrics）互不冲突。predictor 在版本被批准/
    # 钉住后会经由 version_set_for 真正加载它。
    vroot = root / "versions"
    if vroot.is_dir():
        for vdir in sorted(d for d in vroot.iterdir() if d.is_dir()):
            vid = vdir.name
            model_p = vdir / "model.pkl"
            if not model_p.exists():
                continue  # 不是模型版本目录（可能是别的子目录）
            rank_ic = float("nan")
            for jf in sorted(vdir.glob("*.json")):
                try:
                    d = json.loads(jf.read_text(encoding="utf-8"))
                except Exception:
                    continue
                wf = d.get("walk_forward") or {}
                ric = _safe_num(wf.get("rank_ic"))
                if np.isnan(ric):
                    ric = _safe_num(d.get("rank_ic"))
                if np.isnan(ric):
                    ric = _safe_num(d.get("oos_rank_ic"))
                if not np.isnan(ric):
                    rank_ic = ric
                    break
            files_ok = (
                model_p.exists()
                and (vdir / "scaler.pkl").exists()
                and (vdir / "label_encoder.pkl").exists()
                and (vdir / "features.json").exists()
            )
            picked[vid] = ModelInfo(
                key=vid,
                display_name=f"版本 {vid}（已训练产物）",
                model_path=model_p,
                prefix=vid,
                rank_ic=rank_ic,
                loadable=bool(files_ok),
                role="main",
                note=("四件套齐全，可作为主模型候选"
                      if files_ok else "版本四件套（model/scaler/label_encoder/features）不齐，暂不可部署"),
            )

    infos = list(picked.values())
    # 按 RankIC 降序（NaN 排最后）
    infos.sort(key=lambda m: (np.isnan(m.rank_ic), -m.rank_ic if not np.isnan(m.rank_ic) else 0))

    # 多个指标文件可能描述的是**同一个模型文件**（例如 production_metrics 与
    # production_metrics_hs300_bak 都指向 xgboost_model.pkl，只是一次改了评估窗口）。
    # 不标注的话，注册表会并排列出两行「✅可作主模型」，而模型其实只有一个 ——
    # 用户会以为有两个可部署的候选，甚至以为可以「在两者之间切换」。
    by_path: Dict[str, List[ModelInfo]] = {}
    for m in infos:
        if m.model_path is not None:
            by_path.setdefault(str(m.model_path.resolve()), []).append(m)
    for group in by_path.values():
        if len(group) > 1:
            for g in group:
                others = "、".join(x.key for x in group if x.key != g.key)
                g.note = f"{g.note} ｜ 与 {others} 指向同一个模型文件（并非两个可切换的候选）"
    return infos


def _ranks_above(a: ModelInfo, b: ModelInfo) -> bool:
    """a 是否比 b 更值得保留：RankIC 高者胜；无法比较时**保留已有的**，"""

    def _r(m: ModelInfo) -> float:
        return -np.inf if np.isnan(m.rank_ic) else m.rank_ic

    return _r(a) > _r(b)


def best_overall(model_dir: Optional[Path] = None) -> Optional[ModelInfo]:
    """全部模型中 RankIC 最高者（可能是不可部署的实验模型）"""
    infos = scan_models(model_dir)
    if not infos:
        return None
    return infos[0]


def pinned_version() -> str:
    """自我进化「钉住」的模型版本键（``config.ml.active_version``）。"""
    try:
        from config.settings import config

        return str(getattr(config.ml, "active_version", "") or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def pinned_model(model_dir: Optional[Path] = None) -> Optional[ModelInfo]:
    """被钉住的**可部署**模型；未钉住 / 键不存在 / 不可部署 → None。

    「钉住」是自我进化落地模型重训候选的唯一方式（见
    :mod:`src.evolution.generators.retrain`）。这里只负责**如实查表**：
    查不到就返回 None，让调用方回退到自动挑选 —— 一个失效的版本 id
    绝不能让系统无模型可用。
    """
    key = pinned_version()
    if not key:
        return None
    for m in scan_models(model_dir):
        if m.key == key and m.loadable and m.model_path and m.model_path.exists():
            return m
    logger.warning(f"钉住的模型版本 {key!r} 不可部署或不存在，回退到自动挑选")
    return None


def best_loadable(model_dir: Optional[Path] = None) -> Optional[ModelInfo]:
    """**查询**：可部署模型里 RankIC 最高者；被钉住且可部署时优先返回钉住的那个。

    这是「最好能到多好」，**不是**「现在该用哪个」—— 后者是
    :func:`resolve_active`（多一层人工批准白名单）。两者分开，
    是为了让「注册表里躺着一个更优产物」不再等于「它已经生效」。
    """
    pinned = pinned_model(model_dir)
    if pinned is not None:
        return pinned
    infos = [m for m in scan_models(model_dir) if m.loadable]
    return infos[0] if infos else None


# ---------------------------------------------------------------------------
# 人工批准名单 —— 封堵「往 models/ 丢个文件就静默生效」
# ---------------------------------------------------------------------------
# 为什么需要它：注册表按 RankIC 排序，意味着**写文件即上线**。
# 一个新训练的产物只要把指标 JSON 丢进 models/，它就可能变成「可部署最优」，
# 全程没有任何人按过按钮。闸门与人工批准在这里被整体绕开。
#
# 破法很朴素：把「能不能生效」与「指标多高」解耦。生效必须来自一份**显式的
# 批准名单**；指标只决定「已批准的这些里面挑哪个」。
#
# 而这份名单**默认不存在**，等价于空名单 —— 空名单下没有任何版本能生效，
# 于是必然回退到硬编码的生产默认 xgboost_model.pkl。
# 刻意不做成「文件不存在=全部放行」：那样一个忘了创建文件的部署环境
# 就会把刚封上的通道重新打开。
APPROVALS_FILE_NAME = "model_approvals.json"


def approvals_file() -> Path:
    """人工批准名单文件路径（``<config>/model_approvals.json``）。"""
    return Path(CONFIG_DIR) / APPROVALS_FILE_NAME


def load_approvals() -> Dict[str, Dict]:
    """读人工批准名单 → ``{版本键: 批准记录}``。

    文件不存在 / 损坏 → **空字典**（不是全部放行）。
    """
    path = approvals_file()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"模型批准名单解析失败，按「空名单」处理（更保守）: {e}")
        return {}

    items = raw.get("approved") if isinstance(raw, dict) else raw
    out: Dict[str, Dict] = {}
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict) and it.get("key"):
                out[str(it["key"])] = it
    elif isinstance(items, dict):
        for k, v in items.items():
            out[str(k)] = v if isinstance(v, dict) else {}
    return out


def approved_keys() -> List[str]:
    """已批准可生效的模型版本键（排序后）。"""
    return sorted(load_approvals())


def approve_version(key: str, operator: str = "", note: str = "") -> Dict:
    """把某个版本键加入批准名单。

    这是**运维显式动作**，不做任何自动调用 —— 一旦它被自动调用，
    这份名单就退化成了它本来要封堵的那条通道。

    返回更新后的完整名单（便于调用方回显/审计）。
    """
    key = str(key).strip()
    if not key:
        raise ValueError("模型版本键不能为空")
    data = load_approvals()
    data[key] = {
        "key": key,
        "approved_by": operator or "unknown",
        "approved_at": datetime.now().isoformat(timespec="seconds"),
        "note": note or "",
    }
    _write_approvals(data)
    logger.info(f"✅ 已批准模型版本 {key!r}（{operator or 'unknown'}）")
    return data


def revoke_version(key: str) -> bool:
    """从批准名单里移除某个版本键。返回是否真的移除了。"""
    key = str(key).strip()
    data = load_approvals()
    if key not in data:
        return False
    data.pop(key)
    _write_approvals(data)
    logger.info(f"🗑️ 已撤销模型版本批准 {key!r}")
    return True


def _write_approvals(data: Dict[str, Dict]) -> None:
    """原子写批准名单（先写临时文件再替换，避免半截文件把名单读空）。"""
    path = approvals_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "approved": [data[k] for k in sorted(data)],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def resolve_active(model_dir: Optional[Path] = None) -> Optional[ModelInfo]:
    """**裁决**：当前该用哪个模型版本。

    优先级：

    1. ``ml.active_version`` 钉住的版本（自我进化的重训候选，经样本外闸门
       + 人工批准后由 ``overrides.apply_changes`` 写入 —— 它本身即一次批准）；
    2. 批准名单里样本外 RankIC 最高的**可部署**版本；
    3. ``None`` —— 没有任何裁决依据。调用方必须回退到硬编码的生产默认
       ``models/xgboost_model.pkl``，而不是「顺手挑个指标高的」。

    第 3 条是整个函数存在的理由：把「没批准」与「批准了但没想好」统一
    收敛到同一个安全动作上。返回 ``None`` 不是失败，是**正确的保守**。
    """
    pinned = pinned_model(model_dir)
    if pinned is not None:
        return pinned

    approved = load_approvals()
    if not approved:
        return None
    cands = [m for m in scan_models(model_dir) if m.loadable and m.key in approved]
    return cands[0] if cands else None


def resolve_active_path(model_dir: Optional[Path] = None) -> Path:
    """裁决后应当加载的模型文件路径；无裁决 → 生产默认路径。"""
    m = resolve_active(model_dir)
    if m and m.model_path and m.model_path.exists():
        return m.model_path
    return MODEL_DIR / "xgboost_model.pkl"


def active_status(model_dir: Optional[Path] = None) -> str:
    """一句话说明当前「加载什么 / 为什么」，供日志与看板展示。"""
    m = resolve_active(model_dir)
    if m is not None:
        src = "被钉住（ml.active_version）" if pinned_version() else "在人工批准名单内"
        return f"生效模型：{m.key}（{src}，RankIC={m.rank_ic:+.4f}）"
    if pinned_version():
        return (f"ml.active_version={pinned_version()!r} 不可部署或不存在，"
                f"回退到生产默认 xgboost_model.pkl")
    if not load_approvals():
        return ("无人工批准名单（config/model_approvals.json 不存在），"
                "按生产默认加载 xgboost_model.pkl；"
                "新产物不会因为指标更高而自动生效")
    return "批准名单里没有可部署的版本，回退到生产默认 xgboost_model.pkl"


def best_model_path(model_dir: Optional[Path] = None) -> Path:
    """返回主流 predictor 应加载的最优模型路径（无则回退默认）"""
    b = best_loadable(model_dir)
    if b and b.model_path and b.model_path.exists():
        return b.model_path
    return MODEL_DIR / "xgboost_model.pkl"


def report(model_dir: Optional[Path] = None) -> str:
    """生成人类可读的模型登记报告"""
    infos = scan_models(model_dir)
    if not infos:
        return "（未找到任何带指标的模型产物）"
    lines = ["📦 模型注册中心（按样本外 RankIC 排序）", ""]
    approved = load_approvals()
    pinned = pinned_version()
    for i, m in enumerate(infos, 1):
        ric = f"{m.rank_ic:+.4f}" if not np.isnan(m.rank_ic) else "n/a"
        if m.key == pinned:
            tag = "📌已钉住"
        elif m.loadable:
            tag = "✅可作主模型"
        elif m.role == "ensemble_leg":
            tag = "🧩集成腿（非主模型）"
        else:
            tag = "⚠️不可部署"
        if m.key in approved:
            tag += "｜已批准"
        lines.append(
            f"{i}. {m.display_name} — RankIC={ric} [{tag}]\n"
            f"   IC_IR={m.ic_ir:+.2f} 胜率={m.win_rate:.1%} "
            f"样本={m.n_samples}\n"
            f"   说明：{m.note}"
        )
    bo = best_overall(model_dir)
    bl = best_loadable(model_dir)
    lines.append("")
    lines.append(f"🏆 全局最优（含实验）：{bo.display_name} (RankIC={bo.rank_ic:+.4f})"
                 if bo else "（无模型）")
    if bl:
        lines.append(f"🔍 可作主模型的最优：{bl.display_name} (RankIC={bl.rank_ic:+.4f})")
    lines.append(f"🧭 裁决：{active_status(model_dir)}")
    if not approved:
        lines.append("")
        lines.append("⚠️ 尚无人工批准名单（config/model_approvals.json）。"
                     "此时**不会有任何新产物生效** —— 这是刻意的保守默认。")
        lines.append("   要批准某个版本：scripts/evolve.py --approve-model <键> --operator <你>")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
