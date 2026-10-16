# -*- coding: utf-8 -*-
"""自我进化 · 生成器 B：LLM 自我复盘。

它比生成器 A 多什么
===================
生成器 A 是「硬编码阈值 → 固定处方」：换手 > 3 倍就调容忍带，一行 if 到底。
它读不出**跨信号**的因果，也读不出「这次和上次有什么不同」。

生成器 B 把**同一份事实底稿**交给 LLM，让它在几个信号之间做权衡，
例如：

* T+5 准确率只有 20% 但 T+10 明显更好 → 可能不是选股坏了，是**持有期太短**；
* 日胜率低但盈亏比高 → 选股没错，是**等权摊平**了优势；
* 回撤逼近熔断线且成交偏碎 → 优先**防守**，而不是顺手多开仓位。

安全设计（本模块的核心，请勿削弱）
==================================

**① LLM 不产出任何数值。**
它只能从 :data:`ACTIONS` 这份**封闭目录**里挑一个动作 id；
真正写进候选的数值由本模块按 ``ParamSpec.step`` 算出来。
这样「让 LLM 把熔断线调到 50%」在结构上就不可能 —— 它没有地方说这个数。

**② 目录本身也不能越界。**
:func:`catalog_problems` 会校验目录里每个 action 指向的参数都在
:data:`src.evolution.space.PARAM_SPACE` 白名单内、且不属风控硬线。
即便有人手改目录塞进 ``guard.*``，:func:`draft` 里的 ``space.validate``
仍会二次拦下 —— 两道锁，不靠单一防线。

**③ LLM 的 confidence 只记录、不决策。**
它写进 ``meta``，既不能改 ``evaluation.passed``，也不能影响
``space`` 对该参数的方向判定（是否可自动落地）。

**④ 输出必须可解析、可审计。**
强制 JSON；解析失败、动作未知、理由为空、confidence 过低 —— 一律丢弃并告警。
理由文案会原样保留在候选里，人在批准时能看到 LLM 到底怎么想的。
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from src.evolution import space
from src.evolution.feedback import PerformanceSnapshot
from src.evolution.generators.base import atomize, require_evidence
from src.evolution.types import Candidate

NAME = "llm_review"

#: 低于此置信度的动作直接丢掉（LLM 自己也说不准的事不值得占队列）。
MIN_CONFIDENCE = 0.5

#: 最多向 LLM 要几个动作。
DEFAULT_MAX_ACTIONS = 5

#: 事实底稿里「正确率序列」最多列几期，避免 prompt 无限增长。
MAX_SERIES = 8


# ---------------------------------------------------------------------------
# 封闭动作目录 —— LLM 能做的**全部**事情
# ---------------------------------------------------------------------------
# op="bump": new = clamp(cur + sign * steps * spec.step)
# op="set" : new = clamp(value)
#
# 目录里**没有**任何指向 guard.* / trading_guard.* / risk.* 的动作。
# 想加动作就必须改这份代码并过 catalog_problems() 的校验，是显式的。

ACTIONS: Dict[str, Dict[str, Any]] = {
    # ----- 交易频率 -----
    "widen_band": {
        "key": "trading_loop.band_pct", "op": "bump", "sign": +1, "steps": 1,
        "title": "放宽调仓容忍带，降低交易频率",
        "meaning": "小幅偏离不触发调仓，省下双边手续费与滑点",
    },
    "tighten_band": {
        "key": "trading_loop.band_pct", "op": "bump", "sign": -1, "steps": 1,
        "title": "收紧调仓容忍带，更贴近目标仓位",
        "meaning": "更频繁地回归计划权重，代价是更多交易成本",
    },
    # ----- 仓位与现金 -----
    "raise_cash": {
        "key": "trading_loop.cash_reserve_pct", "op": "bump", "sign": +1, "steps": 1,
        "title": "提高现金保留比例，防守回撤",
        "meaning": "降低仓位敞口，回撤更小但上涨时也赚得少（更保守）",
    },
    "lower_cash": {
        "key": "trading_loop.cash_reserve_pct", "op": "bump", "sign": -1, "steps": 1,
        "title": "降低现金保留比例，减少拖累",
        "meaning": "更接近满仓，收益弹性更大、回撤也更大（更激进）",
    },
    "cut_positions": {
        "key": "trading_loop.max_positions", "op": "bump", "sign": -1, "steps": 2,
        "title": "下调目标持仓只数，避免仓位被稀释",
        "meaning": "可选标的买不满时，减少目标只数让每只仓位更实",
    },
    "add_positions": {
        "key": "trading_loop.max_positions", "op": "bump", "sign": +1, "steps": 2,
        "title": "上调目标持仓只数，提高分散度",
        "meaning": "单票集中度过高时分散风险（更分散，但也可能摊薄 alpha）",
    },
    "lower_weight_cap": {
        "key": "trading_loop.weight_cap", "op": "bump", "sign": -1, "steps": 1,
        "title": "收紧单票权重上限，防止单票拖累",
        "meaning": "任一标的的权重天花板降低，组合更均衡（更保守）",
    },
    "raise_weight_cap": {
        "key": "trading_loop.weight_cap", "op": "bump", "sign": +1, "steps": 1,
        "title": "放宽单票权重上限，让高分标的占更重",
        "meaning": "允许更集中的下注（更激进，且受风控 max_single_position_pct 硬约束）",
    },
    # ----- 权重方式 -----
    "use_score_weighting": {
        "key": "trading_loop.weighting", "op": "set", "value": "score",
        "title": "权重改为按评分加权",
        "meaning": "不改进池标准，只让高分标的拿更多资金",
    },
    "use_equal_weighting": {
        "key": "trading_loop.weighting", "op": "set", "value": "equal",
        "title": "权重改回等权",
        "meaning": "不依赖评分刻度，分散度最高",
    },
    # ----- 碎单 -----
    "raise_min_trade": {
        "key": "trading_loop.min_trade_amount", "op": "bump", "sign": +1, "steps": 1,
        "title": "提高最小交易金额，过滤碎单",
        "meaning": "把手续费占比过高的小单挡在门外",
    },
    "lower_min_trade": {
        "key": "trading_loop.min_trade_amount", "op": "bump", "sign": -1, "steps": 1,
        "title": "降低最小交易金额，允许小额调仓",
        "meaning": "让更小的偏离也能成交，贴计划但更碎",
    },
}


def catalog_problems() -> List[str]:
    """自检：动作目录是否越界。返回问题列表（空 = 合规）。

    这是「目录不能越过参数空间」这条约束的**可测形式** ——
    有它才能在 CI 里挡住「顺手加个调熔断线的动作」这种改动。
    """
    problems: List[str] = []
    for name, spec in ACTIONS.items():
        key = spec.get("key")
        if not key:
            problems.append(f"{name}: 缺少 key")
            continue
        try:
            space.get_spec(key)          # 受保护 / 未登记都会在这里抛
        except space.ParamError as e:
            problems.append(f"{name} → {key}: {e}")
            continue
        if spec.get("op") not in ("bump", "set"):
            problems.append(f"{name}: 未知 op {spec.get('op')!r}")
        if spec.get("op") == "set" and "value" not in spec:
            problems.append(f"{name}: op=set 但缺 value")
        if spec.get("op") == "bump" and spec.get("sign") not in (-1, 1):
            problems.append(f"{name}: op=bump 的 sign 必须是 ±1")
    return problems


# ---------------------------------------------------------------------------
# 事实底稿（纯函数，可离线断言）
# ---------------------------------------------------------------------------

def build_factsheet(snapshot: PerformanceSnapshot) -> str:
    """把快照渲染成给 LLM 看的**事实底稿**。

    只放数字与事实，不放结论 —— 结论留给 LLM 给。
    也不放任何风控参数名，避免它以为可以调。
    """
    m = snapshot.metrics or {}
    L: List[str] = []

    L.append("## 一、预测验证（选股准不准）")
    L.append(f"- 预测批次：{snapshot.n_batches} 批，已验证 {snapshot.n_verified} 批，"
             f"到期未验证 {snapshot.pending_verify} 个（批次×窗口）")
    for h in sorted(snapshot.accuracy_by_horizon, key=lambda x: int(x)):
        vals = snapshot.accuracy_by_horizon[h] or []
        if not vals:
            continue
        recent = vals[-MAX_SERIES:]
        L.append(f"- T+{h} 准确率（最近 {len(recent)} 期，早→晚）："
                 + "、".join(f"{v:.1%}" for v in recent)
                 + f"；均值 {sum(recent) / len(recent):.1%}")
        ex = snapshot.excess_by_horizon.get(h) or []
        if ex:
            L.append(f"- T+{h} 超额收益均值：{sum(ex) / len(ex):+.2%}")
        wr = snapshot.win_rate_by_horizon.get(h) or []
        if wr:
            L.append(f"- T+{h} 上涨占比均值：{sum(wr) / len(wr):.1%}")

    if snapshot.score_buckets:
        rows = []
        for name, b in sorted((snapshot.score_buckets or {}).items()):
            if isinstance(b, dict) and b.get("n"):
                rows.append(f"{name}: n={b['n']}"
                            + (f", 平均收益 {b['mean_return']:+.2%}"
                               if b.get("mean_return") is not None else ""))
        if rows:
            L.append("- 评分分桶表现：" + "；".join(rows))

    L.append("")
    L.append("## 二、交易与绩效（真金白银）")
    if m.get("insufficient"):
        L.append("- ⚠️ 净值序列不足 2 个点，收益/回撤/换手**全部不可用**（不是 0，是测不出来）")
    else:
        pairs = [
            ("total_return", "累计收益", "{:+.2%}"),
            ("ann_return", "年化收益", "{:+.2%}"),
            ("max_drawdown", "最大回撤", "{:.2%}"),
            ("daily_win_rate", "日胜率", "{:.1%}"),
            ("profit_loss_ratio", "盈亏比", "{:.2f}"),
            ("turnover", "年换手", "{:.1f} 倍"),
            ("sharpe", "夏普", "{:.2f}"),
        ]
        for k, label, fmt in pairs:
            v = m.get(k)
            if isinstance(v, (int, float)) and v == v:
                L.append(f"- {label}：{fmt.format(v)}")
    L.append(f"- 结算日数：{snapshot.n_settlements}；"
             f"成交 {snapshot.trade_stats.get('fill', 0)} 笔；"
             f"风控拒单 {snapshot.trade_stats.get('risk_block', 0)} 笔")

    L.append("")
    L.append("## 三、持仓与仓位结构")
    rows = snapshot.equity_series or []
    if not rows:
        L.append("- ⚠️ 没有净值序列，持仓结构不可得")
    else:
        eqs = [_f(r.get("equity")) for r in rows]
        eqs = [v for v in eqs if v]
        cash = [_f(r.get("cash")) for r in rows]
        pos = [_f(r.get("n_positions")) for r in rows if _f(r.get("n_positions")) is not None]
        if eqs:
            L.append(f"- 平均总资产：¥{sum(eqs) / len(eqs):,.0f}（{len(eqs)} 个观测）")
        if cash and eqs:
            ratios = [c / e for c, e in zip(cash, eqs) if c is not None and e]
            if ratios:
                L.append(f"- 现金占总资产（均值）：{sum(ratios) / len(ratios):.1%}")
        if pos:
            L.append(f"- 实际持仓只数（均值）：{sum(pos) / len(pos):.1f} 只")
            L.append(f"- 目标持仓只数（当前）："
                     f"{_current('trading_loop.max_positions')} 只")

    L.append("")
    L.append("## 四、风控状态")
    g = snapshot.guard or {}
    L.append(f"- 是否处于熔断：{'是' if g.get('halted') else '否'}"
             + (f"（原因：{g.get('halt_reason')}）" if g.get("halt_reason") else ""))
    if snapshot.last_halt:
        L.append(f"- 最近一次熔断：{snapshot.last_halt}")

    if snapshot.notes:
        L.append("")
        L.append("## 五、数据完备度警告（请据此降低你的置信度）")
        for n in snapshot.notes:
            L.append(f"- {n}")

    L.append("")
    L.append("## 六、可调参数现状（只能通过下面的动作间接改）")
    for key in sorted(space.PARAM_SPACE):
        try:
            cur = _current(key)
            spec = space.get_spec(key)
        except Exception:  # noqa: BLE001
            continue
        rng = ""
        if spec.kind in ("int", "float"):
            rng = f"，范围 [{spec.low}, {spec.high}]"
        elif spec.choices:
            rng = f"，可选 {list(spec.choices)}"
        L.append(f"- {key} = {cur!r}{rng}")

    L.append("")
    L.append("## 七、可执行动作（只能从这份清单里选 id）")
    for name, spec in sorted(ACTIONS.items()):
        L.append(f"- {name}：{spec['title']} —— {spec['meaning']}"
                 f"（作用于 {spec['key']}）")

    return "\n".join(L)


SYSTEM_PROMPT = (
    "你是一个 A 股量化交易系统的策略复盘员。你的任务：基于给出的**事实底稿**，"
    "指出当前策略最值得修正的地方，并从允许的动作清单里挑选应当执行的动作。\n\n"
    "硬性规则（违反则整份回复作废）：\n"
    "1. 只能输出一个 JSON 对象，不要任何解释性文字、不要 Markdown 代码块以外的内容。\n"
    "2. `action` 字段**必须**严格等于给定动作清单里的 id，不得自创、不得改写。\n"
    "3. 你**不能**给出任何数值。所有参数取值由系统按步长计算，你只负责选择方向。\n"
    "4. 每个动作都要在 `reason` 里引用底稿中的**具体数字**；引用不到数字的动作不要提。\n"
    "5. 底稿里标注为「不可用 / 数据不足」的指标，不得作为推理依据。\n"
    "6. 如果没有值得修正的地方，`actions` 返回空数组 —— 按兵不动是合法答案。\n\n"
    "输出格式：\n"
    '{"verdict": "一句话总体判断", '
    '"actions": [{"action": "动作id", "confidence": 0.0~1.0, '
    '"reason": "引用具体数字的理由", "evidence": "你依据的那个指标"}]}\n'
    "confidence 表示你对该动作的把握程度，不要为了凑数而给高分。"
)


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

def parse_review(text: str) -> Dict[str, Any]:
    """从 LLM 文本里严格解析出 ``{"verdict": str, "actions": [...]}``。

    容错仅限「外层包了 Markdown 代码块」；一旦进入 JSON 就按 schema 严格校验，
    不合格的条目**逐条丢弃并告警**，而不是整份作废 —— 好动作不该被坏动作连坐。
    """
    raw = _extract_json(text)
    if raw is None:
        logger.warning("🧬 生成器 B：LLM 回复里找不到 JSON，已丢弃")
        return {"verdict": "", "actions": []}

    if not isinstance(raw, dict):
        logger.warning(f"🧬 生成器 B：LLM 顶层不是对象（{type(raw).__name__}），已丢弃")
        return {"verdict": "", "actions": []}

    verdict = str(raw.get("verdict") or "").strip()
    items = raw.get("actions")
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        logger.warning("🧬 生成器 B：actions 不是数组，已丢弃")
        return {"verdict": verdict, "actions": []}

    out: List[Dict[str, Any]] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            logger.warning(f"🧬 生成器 B：第 {i} 条动作不是对象，已丢弃")
            continue
        name = str(it.get("action") or "").strip()
        if name not in ACTIONS:
            logger.warning(f"🧬 生成器 B：动作 {name!r} 不在允许清单里，已丢弃")
            continue
        reason = str(it.get("reason") or "").strip()
        if not reason:
            logger.warning(f"🧬 生成器 B：动作 {name} 没有给理由，已丢弃")
            continue
        conf = _f(it.get("confidence"))
        if conf is None:
            conf = 0.5
        conf = min(1.0, max(0.0, conf))
        out.append({
            "action": name,
            "confidence": conf,
            "reason": reason,
            "evidence": str(it.get("evidence") or "").strip(),
        })
    return {"verdict": verdict, "actions": out}


def _extract_json(text: str) -> Optional[Any]:
    """从回复里抠出第一个 JSON 对象（容忍 ```json 包裹与前后废话）。"""
    if not text or not text.strip():
        return None
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", s, re.S)
    if fence:
        s = fence.group(1).strip()
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(s[start:end + 1])
    except json.JSONDecodeError as e:
        logger.warning(f"🧬 生成器 B：JSON 解析失败（{e}）")
        return None


# ---------------------------------------------------------------------------
# 动作 → 具体数值
# ---------------------------------------------------------------------------

def resolve(action: str) -> Optional[Dict[str, Any]]:
    """把动作 id 解析成 ``{"key", "value", "title"}``；非法动作返回 None。

    数值在这里算，**不在 LLM 那边** —— 这是本模块安全设计的落点。
    """
    spec = ACTIONS.get(action)
    if spec is None:
        return None
    key = spec["key"]
    p = space.get_spec(key)                      # 受保护/未登记 → ParamError
    cur = _current(key)
    try:
        cur_typed = p.cast(cur)
    except Exception:  # noqa: BLE001
        cur_typed = p.default

    if spec["op"] == "set":
        new = p.clamp(spec["value"])
    else:
        step = p.step if p.step else (1 if p.kind == space.KIND_INT else 0.05)
        new = p.clamp(float(cur_typed) + spec["sign"] * float(step) * int(spec.get("steps", 1)))

    if new == cur_typed:
        return None                              # 已在边界，动作无效
    return {"key": key, "value": new, "title": spec["title"]}


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def generate(
    snapshot: PerformanceSnapshot,
    chat_fn: Optional[Callable[[str, str], str]] = None,
    max_candidates: int = 3,
    min_confidence: float = MIN_CONFIDENCE,
    max_actions: int = DEFAULT_MAX_ACTIONS,
) -> List[Candidate]:
    """LLM 复盘 → 候选。

    Args:
        chat_fn: ``(system, user) -> str``。默认走 :mod:`src.models.llm` 的
            DeepSeek 客户端；测试注入假实现即可完全离线。
        max_candidates: 最多采纳几条（按 confidence 降序）。
        min_confidence: 低于此置信度的动作丢弃。
        max_actions: 向 LLM 要几个动作的上限（写进 prompt）。

    Returns:
        候选列表；无 API Key / 调用失败 / 无有效动作时返回空列表（不抛错）。
    """
    problems = catalog_problems()
    if problems:
        # 目录坏了就停手 —— 宁可不动，也不要按一份可能越界的目录去改参数
        logger.error("🧬 生成器 B：动作目录自检未通过，本轮不生成候选：" + "；".join(problems))
        return []

    if not require_evidence(snapshot):
        return []

    if chat_fn is None:
        if not _llm_ready():
            logger.info("🧬 生成器 B：未配置 DeepSeek API Key，跳过 LLM 复盘")
            return []
        chat_fn = _default_chat

    facts = build_factsheet(snapshot)
    user = (f"以下是系统事实底稿，请复盘并给出 JSON：\n\n{facts}\n\n"
            f"最多给 {max_actions} 个动作。")

    try:
        reply = chat_fn(SYSTEM_PROMPT, user)
    except Exception as e:  # noqa: BLE001
        # LLM 是外部依赖：它挂了不该让整个进化回合看起来像出了 bug
        logger.warning(f"🧬 生成器 B：LLM 调用失败（{e}），本轮跳过")
        return []

    parsed = parse_review(reply or "")
    actions = parsed["actions"]
    if not actions:
        logger.info("🧬 生成器 B：LLM 未提出可执行动作（或全部被校验丢弃）")
        return []

    # 同一参数只保留置信度最高的一条（避免自相矛盾的两个方向）
    by_key: Dict[str, Dict[str, Any]] = {}
    for a in sorted(actions, key=lambda x: -x["confidence"]):
        res = resolve(a["action"])
        if res is None:
            logger.info(f"🧬 生成器 B：动作 {a['action']} 在当前取值下无效（可能已到边界），跳过")
            continue
        if a["confidence"] < min_confidence:
            logger.info(f"🧬 生成器 B：动作 {a['action']} 置信度 {a['confidence']:.2f} "
                        f"低于门槛 {min_confidence:.2f}，丢弃")
            continue
        if res["key"] in by_key:
            logger.info(f"🧬 生成器 B：{res['key']} 已有更高置信度的动作，丢弃 {a['action']}")
            continue
        by_key[res["key"]] = {**a, "resolved": res}

    picked = sorted(by_key.values(), key=lambda x: -x["confidence"])[:max_candidates]
    if not picked:
        return []

    pairs = []
    for a in picked:
        res = a["resolved"]
        why = (f"LLM 复盘：{a['reason']}"
               + (f"（依据：{a['evidence']}）" if a["evidence"] else "")
               + f"\n　→ 执行动作 {a['action']}，{res['key']} 由 {_current(res['key'])} "
               f"改为 {res['value']}（数值由参数步长算出，非 LLM 指定）")
        pairs.append((res["title"], {res["key"]: res["value"]}, why))

    cands = atomize(NAME, pairs)
    # 把 LLM 的置信度与总判断留痕，供人审阅；它不参与任何自动决策
    for c, a in zip(cands, picked):
        c.meta["llm"] = {
            "action": a["action"],
            "confidence": a["confidence"],
            "evidence": a["evidence"],
            "verdict": parsed["verdict"],
        }
    if cands:
        logger.info(f"🧬 生成器 B 产出 {len(cands)} 条候选："
                    + "、".join(c.title for c in cands))
    return cands


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _llm_ready() -> bool:
    """是否已配置可用的 LLM 通道（只读探测，不发起请求）。"""
    try:
        from src.models.llm import deepseek

        return bool(deepseek.is_ready)
    except Exception:  # noqa: BLE001
        return False


def _default_chat(system: str, user: str) -> str:
    """默认 LLM 通道；未配置 Key 时抛 RuntimeError（由 generate 兜住）。"""
    from src.models.llm import deepseek

    return deepseek.chat(system, user, temperature=0.2, max_tokens=1500)


def _current(key: str) -> Any:
    try:
        from src.evolution.overrides import read_effective

        return read_effective(key)
    except Exception:  # noqa: BLE001
        return space.get_default(key)


def _f(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f
