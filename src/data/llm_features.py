# -*- coding: utf-8 -*-
"""
LLM 定性叠加特征适配器
======================
把「最先进 AI」（大模型）看文本的能力接进选股模型：对每只候选股，
汇聚其近期新闻/研报（仅用发布时间 <= 预测日的文本，零泄漏），
调用 LLM 输出结构化评分，作为 3 个新特征喂给 xgboost：
    llm_sentiment : 情绪分 (-1~1, +看多 / -看空)
    llm_risk      : 风险标记 (0/1, 1=利空/风险事件)
    llm_confidence: 置信度 (0~1)

提供方（provider）：
  - "local"  ：本机 OpenAI 兼容服务（如 Ollama，默认 http://localhost:11434/v1），无需云端 key
  - "deepseek"/"openai"/"claude"：云端，key 从环境变量 DEEPSEEK_API_KEY / LLM_API_KEY 读取，或显式传入
默认 provider="local"，开箱即用、不联网、不接 API。

⚠️ 零泄漏铁律：gather_news 只返回 publish_time <= end_date 的新闻。
回测里若混入未来新闻，RankIC 会虚高，等于作弊。

⚠️ 成本现实：walk-forward 全量逐日逐股调 LLM 不现实；
度量贡献时用 mock/代理情绪（零成本、离线可跑），真实 LLM 只放
每日实盘打分那一步（候选股就几十只，成本可控）。
"""

from __future__ import annotations

import os
import json
import time
import hashlib
import logging
from dataclasses import dataclass
from typing import Optional, Dict

logger = logging.getLogger("llm_features")


@dataclass
class LLMFeatureConfig:
    # provider: "local"(Ollama 等本机 OpenAI 兼容服务，无需 key) /
    #           "deepseek" / "openai" / "claude"(云端，需 api_key)
    provider: str = "local"
    api_key: str = ""                             # 本地模式可为空（Ollama 不校验）
    base_url: str = "http://localhost:11434/v1"   # Ollama 默认；云端填对应地址
    model: str = "qwen2.5:7b"                     # 本机模型名；云端用 deepseek-chat 等
    temperature: float = 0.0
    max_tokens: int = 400
    timeout: int = 20
    retry: int = 3
    rate_limit: float = 0.2   # 每次调用间隔秒（限流）
    mock: bool = False        # True=离线伪评分（不用真 LLM，用于回测/测试/无本地服务时）


_PROMPT = """你是一名严谨的 A 股量化研究员。下面是一支股票在【{date}】及之前可得的新闻/公告/研报摘要。
请只依据这些文本，输出 JSON（不要任何额外说明）：
{{
  "sentiment_score": 介于 -1 到 1 的浮点数（+ 偏多 / - 偏空，反映文本对股价的净情绪），
  "risk_flag": 0 或 1（1 表示文本含明显利空/风险事件，如业绩暴雷、监管处罚、减持、诉讼），
  "confidence": 介于 0 到 1 的浮点数（文本信息量与确定度），
  "rationale": 不超过 30 字的一句话理由
}}
只输出 JSON。文本：
{text}"""


class LLMFeatureProvider:
    """把 LLM 的文本判断转成 3 个数值特征，供选股模型使用。"""

    def __init__(self, config: Optional[LLMFeatureConfig] = None):
        self.cfg = config or LLMFeatureConfig()
        # 本地模式（Ollama 等）：不需要云端 key，openai SDK 要求非空则填占位
        if self.cfg.provider == "local" and not self.cfg.api_key:
            self.cfg.api_key = "ollama"
        # 云端模式无 key 且无显式 mock -> 自动转 mock，避免误调真实 API 报错/烧钱
        if (not self.cfg.mock and self.cfg.provider != "local"
                and not self.cfg.api_key
                and not os.environ.get("DEEPSEEK_API_KEY")
                and not os.environ.get("LLM_API_KEY")):
            logger.warning("未检测到 LLM API key，LLMFeatureProvider 自动进入 mock 模式")
            self.cfg.mock = True
        self._client = None
        self._last_call = 0.0

    # ---- 客户端（懒加载，避免无 key 时也 import openai）----
    def _get_client(self):
        if self._client is not None:
            return self._client
        from openai import OpenAI
        key = self.cfg.api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY", "")
        self._client = OpenAI(api_key=key, base_url=self.cfg.base_url)
        return self._client

    # ---- 文本评分 ----
    def score_text(self, text: str, as_of_date: str = "") -> Dict:
        if not text or not text.strip():
            return {"sentiment_score": 0.0, "risk_flag": 0, "confidence": 0.0, "rationale": ""}
        if self.cfg.mock:
            return self._mock_score(text)
        prompt = _PROMPT.format(date=as_of_date, text=text[:6000])
        for attempt in range(self.cfg.retry):
            try:
                self._throttle()
                resp = self._get_client().chat.completions.create(
                    model=self.cfg.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    response_format={"type": "json_object"},
                    timeout=self.cfg.timeout,
                )
                content = resp.choices[0].message.content or ""
                return self._parse(content)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"LLM 评分失败(第{attempt+1}次): {e}")
                time.sleep(1.5 * (attempt + 1))
        return {"sentiment_score": 0.0, "risk_flag": 0, "confidence": 0.0, "rationale": ""}

    def _throttle(self):
        now = time.time()
        wait = self.cfg.rate_limit - (now - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    @staticmethod
    def _parse(content: str) -> Dict:
        try:
            obj = json.loads(content)
            s = float(obj.get("sentiment_score", 0.0))
            r = int(obj.get("risk_flag", 0))
            c = float(obj.get("confidence", 0.0))
            return {
                "sentiment_score": max(-1.0, min(1.0, s)),
                "risk_flag": 1 if r else 0,
                "confidence": max(0.0, min(1.0, c)),
                "rationale": str(obj.get("rationale", ""))[:60],
            }
        except Exception:  # noqa: BLE001
            return {"sentiment_score": 0.0, "risk_flag": 0, "confidence": 0.0, "rationale": ""}

    @staticmethod
    def _mock_score(text: str) -> Dict:
        # 离线确定性伪评分：仅用于回测/管线联通测试，不代表真实 LLM 判断
        h = int(hashlib.md5(text[:200].encode("utf-8")).hexdigest(), 16)
        sentiment = ((h % 200) - 100) / 100.0
        risk = 1 if any(k in text for k in ("处罚", "暴雷", "减持", "诉讼", "立案", "退市")) else 0
        conf = 0.3 + (h % 50) / 100.0
        return {
            "sentiment_score": round(sentiment, 3),
            "risk_flag": risk,
            "confidence": round(conf, 3),
            "rationale": "mock",
        }

    # ---- 新闻汇聚（零泄漏）----
    def gather_news(self, code: str, end_date: str, days: int = 20) -> str:
        """汇聚 code 在 end_date 及之前的新闻文本（泄漏安全）。

        end_date 形如 '2026-03-31'；只保留发布时间 <= end_date 的新闻。
        """
        try:
            import akshare as ak
            df = ak.stock_news_em(symbol=code)
            if df is None or len(df) == 0:
                return ""
            cols = list(df.columns)
            date_col = "日期" if "日期" in cols else (cols[0] if cols else None)
            title_col = "新闻标题" if "新闻标题" in cols else None
            body_col = "新闻内容" if "新闻内容" in cols else ("摘要" if "摘要" in cols else None)
            pieces = []
            for _, row in df.iterrows():
                d = str(row[date_col]) if date_col else ""
                if end_date and d > end_date:
                    continue
                t = str(row[title_col]) if title_col else ""
                b = str(row[body_col]) if body_col else ""
                pieces.append(f"[{d}] {t} {b}")
                if len(pieces) >= days:
                    break
            return "\n".join(pieces)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"新闻汇聚失败 {code}: {e}")
            return ""

    # ---- 对外：一支股的 3 个新特征 ----
    def features_for_stock(self, code: str, end_date: str, days: int = 20) -> Dict:
        news = self.gather_news(code, end_date, days=days)
        sc = self.score_text(news, as_of_date=end_date)
        return {
            "llm_sentiment": sc["sentiment_score"],
            "llm_risk": sc["risk_flag"],
            "llm_confidence": sc["confidence"],
        }


if __name__ == "__main__":
    # 离线自检：mock 模式，不联网、不耗 key
    p = LLMFeatureProvider(LLMFeatureConfig(mock=True))
    sample = "公司发布增持计划，业绩预增；同时股东拟减持。"
    print("mock score:", p.score_text(sample, as_of_date="2026-03-31"))
