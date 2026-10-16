# -*- coding: utf-8 -*-
"""
OpenAI 兼容的 LLM 接入模块（默认指向 DeepSeek，也可指向本地服务）
============================================================
为选股神 AI 提供自然语言分析报告 / 推荐理由生成能力。

用法:
    from src.models.llm import deepseek

    # 个股分析报告
    report = deepseek.analyze_stock(股票名称, 代码, 指标字典)

    # 选股组合推荐理由
    rationale = deepseek.explain_recommendations(推荐列表)

去哪找模型（``config/deepseek.yaml`` 的 ``provider``）
----------------------------------------------------
* ``deepseek`` / ``auto``：云端 DeepSeek（需真 ``api_key`` 与余额）
* ``local``：**本机任意 OpenAI 兼容服务**，无需真 key。例如
    - Ollama      ``base_url: http://localhost:11434/v1``   ``model: qwen2.5:7b``
    - LM Studio   ``base_url: http://localhost:1234/v1``    ``model: <模型名>``
    - vLLM        ``base_url: http://localhost:8000/v1``    ``model: <模型名>``
  这些服务通常不校验 Authorization，本模块会自动填一个占位 key。

API Key 来源优先级:
    1. config/deepseek.yaml 中的 api_key
    2. 环境变量 DEEPSEEK_API_KEY
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import requests
from loguru import logger

from config.settings import ROOT_DIR

CONFIG_DIR = ROOT_DIR / "config"
DEEPSEEK_CONFIG = CONFIG_DIR / "deepseek.yaml"

# DeepSeek API 端点（OpenAI 兼容）。仅作为**默认值**——
# 实际请求地址由 ``DeepSeekClient._chat_url()`` 从 ``config.base_url`` 现算，
# 这样才能通过配置切到本地服务或代理。详见该方法上的说明。
API_BASE = "https://api.deepseek.com"


@dataclass
class DeepSeekConfig:
    """OpenAI 兼容 LLM 配置"""
    api_key: str = ""
    model: str = "deepseek-chat"      # deepseek-chat(V3) / deepseek-reasoner(R1) / 本地模型名
    base_url: str = API_BASE
    # provider: "auto"/"deepseek" = 云端（需真 key）；
    #           "local"           = 本机 OpenAI 兼容服务（无需真 key）
    provider: str = "auto"
    timeout: int = 60
    temperature: float = 0.3
    max_tokens: int = 1500
    enabled: bool = False


class DeepSeekClient:
    """OpenAI 兼容的聊天客户端 — 基于 requests 直连（无需 openai 包）"""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = Path(config_dir) if config_dir else CONFIG_DIR
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config = DeepSeekConfig()
        self._load()

    # ========== 配置管理 ==========

    @property
    def config_path(self) -> Path:
        """本客户端的配置文件路径，**跟随 ``self.config_dir``**。

        ⚠️ 这里修的是与 ``base_url`` 同类的第二个死参数：``__init__`` 收下了
        ``config_dir`` 却只拿它 ``mkdir``，真正的读写一律走模块级常量
        ``DEEPSEEK_CONFIG``。于是传什么 ``config_dir`` 都没用——
        测试也因此无法隔离，只能去碰真实 ``config/deepseek.yaml``。
        """
        return self.config_dir / "deepseek.yaml"

    def _load(self):
        """从配置文件 + 环境变量加载"""
        path = self.config_path
        if path.exists():
            try:
                import yaml
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                for k in ("api_key", "model", "base_url", "provider",
                          "timeout", "temperature", "max_tokens", "enabled"):
                    if k in data and data[k] is not None:
                        setattr(self.config, k, data[k])
            except Exception as e:
                logger.warning(f"DeepSeek 配置加载失败: {e}")

        # 环境变量优先
        env_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if env_key:
            self.config.api_key = env_key

    def save(self):
        """保存配置（写入 yaml）"""
        import yaml
        data = {
            "api_key": self.config.api_key,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "provider": self.config.provider,
            "timeout": self.config.timeout,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "enabled": self.config.enabled,
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
        logger.info("LLM 配置已保存 → %s", self.config_path)

    def set_api_key(self, key: str):
        """设置 API Key"""
        self.config.api_key = key.strip()
        self.config.enabled = bool(self.config.api_key)
        self.save()

    def configure(self, *, provider: Optional[str] = None,
                  base_url: Optional[str] = None,
                  model: Optional[str] = None,
                  api_key: Optional[str] = None) -> DeepSeekConfig:
        """一次性更新连接配置并落盘。

        只改**显式传入**的字段，其余保持原样 —— 于是「只切 provider」
        不会顺手把 base_url 冲掉，反之亦然。

        切到 ``local`` 时**不清空**云端 key：本地与云端来回切是常态，
        留着 key 才能切回去不重新填。
        """
        if provider is not None:
            self.config.provider = str(provider).strip().lower() or "auto"
        if base_url is not None:
            self.config.base_url = str(base_url).strip()
        if model is not None:
            self.config.model = str(model).strip()
        if api_key is not None:
            self.config.api_key = str(api_key).strip()

        # 空字符串归一到默认云端地址，避免 base_url="" 拼出 "/chat/completions"
        if not self.config.base_url:
            self.config.base_url = API_BASE
        self.config.enabled = self.is_ready
        self.save()
        return self.config

    def describe(self) -> str:
        """一行人类可读的连接状态，给看板/日志用。"""
        if self.is_local:
            return f"本地服务 · {self.config.base_url} · 模型 {self.config.model}"
        return f"云端 DeepSeek · {self.config.base_url} · 模型 {self.config.model}"

    def list_models(self, timeout: Optional[int] = None,
                    base_url: Optional[str] = None) -> List[str]:
        """列出服务端可用模型，失败返回空列表（不抛）。

        云端 DeepSeek 只有固定的两个模型名，列了意义不大；
        这个方法主要给**本地服务**用 —— Ollama / LM Studio 的模型名
        又长又带 tag（``qwen2.5:7b-instruct-q4_K_M``），手打必错，
        看板靠它做下拉选择。

        ``base_url`` 用于**探测界面上尚未保存的地址**：不落盘就去试连，
        免得用户改了地址却拿旧配置探测，得到「没找到」的错误结论。
        """
        base = str(base_url or self.config.base_url or API_BASE).strip().rstrip("/")
        url = f"{base}/models"
        try:
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {self._auth_key()}"},
                timeout=timeout or min(self.config.timeout, 10),
                **self._bypass_proxy_kwargs(url),
            )
            resp.raise_for_status()
            data = resp.json() or {}
        except Exception as e:  # noqa: BLE001 —— 探测失败是预期结果，不该炸 UI
            logger.debug(f"拉取模型列表失败 ({url}): {e}")
            return []

        items = data.get("data") or []
        names = [str(it.get("id") or it.get("name") or "").strip()
                 for it in items if isinstance(it, dict)]
        return [n for n in names if n]

    @property
    def is_local(self) -> bool:
        """是否按「本机 OpenAI 兼容服务」模式工作。"""
        return str(self.config.provider or "").strip().lower() == "local"

    @property
    def is_ready(self) -> bool:
        """是否具备发起调用的条件。

        ⚠️ ``local`` 模式**不要求真 key**：Ollama / vLLM / LM Studio 等
        本机服务通常不校验 ``Authorization``。若沿用「必须有 key」的判定，
        本地部署会被一道无意义的门槛拦住 —— 那正是「连本地怎么都不行」
        的第二个原因（第一个是 ``base_url`` 没生效，见 :meth:`_chat_url`）。
        """
        if self.is_local:
            return bool(self.config.base_url)
        return bool(self.config.api_key)

    def _chat_url(self) -> str:
        """按 ``config.base_url`` 现算 chat/completions 端点。

        ⚠️ 这里修的是一个**真 bug**：``base_url`` 一直被 ``_load()`` 读进
        ``self.config``，但 ``chat()`` 发的是模块级常量 ``CHAT_URL``
        （硬编码 ``https://api.deepseek.com/chat/completions``）——
        配置字段**从来没有生效过**，是个死开关。后果是：

        * 想在 ``deepseek.yaml`` 里把地址改成自建代理 / 中转，**毫无效果**；
        * 想连本机的 Ollama / vLLM / LM Studio，**更不可能** ——
          请求依然发往 api.deepseek.com，然后拿回 402。

        现在改为现算。三种写法都接受，省得让人去猜格式：

            https://api.deepseek.com          → .../chat/completions
            http://localhost:11434/v1         → .../v1/chat/completions
            http://localhost:11434/v1/chat/completions   → 原样使用
        """
        base = str(self.config.base_url or API_BASE).strip().rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _auth_key(self) -> str:
        """发给服务端的 key。本地服务不需要真 key，但字段不能空。"""
        if self.config.api_key:
            return self.config.api_key
        return "local" if self.is_local else ""

    @staticmethod
    def _is_loopback(url: str) -> bool:
        """目标地址是否为本机回环。"""
        from urllib.parse import urlparse
        try:
            host = (urlparse(str(url)).hostname or "").strip().lower()
        except Exception:  # noqa: BLE001
            return False
        return host in {"localhost", "127.0.0.1", "::1", ""}

    @classmethod
    def _bypass_proxy_kwargs(cls, url: str) -> Dict[str, Any]:
        """访问**本机回环地址时绕过代理**。

        ⚠️ 这是一个真实且很隐蔽的坑。命令行环境里常设有 ``HTTP_PROXY``
        （本机此刻就有 ``http://127.0.0.1:57236``），而 ``NO_PROXY``
        往往没人配。于是连 ``http://localhost:11434`` 这种回环请求
        也会被 requests 交给代理 —— 代理不认识它，回一个 **502 Bad
        Gateway**。表现出来就是：

            「Ollama 明明 `ollama serve` 起来了，模型也拉了，就是连不上。」

        502/503 不会指向「代理劫持」，所以极难自己想到。回环地址没有
        理由走代理，这里直接绕过。云端地址照旧（可能确实需要代理）。
        """
        if cls._is_loopback(url):
            return {"proxies": {"http": None, "https": None}}
        return {}


    # ========== 核心调用 ==========

    def chat(self, system: str, user: str, temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> str:
        """发送聊天请求，返回文本"""
        if not self.is_ready:
            if self.is_local:
                raise RuntimeError(
                    "本地模式未配置服务地址，请在「设置」页填写 base_url"
                    "（如 http://localhost:11434/v1）"
                )
            raise RuntimeError("未配置 DeepSeek API Key，请在「设置」页填写")

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._auth_key()}",
            "Content-Type": "application/json",
        }
        url = self._chat_url()

        try:
            resp = requests.post(
                url, json=payload, headers=headers,
                timeout=self.config.timeout,
                **self._bypass_proxy_kwargs(url),
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return content.strip()
        except requests.exceptions.Timeout:
            logger.error(f"LLM 请求超时: {url}")
            raise RuntimeError(
                f"LLM 请求超时（{self.config.timeout}s）。"
                + ("本地服务未启动或模型仍在加载？" if self.is_local
                   else "请稍后重试")
            )
        except requests.exceptions.ConnectionError:
            logger.error(f"LLM 连接失败: {url}")
            raise RuntimeError(
                f"连不上 LLM 服务：{url}"
                + (" —— 请确认本地推理服务已启动（Ollama: ollama serve）"
                   if self.is_local else "")
            )
        except requests.exceptions.HTTPError as e:
            logger.error(f"LLM HTTP 错误: {e}")
            try:
                detail = e.response.json().get("error", {}).get("message", str(e))
            except Exception:
                detail = str(e)
            status = e.response.status_code if e.response is not None else 0
            if status == 402:
                if self.is_local:
                    # 自相矛盾的典型场景：provider 已切 local，但 base_url
                    # 还指向 api.deepseek.com —— 请求照样打到云端。此时再提示
                    # 「可改用本地模式」是错的（用户已经 local 了），必须指出
                    # 真正的病因：地址没改、且本机没起服务。
                    detail += (
                        "（已是本地模式，但请求仍发到了云端地址 "
                        f"{self._chat_url()}。"
                        "本地模式的 base_url 必须指向你电脑上的服务"
                        "（如 http://localhost:11434/v1），且本机需先"
                        "`ollama serve` + `ollama pull <模型>`）"
                    )
                else:
                    detail += "（云端 DeepSeek 余额不足；可改用本地模式 provider: local）"
            elif self.is_local:
                # 502/503 通常是「端口有人在听，但不是你的模型服务」——
                # 最常见的原因是 HTTP_PROXY 劫持了回环请求（本模块已自动绕过），
                # 其次是服务没起来 / 模型还在加载。
                detail += (
                    f"（本地服务返回 HTTP {status}。"
                    "若用 Ollama：确认 `ollama serve` 已启动、"
                    "`ollama pull <模型>` 已拉取，地址为 http://localhost:11434/v1；"
                    "若仍在用 LM Studio，注意先在界面里 Start Server）"
                )
            raise RuntimeError(f"LLM 接口错误: {detail}")
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM 请求异常: {e}")
            raise RuntimeError(f"LLM 请求异常: {e}")

    def test_connection(self) -> str:
        """测试连接，返回模型名"""
        resp = self.chat("你是连接测试助手。", "回复：连接成功", max_tokens=20)
        return f"连接成功 · 模型 {self.config.model} · {resp[:30]}"

    # ========== 业务方法 ==========

    def analyze_stock(self, name: str, code: str, metrics: Dict) -> str:
        """
        生成个股分析报告
        metrics 应包含: 收盘价/涨跌幅/AI评分/概率/技术指标/基本面/信号等
        """
        system = (
            "你是资深A股证券分析师，擅长用简洁专业的中文撰写个股分析报告。"
            "基于给定数据给出客观、有依据的分析，明确风险提示，不构成投资建议。"
            "输出用 Markdown 格式，包含：一、核心结论；二、技术面；三、基本面；"
            "四、风险提示；五、操作参考。控制在400字以内，避免空话套话。"
        )
        user = f"请分析以下股票：\n\n名称：{name}（{code}）\n"
        for k, v in metrics.items():
            user += f"- {k}: {v}\n"
        user += "\n请给出分析报告。"
        return self.chat(system, user)

    def explain_recommendations(self, stocks: List[Dict]) -> str:
        """
        生成选股组合推荐理由
        stocks: [{代码, 名称, 评分, 上涨概率, 下跌概率, 推荐理由}, ...]
        """
        system = (
            "你是资深A股策略分析师。基于AI量化模型的选股结果，"
            "用专业但通俗的中文解释为什么这些股票值得关注，"
            "提炼共性特征和板块逻辑，同时提示风险。不构成投资建议。"
            "输出用 Markdown，控制在500字以内。"
        )
        user = "AI 选股模型今日推荐以下股票（按评分排序）：\n\n"
        for i, s in enumerate(stocks, 1):
            user += (
                f"{i}. {s.get('名称', '')}（{s.get('代码', '')}）"
                f" | 评分 {s.get('评分', '?')}"
                f" | 上涨概率 {s.get('上涨概率', '?')}%"
                f" | 下跌概率 {s.get('下跌概率', '?')}%"
            )
            if s.get("推荐理由"):
                user += f" | 理由: {s.get('推荐理由')}"
            user += "\n"
        user += "\n请总结这批推荐的整体逻辑、共性特征和风险提示。"
        return self.chat(system, user)


# 全局单例
deepseek = DeepSeekClient()
