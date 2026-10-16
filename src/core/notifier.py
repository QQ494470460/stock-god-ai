# -*- coding: utf-8 -*-
"""
推送通知（Notifier）
===================

把调度产物（每日推荐 / 回测摘要 / 进化结果）推送到外部渠道：

  · log     ：仅写日志 + 本地文件（默认，零外部依赖）
  · webhook ：向自定义 URL POST JSON（{title, markdown, payload}）
  · wecom   ：企业微信机器人（POST 到 qyapi.weixin.qq.com，markdown 消息）

设计原则
--------
- 默认关闭（config.notifier.enabled=False），关闭时 send() 为安全无操作；
- 网络失败绝不抛出（调度任务不能因推送失败而中断），只记录警告并降级；
- HTTP 发送器可注入（便于离线单测，不真发请求）。

用法
----
    from src.core.notifier import build_notifier
    notifier = build_notifier()             # 读 config.notifier
    notifier.send("今日推荐", markdown_text, payload={"picks": [...]})
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

from loguru import logger

from config.settings import config as app_config, LOG_DIR

try:
    from config.settings import config as _cfg  # noqa: F401
except Exception:  # noqa: BLE001
    pass


# 默认 HTTP 发送器：用标准库 urllib，避免强依赖 requests
def _default_poster(url: str, payload: dict, timeout: float = 5.0) -> tuple:
    """返回 (ok, status_or_err)。"""
    import urllib.request
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return (200 <= resp.status < 300, resp.status)
    except Exception as e:  # noqa: BLE001
        return (False, str(e))


@dataclass
class NotifierConfig:
    enabled: bool = False
    channel: str = "log"       # log / webhook / wecom
    webhook_url: str = ""
    wecom_key: str = ""
    timeout: float = 5.0
    out_dir: str = ""          # 落盘目录（默认 LOG_DIR/notifications）


class Notifier:
    def __init__(
        self,
        cfg: NotifierConfig,
        poster: Optional[Callable[[str, dict, float], tuple]] = None,
    ):
        self.cfg = cfg
        self.poster = poster or _default_poster
        self.history: list = []

    # ---------- 渠道实现 ----------
    def _send_log(self, title: str, markdown: str, payload: Optional[dict]) -> bool:
        out_dir = Path(self.cfg.out_dir) if self.cfg.out_dir else (LOG_DIR / "notifications")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fp = out_dir / f"{ts}_{title[:20]}.md"
            fp.write_text(f"# {title}\n\n{markdown}\n", encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"通知落盘失败: {e}")
        logger.info(f"🔔 [通知·log] {title}")
        return True

    def _send_webhook(self, title: str, markdown: str, payload: Optional[dict]) -> bool:
        if not self.cfg.webhook_url:
            logger.warning("webhook 渠道未配置 URL，降级为 log")
            return self._send_log(title, markdown, payload)
        body = {"title": title, "markdown": markdown, "payload": payload or {}}
        ok, info = self.poster(self.cfg.webhook_url, body, self.cfg.timeout)
        if ok:
            logger.info(f"🔔 [通知·webhook] {title} 已发送 ({info})")
        else:
            logger.warning(f"webhook 推送失败({info})，已忽略")
        return bool(ok)

    def _send_wecom(self, title: str, markdown: str, payload: Optional[dict]) -> bool:
        if not self.cfg.wecom_key:
            logger.warning("企业微信渠道未配置 key，降级为 log")
            return self._send_log(title, markdown, payload)
        url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={self.cfg.wecom_key}"
        body = {"msgtype": "markdown", "markdown": {"content": f"**{title}**\n\n{markdown}"}}
        ok, info = self.poster(url, body, self.cfg.timeout)
        if ok:
            logger.info(f"🔔 [通知·企业微信] {title} 已发送 ({info})")
        else:
            logger.warning(f"企业微信推送失败({info})，已忽略")
        return bool(ok)

    # ---------- 对外接口 ----------
    def send(self, title: str, markdown: str, payload: Optional[dict] = None) -> bool:
        """发送通知；任何异常都被吞掉（调度不可中断）。返回是否成功。"""
        self.history.append({"time": datetime.now().isoformat(), "title": title})
        if not self.cfg.enabled:
            logger.debug(f"通知未启用，跳过：{title}")
            return False
        try:
            ch = (self.cfg.channel or "log").lower()
            if ch == "wecom":
                return self._send_wecom(title, markdown, payload)
            if ch == "webhook":
                return self._send_webhook(title, markdown, payload)
            return self._send_log(title, markdown, payload)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"通知发送异常（已忽略）: {e}")
            return False


def build_notifier(poster: Optional[Callable] = None) -> Notifier:
    """从全局 config.notifier 构建 Notifier。"""
    nc = getattr(app_config, "notifier", None)
    if nc is None:
        cfg = NotifierConfig()
    else:
        cfg = NotifierConfig(
            enabled=getattr(nc, "enabled", False),
            channel=getattr(nc, "channel", "log"),
            webhook_url=getattr(nc, "webhook_url", ""),
            wecom_key=getattr(nc, "wecom_key", ""),
            timeout=getattr(nc, "timeout", 5.0),
        )
    return Notifier(cfg, poster=poster)
