# -*- coding: utf-8 -*-
"""推送通知单元测试（离线，注入 Fake poster，不发真实请求）"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.core.notifier import Notifier, NotifierConfig  # noqa: E402


class _FakePoster:
    def __init__(self, ok=True, status=200):
        self.ok = ok
        self.status = status
        self.calls = []

    def __call__(self, url, payload, timeout):
        self.calls.append((url, payload, timeout))
        return (self.ok, self.status)


def test_disabled_is_noop():
    n = Notifier(NotifierConfig(enabled=False))
    assert n.send("t", "md") is False


def test_log_channel_writes_file(tmp_path):
    cfg = NotifierConfig(enabled=True, channel="log", out_dir=str(tmp_path))
    n = Notifier(cfg)
    assert n.send("测试标题", "正文内容") is True
    files = list(tmp_path.glob("*.md"))
    assert len(files) == 1
    assert "正文内容" in files[0].read_text(encoding="utf-8")


def test_webhook_posts_json():
    poster = _FakePoster(ok=True)
    cfg = NotifierConfig(enabled=True, channel="webhook", webhook_url="http://x/hook")
    n = Notifier(cfg, poster=poster)
    assert n.send("T", "M", payload={"a": 1}) is True
    url, payload, _ = poster.calls[0]
    assert url == "http://x/hook"
    assert payload["title"] == "T"
    assert payload["payload"] == {"a": 1}


def test_webhook_missing_url_falls_back_to_log(tmp_path):
    poster = _FakePoster()
    cfg = NotifierConfig(enabled=True, channel="webhook", webhook_url="", out_dir=str(tmp_path))
    n = Notifier(cfg, poster=poster)
    assert n.send("T", "M") is True
    assert len(list(tmp_path.glob("*.md"))) == 1  # 降级为落盘
    assert poster.calls == []


def test_wecom_url_format():
    poster = _FakePoster(ok=True)
    cfg = NotifierConfig(enabled=True, channel="wecom", wecom_key="KEY123")
    n = Notifier(cfg, poster=poster)
    assert n.send("标题", "内容") is True
    url, payload, _ = poster.calls[0]
    assert "qyapi.weixin.qq.com" in url and "key=KEY123" in url
    assert payload["msgtype"] == "markdown"


def test_network_failure_swallowed():
    poster = _FakePoster(ok=False, status="boom")
    cfg = NotifierConfig(enabled=True, channel="webhook", webhook_url="http://x")
    n = Notifier(cfg, poster=poster)
    # 网络失败不抛异常，返回 False
    assert n.send("T", "M") is False
