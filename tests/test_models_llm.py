# -*- coding: utf-8 -*-
"""LLM 接入模块（``src/models/llm.py``）测试。

重点在**回归两个「死配置」bug**。它们都不是崩溃型的，而是
「你改了配置，系统毫无反应」——这种病最容易被忽略，因为日志里
一片祥和，只是功能永远不生效。所以只能靠断言**请求的实际落点**
来防住：

1. ``base_url`` 从来没生效过 —— ``chat()`` 发的是硬编码常量 ``CHAT_URL``
   （写死 ``https://api.deepseek.com/chat/completions``）。
   后果：想连本机的 Ollama / vLLM / LM Studio 根本不可能，
   请求照样发去云端，然后拿回 402。
2. ``config_dir`` 从来没生效过 —— ``__init__`` 收下它只拿来 ``mkdir``，
   真正的读写一律走模块级 ``DEEPSEEK_CONFIG``。
   后果：测试无法隔离，只能去碰真实 ``config/deepseek.yaml``。

另外证明本地模式确实**不需要 API Key** —— 这是「连本地」能否成立的前提：
若沿用「没 key 就不 ready」的判定，本地部署会被一道无意义的门槛拦死。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models import llm as llm_mod          # noqa: E402
from src.models.llm import DeepSeekClient       # noqa: E402


# ------------------------------------------------------------------ 夹具

class _Resp:
    """最小可用的 ``requests.Response`` 替身——够跑通 chat() 的取值路径即可。"""

    def __init__(self, payload=None, status=200, text=""):
        self._payload = payload if payload is not None else {}
        self.status_code = status
        self._text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            err = requests.exceptions.HTTPError(f"HTTP {self.status_code}")
            err.response = self
            raise err

    def json(self):
        if self._text:
            import json
            return json.loads(self._text)
        return self._payload


@pytest.fixture(autouse=True)
def _no_env_key(monkeypatch):
    """环境变量会盖掉配置——测试里必须清掉，否则结果随机器环境漂移。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)


@pytest.fixture
def client(tmp_path):
    """隔离到 ``tmp_path`` 的客户端。

    这个夹具本身就在回归第 2 个 bug：若 ``config_dir`` 又被忽略，
    它就会读到真实的 ``config/deepseek.yaml`` 而不是空目录。
    """
    return DeepSeekClient(config_dir=tmp_path)


def _capture_post(monkeypatch):
    """截获 ``requests.post``，返回记录表。"""
    calls = []

    def fake_post(url, **kw):
        rec = {"url": url}
        rec.update(kw)          # 连 proxies 一起记，才能断言代理是否被绕过
        calls.append(rec)
        return _Resp({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    return calls


# ------------------------------------------------------- ① base_url 真的生效

def test_chat_url_normalizes_the_common_spellings(client):
    """三种写法都要拼对，别让人去猜格式。"""
    cases = [
        ("https://api.deepseek.com", "https://api.deepseek.com/chat/completions"),
        ("http://localhost:11434/v1", "http://localhost:11434/v1/chat/completions"),
        ("http://localhost:11434/v1/chat/completions",
         "http://localhost:11434/v1/chat/completions"),
        ("http://localhost:11434/v1/", "http://localhost:11434/v1/chat/completions"),
    ]
    for base, want in cases:
        client.config.base_url = base
        assert client._chat_url() == want, f"base_url={base!r}"


def test_chat_actually_posts_to_the_configured_base_url(client, monkeypatch):
    """核心回归：配置里的地址必须真的成为请求落点。

    旧实现发的是硬编码的 ``CHAT_URL``，于是无论配置写什么，请求都发去
    api.deepseek.com —— 这正是「想连本地却永远拿到 402」的直接原因。
    """
    client.config.provider = "local"
    client.config.base_url = "http://localhost:11434/v1"
    client.config.model = "qwen2.5:7b"
    calls = _capture_post(monkeypatch)

    assert client.chat("sys", "hi") == "ok"
    assert calls[0]["url"] == "http://localhost:11434/v1/chat/completions"
    assert calls[0]["json"]["model"] == "qwen2.5:7b"


def test_empty_base_url_falls_back_to_cloud(client):
    """base_url 被清空时不能拼出 ``/chat/completions`` 这种残缺地址。"""
    client.config.base_url = ""
    assert client._chat_url() == "https://api.deepseek.com/chat/completions"


# ------------------------------------------------- ② 本地模式不需要 API Key

def test_local_mode_is_ready_without_an_api_key(tmp_path):
    c = DeepSeekClient(config_dir=tmp_path)
    c.config.provider = "local"
    c.config.base_url = "http://localhost:11434/v1"
    c.config.api_key = ""
    assert c.is_local is True
    assert c.is_ready is True


def test_cloud_mode_is_not_ready_without_an_api_key(tmp_path):
    c = DeepSeekClient(config_dir=tmp_path)
    c.config.provider = "deepseek"
    c.config.api_key = ""
    assert c.is_ready is False


def test_local_mode_sends_a_placeholder_key(client, monkeypatch):
    """本地服务不看 Authorization，但字段空着有些实现会直接 400。"""
    client.config.provider = "local"
    client.config.base_url = "http://localhost:11434/v1"
    client.config.api_key = ""
    calls = _capture_post(monkeypatch)
    client.chat("sys", "hi")
    assert calls[0]["headers"]["Authorization"] == "Bearer local"


def test_an_explicit_key_is_never_overwritten(client, monkeypatch):
    """真 key 在手时不能被占位符顶掉——本地中转也可能是要校验的。"""
    client.config.provider = "local"
    client.config.base_url = "http://localhost:11434/v1"
    client.config.api_key = "sk-real"
    calls = _capture_post(monkeypatch)
    client.chat("sys", "hi")
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-real"


def test_local_mode_with_a_cloud_url_points_out_the_contradiction(client, monkeypatch):
    """已切 local 但 base_url 仍指向 api.deepseek.com 时，提示不能再说
    「可改用本地模式」—— 用户已经是 local 了，真正的病是地址没改。

    这正是用户真实踩到的坑：把 provider 切到 local 之后，
    地址框还停在 api.deepseek.com，于是请求照样打到云端、拿回 402，
    而旧提示却让他「改用本地模式」，等于原地打转。
    """
    client.config.provider = "local"
    client.config.base_url = "https://api.deepseek.com"   # 矛盾配置
    client.config.api_key = ""

    def boom(url, **kw):
        return _Resp(status=402, text='{"error":{"message":"Insufficient Balance"}}')

    monkeypatch.setattr(llm_mod.requests, "post", boom)
    with pytest.raises(RuntimeError) as exc:
        client.chat("sys", "hi")
    msg = str(exc.value)
    assert "云端地址" in msg
    assert "ollama serve" in msg
    # 关键：绝不能出现自相矛盾的「可改用本地模式」
    assert "可改用本地模式" not in msg


# ------------------------------------------------------------ ③ 可操作的报错

def test_connection_error_tells_you_how_to_start_the_service(client, monkeypatch):
    """连不上本机时，提示要落在「下一步做什么」上，而不是一句 ConnectionError。"""
    import requests
    client.config.provider = "local"
    client.config.base_url = "http://localhost:11434/v1"

    def boom(url, **kw):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(llm_mod.requests, "post", boom)
    with pytest.raises(RuntimeError, match="ollama serve"):
        client.chat("sys", "hi")


def test_http_402_hints_at_local_mode(client, monkeypatch):
    """余额不足时给出真正的出路 —— 本地模式不花钱，这是最实际的建议。"""
    client.config.provider = "deepseek"
    client.config.api_key = "sk-x"

    def boom(url, **kw):
        return _Resp(status=402, text='{"error":{"message":"Insufficient Balance"}}')

    monkeypatch.setattr(llm_mod.requests, "post", boom)
    with pytest.raises(RuntimeError, match="provider: local"):
        client.chat("sys", "hi")


def test_a_programming_error_is_not_disguised_as_a_call_failure(client, monkeypatch):
    """``except Exception`` 曾把 NameError 包装成「LLM 调用失败」。

    那正是 ``CHAT_URL`` 这个 bug 能一直藏着的原因——真正的错误
    （``name 'CHAT_URL' is not defined``）被自己的兜底逻辑吃掉了，
    只留下一句看起来像网络问题的 RuntimeError。
    现在只兜网络异常，编程错误必须原样抛出来。
    """
    # 先让客户端进入「可发起调用」状态，否则会先被 is_ready 拦下，
    # 测不到下面的兜底逻辑。
    client.config.provider = "local"
    client.config.base_url = "http://localhost:11434/v1"

    def boom(url, **kw):
        raise NameError("name 'CHAT_URL' is not defined")

    monkeypatch.setattr(llm_mod.requests, "post", boom)
    with pytest.raises(NameError):
        client.chat("sys", "hi")


# ------------------------------------------------------- ④ config_dir 真生效

def test_config_dir_decides_which_file_is_read(tmp_path):
    """第二个死参数回归：``config_dir`` 必须真的决定读哪个文件。"""
    import yaml

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    yaml.safe_dump({"provider": "local", "base_url": "http://a:1/v1",
                    "model": "model-a", "api_key": ""},
                   open(a / "deepseek.yaml", "w", encoding="utf-8"),
                   allow_unicode=True)

    ca = DeepSeekClient(config_dir=a)
    cb = DeepSeekClient(config_dir=b)

    assert ca.config.model == "model-a"          # 读到 a 的文件
    assert ca.is_local is True
    assert cb.config.model == "deepseek-chat"    # b 无文件 → 走默认


def test_config_dir_decides_where_configure_writes(tmp_path):
    """保存也必须写进 ``config_dir``，否则「配置改了但没人认」。"""
    import yaml

    target = tmp_path / "isolated"
    c = DeepSeekClient(config_dir=target)
    c.configure(provider="local", base_url="http://localhost:1234/v1",
                model="qwen2.5:3b")

    assert (target / "deepseek.yaml").exists()
    raw = yaml.safe_load(open(target / "deepseek.yaml", encoding="utf-8"))
    assert raw["provider"] == "local"
    assert raw["model"] == "qwen2.5:3b"
    assert raw["enabled"] is True


def test_meta__the_config_dir_assertions_have_teeth(tmp_path, monkeypatch):
    """负向对照：证明上面那组断言真能抓到「忽略 config_dir」的实现。

    把模块级常量指到目录 a，再要求客户端用目录 b。若实现退回
    「读模块常量」，它就会读到 a 的内容，上一组断言会立刻变红。
    """
    import yaml

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    yaml.safe_dump({"provider": "local", "base_url": "http://a:1/v1",
                    "model": "model-a", "api_key": ""},
                   open(a / "deepseek.yaml", "w", encoding="utf-8"),
                   allow_unicode=True)

    monkeypatch.setattr(llm_mod, "DEEPSEEK_CONFIG", a / "deepseek.yaml")
    cb = DeepSeekClient(config_dir=b)

    # 坏实现会拿到 "model-a"；正确实现应拿默认值
    assert cb.config.model == "deepseek-chat"


# ------------------------------------------------------------ ⑤ 回环地址绕过代理

def test_loopback_is_detected():
    for u in ("http://localhost:11434/v1", "http://127.0.0.1:8000/v1",
              "http://[::1]:1234/v1"):
        assert DeepSeekClient._is_loopback(u) is True, u
    for u in ("https://api.deepseek.com", "http://192.168.1.9:11434/v1",
              "http://my-ollama.local:11434"):
        assert DeepSeekClient._is_loopback(u) is False, u


def test_local_calls_bypass_the_proxy(client, monkeypatch):
    """回环请求必须绕过代理，否则会被代理吞掉并回一个莫名的 502。

    这个坑的代价是「服务明明起来了却连不上」，且 502 完全不会指向代理，
    极难自诊——所以只能靠断言请求参数把它钉死。
    """
    client.config.provider = "local"
    client.config.base_url = "http://localhost:11434/v1"
    calls = _capture_post(monkeypatch)

    client.chat("sys", "hi")
    assert calls[0]["url"].startswith("http://localhost")
    assert calls[0].get("proxies") == {"http": None, "https": None}


def test_cloud_calls_still_honour_the_proxy(client, monkeypatch):
    """云端地址不强制绕过——人家可能真的需要代理才能出网。"""
    client.config.provider = "deepseek"
    client.config.api_key = "sk-x"
    calls = _capture_post(monkeypatch)

    client.chat("sys", "hi")
    assert calls[0]["url"].startswith("https://api.deepseek.com")
    assert "proxies" not in calls[0]


def test_list_models_bypasses_the_proxy_too(client, monkeypatch):
    seen = []

    def fake_get(url, **kw):
        seen.append(kw)
        return _Resp({"data": []})

    monkeypatch.setattr(llm_mod.requests, "get", fake_get)
    client.config.base_url = "http://127.0.0.1:11434/v1"
    client.list_models()
    assert seen[0].get("proxies") == {"http": None, "https": None}


# ------------------------------------------------------------ ⑥ 默认值与探测

def test_defaults_still_point_at_the_cloud(tmp_path):
    """默认行为必须与改动前一致 —— 仍走云端，不悄悄改变既有部署。"""
    c = DeepSeekClient(config_dir=tmp_path)
    assert c.config.base_url == llm_mod.API_BASE
    assert c.is_local is False
    assert c._chat_url() == "https://api.deepseek.com/chat/completions"


def test_configure_keeps_unpassed_fields(client):
    """只改传入的字段：切 provider 不该顺手冲掉 base_url，反之亦然。"""
    client.configure(provider="local", base_url="http://localhost:1234/v1",
                     model="my-model")
    assert client.config.base_url == "http://localhost:1234/v1"

    client.configure(model="another")
    assert client.config.base_url == "http://localhost:1234/v1"
    assert client.config.provider == "local"
    assert client.config.model == "another"


def test_describe_distinguishes_local_from_cloud(client):
    client.configure(provider="local", base_url="http://localhost:11434/v1",
                     model="qwen2.5:7b")
    assert "本地" in client.describe()
    assert "qwen2.5:7b" in client.describe()

    client.configure(provider="deepseek", api_key="sk-x",
                     model="deepseek-chat")
    assert "云端" in client.describe()


def test_list_models_returns_names(client, monkeypatch):
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        return _Resp({"data": [{"id": "qwen2.5:7b"}, {"id": "llama3.2:3b"}]})

    monkeypatch.setattr(llm_mod.requests, "get", fake_get)
    client.config.base_url = "http://localhost:11434/v1"

    assert client.list_models() == ["qwen2.5:7b", "llama3.2:3b"]
    assert calls[0] == "http://localhost:11434/v1/models"


def test_list_models_probes_an_unsaved_address(client, monkeypatch):
    """探测界面上还没保存的地址 —— 否则用户改了地址却拿到旧的「没找到」。"""
    seen = []

    def fake_get(url, **kw):
        seen.append(url)
        return _Resp({"data": [{"id": "unsaved-model"}]})

    monkeypatch.setattr(llm_mod.requests, "get", fake_get)
    client.config.base_url = "http://localhost:11434/v1"

    assert client.list_models(base_url="http://localhost:1234/v1") == ["unsaved-model"]
    assert seen[0] == "http://localhost:1234/v1/models"


def test_list_models_swallows_service_errors(client, monkeypatch):
    """探测跑在看板 UI 路径上，失败要静默返回空列表而不是弹红。"""
    import requests

    def boom(url, **kw):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(llm_mod.requests, "get", boom)
    assert client.list_models() == []
