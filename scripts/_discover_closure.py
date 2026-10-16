# -*- coding: utf-8 -*-
"""用 __import__ 钩子记录 app 导入闭包里出现的所有第三方顶层模块，
区分「包」(collect-all) 与「单文件模块」(add-data)。"""
import builtins
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

seen = set()
_real_import = builtins.__import__


def _hook(name, *a, **k):
    top = name.split(".")[0]
    seen.add(top)
    return _real_import(name, *a, **k)


builtins.__import__ = _hook

SEEDS = [
    "src.ui.app",
    "streamlit", "pandas", "numpy", "scipy", "sklearn", "xgboost",
    "imblearn", "optuna", "shap", "plotly", "akshare", "pywebview",
    "matplotlib", "altair", "pyarrow", "openpyxl",
]
for s in SEEDS:
    try:
        _real_import(s)
    except Exception as e:  # noqa: BLE001
        print(f"# seed 跳过 {s}: {e}", file=sys.stderr)

builtins.__import__ = _real_import

pkgs, single, other = [], [], []
for top in sorted(seen):
    try:
        spec = importlib.util.find_spec(top)
    except Exception:
        continue
    if spec is None:
        continue
    origin = spec.origin or ""
    if "site-packages" not in origin:
        continue
    is_pkg = bool(getattr(spec, "submodule_search_locations", None))
    if not is_pkg and origin.endswith(".py") and not origin.endswith("__init__.py"):
        single.append(top)
    else:
        pkgs.append(top)

print("=== 第三方 包 ===")
print("\n".join(pkgs))
print("\n=== 第三方 单文件模块 ===")
print("\n".join(single))
