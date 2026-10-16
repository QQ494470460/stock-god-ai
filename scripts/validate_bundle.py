# -*- coding: utf-8 -*-
"""无界面校验打包产物：只用 bundle 的 _internal + 标准库 import src.ui.app，
模拟冻结运行时的模块解析。任何缺失依赖都会以 ModuleNotFoundError 暴露。

用法：py -3.12 scripts/validate_bundle.py [bundle_internal_dir]
"""
import os
import sys
import traceback

BUNDLE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dist", "选股神AI", "_internal",
)
INTERNAL = os.path.abspath(BUNDLE)
if not os.path.isdir(INTERNAL):
    print(f"✗ 找不到 bundle 目录: {INTERNAL}")
    sys.exit(2)

# 模拟冻结运行时：sys.path 只保留 标准库 + bundle 的 _internal
stdlib_lib = os.path.dirname(os.__file__)
sys.path = [INTERNAL, os.path.join(INTERNAL, "src"), stdlib_lib]

# 与 launcher 一致的必要环境变量
base = os.path.dirname(INTERNAL)
os.environ["STOCK_GOD_MODEL_DIR"] = os.path.join(base, "models")
os.environ["STOCK_GOD_DATA_DIR"] = os.path.join(base, "data")
os.environ["STOCK_GOD_LOG_DIR"] = os.path.join(base, "logs")
os.environ["STREAMLIT_SERVER_PORT"] = "0"

print(f"校验 bundle: {INTERNAL}")
before = set(sys.modules)
try:
    import src.ui.app  # noqa: F401
    added = sorted(set(sys.modules) - before)
    print(f"✓ IMPORT OK —— src.ui.app 导入成功，闭包新增 {len(added)} 个模块")
    sys.exit(0)
except Exception:
    tb = traceback.format_exc()
    print("✗ IMPORT FAILED —— 看板入口导入失败：\n")
    print(tb)
    sys.exit(1)
