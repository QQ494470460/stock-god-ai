# -*- coding: utf-8 -*-
"""业绩档案 CLI —— 把每日结算快照汇成一条可判断的净值曲线。

用法::

    python scripts/performance.py                 # 打印并落盘 data/performance.json
    python scripts/performance.py --rf 2.0        # 指定无风险年化 2%
    python scripts/performance.py --out /tmp/p.json
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.trading.performance import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
