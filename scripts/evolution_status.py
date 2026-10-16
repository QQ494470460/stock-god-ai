# -*- coding: utf-8 -*-
"""查看「自我进化」的反馈底稿与参数空间 —— 纯只读，不改任何东西。

为什么单独给一个命令：

    进化的第一步不是「改参数」，而是「证明现在到底行不行」。
    如果连准确率、净值、回撤都拿不到，任何改动都是盲猜。
    本命令把三路反馈信号 + 数据完备度提示一次摊开，作为进化的起点。

用法
----
    py -3.12 scripts/evolution_status.py              # 人读报告
    py -3.12 scripts/evolution_status.py --json       # 机器读（JSON）
    py -3.12 scripts/evolution_status.py --space      # 额外打印可进化参数空间
    py -3.12 scripts/evolution_status.py --verify     # ⚠️ 会写盘：补验所有到期的预测批次

退出码：0 = 正常；1 = 采集异常。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 必须先导入 config.settings：它会注入外挂 libs 目录（打包版需要）
try:  # noqa: SIM105
    import config.settings  # noqa: F401
except Exception:  # noqa: BLE001
    pass


def main() -> int:
    ap = argparse.ArgumentParser(description="自我进化 · 反馈底稿查看（只读）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--space", action="store_true", help="附加打印可进化参数空间")
    ap.add_argument("--verify", action="store_true",
                    help="补验到期预测批次（会写 prediction_history.json）")
    ap.add_argument("--no-trades", action="store_true", help="跳过交易侧采集")
    args = ap.parse_args()

    from src.evolution import describe_space, feedback

    if args.verify:
        print("⏳ 正在补验到期预测批次（会写 data/prediction_history.json）…")
        results = feedback.verify_pending()
        print(f"✅ 新增验证结果 {len(results)} 条\n")

    snap = feedback.collect(with_trades=not args.no_trades)

    if args.json:
        print(feedback.snapshot_to_json(snap))
    else:
        print(feedback.render_text(snap))
        if args.space:
            print()
            print(describe_space())

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:  # noqa: BLE001
        print(f"❌ 采集失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
