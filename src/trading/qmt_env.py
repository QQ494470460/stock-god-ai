# -*- coding: utf-8 -*-
"""
QMT 安装路径自动探测
====================

用户装完 QMT 后通常不知道「userdata_mini」到底在哪，手填极易写错。
本模块在本机有限深度地扫描常见位置，找出 QMT 的 userdata_mini 目录。

探测依据（按可信度）：
  1. 目录名 == "userdata_mini" 且同级/上级存在 QMT 特征文件
     （XtMiniQmt.exe / bin.x64 / xtquant / XtItClient.exe）
  2. 目录名 == "userdata_mini"（单独出现也大概率是它）
  3. 目录内含 xtquant 或 bin.x64 的 QMT 安装根目录 → 推断同级 userdata_mini

不依赖任何第三方库，纯 os.scandir 广度优先，带深度与访问量上限，避免卡死。
"""

from __future__ import annotations

import os
import string
from pathlib import Path
from typing import Iterable

# QMT 安装目录常见的特征文件/目录名（小写比较）
_QMT_MARKERS = (
    "xtminiqmt.exe",
    "xtitclient.exe",
    "bin.x64",
    "xtquant",
    "xtminiqmt",
)
_TARGET_DIRNAME = "userdata_mini"

# 优先扫描的根目录（存在才算）
_PRIORITY_ROOTS = [
    r"C:\QMT", r"D:\QMT", r"E:\QMT", r"F:\QMT",
    r"C:\迅投", r"D:\迅投", r"C:\国金QMT", r"D:\国金QMT",
    r"C:\Program Files", r"C:\Program Files (x86)",
    r"C:\ProgramData",
    r"C:\Users\%s".replace("%s", os.environ.get("USERNAME", "") or "Public"),
]


def _existing_drives() -> list[str]:
    out = []
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if os.path.exists(root):
            out.append(root)
    return out


def _iter_roots() -> list[str]:
    """构造待扫描根目录：优先目录 + 所有盘符根。"""
    roots: list[str] = []
    for r in _PRIORITY_ROOTS:
        if r and os.path.isdir(r):
            roots.append(r)
    for d in _existing_drives():
        if d not in roots:
            roots.append(d)
    # 去重保序
    seen, uniq = set(), []
    for r in roots:
        if r.lower() not in seen:
            seen.add(r.lower())
            uniq.append(r)
    return uniq


def _has_marker(path: str) -> bool:
    """目录内是否含 QMT 特征（只查一层，够用且快）。"""
    try:
        with os.scandir(path) as it:
            for e in it:
                if e.name.lower() in _QMT_MARKERS:
                    return True
    except (PermissionError, OSError):
        pass
    return False


def detect_qmt_userdata_mini(
    max_depth: int = 4,
    max_visits: int = 200_000,
    timeout_sec: float = 20.0,
) -> tuple[str | None, str]:
    """扫描本机找 QMT 的 userdata_mini 目录。

    返回 (路径 or None, 证据说明)。
    """
    import time

    t0 = time.time()
    best: str | None = None
    best_score = -1
    best_evidence = ""
    visits = 0

    for root in _iter_roots():
        if time.time() - t0 > timeout_sec:
            break
        if not os.path.isdir(root):
            continue

        # 广度优先，限制深度
        queue: list[tuple[str, int]] = [(root, 0)]
        while queue:
            if time.time() - t0 > timeout_sec or visits > max_visits:
                break
            cur, depth = queue.pop(0)
            visits += 1

            # 命中 userdata_mini：评分
            if os.path.basename(cur).lower() == _TARGET_DIRNAME:
                score = 1
                evidence = "目录名为 userdata_mini"
                # 上级是否是 QMT 安装根
                parent = os.path.dirname(cur)
                if os.path.isdir(parent) and _has_marker(parent):
                    score = 3
                    evidence = "上级目录含 QMT 特征文件（bin.x64 / XtMiniQmt.exe / xtquant）"
                elif _has_marker(cur):
                    score = 2
                    evidence = "目录内含 QMT 特征文件"
                if score > best_score:
                    best, best_score, best_evidence = cur, score, evidence
                continue  # 不再往 userdata_mini 内部钻

            if depth >= max_depth:
                continue

            try:
                with os.scandir(cur) as it:
                    for e in it:
                        try:
                            if e.is_dir(follow_symlinks=False):
                                queue.append((e.path, depth + 1))
                        except OSError:
                            continue
            except (PermissionError, OSError):
                continue

    if best:
        return best, best_evidence
    return None, "未在本机找到 userdata_mini（QMT 可能尚未安装）"


def detect_qmt_install_root() -> tuple[str | None, str]:
    """找 QMT 安装根目录（含 bin.x64 / XtMiniQmt.exe 的目录）。"""
    import time
    t0 = time.time()
    for root in _iter_roots():
        if time.time() - t0 > 15.0:
            break
        if not os.path.isdir(root):
            continue
        queue: list[tuple[str, int]] = [(root, 0)]
        while queue:
            if time.time() - t0 > 15.0:
                break
            cur, depth = queue.pop(0)
            if _has_marker(cur):
                # 只要不是 userdata_mini 自身即可
                if os.path.basename(cur).lower() != _TARGET_DIRNAME:
                    return cur, f"找到 QMT 安装特征目录（{os.path.basename(cur)}）"
            if depth >= 4:
                continue
            try:
                with os.scandir(cur) as it:
                    for e in it:
                        try:
                            if e.is_dir(follow_symlinks=False):
                                queue.append((e.path, depth + 1))
                        except OSError:
                            continue
            except (PermissionError, OSError):
                continue
    return None, "未找到 QMT 安装目录"


if __name__ == "__main__":  # 手动跑：py scripts/... 或 py -c
    p, why = detect_qmt_userdata_mini()
    print("userdata_mini:", p)
    print("证据:", why)
    r, why2 = detect_qmt_install_root()
    print("安装根目录:", r)
    print("证据:", why2)
