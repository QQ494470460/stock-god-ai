# AI选股缩进修复 - 2026-07-06

## 问题
用户反馈 AI 选股功能"没修复好"，排查发现 `app.py` 第 386 行缩进错误。

## 根因
```python
# 修复前
            import sys; print(...)
display_max = scan_max if scan_max != "全" else "全部"  # 缺少12空格缩进
            with st.spinner(...):
```

`display_max` 行缩进丢失，导致 `if st.button(...)` 代码块语法断裂，Streamlit 启动时 Python 解析失败。

## 修复
- 删除调试行 `import sys; print(...)` 
- `display_max` 补回正确缩进（12空格）
- 语法验证：`ast.parse()` 通过

## 验证
- app.py 语法 OK
- 全链路端到端测试：10只股 fetch→features(111列)→predict→DataFrame(13列) 正常
- 所有 import 链验证通过
- Streamlit 已重启 (PID 17284, 端口8501)
