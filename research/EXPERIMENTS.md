---
name: EXPERIMENTS.md
type: source of truth (log thực nghiệm, authoritative — RESEARCH.md chỉ roll-up từ đây)
---

# Experiment Log

Quy ước: `EXP-xxx` là ID nội bộ của file này. Khi thêm experiment mới:
thêm 1 dòng vào index NGAY, rồi mới viết block chi tiết (Question / Input
/ Output / Decision). Không xoá dòng khỏi index kể cả khi experiment bị
huỷ giữa chừng (đánh dấu ABORTED).

## Index (đọc bảng này trước — chỉ mở block chi tiết khi cần review sâu 1 case)

| ID | Tên | Câu hỏi | Quyết định | avg BB | File kết quả |
|---|---|---|---|---:|---|
| _(trống)_ | | | | | |

_(trống sau reset 2026-08-21 — `results/*.json` cũ đã bị xóa khỏi working
tree, vẫn còn trong git history trước commit reset nếu cần tra cứu số
liệu cũ.)_
