---
name: DECISIONS.md
type: source of truth (quyết định của researcher — có thể KHÁC kết luận khoa học ở FINDINGS.md)
---

# Research Decisions

Khác với `FINDINGS.md` (chỉ ghi điều đã chứng minh bằng experiment), file
này ghi **quyết định** — nhiều khi một hướng bị gác lại không phải vì nó
sai, mà vì chi phí, phạm vi, hoặc ưu tiên.

## D-000 — Reset toàn bộ research state, bắt đầu lại từ baseline MI-FGSM + RRB gốc

**Date:** 2026-08-21

**Decision:** Xóa `src/evasion_od` (toàn bộ implementation cũ, gồm cả
nhánh RaPA-mask và spectral augmentation), xóa `results/*.json`, và reset
`research/` về trống — không giữ lại finding/idea/decision của phase
trước (OSFD baseline → RaPA-mask → spectral augmentation) làm căn cứ cho
phase mới.

**Reason:** Người dùng chủ động chọn "làm mới context hoàn toàn" thay vì
archive — coi đây là điểm khởi đầu mới, không kế thừa giả định từ phase
trước.

**Status:** Closed — đã thực hiện. Lịch sử đầy đủ của phase trước (bao
gồm bảng xếp hạng E1-I4/E6-E9, mọi finding/idea/decision) vẫn nằm trong
git history ở các commit trước commit reset này, có thể tra lại bằng
`git log`/`git show` nếu cần đối chiếu, nhưng không được coi là ràng buộc
cho hướng đi mới.

**Revisit Condition:** N/A.

_(các quyết định tiếp theo của phase mới thêm bên dưới)_
