---
name: research
description: Research-state protocol cho project evasion-attack-od — bắt buộc đọc /research trước khi reasoning, chống đề xuất trùng, và checklist ép cập nhật cuối phiên. Dùng khi thảo luận hướng nghiên cứu mới, review experiment cũ, hoặc quyết định GO/NOGO cho một idea.
---

# Research Context Protocol

Repo này dùng thư mục `research/` làm **nguồn sự thật** cho trạng thái
nghiên cứu — không phải conversation history. Lý do: session bị compact,
bị đóng, hoặc mở lại sau vài tuần, nhưng `research/` luôn phản ánh đúng
trạng thái tại thời điểm commit gần nhất.

## Authority — file nào thắng khi có mâu thuẫn

```
FINDINGS.md    > RESEARCH.md   (kết luận khoa học)
IDEAS.md       > RESEARCH.md   (trạng thái từng ý tưởng)
DECISIONS.md   > RESEARCH.md   (quyết định researcher, có thể khác kết luận khoa học)
EXPERIMENTS.md > mọi nơi khác  (dữ liệu thô — nếu số ở đâu đó khác EXPERIMENTS.md, EXPERIMENTS.md đúng)
```

`RESEARCH.md` chỉ là bản roll-up để đọc nhanh — nếu nó mâu thuẫn với 4 file
kia, nó đang stale, phải sync lại (xem checklist bên dưới), không được coi
nó là đúng.

## Bước bắt buộc TRƯỚC khi reasoning về hướng nghiên cứu

1. Đọc `research/RESEARCH.md` (bức tranh tổng, bảng xếp hạng hiện tại).
2. Đọc `research/FINDINGS.md` (điều đã chứng minh — không được contradict).
3. Đọc `research/DECISIONS.md` (đừng đề xuất lại thứ đã bị gác vì lý do
   ngoài khoa học).
4. Đọc `research/NEXT.md` (câu hỏi đang mở, đã có next-step cụ thể chưa).
5. Search `research/IDEAS.md` (bảng index ở đầu file) để kiểm tra idea
   tương tự đã REJECTED/NO-GO/DEFERRED chưa.
6. Nếu cần đánh giá lại một experiment cụ thể, mở đúng block đó trong
   `research/EXPERIMENTS.md` (dùng bảng index ở đầu file để tìm ID, đừng
   đọc toàn bộ file nếu không cần).

Không đề xuất idea mới trước khi hoàn thành 5 bước trên.

## Duplicate Idea Check — trước khi đề xuất bất kỳ idea nào

Với mỗi idea mới, bắt buộc trả lời rõ ràng (bằng văn bản, không chỉ trong
đầu):

1. So sánh với mọi dòng trong `IDEAS.md` có status REJECTED / NO-GO /
   DEFERRED.
2. Trả lời: "Idea này khác gì với những gì đã thử?"
3. Nếu không chỉ ra được khác biệt có ý nghĩa → không đề xuất.
4. Nếu đây là revisit của một idea cũ nhưng có bằng chứng mới (ví dụ:
   paper mới đọc, kết quả experiment khác vừa cho tín hiệu trái ngược) →
   phải ghi rõ trong `IDEAS.md`:
   ```
   Revisit of: IDEA-XXX
   New evidence: ...
   Why reconsider: ...
   ```

## Session-end checklist — BẮT BUỘC trước khi kết thúc bất kỳ phiên nào
chạy experiment mới hoặc thay đổi kết luận

Sau khi chạy 1 experiment mới hoặc đổi trạng thái 1 idea, xác nhận đã cập
nhật ĐỦ các file sau — nếu chưa, liệt kê rõ file nào còn thiếu và lý do
(không được im lặng bỏ qua):

- [ ] `EXPERIMENTS.md`: thêm dòng vào bảng index + viết block chi tiết
      (Question / Input / Output / Decision) cho experiment mới.
- [ ] `FINDINGS.md`: CHỈ thêm finding nếu có bằng chứng trực tiếp từ
      experiment (trỏ tới `EXP-xxx`). Không ghi giải thích cơ chế
      (mechanism) chưa được một experiment riêng kiểm chứng — điều đó
      thuộc `NEXT.md` dưới dạng hypothesis.
- [ ] `IDEAS.md`: cập nhật status của idea liên quan (bảng index +
      block chi tiết). Nếu experiment mới bác bỏ hoặc xác nhận một idea
      đang "CANDIDATE"/"chưa đóng", phải đổi status.
- [ ] `DECISIONS.md`: nếu experiment mới đóng một "Revisit Condition" đang
      mở, cập nhật `Status` của decision đó.
- [ ] `NEXT.md`: đóng câu hỏi (Q-xxx) nếu đã trả lời được, hoặc cập nhật
      Active Hypotheses nếu tín hiệu mới thay đổi ưu tiên. Thêm câu hỏi
      mới nếu experiment mở ra hướng chưa có trong hàng đợi.
- [ ] `RESEARCH.md`: sync lại bảng xếp hạng / phần "Frontier hiện tại" nếu
      thứ hạng hoặc kết luận tổng đổi. Đây là bước cuối cùng — chỉ làm sau
      khi 5 file trên đã nhất quán với nhau.

**Tự kiểm tra nhanh (grep, không cần công cụ ngoài):** mọi `IDEA-xxx` được
nhắc trong `EXPERIMENTS.md`/`DECISIONS.md`/`NEXT.md` phải tồn tại trong
index của `IDEAS.md`, và ngược lại mọi `EXP-xxx` được nhắc trong
`IDEAS.md`/`FINDINGS.md` phải tồn tại trong index của `EXPERIMENTS.md`.
Nếu tìm thấy ID mồ côi (orphan) — báo cho người dùng, đừng tự ý xoá.

## Quy tắc viết FINDINGS.md (nhắc lại vì hay bị vi phạm)

Không được ghi dạng: *"Resize tạo gradient bất biến kiến trúc."*
nếu chưa có experiment đo trực tiếp điều đó. Chỉ được ghi quan sát + evidence
cụ thể, ví dụ: *"E6 (DIM, chỉ resize) thắng E1 nhưng thua E2 — F-006."*
Cơ chế/giải thích "tại sao" luôn là hypothesis cho tới khi có experiment
kiểm chứng, và thuộc về `NEXT.md`, không phải `FINDINGS.md`.

## Nguyên tắc cuối cùng

**Repository research files are the source of truth. Do not rely on
conversation memory when repository evidence is available.** Nếu một
memory/context nào đó (kể cả từ hệ thống memory cá nhân của AI) nói khác
với `research/`, ưu tiên `research/` — nó luôn mới hơn và versioned cùng
code.
