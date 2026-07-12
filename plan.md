# Evasion Attack OD 

## Approach
 Apply RaPA len OD

## Dataset: COCO
**Surrogate:** Faster R-CNN ResNet-50-FPN
## Target Models (Cross-family)

| Group | Name | Backbone | Paradigm |
|---|---|---|---|
| **A — In-family (ResNet-50)** | fcos_r50 | ResNet-50 | anchor-free |
| **A — In-family (ResNet-50)** | deformable_detr | ResNet-50 | transformer |
| **B — Near-family (non-ResNet CNN)** | yolov3_d53 | Darknet-53 | anchor |
| **B — Near-family (non-ResNet CNN)** | yolox_l | CSPNet | anchor-free |
| **C — Cross-family (Swin ViT)** | mask_rcnn_swin_t | Swin-T | two-stage |
| **C — Cross-family (Swin ViT)** | dino_swin_l | Swin-L | full-transformer |
**Expected transfer difficulty:** A (easy) → B (medium) → C (hard)
---

## Metrics đánh giá:
ASR / Object Disappearance Rate => Trả lời câu hỏi: bao nhiêu phần trăm object không detect được khi bị evasion attack?
mAP drop => Trả lời câu hỏi: mAP giảm bao nhiêu khi bị evasion attack?

---

## Attack Scope (đã chốt)
- **Loss & masking đều chỉ nhắm vào backbone feature** của surrogate (không đụng neck/FPN, RPN, ROI head).
- Lý do: đánh cross-family, neck/head khác biệt kiến trúc rất lớn giữa các họ detector (FPN vs PAFPN, RPN+ROI vs anchor-free dense head vs DETR/DINO decoder) → attack/mask ở đó dễ overfit vào surrogate, không transfer. Backbone là phần "chia sẻ" giữa các detector nên là điểm đòn bẩy tốt nhất cho transferability (theo lập luận OSFD).
- Hệ quả: ResNet-50 backbone không có Linear layer (RaPA gốc chỉ mask Linear+Norm) → phạm vi mask khả dụng ban đầu chỉ còn **BatchNorm affine (weight+bias)**, mở rộng sang Conv (channel-wise) là bước cải tiến sau (I2), không phải mặc định.

## Danh sách thực nghiệm

### 0. Pilot Study — xác định (p, S) cho RaPA-mask
Quét lưới trước khi khóa cấu hình cho E4/E5:
- **Cố định:** surrogate Faster R-CNN R50-FPN, loss NRDM (k=1), mask toàn bộ BN affine backbone (~53 layer), không RRB, MI-FGSM.
- **p (drop probability):** {0.02, 0.05, 0.1, 0.15, 0.2}
- **S (n-mask, số forward-backward mask khác nhau/iteration, gradient trung bình):** {1, 3, 5}
- Tổng: 5 × 3 = 15 cấu hình + 1 baseline no-mask = 16 run.
- Đánh giá trên: white-box (chính Faster R-CNN, để loại cấu hình phá vỡ surrogate) + 2 model đại diện black-box (FCOS-R50 — Group A dễ, Mask R-CNN Swin-T — Group C khó).
- Ảnh: 50 ảnh (subset đầu của `dev_300`), T=50–100 iteration (không cần hội tụ hoàn toàn, chỉ cần thấy xu hướng).
- Output: chọn (p\*, S\*) cho mAP-drop black-box trung bình cao nhất, với điều kiện white-box mAP-drop không tệ hơn no-mask baseline quá nhiều (surrogate vẫn phải "sống").

### 1. Main Experiments (giữ tới E5)

| # | Tên | Mask (backbone) | Loss | RRB | Vai trò |
|---|---|---|---|---|---|
| E1 | OSFD Baseline | Không | OSFD (k=3, suppress+amplify) | Không | Reproduce SOTA gốc, dataset đổi sang COCO |
| E2 | OSFD + RRB | Không | OSFD (k=3) | Có | Reproduce full method gốc, dataset COCO |
| E3 | NRDM control | Không | NRDM (k=1) | Không | Control — cô lập đóng góp của loss OSFD (k=3) so với loss trần |
| E4 | RaPA-OD Baseline | BN affine, p=p\*, S=S\* (từ Pilot) | NRDM (k=1) | Không | Cô lập đóng góp thuần của cơ chế RaPA-mask, tách khỏi loss engineering của OSFD |
| E5 (I1) | RaPA-OD + loss OSFD | BN affine, p=p\*, S=S\* | OSFD (k=3) | Không | Kiểm tra RaPA-mask + loss OSFD có cộng dồn (synergy) hay không |

Ma trận đối chiếu (đọc theo hàng/cột để quy kết đóng góp):

| | Không mask | RaPA-mask |
|---|---|---|
| Loss NRDM (k=1) | E3 | E4 |
| Loss OSFD (k=3) | E1 | E5 |

E2 (OSFD+RRB) là cột mốc SOTA riêng, so sánh cuối cùng khi có bước I4 (+RRB) sau này (chưa nằm trong phạm vi tới E5).

Các bước tiếp theo (I2 mở rộng mask sang Conv, I3 stage-wise ablation, I4 +RRB) — **tạm hoãn**, chỉ chạy nếu E5 cho tín hiệu GO và cần cải thiện thêm.

## Dataset dùng cho thực nghiệm
Dùng chung manifest COCO val2017 do `setup_env.sh` sinh sẵn (seed=42), chạy theo 3 giai đoạn tăng dần quy mô:

| Giai đoạn | Số ảnh | Nguồn | Dùng cho |
|---|---|---|---|
| **GO/NOGO screening** | **50** | 50 ảnh đầu của `dev_300` | Pilot Study (p×S sweep) + lượt chạy đầu của E1–E5. Mục tiêu: tín hiệu định hướng nhanh, chưa phải số báo cáo. |
| **Confirm** | 300 | `dev_300` (đầy đủ) | Chạy lại các cấu hình đã GO ở bước trên để có số ổn định hơn trước khi viết kết luận. |
| **Final / held-out** | 100 | `val_100` (không đụng tới cho đến khi config đã khóa) | Chốt số báo cáo cuối cùng, đúng 1 lần, tránh tune ngầm lên tập báo cáo. |

## Việc cần quyết định / TODO
- ~~mmdet version cho model zoo~~ — **đã xử lý**: `setup_env.sh` dùng mmdet 3.3.0 + mmcv 2.1.0, có sẵn Deformable-DETR và DINO-Swin-L trong model zoo, checkpoint list khớp đủ 6 target + surrogate.
- **Cần scaffold trước khi chạy `setup_env.sh` trọn vẹn:** file gọi `pip install -e .` và `scripts/check_env.py`, `scripts/run_attack.py`, `scripts/run_sweep.py` nhưng repo chưa có `setup.py`/`pyproject.toml` lẫn thư mục `scripts/`.