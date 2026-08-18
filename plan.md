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

**Kết quả — đã chốt: p\* = 0.05, S\* = 3**
- Quy trình thực tế: quét rút gọn (20 ảnh, T=30, rates={0.01,0.02,0.05,0.1,0.2}, masks={1,3,5}, 16 cấu hình) để định hướng nhanh, sau đó xác nhận lại đúng (p\*,S\*) chọn được ở quy mô gốc (50 ảnh, T=100) — xem `results/pilot_sweep.json` (bản rút gọn) và `results/pilot_confirm_p005_s3.json` (bản xác nhận).
- Đường cong theo p có dạng đỉnh (peak) tại p=0.05, không đơn điệu: p=0.01 quá thấp (ít lợi ích transfer), p≥0.1 quá cao (phá chất lượng gradient, cả white-box lẫn black-box đều yếu đi rõ, ASR white-box rơi từ ~0.9 xuống ~0.3-0.4 ở p=0.2).
- Số liệu xác nhận full-scale (50 ảnh, T=100, p=0.05, S=3): white-box ASR=0.869, mAP-drop=0.464; FCOS-R50 (Group A) mAP-drop=0.335; Mask R-CNN Swin-T (Group C) mAP-drop=0.165. Mạnh hơn hẳn baseline no-mask (mAP-drop=0 ở cả 3 model, đúng sanity check).
- S=3 được chọn thay vì S=5 vì hiệu quả tương đương/tốt hơn với chi phí tính toán thấp hơn ~1.6x.

### 1. Main Experiments (giữ tới E5)

| # | Tên | Mask (backbone) | Loss | RRB | Vai trò |
|---|---|---|---|---|---|
| E1 | OSFD Baseline | Không | OSFD (k=3, suppress+amplify) | Không | Reproduce SOTA gốc, dataset đổi sang COCO |
| E2 | OSFD + RRB | Không | OSFD (k=3) | Có | Reproduce full method gốc, dataset COCO |
| E3 | NRDM control | Không | NRDM (k=1) | Không | Control — cô lập đóng góp của loss OSFD (k=3) so với loss trần |
| E4 | RaPA-OD Baseline | BN affine, p=p\*, S=S\* (từ Pilot) | NRDM (k=1) | Không | Cô lập đóng góp thuần của cơ chế RaPA-mask, tách khỏi loss engineering của OSFD |
| E5 (I1) | RaPA-OD + loss OSFD | BN affine, p=p\*, S=S\* | OSFD (k=3) | Không | Kiểm tra RaPA-mask + loss OSFD có cộng dồn (synergy) hay không |
| **I4** | **RaPA-OD + OSFD + RRB** | BN affine, p=p\*, S=S\* | OSFD (k=3) | **Có** | = E5 + RRB. Kiểm tra RaPA-mask có cộng dồn thêm lên trên cấu hình mạnh nhất hiện tại (E2) hay không — RRB tác động ở input-level, RaPA ở parameter-level, kỳ vọng không bị interference âm như đã thấy giữa RaPA và loss OSFD (E5<E2) |

Ma trận đối chiếu (đọc theo hàng/cột để quy kết đóng góp):

| | Không mask | RaPA-mask |
|---|---|---|
| Loss NRDM (k=1) | E3 | E4 |
| Loss OSFD (k=3), không RRB | E1 | E5 |
| Loss OSFD (k=3), có RRB | E2 | I4 |

Điều kiện mở khóa I4 (đã đạt): E5 phải thắng E1 (GO signal) — xem kết quả E5 bên dưới. Lý do ưu tiên I4 trước I2 (mở rộng mask sang Conv)/I3 (stage-wise ablation): E2 (OSFD+RRB) đang là baseline mạnh nhất, I4 trả lời trực tiếp câu hỏi "RaPA có giúp vượt qua baseline mạnh nhất hay không" — thông tin giá trị cao nhất cho quyết định tiếp tục đầu tư vào RaPA hay dừng lại ở kết luận "RRB là đủ".

**⚠️ Bug đã biết — E3 bị vô hiệu hóa hoàn toàn (chưa fix, tạm hoãn):** `AttackConfig(k=1.0, mask_enabled=False)` (E3 và các dòng "baseline no-mask" trong pilot sweep) cho gradient = 0 chính xác ngay từ iteration đầu tiên trong `attack.py:57-75` — vì `delta` khởi tạo = 0 nên `feats_adv == feats_clean` ở bước đầu, và với k=1: `dL/df_adv = 2*(1-k)*f_clean = 0`. Delta đứng yên suốt T iteration, attack không sinh ra perturbation nào (ASR=0 tuyệt đối, không phải "yếu"). k=3 (E1/E2, `2*(1-3)*f_clean≠0`) và mọi config có mask (E4/E5, `feats_adv` tính qua mạng đã bị DropConnect nên khác `feats_clean` ngay từ đầu) **không bị lỗi này**. Cần fix (ví dụ: khởi tạo `delta` bằng nhiễu ngẫu nhiên nhỏ trong epsilon-ball thay vì đúng 0) trước khi chạy lại E3 và trước khi dùng số "baseline no-mask" trong `pilot_sweep.json` cho bất kỳ so sánh định lượng nào.

**Kết quả E1 (OSFD Baseline) — GO/NOGO, 50 ảnh, T=100:**

| Group | Model | ASR | mAP-drop |
|---|---|---|---|
| White-box | faster_rcnn_r50_fpn | 0.678 | 0.423 |
| A | fcos_r50 | 0.281 | 0.161 |
| A | deformable_detr | 0.492 | 0.327 |
| B | yolov3_d53 | 0.122 | 0.054 |
| B | yolox_l | 0.077 | 0.045 |
| C | mask_rcnn_swin_t | 0.120 | 0.059 |
| C | dino_swin_l | 0.073 | 0.023 |

Đúng thứ tự transfer difficulty A (avg drop ~0.244) > B (~0.050) ≈ C (~0.041) như giả thuyết. Deformable-DETR (A) transfer bất ngờ tốt hơn FCOS cùng nhóm. File: `results/E1_osfd_baseline_go.json`.

**Kết quả E2 (OSFD + RRB) — GO/NOGO, 50 ảnh, T=100:**

| Group | Model | ASR | mAP-drop |
|---|---|---|---|
| White-box | faster_rcnn_r50_fpn | 0.910 | 0.486 |
| A | fcos_r50 | 0.797 | 0.413 |
| A | deformable_detr | 0.875 | 0.505 |
| B | yolov3_d53 | 0.507 | 0.291 |
| B | yolox_l | 0.448 | 0.324 |
| C | mask_rcnn_swin_t | 0.429 | 0.343 |
| C | dino_swin_l | 0.215 | 0.133 |

RRB tăng transfer rất mạnh so với E1 (gấp 2.5–14x tùy model, mạnh nhất ở Group B/C nơi E1 gần như vô hiệu). Vẫn giữ đúng thứ tự A > B ≈ C nhưng khoảng cách thu hẹp. File: `results/E2_osfd_rrb_go.json`.

**Kết quả E4 (RaPA-OD Baseline: NRDM k=1 + RaPA mask p=0.05,S=3) — GO/NOGO, 50 ảnh, T=100:**

| Group | Model | ASR | mAP-drop |
|---|---|---|---|
| White-box | faster_rcnn_r50_fpn | 0.857 | 0.460 |
| A | fcos_r50 | 0.625 | 0.327 |
| A | deformable_detr | 0.804 | 0.470 |
| B | yolov3_d53 | 0.221 | 0.113 |
| B | yolox_l | 0.169 | 0.110 |
| C | mask_rcnn_swin_t | 0.207 | 0.139 |
| C | dino_swin_l | 0.109 | 0.032 |

**E4 thắng E1 tuyệt đối trên cả 7/7 model** dù dùng loss yếu hơn (NRDM k=1 vs OSFD k=3) — bằng chứng RaPA-mask đóng góp transferability thật, độc lập với loss engineering. Trung bình black-box (A+B+C): E1=0.111 → E4=0.199 (+79%). Vẫn thua E2 (avg 0.335), đặc biệt ở Group B/C — RRB vẫn là đòn bẩy mạnh nhất trong 3 kỹ thuật đã test tới nay. Xếp hạng nhất quán ở mọi model: E2 > E4 > E1. File: `results/E4_rapa_od_baseline_go.json`.

**Kết quả E5 (RaPA-OD + loss OSFD: OSFD k=3 + RaPA mask p=0.05,S=3) — GO/NOGO, 50 ảnh, T=100:**

| Group | Model | ASR | mAP-drop |
|---|---|---|---|
| White-box | faster_rcnn_r50_fpn | 0.837 | 0.448 |
| A | fcos_r50 | 0.536 | 0.305 |
| A | deformable_detr | 0.738 | 0.466 |
| B | yolov3_d53 | 0.197 | 0.081 |
| B | yolox_l | 0.145 | 0.107 |
| C | mask_rcnn_swin_t | 0.199 | 0.147 |
| C | dino_swin_l | 0.095 | 0.046 |

**Cổng quyết định GO/NOGO (E5 vs E1): GO — E5 thắng E1 tuyệt đối trên cả 7/7 model** (avg black-box: E1=0.111 → E5=0.192, +73%). RaPA-mask cộng dồn được với loss OSFD mạnh nhất, không bị triệt tiêu.

**Nhưng bất ngờ: E5 không vượt được E4 — thua nhẹ ở 5/7 model** (White-box, FCOS, DDETR, YOLOv3, YOLOX), **chỉ thắng E4 ở đúng Group C** (Swin-T: 0.147 vs 0.139; DINO: 0.046 vs 0.032). Diễn giải: RaPA-mask có vẻ tương tác âm nhẹ với phần amplify của loss OSFD (k=3) trên target dễ/trung bình, nhưng phát huy đúng ở phần khó nhất (Group C, cross-family thật sự) — nơi đề tài quan tâm nhất. Đây là finding cụ thể hơn "RaPA cộng dồn tuyến tính": **RaPA đặc biệt hữu ích cho hard cross-family transfer**, không nhất thiết cải thiện đều mọi nhóm.

**So với E2 (OSFD+RRB) — vẫn là cấu hình mạnh nhất tính đến nay:** E5 (avg black-box 0.192) vẫn thua rõ E2 (avg 0.335) trên toàn bộ 7/7 model, không riêng Group B/C. RRB vẫn là đòn bẩy transfer mạnh nhất trong các kỹ thuật đã test — RaPA-mask (dù kết hợp loss mạnh nhất) chưa đủ thay thế lợi ích của augmentation-level RRB. Xếp hạng tổng thể nhất quán: **E2 > E4 ≈ E5 > E1**. File: `results/E5_rapa_od_osfd_loss_go.json`.

**Kết quả I4 (RaPA-OD + OSFD + RRB = E5 + RRB) — GO/NOGO, 50 ảnh, T=100:**

| Group | Model | ASR | mAP-drop | So với E2 |
|---|---|---|---|---|
| White-box | faster_rcnn_r50_fpn | 0.906 | 0.484 | −0.002 |
| A | fcos_r50 | 0.833 | 0.413 | = |
| A | deformable_detr | 0.846 | 0.494 | −0.011 |
| B | yolov3_d53 | 0.587 | 0.306 | **+0.015** ✅ |
| B | yolox_l | 0.383 | 0.264 | −0.060 |
| C | mask_rcnn_swin_t | 0.455 | 0.309 | −0.034 |
| C | dino_swin_l | 0.200 | 0.133 | = |

**I4 không vượt được E2** (avg black-box: E2=0.335 → I4=0.320, **−4.5%**) — chỉ thắng rõ 1/6 target (YOLOv3), hòa 2/6, thua 3/6. Giả thuyết "cộng hưởng" (RRB input-level + RaPA parameter-level độc lập nhau, kỳ vọng cộng dồn) **không được xác nhận** — thêm RaPA vào cấu hình đã có RRB không sinh thêm giá trị, nằm trong biên nhiễu ở quy mô 50 ảnh (khác với case E5 vs E4: không phải interference âm rõ rệt, mà là bão hòa/không thêm được gì).

**Kết luận tổng hợp GO/NOGO (E1→E2→E3[bug]→E4→E5→I4):**
- RaPA-mask có đóng góp transferability thật khi *chưa có* RRB: E4>E1 (+79% avg), E5>E1 (+73% avg) — nhất quán trên mọi model.
- RaPA-mask *bão hòa* khi đã có RRB: I4 ≈ E2 (chênh −4.5%, trong biên nhiễu) — không cộng dồn thêm.
- RRB vẫn là kỹ thuật đơn lẻ mạnh nhất đã test (E2 đứng đầu bảng xếp hạng toàn bộ 6 config).
- Xếp hạng tổng thể: **E2 ≈ I4 > E4 ≈ E5 > E1**.
- Hàm ý cho hướng tiếp theo: nếu muốn RaPA tạo giá trị *trên* baseline mạnh nhất, cần hướng khác hẳn việc chỉ kết hợp thêm (I2 mở rộng mask sang Conv, I3 stage-wise ablation nhiều khả năng cũng chỉ dao động quanh biên hiện tại) — có thể cần thiết kế lại loss/mechanism cho RaPA-OD thay vì tối ưu tiếp trong không gian cấu hình đã thử. Quyết định hướng tiếp theo đang chờ thảo luận, chưa chốt.

File: `results/I4_rapa_od_rrb_go.json`.

### 2. Augmentation Comparison — DIM / SSA / SRS (new-plan.txt Sec 5.2)

Nguồn: `new-plan.txt` (kế hoạch riêng, "Spectral Augmentation for Transferable Adversarial Attacks on OD") đề xuất so sánh 5 augmentation input-level: none, RRB, DIM, SSA, SRS (SRS = phương pháp mới, spectral band attenuation + adaptive resize + rotation nhẹ). Tài liệu đó không nhắc tới RaPA-mask và giả định quy mô GPU lớn hơn nhiều (VOC2012, hàng nghìn ảnh, 300 bước PGD). Quyết định áp dụng ở đây (khác với `new-plan.txt`):

- **Dataset: vẫn dùng COCO** (val2017, manifest `dev_300`/`val_100` có sẵn), không chuyển sang VOC2012.
- DIM/SSA/SRS được thêm như một lựa chọn mới trên trục `augmentation` sẵn có trong `AttackConfig` (thay cho `use_rrb: bool` cũ), tái dùng toàn bộ hạ tầng attack.py/RaPA-mask hiện có — không xây pipeline riêng. Nhờ vậy tổ hợp RaPA-mask × augmentation mới vẫn khả thi ở bước sau nếu cần.
- Hoãn Phase 2 của `new-plan.txt` (alternating CNN/ViT surrogate với RT-DETR-L) — RT-DETR-L chưa có trong model zoo, và chưa có tín hiệu Phase 1 để biện minh cho việc đầu tư thêm.
- Quy mô chạy: giữ đúng phương pháp GO/NOGO screening đã dùng cho E1–I4 (50 ảnh, T=100, manifest `dev_300`), **không** nhảy thẳng lên quy mô Phase 1 gốc của `new-plan.txt` (1000+ ảnh, 300 bước, 3 mức epsilon).

Bốn experiment mới, cùng surrogate/loss/eps/alpha/iterations như E1/E2 (chỉ đổi trục augmentation) để so sánh trực tiếp, không cần chạy lại E1/E2:

| # | Tên | Augmentation | Loss | Mask | Vai trò |
|---|---|---|---|---|---|
| E6 | OSFD + DIM | dim (Xie et al. CVPR'19, resize [0.9,1.1] + pad, p=0.7) | OSFD (k=3) | Không | So sánh với RRB bằng một augmentation cổ điển khác (baseline cho DIM) |
| E7 | OSFD + SSA | ssa (Long et al. ECCV'22, FFT × random spectral scale U(1-ρ,1+ρ), ρ=0.5, N=20 copies/iter qua `num_masks`) | OSFD (k=3) | Không | Baseline frequency-domain đã chứng minh hiệu quả cho classification transfer, chưa từng áp dụng cho OD |
| E8 | OSFD + SRS (đề xuất) | srs (band attenuation ngẫu nhiên trong FFT + adaptive resize + rotation ±5°, tái dùng `adaptive_random_resizing`/`random_axis_rotation` từ `rrb.py`) | OSFD (k=3) | Không | Phương pháp đề xuất của `new-plan.txt` — câu hỏi chính: SRS có vượt RRB (đặc biệt ở Group C) hay không |
| E9 | OSFD + RRB + Spectral (Option A) | rrb_spectral (spectral band attenuation → RRB nguyên bản không đổi: rotate θ=7 + adaptive resize + blur, tái dùng `apply_rrb` từ `rrb.py` nguyên vẹn) | OSFD (k=3) | Không | Bổ sung spectral lên trên RRB thay vì thay thế — kiểm tra giả thuyết "E8 thua E2 ở Group B/C vì SRS bỏ blur + thu hẹp θ 7→5, không phải vì bản thân spectral vô dụng" |

Lưu ý chi phí: E7 (SSA) dùng `num_masks=20` (N spectral copies theo đúng spec SSA gốc) nên tốn ~20x forward/backward so với E6/E8/E9/E1/E2 ở cùng số ảnh/iteration — đây là đặc tính thuật toán, không phải lỗi cấu hình. E9 không có chi phí phụ trội này (vẫn 1 forward/backward mỗi iteration, chỉ thêm 1 bước FFT/iFFT rẻ trước RRB).

Chạy: `python scripts/run_attack.py --experiment E9_osfd_rrb_spectral --out results/E9_osfd_rrb_spectral_go.json` (tương tự cho E6/E7/E8, đổi tên experiment).

**Kết quả (50 ảnh, T=100) — E6, E8 đã chạy; E7 bị loại giữa chừng (quá chậm, xem ghi chú dưới); E9 chưa chạy:**

| Group | Model | E1 (none) | E2 (RRB) | E6 (DIM) | E8 (SRS) |
|---|---|---|---|---|---|
| White-box | faster_rcnn_r50_fpn | 0.423 | 0.486 | 0.475 | 0.498 |
| A | fcos_r50 | 0.161 | 0.413 | 0.359 | 0.431 |
| A | deformable_detr | 0.327 | 0.505 | 0.481 | 0.499 |
| B | yolov3_d53 | 0.054 | 0.291 | 0.121 | 0.240 |
| B | yolox_l | 0.045 | 0.324 | 0.193 | 0.312 |
| C | mask_rcnn_swin_t | 0.059 | 0.343 | 0.191 | 0.329 |
| C | dino_swin_l | 0.023 | 0.133 | 0.053 | 0.123 |

Trung bình black-box (6 target, mAP-drop): E1=0.111 → E6=0.233 → **E8=0.322** → E2=0.335.

**E6 (DIM):** thắng E1 tuyệt đối 6/6 (DIM là augmentation thật, có đóng góp) nhưng thua E2 tuyệt đối 6/6. Đáng chú ý: DIM bắt được ~82% lợi ích của RRB (so với E1) ở Group A (cùng backbone) nhưng chỉ ~41% ở Group B và C — gợi ý phần rotate+blur riêng của RRB (không chỉ resize) mới là thứ tạo giá trị cho cross-architecture transfer.

**E8 (SRS):** thắng E1/E6 tuyệt đối (7/7 kể cả white-box so với E6) nhưng **không đạt success criteria "Minimum" của `new-plan.txt` mục 8**: trung bình black-box thấp hơn RRB ~3.9% (yêu cầu phải cao hơn ≥5%), và giảm hơn 3% so với RRB đúng ở Group B (−10.4%) và Group C (−5.0%) — hai nhóm mà tiêu chí yêu cầu không được giảm. Ngược với giả thuyết gốc, SRS mạnh nhất (ngang hoặc nhỉnh hơn RRB) ở Group A dễ, nhưng yếu nhất so với RRB đúng ở Group C khó — cùng pattern như E6, chỉ nhẹ hơn.

**E7 (SSA) — bị loại:** `num_masks=20` khiến tổng forward-backward gấp ~20x E6/E8 ở cùng 50 ảnh/T=100 (ước tính ~100.000 lượt chỉ riêng phần tấn công), vượt thời gian cho phép nên phải dừng giữa chừng. Hệ quả: thiếu một ô đối chứng quan trọng — success criteria yêu cầu SRS phải thắng rõ SSA để chứng minh không chỉ là "port SSA trần", hiện chưa kiểm chứng được. Hướng xử lý (chưa quyết định): giảm N (vd 5) chạy lại để có số liệu tham chiếu, hoặc chấp nhận thiếu ô này giống cách đã tạm gác bug E3.

**Xếp hạng tạm thời:** E2 ≳ E8 > E6 > E1. E9 (đang chờ chạy) là phép kiểm tra trực tiếp giả thuyết rút ra từ E6/E8: giữ nguyên RRB (rotate+blur) và chỉ cộng thêm spectral lên trên, thay vì để SRS tự thay thế phần rotate/blur của RRB bằng phiên bản yếu hơn (θ=5, không blur).

## Dataset dùng cho thực nghiệm
Dùng chung manifest COCO val2017 do `setup_env.sh` sinh sẵn (seed=42), chạy theo 3 giai đoạn tăng dần quy mô:

| Giai đoạn | Số ảnh | Nguồn | Dùng cho |
|---|---|---|---|
| **GO/NOGO screening** | **50** | 50 ảnh đầu của `dev_300` | Pilot Study (p×S sweep) + lượt chạy đầu của E1–E5. Mục tiêu: tín hiệu định hướng nhanh, chưa phải số báo cáo. |
| **Confirm** | 300 | `dev_300` (đầy đủ) | Chạy lại các cấu hình đã GO ở bước trên để có số ổn định hơn trước khi viết kết luận. |
| **Final / held-out** | 100 | `val_100` (không đụng tới cho đến khi config đã khóa) | Chốt số báo cáo cuối cùng, đúng 1 lần, tránh tune ngầm lên tập báo cáo. |

## Việc cần quyết định / TODO
- ~~mmdet version cho model zoo~~ — **đã xử lý**: `setup_env.sh` dùng mmdet 3.3.0 + mmcv 2.1.0, có sẵn Deformable-DETR và DINO-Swin-L trong model zoo, checkpoint list khớp đủ 6 target + surrogate.
- ~~Scaffold `pyproject.toml`, `scripts/check_env.py`, `scripts/run_attack.py`, `scripts/run_sweep.py`~~ — **đã xử lý**: venv tại `/workspace/evasion-venv`, `check_env.py` pass đầy đủ (mmdet/mmcv/mmengine + toàn bộ checkpoint + manifest).
- ~~Pilot Study (p, S)~~ — **đã xử lý**: chốt p\*=0.05, S\*=3 (xem mục "0. Pilot Study" ở trên).
- **Tiếp theo:** chạy E1 (OSFD Baseline) ở giai đoạn GO/NOGO screening (50 ảnh).

Nhận xét tổng quan (6 config đã chạy, GO/NOGO scale 50 ảnh):

RaPA-mask có đóng góp transferability thật và đo được trên object detection (không phải chỉ classification như paper gốc) — E4>E1, E5>E1 nhất quán trên mọi model. Nhưng nó không thắng được RRB, và khi đã có RRB thì RaPA gần như không cộng thêm gì (I4≈E2, chênh trong biên nhiễu). Điểm sáng cụ thể nhất: RaPA giúp rõ nhất ở Group C (cross-family khó nhất, Swin-T/DINO) — đây có lẽ là câu chuyện đáng kể nhất để viết thành đóng góp, không phải "RaPA tốt chung chung".

Đề xuất, theo thứ tự ưu tiên:

Nghi ngờ lớn nhất về tính chặt chẽ hiện tại: (p*=0.05, S*=3) được chọn từ Pilot Study không có RRB trong vòng lặp sweep. Rất có thể khi đã có RRB (vốn tự nó đã tạo diversity ở input-level), operating point tối ưu cho RaPA sẽ khác (ví dụ p thấp hơn, vì không cần mask mạnh nữa) — nên trước khi kết luận chắc "RaPA bão hòa với RRB", nên chạy 1 sweep nhỏ (p,S) với RRB bật sẵn để loại khả năng I4 đang dùng sai hyperparameter chứ không phải RaPA thực sự vô dụng khi kết hợp RRB.
Fix bug E3 (đã chẩn đoán, rẻ) — cần để ma trận đối chiếu E1/E3/E4/E5 hoàn chỉnh, hiện đang thiếu 1 ô kiểm chứng quan trọng (NRDM đứng một mình, không mask).
Việc mở rộng mask sang Conv (I2) hay stage-wise ablation (I3) — nên hoãn tiếp cho tới khi (1) trả lời xong, vì nếu nguyên nhân I4 không thắng là do sai hyperparameter chứ không phải giới hạn cơ chế, thì I2/I3 cũng sẽ dẫm lại vết đó.
Trade-off: hướng (1) rẻ, tái dùng hạ tầng sẵn có, trả lời trực tiếp câu hỏi "RaPA có thật sự bão hòa hay đang bị đo sai điểm vận hành" — nên làm trước. Hướng "thiết kế loss mới cho RaPA-OD" (đã bàn ở lượt trước) chỉ nên cân nhắc sau khi (1) và (2) loại trừ được nguyên nhân do thiết lập thí nghiệm.

## 3. idea.txt exploration — object-context relational hypothesis (Phase 1 / 1b / 2)

Hướng riêng, độc lập với RaPA/RRB ở trên: thay vì suppress+amplify absolute feature (OSFD) hay mask parameter (RaPA), attack trực tiếp **quan hệ tương đối object-vs-vicinity** trong backbone feature space (`idea.txt`). Toàn bộ mã nguồn: `src/evasion_od/regions.py` (mask O/V, các score/prototype), `src/evasion_od/losses.py` (các loss `rel`/`osfd_rel_hybrid`/`spatial`), wiring trong `attack.py`/`config.py` qua `loss_type`.

**Phase 1 — quan hệ `S_O > S_V` có tồn tại nhất quán qua kiến trúc không? (`results/phase0_diagnostic.json`, 200 ảnh × 7 model × 3 giá trị r)**
- Metric `normalized energy` (M2, chuẩn hoá per-channel trước khi so vùng) vượt hẳn `magnitude` thô (M1): pooled `P(S_O>S_V)` 0.85 vs 0.81 trên 25 cặp model×stage, và M1 suy biến gần như hoàn toàn trên backbone Swin (`var_contrast≈0.0000` mọi stage — nghi do LayerNorm khiến norm mỗi vị trí gần constant).
- 1 exception thật, không phải nhiễu: `yolox_l` stage cuối (CSPNet) đảo dấu nhất quán (`P(S_O>S_V)≈0.10-0.15`, `mean_contrast` âm, variance vừa phải — không phải scatter ngẫu nhiên).
- `r` (bề rộng vành vicinity, quét 0.5/1.0/2.0) gần như không ảnh hưởng kết luận.
- **Kết luận: GO có điều kiện** — dùng M2, loại/không dựa vào stage cuối CSPNet.

**Phase 2 — attack trực tiếp quan hệ đó có transfer tốt hơn OSFD không? (`results/P2_*_go.json`, 50 ảnh, T=100, đủ 7 model, so với `E1_osfd_baseline_go.json`)**

| # | Loss | avg mAP-drop black-box | Ghi chú |
|---|---|---|---|
| E1 (baseline) | OSFD (k=3) | **0.111** | — |
| REL_S3/S23/ALL | `C_clean·C_adv` (bounded contrast, mean-pooled/stage) | 0.030–0.038 | Sanity-check xác nhận optimization đúng hướng (loss hội tụ, `C_adv→-1`) nhưng **bão hòa sau ~20/100 iteration**, lãng phí phần lớn budget |
| HYBRID_L10/30/100 | `L_OSFD − λ·relational_diff` (unbounded D, cộng thêm vào OSFD) | 0.036–0.047 | λ hiệu chỉnh theo tỷ lệ đo thực tế (`\|L_OSFD\|/\|L_relD\|≈50-85x` — λ∈{0.1,0.5,1.0} ban đầu đề xuất sẽ vô nghĩa). Kết quả đơn điệu giảm theo λ (càng thêm REL càng tệ) → nghi ngờ gradient hai loss **xung đột hướng**, không chỉ lệch scale |
| SPATIAL_S3/S23/ALL | dense per-pixel `mean_{p∈O}[cos(F_adv(p),μ_O)−cos(F_adv(p),μ_V)]` | 0.032–0.045 | Không bão hòa (loss vẫn giảm tới iteration 99, khác hẳn REL) nhưng vẫn không thắng E1 |

**Phase 1b — trước khi tin SPATIAL, validate quantity mới trên ảnh sạch (`results/P1b_prototype_diagnostic.json`, 200 ảnh × 7 model × 3 r):** `P(margin>0)=1.000` (hoặc 0.995) trên **toàn bộ** model×stage, kể cả `yolox_l` stage cuối — nơi Phase 1 từng đảo dấu — giờ cho tín hiệu **mạnh nhất** bảng (`mean_margin=0.349`). Đây là bằng chứng sạch hơn hẳn Phase 1 (không có exception nào), nhưng attack tương ứng (SPATIAL_*) vẫn NOGO — xem trên.

**Kết luận tổng hợp Phase 1/1b/2 (9 config relational-family đã chạy, tất cả NOGO so với E1):** quan hệ object-context tồn tại thật và **đo được nhất quán qua kiến trúc** (đặc biệt rõ với công thức prototype-cosine ở Phase 1b), và **có thể đảo được bằng attack** (verify trực tiếp qua hội tụ loss ở cả REL và SPATIAL) — nhưng đảo nó **không đủ sức phá decision boundary thật của detector** theo cách OSFD's dense per-pixel MSE làm được, dù thử 3 công thức hoá độc lập (bounded contrast, unbounded diff/hybrid, dense cosine-to-prototype). Đóng nhánh "relational loss thay OSFD" tại đây; E2 (OSFD+RRB) vẫn là baseline mạnh nhất toàn dự án.

**Hướng tiếp theo: Phase G0 — RRB Gradient Mechanism Analysis (mục 4 bên dưới).**

## 4. Phase G0 — RRB Gradient Mechanism Analysis

Câu hỏi đổi từ "loss/augmentation nào tốt hơn?" sang "RRB đang giúp OSFD transfer bằng cơ chế gì?" — vì E1→E2 (thêm RRB) là bước nhảy transferability lớn nhất quan sát được trong toàn dự án, lớn hơn hẳn mọi biến thể loss ở mục 3 hay augmentation ở mục 2. Mã nguồn: `src/evasion_od/gradient_diagnostics.py` (consensus map, pairwise cosine, combine rule), wiring trong `attack.py` (trục `grad_combine`, hook đo checkpoint ngay trên trajectory đang chạy — không tách pass riêng) và `runner.py` (`generate_adversarial_with_diagnostics`, `evaluate_on_model_per_image`); script chạy: `scripts/run_g0_diagnostic.py`.

**Thiết kế:**
- Mỗi iteration, K=5 RRB view độc lập của cùng ảnh (tái dùng cơ chế `num_masks` sẵn có, `augmentation="rrb"`, không RaPA mask), tính gradient loss OSFD (k=3) từng view g_1..g_5, cùng coordinate system với `delta` — rotation/resize/"blur" trong `rrb.py` đều differentiable native (torchvision `rotate`, `F.interpolate`/`F.pad`, cộng nhiễu trực tiếp), autograd tự chain-rule đúng về `delta`, đã verify so khớp với `reference-repo/OSFD/attack/base/RRB.py` (kể cả phát hiện phụ: "gaussian_blur" ở cả bản gốc lẫn port này thực chất là cộng nhiễu Gaussian i.i.d., không phải blur kernel thật — không phải bug, chỉ là tên gây hiểu lầm kế thừa từ code gốc).
- **Phần 1 — correlational, đo trên trajectory thật của `RRB_K5_MEAN` (T=100):** tại checkpoint t∈{0,25,50,75,99}, log mean/median pairwise cosine, consensus map C_p=|mean_k sign(g_k,p)|, E[C_p], P(C_p>0.6), P(C_p>0.8), variance chuẩn hoá V=E_p[Var_k(g_k,p)]/(E_p[ḡ_p²]+ε).
- **Phần 2 — causal, cùng compute budget (K=5 forward/backward/iteration ở cả 4 variant):**
  - `RRB_K5_MEAN` — baseline, ḡ = mean_k g_k (cách kết hợp gradient mặc định trước Phase G0, tương đương cơ chế `num_masks` cũ)
  - `RRB_K5_CONS` — g_cons = C^γ ⊙ ḡ (γ=1)
  - `RRB_K5_DISAGREE` — g_dis = (1-C)^γ ⊙ ḡ, control ngược lại
  - `RRB_K5_CONS_SHUFFLE` — như CONS nhưng C bị shuffle theo spatial (H,W), giữ nguyên channel + phân phối giá trị — control tách "vị trí consensus xảy ra ở đâu" khỏi "trọng số ảnh hưởng magnitude gradient nói chung"
- **Correlation:** A_i^traj (primary) = trung bình E[C_p] qua 5 checkpoint; A_i^final (secondary) = E[C_p] tại t=99 (cùng cặp cho P(C>0.8): high_traj/high_final). Correlate (Pearson + Spearman) với per-image ASR trên từng target + pooled theo Group A/B/C — dùng ASR (không phải mAP) vì mAP trên 1 ảnh đơn lẻ quá nhiễu để làm biến correlate có ý nghĩa.

**Tiêu chí GO/NOGO:** GO nếu CONS thắng MEAN rõ rệt (≥10% relative avg black-box hoặc nhất quán ≥4/6 target), đặc biệt ở Group B/C. Strong GO nếu gain tập trung ở Group B/C (YOLO/Swin), không chỉ white-box/Group A. NOGO nếu CONS ≤ MEAN, hoặc chỉ tăng ở white-box/Group A. CONS > CONS_SHUFFLE ≈ MEAN là bằng chứng thêm rằng vị trí consensus (không chỉ độ lớn) mới là thứ tạo giá trị.

**Trạng thái: đóng nhánh, NO-GO — thay full run (~6h) bằng 2-stage cheap screen (~15 phút tổng) vì mục tiêu chỉ là GO/NO-GO ban đầu.**

**Stage 1 (20 ảnh, T=30, K=3, diagnostic-only, checkpoints 0/15/29):** phát hiện quan trọng khi đọc kết quả — tiêu chí gốc "E[C]≈0 → NO-GO" giả định sai baseline: với K=3, sign-agreement dưới null hypothesis (hoàn toàn không có consensus) đã bằng **E[C]_null=0.5** (không phải 0, tính từ phân phối Binomial(3,0.5) của số view đồng thuận), và P(C=1)_null=0.25. Đọc đúng phải qua **excess-over-null**: E[C] excess tăng từ +0.002 (t=0) lên +0.024 (t=29); `mean_pairwise_cosine` (không có null-offset, vì 2 vector ngẫu nhiên độc lập trong không gian ~triệu chiều có cosine kỳ vọng ≈0, độ lệch chuẩn ~1/√d) tăng từ 0.067→0.138 — hàng trăm lần độ lệch chuẩn null, gần chắc chắn là tín hiệu thật. Kết luận Stage 1: consensus gần null lúc đầu trajectory, xây dần khi attack hội tụ — đủ tín hiệu biên để đáng bỏ thêm 1 pilot causal ngắn.

**Stage 2 (30 ảnh, T=50, K=3, γ=1, `RRB_K5_MEAN` vs `RRB_K5_CONS`, matched RRB views giữa 2 run — xem `AttackConfig.deterministic_augmentation`, reseed RNG theo (seed,image_id,iteration,k_idx) để cô lập grad_combine là biến duy nhất khác nhau, verify bằng unit test cùng-seed-cho-output-giống-hệt):**

| Target | MEAN map_drop | CONS map_drop | Δ relative |
|---|---|---|---|
| White-box (FRCNN) | 0.441 | 0.440 | −0.2% |
| FCOS (A) | 0.394 | 0.383 | −2.8% |
| YOLOX (B) | 0.271 | 0.245 | −9.6% |
| DINO-Swin-L (C) | 0.163 | 0.178 | +9.2% |
| **Avg BB** | **0.276** | **0.269** | **−2.5%** |

**NO-GO theo cả 3 tiêu chí đã đặt:** primary (AvgBB_CONS>AvgBB_MEAN) fail; consistency (≥2/3 target A/B/C) chỉ đạt 1/3 (DINO); cross-family fail vì YOLOX giảm thật (−9.6%, không phải nhiễu). Diễn giải khớp với lo ngại đã nêu trước khi chạy: consensus chỉ rõ dần về sau trajectory (Stage 1), nên linear weighting `C^1⊙ḡ` áp toàn bộ trajectory ngay từ t≈0 (lúc C gần null, chưa phân biệt được vùng consensus thật) làm suy yếu gradient khá đồng đều — thiệt hại rõ ở target cần tín hiệu mạnh sớm (FCOS/YOLOX), trong khi DINO-Swin (khó nhất, có lẽ ít nhạy early-iteration hơn) lại nhích lên nhẹ. Phụ chú: ASR của DINO gần như không đổi dù map_drop tăng — gain đến từ suy giảm confidence trên object vẫn bị detect, không phải object biến mất thêm.

**Đóng nhánh Gradient Consensus tại đây** (không chạy DISAGREE/CONS_SHUFFLE/full K=5×4-variant — không còn giá trị marginal khi causal test chính đã NO-GO). Code (`gradient_diagnostics.py`, `grad_combine` axis, `--variants`/`--diagnostic-only`/matched-views trong `run_g0_diagnostic.py`) giữ nguyên trong repo cho khả năng tái dùng sau (vd nếu có formulation "trajectory-aware" mới muốn test), nhưng không đầu tư thêm compute vào hướng này trừ khi có lý do mới. File: `results/G0_stage1_diagnostic.json`, `results/G0_stage2_cheap_causal.json`.

## 5. Phase B1 — RRB Component Ablation

G0 đóng lại với kết luận "RRB tốt nhưng cơ chế gradient-consensus không giải thích được vì sao" — bước tiếp theo lùi lại một tầng câu hỏi khác: **component nào của RRB (rotation, resize, noise) thực sự tạo ra transferability**, thay vì tiếp tục dò cơ chế gradient. Cùng surrogate/loss (OSFD k=3)/eps/alpha/iterations như E1/E2, chỉ thay augmentation. Mã nguồn: `rrb.py` (`apply_resize_only`, `apply_rotation_resize`, `additive_gaussian_noise` — đổi tên từ `gaussian_blur` cho đúng bản chất, chỉ là cộng nhiễu i.i.d. per-pixel, không phải blur kernel), wiring trong `attack.py` (`rrb_rot`/`rrb_resize`/`rrb_noise`/`rrb_rot_resize`), 5 experiment mới trong `config.py`, script: `scripts/run_b1_ablation.py`.

**5 config, pilot 30 ảnh/T=50, evaluate FRCNN (white-box) + FCOS/YOLOX/DINO-Swin-L (đại diện A/B/C):**
- `OSFD_ROT` — chỉ rotation
- `OSFD_RESIZE` — chỉ adaptive resize
- `OSFD_NOISE` — chỉ additive Gaussian noise
- `OSFD_ROT_RESIZE` — rotation + resize, không noise
- `OSFD_RRB_FULL` — cả 3 (= config E2, chạy lại ở scale pilot để so sánh cùng-scale công bằng với 4 arm kia thay vì dùng thẳng số E2 ở scale 50/T100 khác)

**Tiêu chí đọc:** Strong GO nếu 1 component/tổ hợp giải thích ≥70-80% gain E1→full RRB trên avg BB và tốt ở B/C. GO nếu có ranking rõ (vd RESIZE>ROT>NOISE) và component đứng đầu thắng E1 rõ ở ≥2/3 target đại diện. Interesting GO nếu từng component riêng lẻ yếu nhưng tổ hợp cho synergy rõ rệt (Drop_{A+B} > Drop_A + gain kỳ vọng riêng của B) → hướng nghiên cứu chuyển sang "augmentation interaction". NO-GO nếu mọi component/tổ hợp cho kết quả gần nhau hoặc bất ổn, không có pattern cross-family rõ — khi đó B1 chỉ xác nhận "augmentation diversity nói chung tốt" chứ không chỉ ra được đòn bẩy cụ thể.

**Kết quả pilot (30 ảnh, T=50, evaluate FRCNN+FCOS+YOLOX+DINO-Swin-L):**

| Variant | White-box | FCOS (A) | YOLOX (B) | DINO-Swin (C) | Avg BB |
|---|---|---|---|---|---|
| OSFD_ROT | 0.418 | 0.320 | 0.148 | 0.099 | 0.189 |
| OSFD_RESIZE | 0.442 | 0.375 | 0.217 | 0.096 | 0.230 |
| OSFD_NOISE | 0.413 | 0.264 | 0.132 | 0.082 | 0.159 |
| OSFD_ROT_RESIZE | 0.442 | 0.356 | 0.235 | 0.120 | 0.237 |
| OSFD_RRB_FULL | 0.438 | 0.374 | 0.212 | 0.142 | 0.243 |

Chạy thêm no-augmentation baseline (`E1_osfd_baseline` config, cùng scale 30/T50, cần thiết vì `OSFD_RRB_FULL` ở scale pilot chỉ đạt avg_bb=0.243 so với E2 gốc=0.335 -- thiếu hụt ~27% thuần túy do T=50 thay vì T=100, nên so trực tiếp với số E1/E2 gốc sẽ tính sai %-gain-explained): NoAug avg_bb=0.093 (white-box=0.392), file `results/B1_noaug_baseline_pilot.json`.

**R% = (Drop_variant − Drop_NoAug) / (Drop_FULL − Drop_NoAug), tính trong-scale (sạch, không lệch do T):**

| Variant | R% aggregate | R% FCOS (A) | R% YOLOX (B) | R% DINO-Swin (C) |
|---|---|---|---|---|
| OSFD_ROT | 64.4% | 72.3% | 52.6% | 64.5% |
| OSFD_NOISE | 44.5% | 43.6% | 40.7% | 50.4% |
| OSFD_RESIZE | **91.2%** | 100.5% | 103.7% | 62.0% |
| OSFD_ROT_RESIZE | **96.2%** | 90.8% | **117.0%** | 81.8% |
| OSFD_RRB_FULL | 100% | 100% | 100% | 100% |

**Kết luận: Strong GO** (RESIZE/ROT_RESIZE vượt hẳn ngưỡng 70-80% aggregate). Nhưng phát hiện chính không phải "1 component thắng đều" mà **augmentation cần thiết tỷ lệ thuận với độ khó kiến trúc của target**:
- Group A/B (cùng/gần họ CNN với surrogate): RESIZE một mình đã ≥100% gain của FULL; ROT_RESIZE thậm chí **vượt** FULL trên YOLOX (117%) -- thêm noise vào đang hơi hại ở target này.
- Group C (DINO-Swin, cross-family khó nhất): không component/tổ hợp nào thiếu noise chạm được 100% (ROT_RESIZE chỉ 81.8%) -- đây là target duy nhất cần đủ cả 3 thành phần.

File: `results/B1_rrb_component_ablation.json`.

**Literature check (web scan, Scholar Gateway 0 kết quả cho truy vấn OD-transfer-augmentation 2021-2026 nên dựa chủ yếu vào web scan):** OSFD gốc đã dùng augmentation khai thác spatial consistency/limited equivariance của detector feature; AugTrans (ScienceDirect 2026) đã đi xa hơn với dynamic object-centric rotation, multi-box-aware resizing, composite noise, EOT để cải thiện transferability (AAAI'24 OSFD gốc: https://ojs.aaai.org/index.php/AAAI/article/view/27920; AugTrans: https://www.sciencedirect.com/org/science/article/pii/S1546221826003498). Vì vậy "thêm resize/noise" hay "schedule rotation theo iteration" **không còn novel** -- cả hai baseline literature này đã coi augmentation composition là 1 pipeline/schedule thiết kế trước (kể cả AugTrans's dynamic scheduling: transform/range vẫn do attack pipeline định nghĩa trước, không phải do surrogate tự suy ra). Gap "no direct prior found in this scan" (không claim first-ever): **surrogate-side tự quyết định augmentation composition dựa trên difficulty ước lượng của cross-architecture generalization** -- và bắt buộc phải suy luận thuần từ surrogate-side, vì trong black-box setting không biết kiến trúc target thật để làm kiểu "target=Swin → dùng noise".

## 6. Phase B2 — Augmentation Transfer Signature

Câu hỏi: `Can we predict which augmentation composition produces model-general adversarial directions without accessing the target?` -- tìm signal Q(a) surrogate-side dự đoán được ranking transfer mà B1 đo được (RESIZE>ROT>NOISE tổng thể, nhưng NOISE có vai trò riêng khi kết hợp cho Group C), để sau này dùng `p(a|x) ~ f(Q(a))` thay cho pipeline RRB cố định. Không phải làm lại consensus kiểu G0 (G0 đo K view CÙNG 1 loại augmentation để combine thành 1 update; B2 đo K view của TỪNG loại augmentation khác nhau để characterize riêng từng loại, không combine, không có attack trajectory).

**Thiết kế (diagnostic thuần, không MI-FGSM loop, đo tại delta=0 trên ảnh sạch -- rẻ):** với mỗi ảnh, K=5 draw độc lập mỗi augmentation kind a∈{rot, resize, noise} (1 draw duy nhất cho "none" vì deterministic), mỗi draw là 1 forward+backward (loss OSFD k=3, giống B1) tại delta=0. 3 property surrogate-side:
- **Gradient alignment:** `mean_k cos(g_{a,k}, g_none)` -- augmented gradient còn trỏ về hướng attack "trần" (không augment) hay đã lệch hẳn.
- **Feature distortion stability:** pairwise cosine giữa các distortion vector `d_{a,k} = feats_adv_{a,k}(stage cuối) − feats_clean(stage cuối)` qua K draw (tái dùng `gradient_diagnostics.pairwise_cosine_stats`, chỉ đổi input từ gradient sang feature-distortion) -- augmentation a có tạo distortion ổn định/lặp lại hay ngẫu nhiên mỗi lần khác nhau.
- **Loss sensitivity:** mean/std của `loss_{a,k} − loss_none` qua K draw -- a dịch chuyển loss landscape mạnh/yếu, ổn định/bất định thế nào.

Script: `scripts/run_b2_diagnostic.py`. Lưu ý phạm vi: (1) đo tại delta=0 (đặc trưng nội tại của từng augmentation trên ảnh sạch, không theo trajectory), (2) characterize từng kind RIÊNG LẺ so với "none", không đo tương tác cặp/pairwise giữa các augmentation -- nên phù hợp giải thích ranking đơn lẻ (RESIZE>ROT>NOISE) hơn là hiện tượng "noise chỉ có giá trị marginal khi cộng vào ROT_RESIZE, đặc biệt ở Group C" mà B1 phát hiện; nếu B2 không giải thích được phần combination đó, có thể cần B3 đo joint/pairwise sau. (3) chỉ 3 augmentation kind để rank → Spearman/Pearson có n=3, power rất thấp, đọc theo ranking-match định tính là chính, không theo p-value.

**GO:** signal surrogate-side rank được augmentation gần đúng ranking transfer B1 đo, đặc biệt giải thích được vì sao RESIZE mạnh tổng quát và vì sao NOISE chỉ hữu ích khi composition khó hơn (Group C). **NO-GO:** surrogate-side metric không dự đoán được transfer ordering -- khi đó không theo hướng "adaptive selection", chuyển sang nghiên cứu trực tiếp resize/scale invariance (finding mạnh và ổn định nhất của B1).

**Kết quả (30 ảnh, K=5, đo tại delta=0):**

| Kind | gradient_alignment | feature_stability | loss_sens_mean | loss_sens_std |
|---|---|---|---|---|
| ROT | 0.145 | 0.377 | 1.886 | 0.516 |
| RESIZE | 0.230 | 0.459 | 1.593 | 0.365 |
| NOISE | 0.173 | 0.743 | 0.337 | 0.003 |

So với ranking transfer B1 đo (RESIZE > ROT > NOISE), không metric nào khớp đủ 3/3 pairwise: `gradient_alignment` và `loss_sensitivity_mean` chỉ đạt 2/3 (đúng 1 đầu bảng, sai thứ tự giữa 2 cái còn lại); `feature_stability`/`loss_sensitivity_std` chỉ 1/3 — **gần như đảo ngược hoàn toàn** (NOISE có feature-distortion ổn định nhất qua các draw nhưng lại transfer yếu nhất trong B1).

**Kết luận: NO-GO cho hướng "adaptive selection qua 1 surrogate-side scalar Q(a)".** Nhưng giá trị chính của B2 không phải "không tìm được metric" mà là bác bỏ thêm một giả thuyết: **stability/consistency ⇏ transferability** — NOISE minh chứng rõ nhất (feature stability cao nhất, transfer đơn lẻ thấp nhất trong B1). Khớp với G0 (gradient-consensus weighting cũng không cải thiện avg BB, thậm chí hơi hại). Hai diagnostic độc lập (G0 trên gradient, B2 trên feature-distortion) cùng chỉ về một hướng: cái tạo transferability không phải là "ổn định/nhất quán qua các draw", ngược lại có thể là diversity/exploration mới quan trọng.

File: `results/B2_augmentation_transfer_signature.json`.

**Literature check bổ sung:** OSFD gốc đã khai thác spatial consistency/limited equivariance qua augmentation, resize là 1 thành phần quan trọng; AugTrans (2026) đi xa hơn với multi-box-aware/content-adaptive resizing và explicitly claim scale-invariant feature dùng chung qua kiến trúc — nên "resize giúp transfer" hay "object-aware resize" tự thân **không đủ novel**. Câu hỏi sâu hơn B1 đặt ra mà chưa thấy prior giải quyết trực tiếp: **vì sao scale transformation giải thích gần hết gain transfer cho CNN target nhưng không đủ cho CNN→Swin?**

## 7. Phase S0 — Scale Transfer Mechanism

Thu hẹp câu hỏi về đúng 1 biến: **scale**. Không invent augmentation mới, không random-resize pipeline -- sweep một **fixed global scale factor** xác định để đo transfer response curve T_g(s) theo từng group kiến trúc, tách bạch khỏi mọi randomization khác (không giống `adaptive_random_resizing` của RRB, vốn scale theo kích thước GT box và random hoá cả biên độ lẫn offset crop mỗi iteration).

**Thiết kế:** surrogate Faster R-CNN R50 + OSFD (k=3) như B1/B2. Augmentation mới `fixed_scale` (`rrb.py:apply_fixed_scale`) -- resize toàn ảnh theo hệ số `s` cố định rồi letterbox-pad (s≤1) hoặc center-crop (s>1) về đúng kích thước gốc, không có rotation/noise đi kèm (cô lập biến scale, cùng triết lý ablation của B1). Sweep `s ∈ {0.6, 0.8, 1.0, 1.2, 1.4}`, pilot 30 ảnh/T=50, evaluate FRCNN (white-box) + FCOS/YOLOX/DINO-Swin-L (đại diện A/B/C, cùng bộ target đã dùng ở B1/B2). Script: `scripts/run_s0_scale_sweep.py`.

**Hypothesis:** cross-family transfer (Group C) hưởng lợi từ perturbation hiệu quả trên **phổ scale rộng** hơn, không chỉ 1 phân phối random-resize hẹp quanh 1.0 như RRB mặc định (rho=0.8, s_max=1.1 -- RRB's range gốc chỉ resize *lớn hơn*, không bao giờ nhỏ hơn 1.0, nên grid `{0.6,0.8,1.0,1.2,1.4}` cố tình phủ rộng hơn cả 2 hướng so với range RRB gốc).

**GO:** scale sweep cho response curve có cấu trúc, khác biệt rõ giữa A/B/C -- đặc biệt Group C đòi hỏi range/diversity scale rộng hơn A/B (vd A/B đã plateau sớm quanh s gần 1, C tiếp tục tăng ở scale cực đoan hơn). **NO-GO:** mọi target cho cùng 1 response curve, hoặc random-resize (RRB gốc) chỉ tốt nhờ generic EOT averaging chứ không phải cơ chế scale cụ thể -- khi đó "scale-space coverage mechanism" không đủ mạnh để xây contribution riêng.

**Rerun v1 (30 ảnh, T=50) bị confound:** `apply_fixed_scale` bản đầu center-crop khi s>1, làm mất nội dung/GT box gần biên -- không tách được hiệu ứng scale khỏi hiệu ứng mất nội dung. Đã fix: redefine bằng **occupancy = min(s, 1/s)**, luôn shrink+pad, không bao giờ crop (chứng minh toán học: "zoom in" content-preserving trong canvas cố định là bất khả thi trừ phi là no-op, nên s>1 giờ tương đương shrink theo 1/s thay vì phóng to+crop). Verify: `pad_fraction = 1-occupancy ≥ 0` ở mọi scale trong grid → 0 GT box bị crop, đảm bảo bằng construction (`results/S0_scale_transfer_sweep_v2.json`).

**Kết quả sau fix (30 ảnh, T=50), sắp theo pad_fraction tăng dần (méo ít→nhiều) để lộ đúng biến thật:**

| Scale (pad_fraction) | White-box | FCOS (A) | YOLOX (B) | DINO-Swin (C) |
|---|---|---|---|---|
| 1.0 (0.000) | 0.392 | 0.180 | 0.066 | 0.029 |
| 1.2 (0.167) | 0.202 | 0.136 | 0.090 | 0.045 |
| 0.8 (0.200) | 0.221 | 0.120 | 0.061 | **0.061** |
| 1.4 (0.286) | 0.121 | 0.105 | 0.092 | 0.042 |
| 0.6 (0.400) | 0.074 | 0.075 | 0.057 | 0.051 |

**Kết luận: GO.** White-box/FCOS (Group A, gần surrogate): **monotonic** -- méo càng nhiều càng tệ, đỉnh rõ tại pad=0 (ảnh gốc), khớp trực giác đơn giản. YOLOX (B) và DINO-Swin (C): **không monotonic, và không peak tại ảnh gốc** -- DINO tệ nhất chính tại pad=0 (0.029, thấp nhất bảng), tốt nhất tại pad=0.2 (0.061, gấp đôi); YOLOX cũng có 2 giá trị cao nhất ở vùng có padding (1.2, 1.4), không phải ở pad=0. Đây là response curve khác hệ thống thật giữa group, không đơn thuần yếu hơn mà khác **hướng**: CNN/near-family thích ảnh gốc, target xa (B/C) bị hại bởi nó.

**Liên hệ với RRB gốc:** phân tích lại `adaptive_random_resizing` cho thấy nó cũng luôn là phép "enlarge rồi pad rồi resize xuống" -- net effect là content occupancy nằm trong khoảng hẹp `[1/s_max, 1.0] ≈ [0.91, 1.0]` (s_max=1.1 mặc định), chưa bao giờ chạm tới vùng occupancy≈0.8 mà S0 vừa tìm thấy có lợi cho DINO. Đây là gợi ý mechanism cụ thể cho phase tiếp theo: RRB hiện tại quá hẹp/quá gần 1.0 để khai thác vùng này.

File: `results/S0_scale_transfer_sweep_v2.json` (bản đã fix; `results/S0_scale_transfer_sweep.json` giữ lại làm bản có confound, không dùng để kết luận).

**Literature check bổ sung:** *The Scissors Effect* (arXiv 2606.22516, 2026) cho thấy resize-based input diversity có thể giúp hoặc hại transfer tùy regime, liên hệ trực tiếp gradient geometry/alignment -- gần về mechanism với S0 nhưng chưa áp dụng cho OD/cross-architecture. AugTrans (2026) đã dùng content-adaptive resizing với range gồm cả shrink lẫn enlarge. Resize-invariant attack cũng đã có trong classification (PubMed 38402809). Vì vậy "resize giúp transfer" hay "thêm shrink-range" tự thân không đủ novel -- contribution phải nằm ở **OD cross-family directional asymmetry (shrink vs enlarge) và cách khai thác nó**, không phải bản thân phép resize-padding.

## 8. Phase S1 — Bidirectional / Shrink-aware RRB

Attack pilot nhỏ, hypothesis rõ, chưa phải method cuối: S0 chỉ đo transfer response curve qua 1 phép scale CỐ ĐỊNH mỗi lần chạy (không phải augmentation ngẫu nhiên mỗi iteration như RRB thật) -- S1 kiểm tra liệu finding đó có causal khi đưa vào đúng dạng augmentation ngẫu nhiên/mỗi-iteration của 1 attack thật hay không.

**Thiết kế:** giữ nguyên OSFD (k=3) + rotation/noise của RRB gốc, chỉ thay bước resize. Hàm mới `rrb.py:random_occupancy_resize` (sample occupancy ngẫu nhiên trong `[occ_low, occ_high]` mỗi lần gọi, shrink+pad content-preserving như S0, offset pad ngẫu nhiên giống RRB gốc) + `apply_rrb_occupancy` (rotation + `random_occupancy_resize` + noise, thay cho `adaptive_random_resizing`). Wiring qua augmentation kind mới `"rrb_occupancy"` + field `occ_low`/`occ_high` trong `AttackConfig`.

3 variant, pilot 30 ảnh/T=50, evaluate FRCNN+FCOS/YOLOX/DINO-Swin (tái dùng số `OSFD_RRB_FULL` đã có sẵn từ B1 cho `RRB_ORIG`, không chạy lại vì cùng scale/manifest):
- `RRB_ORIG` = `OSFD_RRB_FULL` (B1, đã có số) -- range gốc, occupancy thực chất chỉ ≈[0.91,1.0]
- `RRB_SHRINK`: `occ_low=0.7, occ_high=0.9` -- quanh sweet-spot 0.8 mà DINO ưa thích
- `RRB_BIDIR`: `occ_low=0.7, occ_high=1.0` -- phủ cả vùng gần gốc lẫn shrink vừa phải, tránh cực đoan (S0 cho thấy occupancy=0.6/pad=0.4 tệ, không đưa vào range)

Script: `scripts/run_s1_shrink_rrb_pilot.py`.

**Kỳ vọng nếu S0 causal:** FCOS ≈ không đổi nhiều (ORIG≈BIDIR); YOLOX BIDIR≥ORIG; **DINO-Swin: SHRINK/BIDIR > ORIG rõ rệt** -- gain tăng dần theo khoảng cách kiến trúc với surrogate.

**Strong GO:** DINO tăng rõ, avg BB không giảm, YOLOX cũng tăng. **GO:** DINO tăng ≥10% relative, avg BB không giảm quá ~3%. **NO-GO:** DINO không tăng, hoặc gain DINO phải đánh đổi bằng collapse FCOS/YOLOX khiến avg BB giảm rõ.

**Kết quả (30 ảnh, T=50, `RRB_ORIG` tái dùng số `OSFD_RRB_FULL` từ B1):**

| Variant | White-box | FCOS (A) | YOLOX (B) | DINO-Swin (C) | Avg BB |
|---|---|---|---|---|---|
| RRB_ORIG | 0.438 | 0.374 | 0.212 | 0.142 | 0.243 |
| RRB_SHRINK | 0.325 | 0.268 | 0.159 | 0.119 | 0.182 |
| RRB_BIDIR | 0.370 | 0.325 | 0.181 | 0.135 | 0.214 |

**Kết luận: NO-GO, bất ngờ hơn dự đoán.** Không chỉ FCOS/YOLOX giảm (đã chấp nhận được theo kỳ vọng) mà **DINO-Swin -- target trọng tâm -- cũng giảm ở cả 2 variant** (BIDIR −4.9%, SHRINK −16.2%), ngược hẳn kỳ vọng "SHRINK/BIDIR > ORIG rõ". Avg BB giảm 12-25%, vượt xa ngưỡng chấp nhận ~3%. White-box cũng giảm mạnh (−25.8% với SHRINK) -- attack yếu đi phổ quát, không phải kiểu đánh đổi CNN-lấy-Swin.

**Giả thuyết nguyên nhân:** S0 đo 1 giá trị scale **cố định xuyên suốt** T=50 iteration; S1 (giống RRB thật) **resample occupancy ngẫu nhiên mỗi iteration**. Hai setup khác nhau về structure, không chỉ về giá trị occupancy trung tâm -- "coherence xuyên suốt trajectory" có thể quan trọng hơn bản thân giá trị scale.

File: `results/S1_shrink_rrb_pilot.json`.

## 9. Phase S2 — Trajectory Consistency

Test cuối để reconcile S0 (fixed suốt trajectory) vs S1 (random mỗi iteration) trước khi đóng hẳn nhánh scale -- cô lập đúng 1 biến, resize-only (không rotation/noise, giữ đúng isolation của S0) để tránh nhầm lẫn với confound khác:

- `FIXED_SHRINK`: occupancy ~ Uniform(0.7,0.9) sample **1 lần/ảnh**, giữ nguyên suốt T=50 (augmentation kind mới `fixed_shrink_per_image`, sample trong `run_attack()` trước vòng lặp, verify bằng cách đếm số lần gọi `random.uniform` -- đúng 1 lần/5-iteration test).
- `RANDOM_SHRINK`: cùng range, resample mỗi iteration (`random_shrink` kind, gọi thẳng `random_occupancy_resize` -- verify đúng 5 lần/5-iteration test).
- `FIXED_0.8`: tái dùng thẳng số scale=0.8 đã có từ S0 (cùng scale/manifest, không chạy lại).
- `RRB_ORIG`: tái dùng số `OSFD_RRB_FULL` từ B1.

Script: `scripts/run_s2_trajectory_consistency.py`.

**GO** nếu FIXED_SHRINK thắng rõ RANDOM_SHRINK, đặc biệt trên DINO-Swin, và pattern gần lại với S0 -- research story chuyển từ "scale range" sang "trajectory-consistent augmentation". **NO-GO** nếu fixed cũng không cứu được DINO hoặc avg BB vẫn thấp hơn RRB_ORIG rõ -- đóng hẳn nhánh scale, không tune thêm.

**Kết quả (30 ảnh, T=50):**

| Variant | White-box | FCOS (A) | YOLOX (B) | DINO-Swin (C) | Avg BB |
|---|---|---|---|---|---|
| RRB_ORIG | 0.438 | 0.374 | 0.212 | 0.142 | 0.243 |
| FIXED_0.8 | 0.221 | 0.120 | 0.061 | 0.061 | 0.081 |
| FIXED_SHRINK | 0.185 | 0.109 | 0.068 | 0.034 | 0.070 |
| RANDOM_SHRINK | 0.350 | 0.271 | 0.180 | 0.105 | 0.185 |

**Kết luận: NO-GO, ngược hoàn toàn hypothesis.** RANDOM_SHRINK thắng FIXED_SHRINK trên **mọi** target, không phải suýt soát: white-box gần gấp đôi (0.350 vs 0.185), DINO-Swin gấp ~3 lần (0.105 vs 0.034) -- đúng ngược lại "coherence xuyên suốt trajectory quan trọng hơn giá trị scale". Resample mỗi iteration tốt hơn hẳn giữ cố định, kể cả trên target trọng tâm nhất. **Đóng nhánh scale-mechanism tại đây theo đúng quyết định đã chốt trước khi chạy.**

File: `results/S2_trajectory_consistency.json`.

**Synthesis xuyên suốt G0→B1→B2→S0→S1→S2:** 3 dòng bằng chứng độc lập cùng chỉ về 1 hướng:
- G0: ép gradient theo "đồng thuận" (consensus) giữa các RRB view → hại transfer.
- B2: augmentation có feature-distortion **ổn định nhất** qua các draw (NOISE) lại transfer **yếu nhất** trong B1.
- S2: giữ 1 view scale **cố định** suốt trajectory → tệ hơn hẳn resample ngẫu nhiên mỗi bước, trên mọi target.

Cả 3 độc lập bác bỏ cùng 1 giả thuyết ngầm ("tìm đúng hướng/giá trị rồi giữ nó ổn định sẽ tốt hơn") theo 3 cách đo hoàn toàn khác nhau (gradient-space, feature-space, augmentation-schedule). Kết luận chung: **cái tạo transferability của RRB không phải là một "điểm vận hành đúng" nào đó, mà là chính sự đa dạng/ngẫu nhiên giữa các iteration** (đúng tinh thần EOT nguyên bản của RRB). Mọi nỗ lực "cải thiện" RRB bằng cách làm nó nhất quán/ổn định/có chủ đích hơn (G0's consensus-weighting, S2's fixed-scale) đều làm attack yếu đi. 6 phase mechanism investigation (G0/B1/B2/S0/S1/S2) đều không tìm ra cấu hình nào vượt được `E2` (OSFD+RRB gốc) -- RRB mặc định vẫn là baseline mạnh nhất toàn dự án.

## 10. Phase S3 — History-Aware RRB (candidate)

Câu hỏi hẹp hơn S0-S2: trong đúng safe range của RRB_ORIG (theta≤7°, occupancy∈[0.91,1.0] -- không phải range rộng [0.7,1.0] của S1 đã biết là tệ hơn trên mọi target), liệu **điều kiện hoá mỗi draw theo lịch sử h=3 draw gần nhất** -- tránh lặp lại transform quá giống (`RRB_ANTI_REPEAT`) hoặc chủ động tối đa hoá khoảng cách (`RRB_OVER_DIVERSE`, positive control) -- có tốt hơn i.i.d. resampling thuần (`RRB_IID`) hay không. Trước khi chạy GPU, `scripts/sanity_check_history_aware_rrb.py` (thuần thống kê, không cần model/ảnh) ước lượng effect size khả dĩ: với h=3 và tau=percentile-10 của phân phối null (~0.44 trong không gian chuẩn hoá 4 chiều [theta, occ, pad_top_frac, pad_left_frac]), ~26% draw i.i.d. đã "quá gần" 3 draw liền trước -- đủ lớn để `RRB_ANTI_REPEAT` thực sự khác `RRB_IID` về hành vi, không phải no-op gần như chắc chắn.

Mã nguồn: `src/evasion_od/rrb.py` (`RRBParams`, `sample_rrb_params`/`sample_anti_repeat_params`/`sample_over_diverse_params`, `params_distance`, `apply_rrb_with_params`), wiring per-image transform history trong `attack.py:run_attack` (reset mỗi ảnh), field `history_tau`/`history_window`/`history_max_tries`/`history_k_candidates` trong `AttackConfig`, 3 experiment `RRB_IID`/`RRB_ANTI_REPEAT`/`RRB_OVER_DIVERSE` trong `config.py`. Script: `scripts/run_s3_history_aware_rrb.py`.

**Kết quả pilot (30 ảnh, T=50, `RRB_ORIG` tái dùng số `OSFD_RRB_FULL` từ B1, cùng scale):**

| Variant | White-box | FCOS (A) | YOLOX (B) | DINO-Swin (C) | Avg BB |
|---|---|---|---|---|---|
| RRB_ORIG | 0.438 | 0.374 | 0.212 | 0.142 | 0.243 |
| RRB_IID | 0.409 | 0.351 | 0.206 | 0.148 | 0.235 |
| RRB_ANTI_REPEAT | 0.432 | 0.340 | 0.215 | 0.162 | 0.239 |
| RRB_OVER_DIVERSE | 0.414 | 0.334 | 0.198 | 0.154 | 0.229 |

**Kết luận: Borderline GO / Needs confirmation.** `RRB_ANTI_REPEAT` > `RRB_IID` ở YOLOX (+4%) và đặc biệt DINO-Swin (0.148→0.162, +9%; so với `RRB_ORIG`: 0.142→0.162, +14%), nhưng thua ở FCOS (-3%) và avg BB chỉ nhỉnh hơn IID +1.7% -- chưa tách khỏi biên nhiễu ở N=30. `RRB_OVER_DIVERSE` (positive control) không nhất quán thắng `RRB_ANTI_REPEAT` (chỉ hơn ở white-box/YOLOX, thua FCOS/DINO) -- không xác nhận "càng diverse càng tốt"; đã hoàn thành vai trò control, không cần chạy lại ở scale confirm.

Pattern đáng chú ý (chưa đủ mạnh để claim chắc): **kiến trúc target càng xa surrogate thì `RRB_ANTI_REPEAT` càng có ích** (FCOS giảm, YOLOX/DINO tăng, DINO tăng nhiều nhất). Nếu là tín hiệu thật, câu chuyện phù hợp hơn "optimal temporal diversity nói chung" là:

> Reducing inter-iteration augmentation redundancy may preferentially benefit harder cross-architecture targets.

**Kế hoạch confirm:** 100 ảnh, T=50, chỉ 3 arm `RRB_ORIG` (chạy lại tươi ở N=100, không tái dùng số N=30 từ B1 vì lệch scale) / `RRB_IID` / `RRB_ANTI_REPEAT` -- bỏ `RRB_OVER_DIVERSE`. **GO** nếu `RRB_ANTI_REPEAT` > `RRB_IID` trên DINO-Swin và avg BB không giảm rõ. **Strong GO** nếu `RRB_ANTI_REPEAT` > `RRB_ORIG` trên DINO-Swin và ≥2/3 target BB không giảm đáng kể. **NO-GO** nếu DINO gain biến mất/đảo chiều ở N=100. Chỉ lên N=300 nếu confirm N=100 giữ được gain DINO ≥8-10% relative và YOLOX không đảo dấu mạnh. Lệnh:

    python scripts/run_s3_history_aware_rrb.py --n-images 100 --n-iters 50 \
        --variants RRB_ORIG,RRB_IID,RRB_ANTI_REPEAT \
        --out results/S3_confirm_n100.json

**Kết quả confirm (100 ảnh, T=50, `RRB_ORIG` chạy tươi ở đúng N=100):**

| Variant | White-box | FCOS (A) | YOLOX (B) | DINO-Swin (C) | Avg BB |
|---|---|---|---|---|---|
| RRB_ORIG | 0.485 | 0.410 | 0.320 | 0.118 | 0.283 |
| RRB_IID | 0.475 | 0.402 | 0.315 | 0.113 | 0.277 |
| RRB_ANTI_REPEAT | 0.475 | 0.403 | 0.313 | 0.112 | 0.276 |

**Kết luận: NO-GO -- đóng nhánh S3 tại đây.** DINO gain của pilot (0.148→0.162, +9%) biến mất hoàn toàn ở N=100: `RRB_ANTI_REPEAT` (0.112) còn thấp hơn nhẹ `RRB_IID` (0.113, -0.9%), YOLOX cũng vậy (0.313 vs 0.315, -0.6%) -- đúng điều kiện NO-GO đã đặt trước ("DINO gain biến mất hoặc đảo chiều ở N=100"). `RRB_ANTI_REPEAT` cũng không thắng `RRB_ORIG` ở bất kỳ target nào (avg BB 0.276 vs 0.283, -2.5%). Ba variant gần như không phân biệt được (chênh lệch avg BB giữa cả 3 chỉ ~2.5%, trong khi bản thân `RRB_ORIG` đã dao động rất mạnh giữa 2 scale -- DINO 0.142 ở N=30 xuống 0.118 ở N=100, YOLOX 0.212 lên 0.320 -- xác nhận N=30 quá nhiễu để kết luận, đúng như lo ngại đã nêu trước khi chạy pilot).

**Bài học:** pattern "kiến trúc target càng xa surrogate thì ANTI_REPEAT càng có ích" ở pilot N=30 là nhiễu, không phải tín hiệu thật -- ví dụ cụ thể cho thấy tại sao pilot 30 ảnh chỉ đủ để định hướng thô (GO/NOGO sàng lọc), không đủ để tin bất kỳ ranking cụ thể nào giữa các cấu hình gần nhau. History-aware sampling (cả anti-repeat lẫn over-diverse) không cộng thêm giá trị lên trên i.i.d. resampling thuần của RRB_ORIG -- củng cố thêm kết luận xuyên suốt G0→B1→B2→S0→S1→S2→S3: **cái tạo transferability của RRB là chính sự ngẫu nhiên per-iteration, không phải bất kỳ cấu trúc/policy nào áp lên trên nó** (i.i.d. resampling đã là gần tối ưu trong không gian đã thử). Không tiếp tục điều chỉnh sampling policy cho RRB trừ khi có hướng khác hẳn.

File: `results/S3_history_aware_rrb.json` (pilot), `results/S3_confirm_n100.json` (confirm).

**Đóng hẳn nhánh cải tiến RRB/augmentation-sampling tại đây** (G0/B1/B2/S0/S1/S2/S3 -- 7 phase). Literature check (scan 2025-2026): AugTrans (2026) đã chiếm phần lớn không gian augmentation-centric cho OD transfer (dynamic object-aware rotation/resize/noise + EOT); paper "Resolving Gradient Conflicts in Multi-Input Transformations" (2026, classification) đã trực tiếp giải quyết đúng câu hỏi "cân bằng diversity vs gradient conflict giữa các transformed view". Không còn lý do khoa học để tiếp tục tune sampling policy cho RRB.

## 11. Phase F0 — Backbone Stage Ablation

Pivot khỏi trục augmentation sang trục khác hẳn: bản thân `L_OSFD = sum_{l=0}^{3} mean((F_adv_l - k*F_clean_l)^2)` cộng dồn distortion qua cả 4 stage backbone của `faster_rcnn_r50_fpn` một cách vô điều kiện. Câu hỏi: **có phải một số stage đang kéo perturbation về vulnerability riêng của ResNet-50 (surrogate-specific, tốt white-box nhưng không transfer), trong khi stage khác mang tín hiệu sống sót tốt hơn qua kiến trúc khác (CSPNet/Darknet/Swin)?** Khác nhánh G0 (vốn consensus giữa các RRB *view* của cùng 1 loss) -- ở đây phân rã chính loss OSFD theo *stage*, không đụng tới augmentation (cố tình **không RRB**, vì randomness per-iteration của RRB sẽ confound việc đánh giá đúng *hướng gradient* của từng stage, không chỉ độ lớn).

Literature check: multi-stage feature attack không mới ở classification (SMP-Attack, ICCV'25), nhưng câu hỏi "stage nào transfer vs stage nào overfit-surrogate" cụ thể cho OSFD/OD chưa thấy trong scan hiện tại.

Mã nguồn: `losses.py:backbone_feature_loss` (tham số mới `stage_weights`, `None` = uniform, tương thích ngược 100% với mọi experiment trước F0), `AttackConfig.osfd_stage_weights`, 8 experiment `OSFD_S0`/`OSFD_S1`/`OSFD_S2`/`OSFD_S3`/`OSFD_S01`/`OSFD_S12`/`OSFD_S23`/`OSFD_ALL` trong `config.py`. Script: `scripts/run_f0_stage_ablation.py`.

**Thiết kế:** 30 ảnh, T=50, không RRB, evaluate FRCNN (white-box) + FCOS/YOLOX/DINO-Swin-L (A/B/C). `OSFD_ALL` chạy tươi ở đúng scale pilot (không tái dùng số E1 gốc 50 ảnh/T=100 -- cùng lý do B1 đã rerun `OSFD_RRB_FULL`). Đọc thêm tỉ lệ **BB/WB** (avg black-box mAP-drop / white-box mAP-drop) bên cạnh bảng số thô -- một stage/subset có white-box drop *thấp hơn* nhưng tỉ lệ BB/WB *cao hơn* rõ, đặc biệt ở YOLOX/DINO-Swin-L, là tín hiệu "stage đó ít overfit surrogate hơn".

**GO:** một stage/subset thắng `OSFD_ALL` trên Group B/C, hoặc có tỉ lệ BB/WB cao hơn rõ. **NO-GO:** `OSFD_ALL` vẫn tốt nhất mọi nơi, hoặc stage ranking nhiễu/không nhất quán giữa các target -- đóng nhánh feature-stage selection.

    python scripts/run_f0_stage_ablation.py --n-images 30 --n-iters 50 \
        --out results/F0_stage_ablation.json

**Kết quả pilot (30 ảnh, T=50):**

| Variant | White-box | FCOS (A) | YOLOX (B) | DINO-Swin (C) | Avg BB | BB/WB |
|---|---|---|---|---|---|---|
| OSFD_S0 | 0.042 | 0.029 | 0.006 | 0.019 | 0.018 | 0.423 |
| OSFD_S1 | 0.149 | 0.102 | 0.036 | 0.040 | 0.059 | 0.398 |
| OSFD_S2 | 0.372 | 0.155 | 0.091 | 0.048 | 0.098 | 0.264 |
| OSFD_S3 | 0.329 | 0.146 | 0.058 | 0.036 | 0.080 | 0.244 |
| OSFD_S01 | 0.141 | 0.100 | 0.054 | 0.052 | 0.069 | 0.489 |
| OSFD_S12 | 0.322 | 0.152 | 0.078 | 0.045 | 0.092 | 0.285 |
| OSFD_S23 | 0.396 | 0.186 | 0.065 | 0.036 | 0.095 | 0.241 |
| OSFD_ALL | 0.390 | 0.182 | 0.069 | 0.028 | 0.093 | 0.239 |

**Kết luận pilot: GO, tín hiệu rõ hơn dự đoán.** `OSFD_S2` (chỉ tấn công stage 2 riêng lẻ) thắng `OSFD_ALL` trên avg BB (0.098 vs 0.093, +5%), YOLOX (0.091 vs 0.069, +32%), và đặc biệt DINO-Swin (0.048 vs 0.028, +71%) -- chỉ thua nhẹ FCOS (-15%) và white-box gần như giữ nguyên (-5%). Khớp đúng giả thuyết "stage-specific surrogate overfitting": bỏ bớt stage 0/1/3 giúp cross-family transfer tốt hơn dù white-box không đổi nhiều.

`OSFD_S01` (stage 0+1) cho pattern khác: DINO tăng mạnh nhất bảng (+86% so ALL) và tỉ lệ BB/WB cao nhất (0.489, gấp đôi ALL), nhưng đánh đổi lớn -- white-box sập còn 0.141 (-64%), FCOS -45%. Ít cân bằng hơn S2 nhưng "hiệu suất transfer trên mỗi đơn vị white-box" cao nhất bảng.

Vì bài học S3 (pilot N=30 tạo false signal đẹp trên DINO, biến mất ở N=100), **cần confirm `OSFD_S2` vs `OSFD_ALL` ở N=100 trước khi tin kết luận này** -- xem mục confirm bên dưới.

**Kế hoạch confirm:** 100 ảnh, T=50, chỉ 2 arm `OSFD_S2` / `OSFD_ALL` (chạy tươi cả hai ở N=100). **GO** nếu `OSFD_S2` > `OSFD_ALL` trên DINO-Swin và không giảm rõ avg BB. **NO-GO** nếu gain DINO/YOLOX biến mất hoặc đảo chiều ở N=100 (đúng kiểu S3 đã gặp). Lệnh:

    python scripts/run_f0_stage_ablation.py --n-images 100 --n-iters 50 \
        --variants OSFD_S2,OSFD_ALL \
        --out results/F0_confirm_n100.json

**Kết quả confirm (100 ảnh, T=50, cả hai variant chạy tươi ở đúng N=100):**

| Variant | White-box | FCOS (A) | YOLOX (B) | DINO-Swin (C) | Avg BB | BB/WB |
|---|---|---|---|---|---|---|
| OSFD_S2 | 0.433 | 0.208 | 0.108 | 0.038 | 0.118 | 0.273 |
| OSFD_ALL | 0.450 | 0.228 | 0.102 | 0.013 | 0.114 | 0.253 |

**Kết luận: GO -- tín hiệu sống sót qua confirm, khác hẳn số phận của S3.** `OSFD_S2` vẫn thắng `OSFD_ALL` trên avg BB (0.118 vs 0.114, +3.5%) và đặc biệt DINO-Swin (0.038 vs 0.013, **+192%** tương đối -- ALL sụp mạnh trên DINO khi tăng N, 0.028→0.013, trong khi S2 giữ được 0.048→0.038). YOLOX vẫn thắng nhưng biên co hẹp nhiều so pilot (+32%→+6%). FCOS/white-box vẫn thua nhẹ, cùng hướng với pilot (-9%/-4%). Không có target nào đảo chiều -- đúng điều kiện GO đã đặt trước, khác hẳn Phase S3 (nơi cả DINO lẫn YOLOX đảo dấu hoàn toàn ở N=100).

**So sánh 2 scale (bài học về độ tin cậy pilot):** cả `OSFD_S2` lẫn `OSFD_ALL` đều tăng đáng kể ở mọi target khi N=30→100 (attack "mạnh lên" nói chung theo scale, hiện tượng đã thấy ở cả S2/S3 pilot trước đó -- có thể do tập con 30 ảnh đầu của `dev_300` khó hơn/dễ hơn trung bình), nhưng **thứ hạng tương đối S2>ALL giữ nguyên** trên avg BB và DINO -- khác biệt quan trọng so với S3 nơi cả giá trị tuyệt đối lẫn thứ hạng tương đối đều đảo lộn. Điều này củng cố thêm rằng tín hiệu F0 (contribution phân hoá theo stage) là cấu trúc thật của loss OSFD, không phải nhiễu sampling như history-aware RRB.

**Bước tiếp theo (đã đề xuất, chưa chạy):** kiểm tra `OSFD_S2 + RRB` so với `E2` (`OSFD_ALL` + RRB) -- đây là câu hỏi quyết định giá trị thực tế: liệu lợi thế của việc chỉ tấn công stage 2 có cộng dồn được lên trên baseline mạnh nhất dự án hay không (giống cách I4 từng kiểm tra RaPA-mask có cộng dồn lên RRB hay bị bão hoà).

## 12. Phase F1 — Best Stage + RRB vs Full-Stage + RRB

Câu hỏi quyết định của Phase F: liệu lợi thế của `OSFD_S2` (đã confirm ở cả N=30 và N=100 khi không có augmentation) có **cộng dồn** lên trên RRB (đòn bẩy mạnh nhất dự án, bước nhảy E1→E2 vẫn là bước nhảy lớn nhất từng đo được), hay bị RRB **nuốt chửng** giống số phận RaPA-mask (I4 vs E2, mục 1 -- I4≈E2, không cộng dồn thêm)? 2 arm, cùng loss/eps/alpha/iterations, chỉ khác `osfd_stage_weights`:

- `E2_ALL_RRB` -- OSFD toàn bộ 4 stage + RRB (= config `E2_osfd_rrb` có sẵn)
- `S2_RRB` -- OSFD chỉ stage 2 + RRB (config mới `S2_RRB` trong `config.py`)

Ở scale pilot N=30/T=50, `E2_ALL_RRB` tái dùng số `OSFD_RRB_FULL` từ Phase B1 (`results/B1_rrb_component_ablation.json` -- cùng config hệt nhau, cùng lý do tái dùng như mọi script Phase S), pilot chỉ tốn compute cho `S2_RRB`. Script: `scripts/run_f1_stage_rrb_combo.py`.

**GO:** `S2_RRB` thắng `E2_ALL_RRB` trên avg BB và/hoặc DINO-Swin-L, không giảm nhiều ở target khác. **NO-GO:** `E2_ALL_RRB` vẫn tốt nhất mọi nơi (RRB nuốt chửng lợi thế stage-2, giống số phận I4) -- đóng nhánh F tại đây, stage-2-only chỉ có giá trị khi không có augmentation.

    python scripts/run_f1_stage_rrb_combo.py --n-images 30 --n-iters 50 \
        --out results/F1_stage_rrb_combo.json

**Kết quả pilot (30 ảnh, T=50, `E2_ALL_RRB` tái dùng số `OSFD_RRB_FULL` từ B1, `S2_RRB` chạy tươi):**

| Variant | White-box | FCOS (A) | YOLOX (B) | DINO-Swin (C) | Avg BB |
|---|---|---|---|---|---|
| E2_ALL_RRB | 0.438 | 0.374 | 0.212 | 0.142 | 0.243 |
| S2_RRB | 0.398 | 0.314 | 0.217 | 0.172 | 0.234 |

**Kết luận pilot: kết quả hỗn hợp, không phải GO sạch.** `S2_RRB` thắng rõ DINO-Swin (0.172 vs 0.142, +21%) và nhỉnh nhẹ YOLOX (+2%), nhưng thua FCOS (-16%) và white-box (-9%), nên avg BB tổng thể thua nhẹ (0.234 vs 0.243, -3.7%). RRB không nuốt chửng hoàn toàn lợi thế stage-2 (khác hẳn số phận I4 -- I4≈E2 gần như phẳng ở mọi target), nhưng cũng không cộng dồn đều lên mọi group -- chỉ giữ được rõ ràng ở đúng target khó nhất. Pattern này khớp với những gì đã thấy ở E4/E5 (RaPA-mask): "hữu ích đặc biệt cho Group C cross-family khó, không cải thiện đều mọi nhóm" -- một motif lặp lại xuyên suốt dự án (RaPA-mask, giờ tới stage-2-only) rằng các can thiệp parameter/loss-level có xu hướng giúp đúng target khó nhất trong khi RRB (input-level) vẫn thắng ở phần còn lại.

**Chưa kết luận chính thức -- cần confirm N=100 trước.** Bài học Phase S3 (tín hiệu DINO đẹp ở N=30 biến mất hoàn toàn ở N=100) áp dụng trực tiếp ở đây: mức tăng DINO-Swin (+21%) nhỏ hơn cả mức F0 (no-RRB) từng thấy ở pilot (+71%, và vẫn giữ được ở N=100 dù yếu đi tương đối). Cần xác nhận trước khi tin bất kỳ kết luận nào về việc "S2 + RRB có đáng dùng hay không".

**Kế hoạch confirm (đề xuất, chưa chạy):** 100 ảnh, T=50, cả 2 arm chạy tươi (B1's N=30 reference không còn hợp lệ ở N=100). **GO** nếu `S2_RRB` vẫn thắng `E2_ALL_RRB` rõ trên DINO-Swin và avg BB không giảm nhiều. **NO-GO** nếu gain DINO biến mất/đảo chiều (như S3), hoặc mọi chỉ số đều thua `E2_ALL_RRB` rõ rệt -- khi đó đóng nhánh F, kết luận "stage-2-only chỉ có giá trị khi không có RRB, RRB đã đủ mạnh để không cần thêm". Lệnh:

    python scripts/run_f1_stage_rrb_combo.py --n-images 100 --n-iters 50 \
        --variants E2_ALL_RRB,S2_RRB \
        --out results/F1_confirm_n100.json

**Kết quả confirm (100 ảnh, T=50, cả hai variant chạy tươi ở đúng N=100):**

| Variant | White-box | FCOS (A) | YOLOX (B) | DINO-Swin (C) | Avg BB |
|---|---|---|---|---|---|
| E2_ALL_RRB | 0.480 | 0.399 | 0.303 | 0.128 | 0.277 |
| S2_RRB | 0.436 | 0.364 | 0.285 | 0.122 | 0.257 |

**Kết luận: NO-GO -- đóng nhánh F1, đúng kịch bản cảnh báo từ Phase S3.** Ở N=100, `S2_RRB` thua `E2_ALL_RRB` trên **toàn bộ** chỉ số, kể cả DINO-Swin -- điểm sáng duy nhất của pilot (0.172 vs 0.142, +21%) đã **đảo dấu hoàn toàn** thành thua (0.122 vs 0.128, -4.7%). YOLOX cũng đảo dấu tương tự (pilot +2% → confirm -6%). avg BB thua rõ hơn cả pilot (-3.7%→-7.2%). Không còn chỉ số nào để biện minh cho `S2_RRB`.

**Kết luận tổng hợp Phase F0+F1:** stage-2-only là finding thật nhưng **có điều kiện chặt** -- chỉ có giá trị khi *không* có augmentation (F0: `OSFD_S2` thắng `OSFD_ALL` bền vững qua cả N=30 và N=100). Ngay khi thêm RRB vào, lợi thế đó không những không cộng dồn mà **bị đảo ngược khi đo đủ ảnh** -- RRB không chỉ "nuốt chửng" (như I4 với RaPA-mask, nơi I4≈E2 gần bằng nhau) mà còn khiến việc bớt stage đi *có hại* so với dùng đủ cả 4 stage. Đóng hẳn nhánh Phase F tại đây. `E2` (OSFD toàn bộ stage + RRB) tiếp tục là baseline mạnh nhất dự án, không có config nào (RRB-sampling-policy ở Phase S, hay stage-selection ở Phase F) vượt qua được khi kết hợp RRB.

File: `results/F1_stage_rrb_combo.json` (pilot), `results/F1_confirm_n100.json` (confirm).

## 13. Nguyên tắc phương pháp luận — pilot N=30 không đủ tin cậy khi hiệu ứng tập trung ở 1 target

Hai lần liên tiếp (Phase S3, Phase F1), một pilot N=30 tạo tín hiệu "sạch" và có vẻ đáng tin -- gain tập trung rõ ràng ở đúng DINO-Swin-L (target khó nhất, được quan tâm nhất) -- rồi **biến mất hoặc đảo dấu hoàn toàn** khi confirm ở N=100:

| Phase | Metric (DINO-Swin) | Pilot N=30 | Confirm N=100 |
|---|---|---|---|
| S3 (`RRB_ANTI_REPEAT` vs `RRB_IID`) | mAP-drop | 0.162 vs 0.148 (+9%) | 0.112 vs 0.113 (-0.9%, đảo dấu) |
| F1 (`S2_RRB` vs `E2_ALL_RRB`) | mAP-drop | 0.172 vs 0.142 (+21%) | 0.122 vs 0.128 (-4.7%, đảo dấu) |

Ngược lại, Phase F0 (`OSFD_S2` vs `OSFD_ALL`, không RRB) là phản ví dụ đáng chú ý: gain DINO-Swin +71% ở N=30 vẫn giữ đúng dấu ở N=100 (tuy giảm biên độ tương đối theo cách khác -- ALL sụp mạnh hơn S2 khi tăng N, nên % tương đối thực ra *tăng* lên +192%, nhưng bản chất "S2 > ALL" không đổi).

**Rút ra:** không có quy tắc đơn giản kiểu "cứ tin pilot nếu tín hiệu tập trung ở 1 target" -- cả tín hiệu thật (F0) lẫn tín hiệu nhiễu (S3, F1) đều có thể biểu hiện giống nhau ở N=30 (thắng rõ ở đúng DINO-Swin, các target khác gần như hòa/thua nhẹ). Điểm phân biệt duy nhất đáng tin là **chạy confirm ở scale lớn hơn (N=100) trước khi ghi nhận bất kỳ GO nào vào kết luận cuối**, không phụ thuộc vào việc pilot "trông có vẻ sạch" hay "khớp với hypothesis". Áp dụng bắt buộc cho mọi phase tương lai của dự án: pilot N=30 chỉ dùng để **định hướng** (loại bỏ hướng rõ ràng vô dụng, tiết kiệm compute), không dùng để **kết luận**.