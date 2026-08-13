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