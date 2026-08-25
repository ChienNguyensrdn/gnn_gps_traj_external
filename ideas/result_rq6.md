# RQ6 — Dual-Axis Evolution

> Báo cáo trên held-out test split của `TIST2015-Tokyo`. Checkpoint được chọn bằng validation; ngưỡng short/medium/long được fit bằng tertile validation rồi khóa trước khi áp dụng test.

## 1. Câu hỏi nghiên cứu

RQ6 kiểm tra:

> Tiến hóa biểu diễn theo chiều layer hay theo chiều thời gian quan trọng hơn, và việc kết hợp hai trục có mang lại lợi ích bổ sung hay không?

Các cấu hình:

| Variant | Thành phần |
|---|---|
| `E1-kd` | CE + response KD |
| `E2-kd-traj` | E1 + layer trajectory |
| `E3-kd-vel` | E1 + layer velocity |
| `E4-layer` | E1 + layer trajectory + layer velocity |
| `E6-temporal` | E1 + temporal evolution only |
| `E5-dual` | E4 + temporal evolution, tức dual-axis đầy đủ |

Mỗi variant dùng seed `42, 43, 44`. Representation alignment được fit trên validation bằng centered orthogonal Procrustes trước khi tính transition cosine trên test.

## 2. Kết quả test tổng thể

| Variant | R@1 ↑ | R@5 ↑ | R@10 ↑ | MRR ↑ | NLL ↓ | Brier ↓ | ECE ↓ | CKA ↑ | Temporal cosine ↑ | Layer cosine ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1-kd | 0.145312 | 0.303215 | 0.365176 | 0.220291 | 7.569135 | 0.946729 | 0.033526 | 0.317990 | 0.223300 | 0.281881 |
| E2-kd-traj | 0.144380 | 0.301111 | 0.364624 | 0.218999 | 7.560588 | 0.947402 | 0.035012 | 0.321180 | 0.227242 | 0.283755 |
| E3-kd-vel | 0.144259 | 0.301646 | 0.365107 | 0.219352 | 7.561420 | 0.946442 | 0.033649 | 0.326535 | 0.229866 | 0.287418 |
| E4-layer | 0.144949 | 0.301784 | 0.365935 | 0.219906 | 7.560033 | 0.946117 | 0.032452 | 0.328819 | 0.234386 | **0.289142** |
| E6-temporal | 0.145398 | 0.301766 | 0.365470 | 0.220004 | **7.512255** | 0.946604 | **0.028289** | 0.323579 | 0.234039 | 0.269527 |
| **E5-dual** | **0.147140** | **0.303595** | **0.367367** | **0.221656** | 7.528161 | **0.945082** | 0.029841 | **0.331437** | **0.246387** | 0.278369 |

Kết quả cho thấy hai điểm khác nhau:

- `E5-dual` đạt mean tốt nhất về toàn bộ ranking metrics, Brier, CKA và temporal transition cosine.
- `E6-temporal` đạt NLL và ECE thấp nhất, tức calibration theo log-loss tốt nhất.
- `E4-layer` đạt layer transition cosine cao nhất, phù hợp với objective layer trajectory + velocity.

## 3. Kết quả theo độ dài trajectory

### 3.1. Short

| Variant | Queries | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|---:|
| E1-kd | 8364 | 0.119759 | 0.248605 | 0.299936 | 0.181581 |
| E2-kd-traj | 8364 | 0.119082 | 0.248804 | 0.301012 | 0.180802 |
| E3-kd-vel | 8364 | 0.118683 | 0.246692 | 0.300375 | 0.180660 |
| E4-layer | 8364 | 0.120277 | 0.248406 | 0.301690 | 0.181620 |
| E6-temporal | 8364 | 0.121433 | 0.248605 | 0.300136 | 0.182472 |
| **E5-dual** | 8364 | **0.122270** | **0.249841** | **0.302447** | **0.183641** |

### 3.2. Medium

| Variant | Queries | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|---:|
| E1-kd | 7326 | 0.159523 | **0.329830** | 0.395168 | 0.240005 |
| E2-kd-traj | 7326 | 0.159114 | 0.323733 | 0.391801 | 0.238054 |
| E3-kd-vel | 7326 | 0.158113 | 0.326963 | 0.392802 | 0.238421 |
| E4-layer | 7326 | 0.158340 | 0.324916 | 0.392756 | 0.238514 |
| E6-temporal | 7326 | 0.158568 | 0.327600 | 0.393757 | 0.238827 |
| **E5-dual** | 7326 | **0.160251** | 0.327873 | **0.395441** | **0.240210** |

### 3.3. Long

| Variant | Queries | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|---:|
| E1-kd | 3634 | 0.175472 | 0.375252 | 0.454871 | 0.269642 |
| E2-kd-traj | 3634 | 0.172904 | 0.375894 | 0.456247 | 0.268497 |
| E3-kd-vel | 3634 | 0.175197 | 0.377087 | 0.458265 | 0.269961 |
| E4-layer | 3634 | 0.174739 | 0.378004 | 0.459732 | 0.270515 |
| E6-temporal | 3634 | 0.174005 | 0.372042 | 0.458815 | 0.268441 |
| **E5-dual** | 3634 | **0.177949** | **0.378371** | **0.460191** | **0.271747** |

E5 đạt Recall@1 và MRR cao nhất trong cả ba nhóm độ dài. Với medium trajectory, E1 vẫn có Recall@5 cao nhất; vì chưa có paired test riêng theo bucket, không được suy diễn các chênh lệch nhỏ này là có ý nghĩa thống kê.

## 4. Paired significance test

Positive effect nghĩa là variant đứng trước tốt hơn. Với NLL/Brier, effect dương tương ứng loss của variant đứng trước thấp hơn. Holm correction được áp dụng đồng thời cho 24 phép kiểm định.

| Comparison | Metric | Effect | Bootstrap 95% CI | Holm p | Significant |
|---|---|---:|---:|---:|---|
| E6-temporal-vs-E1-kd | Recall@1 | 0.000086 | −0.001880–0.002070 | 1 | no |
| E6-temporal-vs-E1-kd | Recall@5 | −0.001449 | −0.003329–0.000414 | 1 | no |
| E6-temporal-vs-E1-kd | Recall@10 | 0.000293 | −0.001535–0.002104 | 1 | no |
| E6-temporal-vs-E1-kd | MRR | −0.000287 | −0.001577–0.001031 | 1 | no |
| E6-temporal-vs-E1-kd | NLL | 0.056880 | 0.048087–0.065739 | 0.00239976 | **yes** |
| E6-temporal-vs-E1-kd | Brier | 0.000124 | −0.000789–0.001040 | 1 | no |
| E4-layer-vs-E1-kd | Recall@1 | −0.000362 | −0.002260–0.001535 | 1 | no |
| E4-layer-vs-E1-kd | Recall@5 | −0.001432 | −0.003346–0.000483 | 1 | no |
| E4-layer-vs-E1-kd | Recall@10 | 0.000759 | −0.001069–0.002674 | 1 | no |
| E4-layer-vs-E1-kd | MRR | −0.000384 | −0.001628–0.000854 | 1 | no |
| E4-layer-vs-E1-kd | NLL | 0.009102 | −0.000637–0.019106 | 1 | no |
| E4-layer-vs-E1-kd | Brier | 0.000612 | −0.000156–0.001375 | 1 | no |
| E5-dual-vs-E4-layer | Recall@1 | 0.002191 | 0.000345–0.004071 | 0.343766 | no |
| E5-dual-vs-E4-layer | Recall@5 | 0.001811 | −0.000017–0.003674 | 0.883112 | no |
| E5-dual-vs-E4-layer | Recall@10 | 0.001432 | −0.000311–0.003226 | 1 | no |
| E5-dual-vs-E4-layer | MRR | 0.001750 | 0.000533–0.003001 | 0.0719928 | no |
| E5-dual-vs-E4-layer | NLL | 0.031872 | 0.023793–0.040144 | 0.00239976 | **yes** |
| E5-dual-vs-E4-layer | Brier | 0.001034 | 0.000263–0.001856 | 0.186181 | no |
| E5-dual-vs-E6-temporal | Recall@1 | 0.001742 | −0.000414–0.003812 | 1 | no |
| E5-dual-vs-E6-temporal | Recall@5 | 0.001828 | −0.000190–0.003916 | 1 | no |
| E5-dual-vs-E6-temporal | Recall@10 | 0.001897 | −0.000103–0.003847 | 0.883112 | no |
| E5-dual-vs-E6-temporal | MRR | 0.001652 | 0.000232–0.003032 | 0.360364 | no |
| E5-dual-vs-E6-temporal | NLL | −0.015906 | −0.025722–−0.006134 | 0.0241976 | **yes** |
| E5-dual-vs-E6-temporal | Brier | 0.001522 | 0.000497–0.002526 | 0.0713929 | no |

## 5. Diễn giải thống kê

### 5.1. Temporal-only chủ yếu cải thiện NLL

E6 không cải thiện có ý nghĩa các ranking metrics so với E1 sau Holm correction. Tuy nhiên, E6 giảm NLL có ý nghĩa với effect `0.056880` và adjusted p `0.00239976`. Điều này cho thấy temporal evolution riêng lẻ đóng góp rõ hơn cho chất lượng phân phối xác suất so với thứ hạng dự đoán.

### 5.2. Layer-only chưa vượt KD-only

Không metric nào của E4 so với E1 còn ý nghĩa sau Holm correction. Do đó, chưa có bằng chứng thống kê rằng layer trajectory + velocity riêng lẻ cải thiện prediction hoặc calibration so với response KD.

### 5.3. Dual cải thiện NLL so với layer-only

E5 có mean ranking cao hơn E4, nhưng các khác biệt ranking không còn ý nghĩa sau Holm correction. NLL của E5 tốt hơn E4 có ý nghĩa, effect `0.031872`, adjusted p `0.00239976`.

### 5.4. Temporal-only có NLL tốt hơn dual

Trong comparison E5-vs-E6, NLL effect là `−0.015906`. Vì effect âm, E5 có NLL cao hơn E6; khác biệt vẫn có ý nghĩa sau Holm correction (`p=0.0241976`). Vì vậy, E5 không thống trị E6 trên mọi tiêu chí: E5 mạnh hơn về mean ranking/Brier, còn E6 mạnh hơn về NLL/ECE.

## 6. Trả lời RQ6

Kết quả hỗ trợ câu trả lời có điều kiện:

1. **Layer evolution** làm representation alignment theo layer tốt hơn, thể hiện qua layer cosine cao nhất ở E4, nhưng chưa tạo gain prediction có ý nghĩa so với KD-only.
2. **Temporal evolution** tạo cải thiện có ý nghĩa về NLL so với KD-only, nhưng không cải thiện ranking có ý nghĩa sau Holm correction.
3. **Dual-axis evolution** đạt mean ranking, Brier, CKA và temporal cosine tốt nhất; tuy nhiên ranking gain so với E4/E6 chưa có ý nghĩa sau multiple-testing correction.
4. Không có một variant thắng tuyệt đối: E5 phù hợp nếu ưu tiên ranking tổng thể, còn E6 phù hợp nếu ưu tiên NLL/ECE.

## 7. Kết luận có thể dùng trong bài báo

> Trên TIST2015-Tokyo, mô hình dual-axis đầy đủ đạt mean tốt nhất về ranking, Brier, CKA và temporal transition similarity trên ba seed. Temporal-only đạt NLL và ECE tốt nhất, đồng thời giảm NLL có ý nghĩa so với KD-only sau Holm correction. Layer-only cải thiện các thước đo alignment biểu diễn nhưng chưa cải thiện có ý nghĩa các metric dự đoán so với KD-only. Dual cải thiện NLL có ý nghĩa so với layer-only, trong khi temporal-only vẫn có NLL tốt hơn dual một cách có ý nghĩa. Các cải thiện ranking giữa những variant evolution không còn ý nghĩa sau multiple-testing correction.

## 8. Giới hạn

- Kết quả hiện chỉ thuộc Tokyo, chưa đại diện cho 12 thành phố TIST2015.
- Paired tests dùng query × seed; nhiều query có thể cùng user/trajectory nên clustered bootstrap sẽ là kiểm tra bổ sung chặt hơn.
- Chưa có paired significance riêng cho short/medium/long buckets.
- Không nên tuyên bố E5 vượt E6 có ý nghĩa về ranking.
- CKA/cosine phản ánh alignment biểu diễn, không tự động đồng nghĩa với tăng accuracy.

## Publication gate

- Sáu variant × ba seed test: **đủ**.
- Length-stratified evaluation: **đủ**.
- Representation metrics: **đủ**.
- Paired bootstrap/permutation + Holm: **đủ**.
- Trạng thái RQ6 Tokyo: **ready**.
- Tổng quát hóa 12 thành phố: **chưa hoàn thành**.
