# BeliefMove-Evo — Kết quả thực nghiệm

> Cập nhật ngày 23/08/2026 từ các raw result của TIST2015-Tokyo. Các bảng tách riêng validation và test để tránh dùng test set trong quá trình chọn mô hình.

## 1. Phạm vi RQ4

RQ4 kiểm tra đóng góp của từng thành phần trong **distillation tiến hóa biểu diễn**. Tất cả thí nghiệm bên dưới sử dụng:

- Dataset: `TIST2015-Tokyo`;
- thứ tự trajectory đúng: `correct`;
- ba seed độc lập: `42, 43, 44`;
- validation để chọn checkpoint;
- test chỉ để báo cáo kết quả cuối cùng.

Có 6 cấu hình × 3 seed × 2 split được tổng hợp, tương ứng **36 raw runs**. Cả sáu cấu hình trên test split đều vượt publication gate tối thiểu ba seed.

## 2. E0–E5 là gì?

Hàm mất mát tổng quát của student:

$$
\mathcal{L}
=
\mathcal{L}_{CE}
+ \lambda_{KD}\mathcal{L}_{KD}
+ \lambda_{traj}\mathcal{L}_{traj}
+ \lambda_{vel}\mathcal{L}_{vel}
+ \lambda_{temp}\mathcal{L}_{temp}.
$$

| Biến thể | Thành phần loss | Giải thích |
|---|---|---|
| **E0-ce** | CE | Baseline student chỉ học từ nhãn thật bằng cross-entropy, không nhận tri thức từ teacher. |
| **E1-kd** | CE + KD | Thêm response knowledge distillation: student học phân phối xác suất mềm từ logits của teacher, ngoài nhãn cứng. |
| **E2-kd-traj** | CE + KD + Trajectory | Thêm trajectory representation loss, buộc các trạng thái biểu diễn trung gian của student gần teacher sau khi chiếu về cùng không gian latent. |
| **E3-kd-vel** | CE + KD + Velocity | Thêm velocity loss, buộc hướng và độ biến đổi giữa các trạng thái biểu diễn liên tiếp của student giống teacher. |
| **E4-layer** | CE + KD + Trajectory + Velocity | Kết hợp matching trạng thái và matching chuyển động biểu diễn theo chiều sâu mạng (layer-wise evolution). |
| **E5-dual** | E4 + Temporal | Mô hình đầy đủ: ngoài tiến hóa theo layer còn distill tiến hóa theo thời gian của trajectory, tạo **dual-axis evolution** gồm trục layer và trục thời gian. |

Trong cấu hình hiện tại, loss được bật có trọng số `1.0` và loss bị loại bỏ có trọng số `0.0`.

### Ý nghĩa các loss

- **CE — Cross-Entropy:** học trực tiếp đích POI tiếp theo từ ground truth.
- **KD — Knowledge Distillation:** chuyển tri thức dự đoán mềm của teacher sang student.
- **Trajectory:** trả lời câu hỏi student cần hình thành *trạng thái biểu diễn nào*.
- **Velocity:** trả lời câu hỏi biểu diễn của student cần thay đổi *theo hướng nào* giữa các layer.
- **Temporal:** đồng bộ quá trình biến đổi biểu diễn qua các bước thời gian của trajectory.

## 3. Kết quả test chính

Ký hiệu $\uparrow$ là càng lớn càng tốt; $\downarrow$ là càng nhỏ càng tốt. Số liệu là mean ± sample standard deviation trên ba seed.

| Experiment | Recall@1 ↑ | Recall@5 ↑ | Recall@10 ↑ | MRR ↑ | NLL ↓ | Brier ↓ | ECE ↓ | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| E0-ce-correct | 0.134496 ± 0.003353 | 0.271717 ± 0.003353 | 0.325812 ± 0.003374 | 0.200272 ± 0.003066 | 8.892263 ± 0.514020 | 0.955618 ± 0.006304 | 0.043941 ± 0.026593 | ready |
| E1-kd-correct | 0.145312 ± 0.000269 | 0.303215 ± 0.000880 | 0.365176 ± 0.001557 | 0.220291 ± 0.000177 | 7.569135 ± 0.014063 | 0.946729 ± 0.000965 | 0.033526 ± 0.001408 | ready |
| E2-kd-traj-correct | 0.144380 ± 0.000762 | 0.301111 ± 0.001896 | 0.364624 ± 0.001748 | 0.218999 ± 0.000070 | 7.560588 ± 0.039233 | 0.947402 ± 0.001333 | 0.035012 ± 0.004422 | ready |
| E3-kd-vel-correct | 0.144259 ± 0.001017 | 0.301646 ± 0.000936 | 0.365107 ± 0.001378 | 0.219352 ± 0.000468 | 7.561420 ± 0.010354 | 0.946442 ± 0.001181 | 0.033649 ± 0.001887 | ready |
| E4-layer-correct | 0.144949 ± 0.002026 | 0.301784 ± 0.000978 | 0.365935 ± 0.000988 | 0.219906 ± 0.001107 | 7.560033 ± 0.008218 | 0.946117 ± 0.001394 | 0.032452 ± 0.003010 | ready |
| **E5-dual-correct** | **0.147140 ± 0.002240** | **0.303595 ± 0.001035** | **0.367367 ± 0.000803** | **0.221656 ± 0.001334** | **7.528161 ± 0.012171** | **0.945082 ± 0.001529** | **0.029841 ± 0.003400** | **ready** |

## 4. Khoảng tin cậy 95% trên test

| Experiment | Recall@1 | Recall@5 | Recall@10 | MRR | NLL | Brier | ECE |
|---|---|---|---|---|---|---|---|
| E0-ce-correct | 0.130977–0.137653 | 0.267853–0.273856 | 0.321983–0.328348 | 0.196752–0.202353 | 8.352638–9.376132 | 0.950670–0.962716 | 0.017782–0.070948 |
| E1-kd-correct | 0.145156–0.145622 | 0.302318–0.304078 | 0.363693–0.366798 | 0.220104–0.220456 | 7.554366–7.582365 | 0.945668–0.947555 | 0.032672–0.035151 |
| E2-kd-traj-correct | 0.143707–0.145208 | 0.299317–0.303095 | 0.363589–0.366643 | 0.218957–0.219079 | 7.535083–7.605765 | 0.945900–0.948444 | 0.030899–0.039688 |
| E3-kd-vel-correct | 0.143086–0.144898 | 0.300662–0.302525 | 0.363589–0.366280 | 0.218896–0.219831 | 7.550418–7.570974 | 0.945087–0.947253 | 0.032153–0.035770 |
| E4-layer-correct | 0.143345–0.147226 | 0.301025–0.302888 | 0.365194–0.367057 | 0.219081–0.221164 | 7.551759–7.568194 | 0.944508–0.946961 | 0.029011–0.034597 |
| **E5-dual-correct** | **0.144587–0.148779** | **0.302577–0.304647** | **0.366539–0.368143** | **0.220205–0.222829** | **7.519984–7.542148** | **0.943489–0.946538** | **0.026094–0.032731** |

Các khoảng tin cậy được bootstrap từ ba giá trị seed. Vì $n=3$ còn nhỏ, chúng chủ yếu phản ánh độ ổn định giữa các lần chạy và không thay thế kiểm định paired trên từng query.

## 5. Kết quả validation

Validation chỉ dùng để chọn checkpoint/hyperparameter; không dùng làm kết quả cuối cùng của paper.

| Experiment | Recall@1 | Recall@5 | Recall@10 | Gate |
|---|---:|---:|---:|---|
| E0-ce-correct | 0.154402 ± 0.003605 | 0.303000 ± 0.001045 | 0.359578 ± 0.001534 | not ready |
| E1-kd-correct | 0.161994 ± 0.001866 | 0.332056 ± 0.002055 | 0.398379 ± 0.003153 | not ready |
| E2-kd-traj-correct | 0.161374 ± 0.003778 | 0.330413 ± 0.000918 | 0.398306 ± 0.002531 | not ready |
| E3-kd-vel-correct | **0.163345 ± 0.001960** | 0.331399 ± 0.002355 | 0.397065 ± 0.001793 | not ready |
| E4-layer-correct | 0.162724 ± 0.003718 | 0.330413 ± 0.002067 | 0.398124 ± 0.001017 | not ready |
| E5-dual-correct | 0.159366 ± 0.002308 | **0.333297 ± 0.001550** | **0.401482 ± 0.002574** | not ready |

`not ready` ở bảng validation là hành vi có chủ đích của publication gate: chỉ test split đủ ít nhất ba seed mới được coi là kết quả báo cáo cuối.

## 6. Phân tích RQ4

### 6.1. Đóng góp chính đến từ knowledge distillation

So với E0, E1 cải thiện toàn bộ ranking metrics. Điều này cho thấy phân phối mềm của teacher cung cấp tín hiệu giàu thông tin hơn so với chỉ học nhãn cứng bằng CE.

### 6.2. Trajectory hoặc velocity riêng lẻ chưa tạo thêm lợi ích

E2 và E3 không vượt E1 trên các Recall và MRR test. Do đó, kết quả hiện tại không ủng hộ tuyên bố rằng trajectory loss hoặc velocity loss độc lập luôn cải thiện dự đoán. Các loss này có thể cần tín hiệu bổ trợ hoặc cách cân bằng trọng số tốt hơn.

### 6.3. E5 đạt mean tốt nhất trên toàn bộ test metrics

E5-dual đạt Recall@1/5/10 và MRR cao nhất, đồng thời có NLL, Brier và ECE thấp nhất. So với E0, E5 cải thiện tương đối:

- Recall@1: **+9.40%**;
- Recall@5: **+11.73%**;
- Recall@10: **+12.75%**;
- MRR: **+10.68%**;
- NLL giảm **15.34%**;
- ECE giảm **32.09%**.

So với E1-kd, E5 chỉ cải thiện nhỏ: +0.001828 Recall@1, +0.000380 Recall@5 và +0.002191 Recall@10. Kết quả gợi ý rằng dual-axis evolution bổ sung lợi ích về ranking và calibration, nhưng phần lớn mức tăng so với E0 vẫn đến từ KD.

### 6.4. Giới hạn diễn giải

- Kết quả hiện chỉ thuộc **Tokyo**, không đại diện cho macro-average 12 thành phố TIST2015.
- Nhiều CI của E1 và E5 còn chồng lấn; chưa được phép viết “E5 vượt E1 có ý nghĩa thống kê”.
- Cần paired test trên cùng query hoặc thêm seed để kiểm chứng phần cải thiện nhỏ của E5.
- Validation và test có generalization gap, nhưng thứ hạng tổng quát E0 < E1/E2/E3/E4 < E5 vẫn được duy trì trên test.

## 7. Kết luận có thể dùng trong bài báo

> Trên TIST2015-Tokyo, response knowledge distillation là thành phần tạo ra mức cải thiện lớn nhất so với mô hình chỉ dùng cross-entropy. Các loss trajectory và velocity khi sử dụng riêng lẻ chưa vượt baseline KD. Mô hình đầy đủ E5-dual đạt mean tốt nhất trên toàn bộ ranking và calibration metrics, cho thấy tiềm năng của việc kết hợp tiến hóa biểu diễn theo chiều layer và thời gian. Tuy nhiên, do thí nghiệm hiện có ba seed và chỉ trên Tokyo, lợi ích nhỏ của E5 so với E1 cần được xác nhận bằng paired significance test và đánh giá đa thành phố trước khi đưa ra kết luận tổng quát.

## Publication gate

- `ready`: test split và đủ ít nhất ba seed.
- `not ready`: validation split hoặc thiếu số seed tối thiểu.
- Trạng thái hiện tại: **RQ4 Tokyo test đã đủ gate cho cả E0–E5; đánh giá 12 thành phố chưa hoàn thành**.
