# RQ11 — Calibration

> **Phiên bản kết quả cũ:** temperature được chọn để tối thiểu hóa NLL trên validation. Test không được dùng để tuning. Kết quả đa mục tiêu NLL/Brier/ECE của pipeline mới sẽ thay thế báo cáo này sau khi hoàn thành.

## 1. Câu hỏi nghiên cứu

RQ11 kiểm tra ảnh hưởng của distillation và Bayesian update tới chất lượng xác suất dự đoán. Ba metric calibration chính là:

- NLL: đánh giá xác suất gán cho nhãn đúng; thấp hơn tốt hơn.
- Brier: đánh giá sai số của toàn bộ phân phối xác suất; thấp hơn tốt hơn.
- ECE: đo khoảng cách giữa confidence và accuracy; thấp hơn tốt hơn.

Temperature scaling là phép biến đổi đơn điệu nên không thay đổi thứ hạng candidate, R@k hoặc MRR. Trong phiên bản này, temperature chỉ tối ưu NLL; vì vậy giảm NLL không đồng nghĩa Brier và ECE cũng phải giảm.

## 2. Protocol

Hai nhóm thí nghiệm được báo cáo riêng:

- `distillation`: last-query protocol của RQ10, gồm `none`, `gru`, `transformer`.
- `bayesian`: all-prefix protocol của RQ7, gồm `B0-static`, `B3-dbn`.

Transition và prior của Bayesian chỉ fit trên train. Trọng số B3 được chọn trên validation từ RQ7. Temperature được fit trên validation và chỉ sau đó mới áp dụng lên test.

## 3. Distillation — last-query

| Variant | T | NLL trước | NLL sau | Brier trước | Brier sau | ECE trước | ECE sau | Adaptive ECE trước | Adaptive ECE sau |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 2.5000 | 9.210952 | 7.864161 | 0.952485 | 0.985876 | 0.030813 | 0.118299 | 0.031762 | 0.118264 |
| gru | 1.5833 | 7.575581 | 7.062748 | 0.944472 | 0.961283 | 0.029656 | 0.093463 | 0.030332 | 0.093256 |
| transformer | 1.7500 | 7.962563 | 7.160651 | 0.949805 | 0.966111 | 0.053708 | 0.102472 | 0.053708 | 0.102440 |

### Phân tích

NLL giảm ở cả ba biến thể:

- `none`: giảm 1.346791, tương đương khoảng 14.62%.
- `gru`: giảm 0.512833, tương đương khoảng 6.77%.
- `transformer`: giảm 0.801912, tương đương khoảng 10.07%.

Tuy nhiên, Brier và ECE đều xấu hơn. Với `none`, ECE tăng từ 0.030813 lên 0.118299; với `gru`, tăng từ 0.029656 lên 0.093463; với `transformer`, tăng từ 0.053708 lên 0.102472. Kết quả cho thấy temperature tối ưu NLL đã làm phân phối mềm hơn theo hướng có lợi cho xác suất nhãn đúng trung bình, nhưng làm confidence top-1 lệch xa accuracy hơn.

Trong nhóm distillation, GRU sau calibration có NLL, Brier và ECE thấp nhất. Tuy vậy, đây mới là so sánh mô tả; báo cáo cũ chưa có paired test trực tiếp giữa GRU/Transformer và `none` sau calibration.

Reliability diagram cũ: `results/beliefmove-evo/aggregated/rq11_distillation_reliability.svg`.

## 4. Bayesian — all-prefix

| Variant | T | NLL trước | NLL sau | Brier trước | Brier sau | ECE trước | ECE sau | Adaptive ECE trước | Adaptive ECE sau |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0-static | 1.5000 | 7.130893 | 6.619977 | 0.949882 | 0.954934 | 0.042341 | 0.069569 | 0.042590 | 0.069368 |
| B3-dbn | 1.7500 | 7.487551 | 6.430621 | 0.975386 | 0.949091 | 0.142256 | 0.060272 | 0.142256 | 0.060056 |

### Phân tích

`B0-static` giảm NLL 0.510916 nhưng Brier tăng 0.005052 và ECE tăng 0.027228. Mô hình này lặp lại xung đột metric đã thấy trong nhóm distillation.

`B3-dbn` là biến thể duy nhất cải thiện đồng thời cả ba metric:

- NLL giảm 1.056930, khoảng 14.12%.
- Brier giảm 0.026295, khoảng 2.70%.
- ECE giảm 0.081984, khoảng 57.63%.

Sau calibration, B3 có NLL 6.430621, Brier 0.949091 và ECE 0.060272; các giá trị này đều thấp hơn B0 sau calibration. Đây là bằng chứng mô tả rằng posterior B3 ban đầu bị miscalibrated mạnh nhưng có thể được sửa đáng kể bằng temperature scaling. Cần paired comparison B3–B0 trong pipeline mới để xác nhận khác biệt trực tiếp.

Reliability diagram cũ: `results/beliefmove-evo/aggregated/rq11_bayesian_reliability.svg`.

## 5. Paired NLL/Brier improvement

Positive effect nghĩa là trạng thái sau calibration tốt hơn; đối với NLL và Brier, dấu đã được quy đổi để giá trị dương luôn biểu thị cải thiện.

| Protocol | Variant | Metric | Effect sau tốt hơn | 95% CI | Holm p | Significant |
|---|---|---|---:|---:|---:|---|
| distillation | none | nll | 1.346791 | 1.312654–1.379519 | 0.0009999 | yes |
| distillation | none | brier | -0.033391 | -0.035038–-0.031756 | 0.0009999 | yes |
| distillation | gru | nll | 0.512834 | 0.496452–0.528952 | 0.0009999 | yes |
| distillation | gru | brier | -0.016812 | -0.018062–-0.015588 | 0.0009999 | yes |
| distillation | transformer | nll | 0.801912 | 0.780156–0.823110 | 0.0009999 | yes |
| distillation | transformer | brier | -0.016306 | -0.017774–-0.014806 | 0.0009999 | yes |
| bayesian | B0-static | nll | 0.510916 | 0.504435–0.517508 | 0.0009999 | yes |
| bayesian | B0-static | brier | -0.005051 | -0.005528–-0.004582 | 0.0009999 | yes |
| bayesian | B3-dbn | nll | 1.056929 | 1.047362–1.066722 | 0.0009999 | yes |
| bayesian | B3-dbn | brier | 0.026295 | 0.025550–0.027023 | 0.0009999 | yes |

Các kiểm định xác nhận NLL giảm có ý nghĩa ở cả năm biến thể. Brier xấu đi có ý nghĩa ở `none`, `gru`, `transformer` và `B0-static`; chỉ B3 cải thiện Brier có ý nghĩa.

## 6. Bootstrap ECE improvement

| Protocol | Variant | Metric | Effect sau tốt hơn | 95% CI |
|---|---|---|---:|---:|
| distillation | none | ece | -0.087485 | -0.092507–-0.082441 |
| distillation | gru | ece | -0.063807 | -0.068848–-0.058567 |
| distillation | transformer | ece | -0.048765 | -0.054520–-0.043417 |
| bayesian | B0-static | ece | -0.027228 | -0.029441–-0.024973 |
| bayesian | B3-dbn | ece | 0.081984 | 0.079585–0.084412 |

Bootstrap CI không chứa 0 trong cả năm trường hợp. Dấu âm xác nhận ECE xấu đi ở bốn biến thể; dấu dương xác nhận B3 cải thiện ECE.

## 7. Kết luận tạm thời

Temperature scaling tối ưu NLL làm giảm NLL nhất quán, nhưng không phải một giải pháp calibration toàn diện. Kết quả thể hiện xung đột rõ ràng giữa log-likelihood với Brier/ECE. Chỉ B3-DBN cải thiện đồng thời NLL, Brier và ECE, cho thấy Bayesian posterior là trường hợp hưởng lợi rõ nhất từ bước calibration.

Không được dùng bảng cũ để tuyên bố rằng distillation hoặc temperature scaling cải thiện calibration nói chung. Kết luận hợp lệ là: **NLL-optimized temperature scaling cải thiện NLL cho mọi biến thể, còn cải thiện calibration đa metric chỉ được quan sát ở B3-DBN.**

## 8. Hạn chế và trạng thái

- Đây là kết quả cũ với một objective chọn temperature là NLL.
- Chưa có temperature tối ưu riêng Brier và ECE.
- Chưa có paired comparison trực tiếp GRU/Transformer–None và B3–B0 sau calibration.
- Distillation last-query và Bayesian all-prefix không được so trực tiếp trị tuyệt đối.
- Kết quả chỉ thuộc TIST2015–Tokyo, seed 42–44; chưa phải kết quả 12-city.
- Pipeline đa mục tiêu mới sẽ thay thế các hạn chế đầu tiên khi hoàn thành.

## 9. Publication gate

Phiên bản cũ đạt gate về split, seed và kiểm định trước–sau calibration, nhưng chưa đạt gate cuối của RQ11 vì thiếu calibration đa mục tiêu và kiểm định trực tiếp giữa các model. Báo cáo này được giữ làm mốc đối chiếu, không phải kết quả RQ11 cuối cùng.
