# RQ11 — Calibration đa mục tiêu

> Identity và temperature tối ưu NLL/Brier/ECE đều được chọn trên validation; test không dùng để tuning. Hai protocol được báo cáo riêng.

## 1. Câu hỏi nghiên cứu và protocol

RQ11 kiểm tra ảnh hưởng của distillation và Bayesian update tới NLL, Brier và ECE, đồng thời xác định liệu một temperature duy nhất có tối ưu được mọi khía cạnh calibration hay không.

- `identity`: không calibration, luôn dùng `T=1`.
- `nll`: temperature tối thiểu hóa NLL trên validation.
- `brier`: temperature tối thiểu hóa Brier trên validation.
- `ece`: temperature tối thiểu hóa ECE trên validation.
- Distillation dùng last-query của RQ10; Bayesian dùng all-prefix của RQ7. Hai nhóm không được so trực tiếp trị tuyệt đối.
- Transition/prior chỉ fit train; B3 weight lấy từ validation RQ7; test chỉ dùng để đánh giá cuối.

Temperature scaling là phép biến đổi đơn điệu nên giữ nguyên ranking, R@k và MRR.

## 2. Distillation — last-query

| Variant | T-NLL | NLL id→opt | T-Brier | Brier id→opt | T-ECE | ECE id→opt |
|---|---:|---:|---:|---:|---:|---:|
| none | 2.5000 | 9.210952→7.864161 | 1.1000 | 0.952485→0.950112 | 1.1000 | 0.030813→0.010426 |
| gru | 1.5833 | 7.575581→7.062748 | 1.1000 | 0.944471→0.942611 | 1.0333 | 0.029656→0.019430 |
| transformer | 1.7500 | 7.962563→7.160651 | 1.1000 | 0.949805→0.944666 | 1.1000 | 0.053708→0.016187 |

### Trade-off trên test

| Variant | Objective | NLL | Brier | ECE | Adaptive ECE | Confidence gap |
|---|---|---:|---:|---:|---:|---:|
| none | identity | 9.210952 | 0.952485 | 0.030813 | 0.031762 | -0.030813 |
| none | nll | 7.864161 | 0.985876 | 0.118299 | 0.118264 | 0.118264 |
| none | brier | 8.953917 | 0.950112 | 0.010426 | 0.011023 | -0.001586 |
| none | ece | 8.953917 | 0.950112 | 0.010426 | 0.011023 | -0.001586 |
| gru | identity | 7.575581 | 0.944471 | 0.029656 | 0.030332 | -0.028993 |
| gru | nll | 7.062748 | 0.961283 | 0.093463 | 0.093256 | 0.093256 |
| gru | brier | 7.382258 | 0.942611 | 0.014729 | 0.014959 | 0.005067 |
| gru | ece | 7.502665 | 0.943379 | 0.019430 | 0.020197 | -0.017393 |
| transformer | identity | 7.962563 | 0.949805 | 0.053708 | 0.053708 | -0.053708 |
| transformer | nll | 7.160651 | 0.966111 | 0.102472 | 0.102440 | 0.102440 |
| transformer | brier | 7.722951 | 0.944666 | 0.016187 | 0.016613 | -0.015606 |
| transformer | ece | 7.722951 | 0.944666 | 0.016187 | 0.016613 | -0.015606 |

Reliability diagram identity–ECE: `results/beliefmove-evo/aggregated/rq11_distillation_reliability.svg`.

### Nhận xét

Calibration đúng objective cải thiện metric mục tiêu ở cả ba biến thể. NLL-optimal giảm NLL mạnh nhưng làm Brier và ECE xấu hơn, tái khẳng định NLL không đại diện đầy đủ cho top-1 calibration. Brier-optimal cho kết quả cân bằng hơn: nó đồng thời giảm Brier và giảm đáng kể ECE trên test.

Với GRU, temperature ECE được chọn trên validation cho ECE test 0.019430, trong khi Brier-optimal đạt ECE test thấp hơn là 0.014729. Đây không phải leakage hay lỗi: selection được khóa trên validation nên thứ tự hai temperature có thể đảo trên test.

Sau calibration theo metric tương ứng, GRU và Transformer đều tốt hơn `none` có ý nghĩa về NLL/Brier. Tuy nhiên, ECE-optimal của `none` lại thấp hơn GRU và Transformer trên test; vì thế distillation cải thiện scoring rules nhưng không tự động đảm bảo ECE thấp nhất.

## 3. Bayesian — all-prefix

| Variant | T-NLL | NLL id→opt | T-Brier | Brier id→opt | T-ECE | ECE id→opt |
|---|---:|---:|---:|---:|---:|---:|
| B0-static | 1.5000 | 7.130893→6.619977 | 1.1000 | 0.949882→0.946192 | 1.1000 | 0.042341→0.016184 |
| B3-dbn | 1.7500 | 7.487550→6.430621 | 1.5000 | 0.975386→0.942895 | 1.5000 | 0.142256→0.019308 |

### Trade-off trên test

| Variant | Objective | NLL | Brier | ECE | Adaptive ECE | Confidence gap |
|---|---|---:|---:|---:|---:|---:|
| B0-static | identity | 7.130893 | 0.949882 | 0.042341 | 0.042590 | -0.042130 |
| B0-static | nll | 6.619977 | 0.954934 | 0.069569 | 0.069368 | 0.069368 |
| B0-static | brier | 6.927140 | 0.946192 | 0.016184 | 0.016706 | -0.009596 |
| B0-static | ece | 6.927140 | 0.946192 | 0.016184 | 0.016706 | -0.009596 |
| B3-dbn | identity | 7.487550 | 0.975386 | 0.142256 | 0.142256 | -0.142256 |
| B3-dbn | nll | 6.430621 | 0.949091 | 0.060272 | 0.060056 | 0.060056 |
| B3-dbn | brier | 6.525716 | 0.942895 | 0.019308 | 0.018314 | 0.018314 |
| B3-dbn | ece | 6.525716 | 0.942895 | 0.019308 | 0.018314 | 0.018314 |

Reliability diagram identity–ECE: `results/beliefmove-evo/aggregated/rq11_bayesian_reliability.svg`.

### Nhận xét

B3-DBN ban đầu bị under-confident mạnh, thể hiện qua confidence gap -0.142256 và ECE 0.142256. Brier/ECE-optimal calibration giảm ECE xuống 0.019308 và Brier xuống 0.942895. So với identity, mức giảm ECE là 0.122948, tương đương khoảng 86.43%.

Sau calibration theo metric tương ứng, B3 tốt hơn B0 có ý nghĩa về NLL và Brier. Tuy nhiên, ECE của B3 vẫn cao hơn B0 0.003124; bootstrap CI không chứa 0. Kết quả cho thấy Bayesian update cải thiện scoring rules sau calibration nhưng B0 vẫn có lợi thế nhỏ về top-1 ECE.

## 4. Paired NLL/Brier tests

Positive effect nghĩa là biến thể/calibration đứng trước tốt hơn; dấu của NLL và Brier đã được quy đổi để giá trị dương biểu thị metric thấp hơn.

| Loại | Protocol | Comparison | Objective | Metric | Effect | 95% CI | Holm p | Significant |
|---|---|---|---|---|---:|---:|---:|---|
| calibration | distillation | none-vs-none | nll | nll | 1.346791 | 1.312654–1.379519 | 0.00159984 | yes |
| calibration | distillation | none-vs-none | brier | brier | 0.002373 | 0.002097–0.002645 | 0.00159984 | yes |
| calibration | distillation | gru-vs-gru | nll | nll | 0.512834 | 0.496066–0.529065 | 0.00159984 | yes |
| calibration | distillation | gru-vs-gru | brier | brier | 0.001861 | 0.001540–0.002171 | 0.00159984 | yes |
| calibration | distillation | transformer-vs-transformer | nll | nll | 0.801912 | 0.780576–0.823269 | 0.00159984 | yes |
| calibration | distillation | transformer-vs-transformer | brier | brier | 0.005139 | 0.004805–0.005471 | 0.00159984 | yes |
| model | distillation | gru-vs-none | nll | nll | 0.801413 | 0.785932–0.817421 | 0.00159984 | yes |
| model | distillation | gru-vs-none | brier | brier | 0.007501 | 0.006446–0.008561 | 0.00159984 | yes |
| model | distillation | transformer-vs-none | nll | nll | 0.703509 | 0.684558–0.721820 | 0.00159984 | yes |
| model | distillation | transformer-vs-none | brier | brier | 0.005446 | 0.004224–0.006671 | 0.00159984 | yes |
| calibration | bayesian | B0-static-vs-B0-static | nll | nll | 0.510916 | 0.504255–0.517654 | 0.00159984 | yes |
| calibration | bayesian | B0-static-vs-B0-static | brier | brier | 0.003690 | 0.003561–0.003820 | 0.00159984 | yes |
| calibration | bayesian | B3-dbn-vs-B3-dbn | nll | nll | 1.056929 | 1.047321–1.066580 | 0.00159984 | yes |
| calibration | bayesian | B3-dbn-vs-B3-dbn | brier | brier | 0.032490 | 0.031902–0.033076 | 0.00159984 | yes |
| model | bayesian | B3-dbn-vs-B0-static | nll | nll | 0.189356 | 0.187090–0.191619 | 0.00159984 | yes |
| model | bayesian | B3-dbn-vs-B0-static | brier | brier | 0.003297 | 0.003042–0.003543 | 0.00159984 | yes |

Toàn bộ calibration theo NLL/Brier đều cải thiện metric mục tiêu có ý nghĩa sau Holm correction. Các so sánh model cũng cho thấy GRU, Transformer và B3 tốt hơn đối chứng tương ứng về NLL/Brier sau khi mỗi model được calibrate theo cùng objective.

## 5. Bootstrap ECE tests

| Loại | Protocol | Comparison | Effect | 95% CI |
|---|---|---|---:|---:|
| calibration | distillation | none-vs-none | 0.020387 | 0.015088–0.022365 |
| calibration | distillation | gru-vs-gru | 0.010225 | 0.007918–0.011197 |
| calibration | distillation | transformer-vs-transformer | 0.037520 | 0.035044–0.037977 |
| model | distillation | gru-vs-none | -0.009004 | -0.011351–-0.005151 |
| model | distillation | transformer-vs-none | -0.005761 | -0.008163–-0.001667 |
| calibration | bayesian | B0-static-vs-B0-static | 0.026157 | 0.024895–0.027091 |
| calibration | bayesian | B3-dbn-vs-B3-dbn | 0.122948 | 0.120463–0.125089 |
| model | bayesian | B3-dbn-vs-B0-static | -0.003124 | -0.005201–-0.001262 |

ECE-optimal calibration cải thiện ECE cho cả năm biến thể và mọi CI đều không chứa 0. Trong so sánh trực tiếp sau ECE calibration, dấu âm cho biết `none` tốt hơn GRU/Transformer và B0 tốt hơn B3 về ECE.

## 6. Kết luận RQ11

Kết quả chứng minh calibration phải được đánh giá theo đúng objective. NLL-optimal temperature cải thiện NLL nhưng có thể làm Brier và ECE xấu đi; Brier/ECE-optimal temperature tạo calibration cân bằng hơn. Distillation và B3-DBN duy trì lợi thế rõ ràng về NLL/Brier sau calibration, nhưng không đạt ECE thấp nhất so với đối chứng.

Kết luận được phép sử dụng là: **validation-selected, objective-specific temperature scaling cải thiện nhất quán metric calibration mục tiêu; distillation và Bayesian update cải thiện proper scoring rules NLL/Brier, trong khi lợi thế về ECE phụ thuộc mô hình và objective.**

Không được tuyên bố một temperature duy nhất tối ưu mọi calibration metric hoặc distillation/Bayesian update luôn cải thiện ECE.

## 7. Hạn chế

- Temperature grid rời rạc; kết quả phản ánh các giá trị trong grid đã khai báo.
- ECE phụ thuộc cách chia bin; báo cáo kèm Adaptive ECE và reliability diagram để giảm phụ thuộc vào một cách binning.
- Temperature được chọn riêng theo seed trên validation; cột T là trung bình của ba seed.
- Distillation và Bayesian dùng hai protocol khác nhau, không so trực tiếp trị tuyệt đối.
- Kết quả chỉ áp dụng cho TIST2015–Tokyo, seed 42–44; chưa phải 12-city.

## 8. Publication gate

RQ11 đạt gate nội bộ cho Tokyo: đủ seed 42–44, đủ identity/NLL/Brier/ECE predictions, temperature chỉ fit validation, test đóng băng, paired NLL/Brier tests đã Holm-correct và ECE có paired bootstrap CI. Gate đa thành phố chưa hoàn thành.
