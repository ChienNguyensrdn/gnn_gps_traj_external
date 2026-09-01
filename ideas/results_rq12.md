# RQ12 — Đánh đổi giữa độ chính xác và hiệu quả

> Batch-1 và batch-256 được báo cáo riêng. Neural/Bayesian dùng warm-up và CUDA synchronization; LLM latency lấy từ live cache-generation của RQ8 và được ghi nhãn bounded.

## 1. Batch-1 — độ trễ single-request

### Neural — last-query

> Timing dùng mẫu xác định 2.000/19.324 query; chất lượng lấy từ full frozen test. Timing, quality và memory là mean ± std qua ba seed 42–44.

| Variant | R@1 | R@5 | R@10 | MRR | Mean ms/q | P50 ms/q | P95 ms/q | Query/s | GPU peak MB | RSS peak MB | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| teacher-gru | 0.143638 ± 0.001971 | 0.290502 ± 0.002346 | 0.344891 ± 0.003259 | 0.213183 ± 0.001935 | 1.0049 ± 0.0606 | 0.9794 ± 0.0556 | 1.1537 ± 0.0790 | 997.50 ± 58.20 | 56.9 ± 0.0 | 755.2 ± 4.5 | 10409646 |
| teacher-transformer | 0.150417 ± 0.001609 | 0.302077 ± 0.001061 | 0.362089 ± 0.000987 | 0.222782 ± 0.000968 | 1.8412 ± 0.0294 | 1.7862 ± 0.0277 | 2.0951 ± 0.0491 | 543.21 ± 8.62 | 50.9 ± 0.0 | 877.4 ± 0.2 | 10736174 |
| student-none | 0.133789 ± 0.000470 | 0.272528 ± 0.003174 | 0.325467 ± 0.000617 | 0.199728 ± 0.001192 | 1.0227 ± 0.0065 | 0.9961 ± 0.0066 | 1.2025 ± 0.0594 | 977.86 ± 6.18 | 50.0 ± 0.0 | 743.2 ± 0.0 | 8619774 |
| student-gru | 0.147485 ± 0.002119 | 0.306734 ± 0.000957 | 0.370851 ± 0.002514 | 0.223206 ± 0.001778 | 1.0711 ± 0.1395 | 1.0478 ± 0.1316 | 1.2122 ± 0.1760 | 943.56 ± 114.38 | 50.0 ± 0.0 | 744.0 ± 0.4 | 8619774 |
| student-transformer | 0.148623 ± 0.002627 | 0.308804 ± 0.002957 | 0.373680 ± 0.003187 | 0.224638 ± 0.000803 | 1.0219 ± 0.0307 | 0.9985 ± 0.0309 | 1.1537 ± 0.0421 | 979.18 ± 28.96 | 50.0 ± 0.0 | 746.6 ± 4.3 | 8619774 |

### Bayesian — all-prefix

> Timing dùng mẫu xác định 2.000/94.587 query all-prefix; chất lượng lấy từ full frozen test. Không so trực tiếp trị tuyệt đối với Neural last-query.

| Variant | R@1 | R@5 | R@10 | MRR | Mean ms/q | P95 ms/q | Query/s | Model s | Post-processing/Fusion s | GPU peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0-static | 0.141972 ± 0.000794 | 0.313228 ± 0.000897 | 0.383372 ± 0.000863 | 0.223297 ± 0.000273 | 1.3272 ± 0.0214 | 1.5088 ± 0.0284 | 753.62 ± 12.18 | 8.958 ± 0.211 | 2.907 ± 0.026 | 50.0 ± 0.0 |
| B3-dbn | 0.148607 ± 0.000749 | 0.325217 ± 0.000270 | 0.396834 ± 0.000560 | 0.232260 ± 0.000460 | 2.7517 ± 0.1870 | 3.2672 ± 0.2152 | 364.50 ± 24.00 | 9.372 ± 1.276 | 16.632 ± 0.438 | 50.0 ± 0.0 |

## 2. Batch-256 — throughput

### Neural — last-query

> Timing chạy toàn bộ 19.324 query; chất lượng lấy từ full frozen test.

| Variant | R@1 | R@5 | R@10 | MRR | Mean ms/q | P50 ms/q | P95 ms/q | Query/s | GPU peak MB | RSS peak MB | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| teacher-gru | 0.143638 ± 0.001971 | 0.290502 ± 0.002346 | 0.344891 ± 0.003259 | 0.213183 ± 0.001935 | 0.0075 ± 0.0008 | 0.0073 ± 0.0008 | 0.0088 ± 0.0010 | 135255.31 ± 14188.73 | 159.9 ± 0.0 | 775.7 ± 0.4 | 10409646 |
| teacher-transformer | 0.150417 ± 0.001609 | 0.302077 ± 0.001061 | 0.362089 ± 0.000987 | 0.222782 ± 0.000968 | 0.0100 ± 0.0001 | 0.0095 ± 0.0001 | 0.0143 ± 0.0001 | 99635.11 ± 776.08 | 173.8 ± 0.0 | 853.2 ± 0.4 | 10736174 |
| student-none | 0.133789 ± 0.000470 | 0.272528 ± 0.003174 | 0.325467 ± 0.000617 | 0.199728 ± 0.001192 | 0.0069 ± 0.0003 | 0.0068 ± 0.0003 | 0.0082 ± 0.0003 | 144102.37 ± 5365.19 | 151.4 ± 0.0 | 761.5 ± 0.4 | 8619774 |
| student-gru | 0.147485 ± 0.002119 | 0.306734 ± 0.000957 | 0.370851 ± 0.002514 | 0.223206 ± 0.001778 | 0.0066 ± 0.0001 | 0.0065 ± 0.0001 | 0.0079 ± 0.0002 | 150472.14 ± 2126.66 | 151.4 ± 0.0 | 760.8 ± 0.4 | 8619774 |
| student-transformer | 0.148623 ± 0.002627 | 0.308804 ± 0.002957 | 0.373680 ± 0.003187 | 0.224638 ± 0.000803 | 0.0067 ± 0.0003 | 0.0066 ± 0.0003 | 0.0081 ± 0.0005 | 148488.21 ± 6031.71 | 151.4 ± 0.0 | 761.5 ± 0.3 | 8619774 |

### Bayesian — all-prefix

> Timing chạy toàn bộ 94.587 query all-prefix; không so trực tiếp trị tuyệt đối với Neural last-query.

| Variant | R@1 | R@5 | R@10 | MRR | Mean ms/q | P95 ms/q | Query/s | Model s | Post-processing/Fusion s | GPU peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0-static | 0.141972 ± 0.000794 | 0.313228 ± 0.000897 | 0.383372 ± 0.000863 | 0.223297 ± 0.000273 | 0.3925 ± 0.0022 | 0.4282 ± 0.0020 | 2547.82 ± 14.61 | 3.655 ± 0.343 | 181.453 ± 0.902 | 153.0 ± 0.0 |
| B3-dbn | 0.148607 ± 0.000749 | 0.325217 ± 0.000270 | 0.396834 ± 0.000560 | 0.232260 ± 0.000460 | 1.7248 ± 0.0121 | 1.8722 ± 0.0122 | 579.79 ± 4.08 | 3.548 ± 0.197 | 811.680 ± 5.537 | 153.0 ± 0.0 |

## 3. LLM routing — bounded Tokyo limit=200

> Latency lấy từ live Ollama cache-generation, không cùng timing harness với PyTorch.

| Policy | R@1 | R@5 | R@10 | MRR | Call rate | Mean latency s/q | P95 latency s/q | Tokens/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| never | 0.125000 | 0.250000 | 0.305000 | 0.186565 | 0.000000 | 0.000000 | 0.000000 | 0.00 |
| entropy | 0.125000 | 0.250000 | 0.305000 | 0.186982 | 0.145000 | 0.390842 | 2.601479 | 141.40 |
| always | 0.120000 | 0.280000 | 0.305000 | 0.193732 | 1.000000 | 2.822543 | 3.637879 | 955.88 |
| random-budget-matched | 0.124200 | 0.253500 | 0.305000 | 0.187044 | 0.145000 | 0.395059 | 2.614543 | 137.60 |

## 4. Phân tích kết quả

### Hiệu quả của distillation

Hai student distillation có 8.619.774 tham số, ít hơn teacher GRU khoảng 17,2% và teacher Transformer khoảng 19,7%. `student-transformer` đạt chất lượng tốt nhất trong nhóm student với R@10 = 0,373680 và MRR = 0,224638. `student-gru` đạt throughput batch-256 cao nhất, khoảng 150.472 query/s. Điều này cho thấy distillation cải thiện chất lượng so với `student-none` mà không tăng kích thước hoặc chi phí inference của student.

Ở single-request, các student nằm quanh 1 ms/query. Teacher Transformer chậm nhất, 1,8412 ms/query, trong khi teacher GRU và các student gần nhau hơn. Vì `student-gru` và `student-transformer` có cùng kiến trúc inference, khác biệt latency nhỏ giữa chúng chủ yếu phản ánh nhiễu hệ thống thay vì khác biệt mô hình.

### Chi phí của Bayesian belief

So với B0-static, B3-dbn tăng R@1 từ 0,141972 lên 0,148607, R@10 từ 0,383372 lên 0,396834 và MRR từ 0,223297 lên 0,232260. Đổi lại, single-request latency tăng từ 1,3272 lên 2,7517 ms/query, còn throughput batch-256 giảm từ khoảng 2.548 xuống 580 query/s.

Nút thắt chính không nằm ở neural forward mà ở `Post-processing/Fusion`: với batch-256, B3-dbn dùng khoảng 811,680 giây qua năm repeat, so với 181,453 giây của B0-static. Vì vậy, nếu triển khai production, tối ưu vector hóa hoặc chuyển Bayesian fusion khỏi vòng lặp Python là ưu tiên quan trọng.

### Chi phí của LLM

LLM routing chậm hơn neural/Bayesian nhiều bậc độ lớn. Policy `entropy` giảm call rate xuống 14,5% và còn khoảng 141 token/query, nhưng không cải thiện R@1/R@5 so với `never` trong bounded experiment này. `always` tăng MRR và R@5 nhưng tốn khoảng 2,82 giây/query. Do RQ8 chỉ có 200 query, các con số này chỉ minh họa trade-off chi phí và chưa đủ để kết luận tổng quát.

## 5. Kết luận RQ12

- Distillation tạo điểm vận hành tốt nhất: student nhỏ hơn teacher, inference nhanh và chất lượng cao hơn student chỉ học cross-entropy.
- `student-gru` phù hợp khi ưu tiên throughput; `student-transformer` phù hợp khi ưu tiên chất lượng ranking, dù chênh lệch giữa hai student nhỏ.
- B3-dbn cải thiện chất lượng nhưng có overhead CPU/fusion đáng kể; cần tối ưu implementation trước khi triển khai quy mô lớn.
- LLM chỉ nên được gọi có chọn lọc, nhưng entropy router hiện chưa chứng minh được lợi ích chất lượng trên bounded RQ8.

## 6. Protocol gate và giới hạn

- Batch-1 đo single-request latency trên mẫu query xác định; batch-256 đo throughput trên toàn bộ test query.
- Neural last-query và Bayesian all-prefix được báo cáo riêng, không so trực tiếp chất lượng/latency tuyệt đối giữa hai protocol.
- Timing loại checkpoint loading, CSV loading, preprocessing và warm-up; có tính CPU→device transfer và online forward/post-processing/fusion.
- Aggregate mặc định từ chối run có GPU process ngoại lai. Nếu summary có `gpu_contention_allowed=true`, kết quả chỉ là provisional và không dùng làm số publication cuối cùng.
- Offline teacher training và LLM cache construction ghi N/A vì chưa có timer chuẩn từ đầu.
- RQ8 là bounded limit hữu hạn và không cùng timing harness với PyTorch.
- Kết quả hiện chỉ áp dụng cho Tokyo, ba seed 42–44 và hardware đã ghi trong JSON; chưa phải kết quả 12-city hoặc cross-hardware.
