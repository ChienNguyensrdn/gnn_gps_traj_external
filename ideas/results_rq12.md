# RQ12 — Accuracy–Efficiency Trade-off

> **Trạng thái legacy:** các số dưới đây được sinh bởi protocol batch-256 cũ. Sau
> khi cập nhật RQ12, cần chạy lại `./scripts/rq12_efficiency.sh run-profiles` rồi
> `aggregate` để thay bằng báo cáo tách `batch-1`/`batch-256`, mean ± std và gate
> GPU contention. Không dùng bảng legacy này làm kết quả xuất bản cuối cùng.

> Neural/Bayesian latency được benchmark trên cùng hardware với warm-up và CUDA synchronization. LLM latency lấy từ live cache-generation của RQ8 và được ghi nhãn bounded.

## 1. Câu hỏi nghiên cứu và cách đọc kết quả

RQ12 kiểm tra trade-off giữa chất lượng dự đoán, latency, throughput, bộ nhớ, số tham số và chi phí gọi LLM.

- Neural dùng last-query protocol và batch size cố định của benchmark.
- Bayesian dùng all-prefix protocol; không so trực tiếp trị tuyệt đối với neural.
- `Mean/P50/P95 ms/q` của PyTorch là **amortized batch time trên mỗi query**, tức thời gian batch chia cho số query trong batch. Đây không phải single-request latency ở batch size 1.
- `Model s` và `Fusion s` là tổng thời gian qua toàn bộ repeat. Với B0, `Fusion s` bao gồm CPU transfer, softmax và post-processing dù không có Bayesian fusion thực sự.
- LLM latency là latency đã ghi khi tạo live Ollama cache trong RQ8 bounded limit 200, không phải cùng timing harness với PyTorch.

## 2. Neural — last-query

| Variant | R@1 | R@5 | R@10 | MRR | Mean ms/q | P50 ms/q | P95 ms/q | Query/s | GPU peak MB | RSS peak MB | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| teacher-gru | 0.143638 | 0.290502 | 0.344891 | 0.213183 | 0.0074 | 0.0072 | 0.0087 | 136780.21 | 159.9 | 770.4 | 10409646 |
| teacher-transformer | 0.150417 | 0.302077 | 0.362089 | 0.222782 | 0.0100 | 0.0094 | 0.0143 | 99609.52 | 173.8 | 852.7 | 10736174 |
| student-none | 0.133789 | 0.272528 | 0.325467 | 0.199728 | 0.0069 | 0.0068 | 0.0084 | 144052.32 | 151.4 | 769.0 | 8619774 |
| student-gru | 0.147485 | 0.306734 | 0.370851 | 0.223206 | 0.0068 | 0.0066 | 0.0081 | 148013.12 | 151.4 | 766.2 | 8619774 |
| student-transformer | 0.148623 | 0.308804 | 0.373680 | 0.224638 | 0.0069 | 0.0068 | 0.0081 | 144600.88 | 151.4 | 766.2 | 8619774 |

### Phân tích

Hai student distillation có cùng kiến trúc và số tham số 8,619,774 nên chi phí inference gần như giống nhau; teacher backbone chỉ ảnh hưởng quá trình huấn luyện. `student-gru` đạt throughput cao nhất 148,013 query/s, trong khi `student-transformer` đạt chất lượng ranking cao nhất trong ba student.

So với teacher GRU, student GRU:

- giảm khoảng 17.19% số tham số;
- giảm mean latency từ 0.0074 xuống 0.0068 ms/query;
- tăng R@1 từ 0.143638 lên 0.147485;
- tăng R@10 từ 0.344891 lên 0.370851;
- tăng MRR từ 0.213183 lên 0.223206.

So với teacher Transformer, student Transformer giảm khoảng 19.71% số tham số và khoảng 31% amortized mean latency. Student có R@5, R@10 và MRR cao hơn teacher, nhưng R@1 thấp hơn nhẹ: 0.148623 so với 0.150417.

Teacher Transformer có ranking tốt hơn teacher GRU nhưng mean latency cao hơn khoảng 35% và throughput thấp hơn khoảng 27%. Điều này củng cố lợi ích của distillation: student giữ hoặc cải thiện phần lớn chất lượng với kiến trúc inference nhỏ và nhanh hơn.

## 3. Bayesian — all-prefix

| Variant | R@1 | R@5 | R@10 | MRR | Mean ms/q | P95 ms/q | Query/s | Model s | Post-processing/Fusion s | GPU peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0-static | 0.141972 | 0.313228 | 0.383372 | 0.223297 | 0.3898 | 0.4238 | 2565.70 | 3.425 | 180.428 | 153.0 |
| B3-dbn | 0.148607 | 0.325217 | 0.396834 | 0.232260 | 1.7482 | 1.8957 | 572.16 | 3.620 | 822.642 | 153.0 |

### Phân tích

B3 cải thiện so với B0:

- R@1 tăng 0.006635, khoảng 4.67%;
- R@5 tăng 0.011989, khoảng 3.83%;
- R@10 tăng 0.013462, khoảng 3.51%;
- MRR tăng 0.008963, khoảng 4.01%.

Đổi lại, mean latency tăng từ 0.3898 lên 1.7482 ms/query, tương đương khoảng 4.49 lần, và throughput giảm từ 2,565.70 xuống 572.16 query/s. GPU peak không đổi vì chi phí tăng chủ yếu đến từ CPU post-processing và transition-prior fusion, không phải backbone GPU.

Phần model forward chỉ chiếm khoảng 3.4–3.6 giây tổng cộng, trong khi post-processing/fusion chiếm 180.4 giây cho B0 và 822.6 giây cho B3. Vì vậy, điểm nghẽn của Bayesian hiện nằm ở NumPy/CPU candidate-distribution processing. B3 vẫn đạt latency dưới 2 ms/query theo batch-amortized benchmark, nhưng cần vector hóa hoặc chuyển fusion sang tensor nếu triển khai throughput cao.

## 4. LLM routing — bounded Tokyo limit 200

> Latency source: recorded live Ollama cache-generation latency. Không so trực tiếp với neural/Bayesian như cùng timing harness.

| Policy | R@1 | R@5 | R@10 | MRR | Call rate | Mean latency s/q | P95 latency s/q | Tokens/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| never | 0.125000 | 0.250000 | 0.305000 | 0.186565 | 0.000000 | 0.000000 | 0.000000 | 0.00 |
| entropy | 0.125000 | 0.250000 | 0.305000 | 0.186982 | 0.145000 | 0.390842 | 2.601479 | 141.40 |
| always | 0.120000 | 0.280000 | 0.305000 | 0.193732 | 1.000000 | 2.822543 | 3.637879 | 955.88 |
| random-budget-matched | 0.124200 | 0.253500 | 0.305000 | 0.187044 | 0.145000 | 0.395059 | 2.614543 | 137.60 |

### Phân tích

Entropy routing giảm call rate từ 100% xuống 14.5%, mean latency từ 2.822543 xuống 0.390842 giây/query, tương đương giảm khoảng 86.15%, và tokens/query giảm khoảng 85.21%. Tuy nhiên, nó không cải thiện chất lượng rõ ràng so với Never hoặc Random-budget-matched trong RQ8. Vì vậy, router thể hiện lợi ích chi phí so với Always-LLM nhưng chưa chứng minh được lựa chọn query tốt hơn random ở cùng budget.

Always-LLM đạt MRR cao nhất trong nhóm routing nhưng có R@1 thấp hơn Never. Với experiment bounded 200 query, không nên diễn giải các khác biệt nhỏ thành kết luận tổng quát.

## 5. Kết luận RQ12

Trong neural last-query, student distillation nằm trên vùng Pareto tốt: chi phí inference thấp hơn teacher trong khi chất lượng ranking tương đương hoặc tốt hơn ở phần lớn metric. `student-transformer` phù hợp khi ưu tiên chất lượng tổng thể; `student-gru` có throughput cao nhất và chênh lệch chất lượng rất nhỏ.

Trong Bayesian all-prefix, B3 đổi khoảng 4.49 lần latency để đạt mức tăng 3.5–4.7% tương đối trên các metric ranking. Đây là trade-off có thật, và bottleneck chủ yếu ở CPU fusion.

Trong LLM routing bounded, Entropy giảm mạnh token và latency so với Always-LLM, nhưng chưa chứng minh được lợi thế chất lượng so với random-budget-matched. Do timing source và query protocol khác nhau, không được dùng tỷ lệ neural–LLM latency như một speedup publication chính thức.

Kết luận được phép sử dụng là: **distilled students cải thiện accuracy–efficiency trade-off so với heavy teachers trên matched last-query benchmark; B3 cải thiện quality với chi phí CPU fusion đáng kể; selective routing giảm mạnh chi phí so với Always-LLM trong bounded experiment.**

## 6. Protocol và giới hạn

- Neural last-query và Bayesian all-prefix được báo cáo riêng; không so trực tiếp latency/quality tuyệt đối giữa hai query protocol.
- Timing loại checkpoint loading, CSV loading, preprocessing và warm-up; có tính CPU→device transfer và online forward/fusion.
- Neural latency là amortized batch latency. Cần benchmark bổ sung batch size 1 nếu muốn claim latency phục vụ từng request.
- Bảng hiện chỉ trình bày mean; raw JSON có repeat/seed timing nhưng báo cáo publication nên bổ sung độ lệch chuẩn hoặc CI.
- Kết quả chỉ đáng tin như benchmark hardware-isolated nếu không có tiến trình khác dùng chung GPU/CPU. Cần ghi nhận trạng thái contention khi chạy.
- Offline teacher training và LLM cache construction chưa có timer chuẩn từ đầu nên ghi N/A, không suy diễn số liệu.
- RQ8 là bounded limit hữu hạn; latency của nó là recorded live Ollama latency, không phải cùng harness với PyTorch.
- Kết quả hiện chỉ áp dụng cho Tokyo và hardware ghi trong JSON, chưa phải 12-city hoặc cross-hardware.

## 7. Publication gate

RQ12 đạt gate nội bộ cho matched batch-throughput benchmark Tokyo nếu tất cả run dùng cùng hardware, batch size, warm-up và repeat như JSON xác nhận. Gate publication cho online latency vẫn cần batch-size-1 benchmark và xác nhận hardware không bị contention. Gate full-query LLM và đa thành phố chưa hoàn thành.
