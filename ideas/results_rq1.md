# RQ1 — Baseline reproducibility

> Báo cáo phân biệt quantitative baseline trên Tokyo matched full-test và baseline TIST2015 bounded 12-city. Hai protocol khác query/scope nên không so sánh trực tiếp trị tuyệt đối hoặc tính paired significance giữa chúng.

## 1. Câu hỏi nghiên cứu

RQ1 kiểm tra liệu các baseline định lượng và AgentMove có được tái lập ổn định trên preprocessing, temporal split và candidate space đã khóa hay không. Quantitative baseline được chạy với seed 42–44; Markov deterministic và AgentMove bounded được tổng hợp riêng trên đúng 12 thành phố TIST2015.

## 2. Quantitative baselines — Tokyo matched full-test

Seeds: 42, 43, 44. Checkpoint được chọn bằng validation; test chỉ dùng báo cáo cuối.

| Baseline | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher-gru | 0.143638 ± 0.001971 | 0.290502 ± 0.002346 | 0.344891 ± 0.003259 | 0.213183 ± 0.001935 | 8.799283 ± 0.278926 | 0.948259 ± 0.002997 | 0.035296 ± 0.013116 |
| teacher-transformer | 0.150417 ± 0.001609 | 0.302077 ± 0.001061 | 0.362089 ± 0.000987 | 0.222782 ± 0.000968 | 9.466769 ± 0.151860 | 0.949876 ± 0.003325 | 0.070451 ± 0.011092 |
| student-ce | 0.133789 ± 0.000470 | 0.272528 ± 0.003174 | 0.325467 ± 0.000617 | 0.199728 ± 0.001192 | 9.210952 ± 0.017885 | 0.952485 ± 0.000825 | 0.030813 ± 0.003537 |

### Phân tích quantitative baseline

Transformer teacher đạt ranking tốt nhất: R@1 = 0,150417, R@10 = 0,362089 và MRR = 0,222782. So với GRU teacher, mức tăng mô tả lần lượt khoảng 0,006779 R@1, 0,017198 R@10 và 0,009599 MRR.

Tuy nhiên, Transformer không tốt nhất về probabilistic calibration. NLL = 9,466769 và ECE = 0,070451 đều xấu hơn GRU. GRU có NLL tốt nhất (8,799283), còn student-CE có ECE thấp nhất (0,030813). Vì vậy không thể chọn backbone chỉ dựa trên Recall/MRR nếu downstream Bayesian inference cần xác suất tin cậy.

Student-CE thấp hơn cả hai teacher về ranking, xác nhận một lightweight student chỉ học cross-entropy chưa tái lập được knowledge của heavy teacher. Độ lệch chuẩn nhỏ qua ba seed cho thấy thứ hạng tổng quát Transformer teacher > GRU teacher > student-CE ổn định trong protocol Tokyo này.

## 3. TIST2015 bounded baseline — 12 thành phố

| Baseline | Status | Cities | Limit/city | Acc@1 | Acc@5 | Acc@10 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| Markov/Bi-gram | ready-bounded | 12 | 200 | 0.112147 | 0.210307 | 0.242303 | 0.158516 |
| AgentMove original | ready-bounded | 12 | 200 | 0.134024 | 0.334174 | N/A | 0.205829 |

### Phân tích bounded baseline

Trên macro average 12 thành phố, AgentMove cao hơn Markov về các metric chung:

- Acc@1 cao hơn 0,021877 điểm tuyệt đối.
- Acc@5 cao hơn 0,123867 điểm tuyệt đối.
- MRR cao hơn 0,047313 điểm tuyệt đối.

Đây chỉ là chênh lệch macro mô tả. Báo cáo không gọi đây là paired improvement vì Markov và AgentMove chưa lưu per-query predictions trong cùng paired statistical harness. AgentMove chỉ trả năm prediction nên Acc@10 là N/A; không nội suy hoặc đồng nhất Acc@5 với Acc@10.

Markov/Bi-gram là deterministic baseline nên một run là đủ cho cùng input/protocol; không tạo pseudo-seed để sinh standard deviation giả. AgentMove sử dụng Qwen2:7b qua Ollama và được ghi nhãn `no-OSM matched`, do đó chưa đại diện cho hàng “Ours/full world knowledge”.

## 4. Quan hệ giữa hai protocol

Không so trực tiếp các số Tokyo full-test với macro bounded 12-city vì chúng khác nhau ở:

- phạm vi thành phố;
- số query và selection protocol;
- candidate/prediction count;
- loại mô hình và stochasticity;
- metric availability.

Quantitative Tokyo trả lời tính ổn định qua seed trên một matched neural protocol. TIST2015 bounded trả lời khả năng tái lập Markov và AgentMove trên cùng giới hạn 200 query mỗi thành phố. Hai phần bổ sung cho nhau nhưng không tạo thành một bảng xếp hạng duy nhất.

## 5. Kết luận RQ1

- Quantitative baseline được tái lập đầy đủ trên Tokyo với seed 42–44 và biến thiên nhỏ.
- Transformer teacher tốt nhất về Recall/MRR nhưng calibration kém hơn GRU và student-CE.
- AgentMove bounded có macro Acc@1/Acc@5/MRR cao hơn Markov trên 12 thành phố, nhưng chưa có paired significance.
- Cả Markov và AgentMove đã hoàn thành đúng 12 thành phố ở limit 200.
- RQ1 được khóa dưới dạng hai protocol báo cáo riêng, không dùng số liệu cross-protocol để claim superiority.

## 6. Publication gate và giới hạn

- Quantitative matched multi-seed: **ready**.
- Markov bounded 12-city: **ready-bounded**.
- AgentMove bounded 12-city: **ready-bounded**.
- Gate tổng: **ready-separated-protocols**.
- Không pseudo-replicate Markov deterministic theo seed.
- Không tính paired delta giữa Tokyo full-test và bounded 12-city.
- AgentMove Acc@10 là N/A vì prediction count bằng 5.
- Baseline bounded limit 200 không được gọi là full-query result.
- Việc kiểm tra RQ1 được thực hiện hồi cứu sau các RQ sau và phải được mô tả trung thực trong manuscript.
