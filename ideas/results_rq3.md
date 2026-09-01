# RQ3 — LLM knowledge distillation

> Trọng số fusion được fit trên validation; quality và paired tests được báo cáo trên bounded matched test gồm 200 query tại Tokyo. Đây là frozen belief-fusion ablation của teacher signals, không phải một lượt huấn luyện lại neural student.

## 1. Câu hỏi nghiên cứu

RQ3 kiểm tra structured mobility beliefs từ LLM có giúp Bayesian student tốt hơn data-only hay không, đồng thời tách đóng góp của quantitative teacher và LLM teacher:

- `M1-data-only`: BN empirical prior theo user và target-time.
- `M2-llm`: M1 kết hợp structured LLM habit/semantic likelihood.
- `M3-quantitative`: M1 kết hợp phân phối từ quantitative teacher.
- `M4-both`: M1 kết hợp cả quantitative teacher và LLM teacher.

Các nguồn được kết hợp bằng log-linear fusion. Trọng số được chọn theo `R@1 + R@10` trên validation; test không tham gia lựa chọn.

## 2. Kết quả test

| Variant | q-weight | LLM-weight | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1-data-only | 0.00 | 0.00 | 0.035000 | 0.115000 | 0.135000 | 0.074670 | 9.091769 | 0.997919 | 0.021684 |
| M2-llm | 0.00 | 1.00 | 0.055000 | 0.120000 | 0.155000 | 0.087913 | 9.041363 | 0.995740 | 0.039496 |
| M3-quantitative | 1.00 | 0.00 | 0.115000 | 0.215000 | 0.240000 | 0.161094 | 8.616508 | 0.994605 | 0.102825 |
| M4-both | 1.00 | 0.75 | 0.125000 | 0.220000 | 0.230000 | 0.169674 | 8.583706 | 0.989478 | 0.116238 |

## 3. Paired comparisons

Positive effect nghĩa là variant đứng trước tốt hơn. Với NLL và Brier, dấu đã được đảo để giá trị dương luôn mang nghĩa tốt hơn. Holm correction được áp dụng chung cho toàn bộ 30 phép kiểm định.

| Comparison | Metric | Effect | 95% CI | Holm p | Significant |
|---|---|---:|---:|---:|---|
| M2-llm-vs-M1-data-only | recall@1 | 0.020000 | -0.010000–0.050000 | 1 | no |
| M2-llm-vs-M1-data-only | recall@5 | 0.005000 | -0.010000–0.025000 | 1 | no |
| M2-llm-vs-M1-data-only | recall@10 | 0.020000 | 0.005000–0.040000 | 1 | no |
| M2-llm-vs-M1-data-only | mrr | 0.013243 | -0.003912–0.032036 | 1 | no |
| M2-llm-vs-M1-data-only | nll | 0.050408 | -0.000746–0.105305 | 0.992901 | no |
| M2-llm-vs-M1-data-only | brier | 0.002179 | 0.000491–0.004245 | 0.358164 | no |
| M3-quantitative-vs-M1-data-only | recall@1 | 0.080000 | 0.045000–0.120000 | 0.0029997 | yes |
| M3-quantitative-vs-M1-data-only | recall@5 | 0.100000 | 0.055000–0.145000 | 0.00569943 | yes |
| M3-quantitative-vs-M1-data-only | recall@10 | 0.105000 | 0.065000–0.150000 | 0.0029997 | yes |
| M3-quantitative-vs-M1-data-only | mrr | 0.086424 | 0.054666–0.121339 | 0.0029997 | yes |
| M3-quantitative-vs-M1-data-only | nll | 0.475262 | -0.113902–1.059787 | 1 | no |
| M3-quantitative-vs-M1-data-only | brier | 0.003315 | -0.016848–0.024844 | 1 | no |
| M4-both-vs-M1-data-only | recall@1 | 0.090000 | 0.050000–0.130000 | 0.0029997 | yes |
| M4-both-vs-M1-data-only | recall@5 | 0.105000 | 0.060000–0.150000 | 0.0029997 | yes |
| M4-both-vs-M1-data-only | recall@10 | 0.095000 | 0.055000–0.135000 | 0.0029997 | yes |
| M4-both-vs-M1-data-only | mrr | 0.095004 | 0.062511–0.130239 | 0.0029997 | yes |
| M4-both-vs-M1-data-only | nll | 0.508063 | -0.096557–1.100106 | 1 | no |
| M4-both-vs-M1-data-only | brier | 0.008441 | -0.015192–0.033960 | 1 | no |
| M4-both-vs-M2-llm | recall@1 | 0.070000 | 0.035000–0.110000 | 0.0029997 | yes |
| M4-both-vs-M2-llm | recall@5 | 0.100000 | 0.060000–0.140000 | 0.0029997 | yes |
| M4-both-vs-M2-llm | recall@10 | 0.075000 | 0.040000–0.110000 | 0.0029997 | yes |
| M4-both-vs-M2-llm | mrr | 0.081761 | 0.051697–0.114935 | 0.0029997 | yes |
| M4-both-vs-M2-llm | nll | 0.457656 | -0.143248–1.040233 | 1 | no |
| M4-both-vs-M2-llm | brier | 0.006262 | -0.016695–0.031249 | 1 | no |
| M4-both-vs-M3-quantitative | recall@1 | 0.010000 | 0.000000–0.025000 | 1 | no |
| M4-both-vs-M3-quantitative | recall@5 | 0.005000 | 0.000000–0.015000 | 1 | no |
| M4-both-vs-M3-quantitative | recall@10 | -0.010000 | -0.025000–0.000000 | 1 | no |
| M4-both-vs-M3-quantitative | mrr | 0.008580 | 0.000972–0.018340 | 0.870313 | no |
| M4-both-vs-M3-quantitative | nll | 0.032801 | 0.002229–0.064457 | 0.746225 | no |
| M4-both-vs-M3-quantitative | brier | 0.005126 | -0.004063–0.014255 | 1 | no |

## 4. Phân tích kết quả

### 4.1. Quantitative teacher là nguồn gain chính

M3 tăng R@1 từ 0,035 lên 0,115, R@10 từ 0,135 lên 0,240 và MRR từ 0,074670 lên 0,161094 so với M1. Các gain Recall@K/MRR đều có ý nghĩa sau Holm correction. Điều này xác nhận learned quantitative mobility pattern bổ sung lượng thông tin lớn mà empirical user/time BN không nắm được.

NLL và Brier của M3 tốt hơn M1 về giá trị trung bình, nhưng paired confidence intervals cắt 0 và Holm p = 1. Vì vậy chỉ có thể khóa kết luận về ranking, chưa thể khẳng định quantitative teacher cải thiện probabilistic quality trong bounded sample này.

### 4.2. LLM-only có tín hiệu dương nhưng chưa đạt significance

M2 cao hơn M1 về tất cả Recall@K/MRR theo điểm ước lượng: R@1 tăng 0,020, R@10 tăng 0,020 và MRR tăng 0,013243. Tuy vậy không phép so sánh M2–M1 nào còn significant sau Holm correction. Một số bootstrap CI như R@10 và Brier nằm trên 0, nhưng adjusted p vẫn không đạt ngưỡng; báo cáo dùng gate bảo thủ và không claim LLM knowledge hữu ích độc lập.

Theo tiêu chí đã định nghĩa trong `ideas/pipline.md`, kết quả hiện tại chưa đủ để khẳng định M2 tốt hơn M1. Cần tăng số query, mở rộng thành phố hoặc cải thiện structured evidence trước khi đưa ra claim này.

### 4.3. M4 tốt nhất ở R@1/MRR nhưng chưa chứng minh gain bổ sung của LLM

M4 đạt R@1 = 0,125 và MRR = 0,169674, cao nhất trong bốn variant. M4 cũng có NLL và Brier trung bình thấp nhất. Tuy nhiên, so sánh quyết định cho giá trị tăng thêm của LLM là M4–M3, và không metric nào significant sau Holm correction.

M4 tăng R@1 thêm 0,010 và MRR thêm 0,008580 so với M3, nhưng làm R@10 giảm 0,010. Vì vậy chưa thể kết luận cả hai teacher tốt hơn quantitative teacher đơn lẻ. Trọng số LLM = 0,75 được chọn trên validation chỉ cho thấy validation objective ưu tiên evidence này; nó không thay thế kiểm định trên test.

### 4.4. M4 vượt M2 chủ yếu phản ánh quantitative teacher

M4 vượt M2 có ý nghĩa trên toàn bộ Recall@K/MRR. Vì M4 khác M2 ở việc thêm quantitative teacher, kết quả này củng cố kết luận quantitative teacher hữu ích. Nó không phải bằng chứng riêng cho LLM, do cả hai phía đều đã chứa LLM signal.

### 4.5. Calibration không đi cùng ranking

M1 có ECE thấp nhất (0,021684), trong khi M3 và M4 cải thiện mạnh ranking nhưng ECE tăng lên 0,102825 và 0,116238. Không có paired ECE test vì ECE là aggregate metric. Không nên diễn giải M1 là mô hình tổng thể tốt hơn chỉ nhờ ECE thấp, cũng không nên gọi M4 calibrated dù NLL/Brier trung bình giảm.

## 5. Trả lời RQ3

- Structured LLM beliefs tạo gain mô tả khi thêm vào BN data-only, nhưng chưa có bằng chứng significance sau multiple-testing correction.
- Quantitative teacher cải thiện rõ và có ý nghĩa thống kê trên toàn bộ Recall@K/MRR.
- Kết hợp cả hai teacher cho R@1/MRR tốt nhất theo điểm ước lượng, nhưng chưa tốt hơn M3 có ý nghĩa; do đó chưa chứng minh được incremental value của LLM khi quantitative teacher đã có mặt.
- Claim hợp lệ hiện tại là “quantitative teacher hữu ích”; claim “LLM knowledge distillation hữu ích” vẫn **chưa được xác nhận** trong bounded RQ3.

## 6. Protocol gate và giới hạn

- Gate kỹ thuật: **ready-bounded-matched**.
- Prior data-only chỉ fit train; fusion weights chỉ chọn bằng validation; test không dùng để tuning.
- Structured LLM evidence được replay từ immutable Qwen2:7b cache; evaluation không gọi lại LLM.
- Cache phủ mọi query; evidence ngoài cached quantitative top-k dùng likelihood trung tính.
- Các variant deterministic chỉ có một paired run, không pseudo-replicate theo seed.
- Cỡ mẫu chỉ 200 test query khiến power thấp sau Holm correction 30 phép kiểm định.
- Đây là frozen belief-fusion ablation, chưa phải end-to-end neural student distillation.
- Kết quả chỉ áp dụng cho Tokyo, Qwen2:7b, limit 200 và `no-OSM`; không được gọi là full-query, full-world-knowledge hoặc 12-city result.
