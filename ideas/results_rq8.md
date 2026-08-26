# RQ8 — Phân tích Uncertainty-aware LLM Routing

## 1. Câu hỏi nghiên cứu và protocol

RQ8 kiểm tra liệu có thể giảm số lần gọi LLM nhưng vẫn giữ chất lượng dự đoán
gần với chính sách gọi LLM cho mọi query hay không.

Đây là **bounded experiment** trên 200 test queries của `TIST2015–Tokyo`, dùng
`qwen2:7b`. Threshold được fit trên validation; test không dùng để tuning. Các
router dùng chung một Always-LLM cache nên được so sánh trên cùng query và cùng
LLM output.

Các policy gồm:

- **Never:** không gọi LLM.
- **Always:** gọi LLM trên mọi query để rerank top-10 candidates.
- **Entropy:** gọi khi entropy vượt threshold validation.
- **Margin:** gọi khi top-2 margin thấp hơn threshold validation.
- **Random-budget-matched:** random control có cùng call rate với Entropy.

Never, Always, Entropy và Margin là deterministic và chỉ được tính một run.
Random được đánh giá bằng 50 permutation seeds; không dùng các bản sao
deterministic để tạo pseudo-replication.

## 2. Kết quả tại primary budget

| Router | Runs | R@1 | R@5 | R@10 | MRR | LLM call rate | Latency mean (s) | Latency p95 (s) | Tokens/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Never | 1 | 0.125000 | 0.250000 | 0.305000 | 0.186565 | 0.000000 | 0.000000 | 0.000000 | 0.00 |
| Always | 1 | 0.120000 | 0.280000 | 0.305000 | 0.193732 | 1.000000 | 2.822543 | 3.637879 | 955.88 |
| Entropy | 1 | 0.125000 | 0.250000 | 0.305000 | 0.186982 | 0.145000 | 0.390842 | 2.601479 | 141.40 |
| Margin | 1 | 0.125000 | 0.250000 | 0.305000 | 0.186565 | 0.000000 | 0.000000 | 0.000000 | 0.00 |
| Random-budget-matched | 50 | 0.124200 | 0.253500 | 0.305000 | 0.187044 | 0.145000 | 0.395059 | 2.614543 | 137.60 |

## 3. Threshold được chọn trên validation

| Router | Threshold | Validation call rate |
|---|---:|---:|
| Entropy | 8.9535921 | 0.150000 |
| Margin | 1.3394625e-06 | 0.000000 |

Entropy dùng 15% call rate trên validation và 14.5% trên test, cho thấy budget
được chuyển sang test tương đối ổn định. Margin bị validation loại bỏ hoàn toàn
và rút gọn về Never.

Giá trị entropy tuyệt đối khá lớn vì phân phối trải trên candidate space của
Tokyo. Khi mở rộng nhiều thành phố có số POI khác nhau, nên dùng normalized
entropy `H(p)/log(K)` hoặc fit threshold riêng cho từng city.

## 4. Phân tích quality và chi phí

### 4.1. Always-LLM không cải thiện nhất quán

So với Never, Always tăng Recall@5 từ `0.250` lên `0.280` và MRR từ `0.186565`
lên `0.193732`. Tuy nhiên Recall@1 giảm từ `0.125` xuống `0.120`, còn Recall@10
không đổi ở `0.305`.

Recall@10 không đổi vì LLM chỉ rerank top-10 do predictor cung cấp. Nếu
ground-truth không nằm trong candidate set, reranking không thể cải thiện
candidate recall.

Sau Holm correction, không metric nào của Always khác Never có ý nghĩa thống
kê. Recall@5 có effect `+0.030` và bootstrap CI không chứa 0, nhưng Holm
`p = 0.49645`; do đó chưa đủ bằng chứng cho claim Always tốt hơn Never.

### 4.2. Entropy tiết kiệm chi phí nhưng không tăng chất lượng

Entropy chỉ gọi LLM trên 14.5% test queries. So với Always:

- số lần gọi LLM giảm **85.5%**;
- tokens/query giảm từ `955.88` xuống `141.40`, khoảng **85.2%**;
- latency trung bình giảm từ `2.822543` xuống `0.390842` giây, khoảng **86.1%**.

Tuy nhiên, Entropy và Never có Recall@1/5/10 giống hệt nhau. MRR chỉ tăng
`0.000417`, với CI `0–0.001250` và Holm `p = 1`. Entropy vì vậy giảm chi phí
nhưng không tạo cải thiện quality có ý nghĩa thống kê.

### 4.3. Random control tốt hơn Entropy tại Recall@5

Entropy và Random có cùng call rate `0.145`. Trung bình qua 50 random
permutations:

| Metric | Entropy | Random | Entropy − Random | Holm p |
|---|---:|---:|---:|---:|
| Recall@1 | 0.125000 | 0.124200 | +0.000800 | 1 |
| Recall@5 | 0.250000 | 0.253500 | **−0.003500** | **0.00159984** |
| Recall@10 | 0.305000 | 0.305000 | 0.000000 | 1 |
| MRR | 0.186982 | 0.187044 | −0.000062 | 1 |

Tại Recall@5, CI của effect Entropy–Random là
`−0.004700 – −0.002400`; Random tốt hơn Entropy có ý nghĩa thống kê sau Holm
correction. Các metric còn lại không khác biệt có ý nghĩa.

Kết quả này bác bỏ giả thuyết rằng entropy hiện tại định tuyến LLM tốt hơn lựa
chọn ngẫu nhiên cùng budget.

## 5. Budget sweep

| Budget tối đa | Router | R@1 | R@5 | MRR | Call rate thực tế | Tokens/query |
|---:|---|---:|---:|---:|---:|---:|
| 0.10 | Entropy | 0.125000 | 0.250000 | 0.186565 | 0.000000 | 0.00 |
| 0.10 | Margin | 0.125000 | 0.250000 | 0.186565 | 0.000000 | 0.00 |
| 0.25 | Entropy | 0.125000 | 0.250000 | 0.186982 | 0.145000 | 141.40 |
| 0.25 | Margin | 0.125000 | 0.250000 | 0.186565 | 0.000000 | 0.00 |
| 0.50 | Entropy | 0.125000 | 0.250000 | 0.186982 | 0.145000 | 141.40 |
| 0.50 | Margin | 0.125000 | 0.250000 | 0.186565 | 0.000000 | 0.00 |

Ở budget 10%, validation chọn không gọi LLM. Tăng budget từ 25% lên 50% không
làm call rate vượt 14.5%, vì validation không tìm thấy thêm vùng uncertainty có
lợi theo objective hiện tại. Do đó vấn đề không phải thiếu budget mà là score
routing chưa dự báo được lợi ích thực tế của LLM.

## 6. Oracle-gain upper bound

Oracle chỉ gọi LLM trên query có positive realized LLM gain và đạt:

| Policy | R@1 | R@5 | MRR | Call rate |
|---|---:|---:|---:|---:|
| Never | 0.125000 | 0.250000 | 0.186565 | 0.000000 |
| Entropy | 0.125000 | 0.250000 | 0.186982 | 0.145000 |
| Oracle | **0.165000** | **0.280000** | **0.220302** | **0.065000** |

Chỉ 6.5% queries có positive realized gain đủ để Oracle tăng Recall@1 thêm
`0.040`, Recall@5 thêm `0.030` và MRR thêm `0.033737` so với Never.

Khoảng cách lớn giữa Oracle và Entropy chứng minh LLM có ích trên một tập query
nhỏ, nhưng entropy/margin không nhận diện được tập đó. Đây là bằng chứng ủng hộ
phát triển **gain-aware router**, không phải tăng call budget một cách cơ học.

Oracle chỉ là upper bound hậu nghiệm vì dùng test outcome để biết query nào có
gain; nó không phải policy triển khai và không được báo cáo như mô hình thực tế.

## 7. Tóm tắt paired significance

- Entropy vs Never: không metric nào significant.
- Entropy vs Always: không metric nào significant sau Holm.
- Always vs Never: không metric nào significant sau Holm.
- Entropy vs Random: Random tốt hơn có ý nghĩa tại Recall@5; các metric khác
  không significant.

Holm correction được áp dụng trên toàn bộ 16 phép kiểm định. Positive effect
nghĩa là policy đứng trước tốt hơn policy đứng sau.

## 8. Kết luận RQ8

Claim an toàn:

> Trong bounded experiment trên 200 queries của TIST2015–Tokyo, entropy routing
> giảm khoảng 85% chi phí LLM nhưng không cải thiện có ý nghĩa chất lượng so với
> Never-LLM và kém Random routing tại Recall@5. Oracle analysis cho thấy một
> gain-aware router có tiềm năng cải thiện đáng kể chất lượng chỉ với 6.5% LLM
> call rate.

Kết quả hiện tại là negative result cho Entropy và Margin, nhưng là positive
diagnostic cho hướng gain-aware routing. Không nên claim uncertainty routing
hiện tại tốt hơn random, Always tốt hơn Never, hoặc kết quả đại diện cho toàn bộ
Tokyo/12-city.

## 9. Hướng phát triển

1. Học trực tiếp xác suất `LLM gain > 0` trên validation thay vì dùng entropy
   tuyệt đối.
2. Kết hợp entropy, margin, trajectory length, candidate coverage,
   DBN–predictor disagreement và calibration confidence.
3. Đánh giá gain-aware router trên cùng Always cache trước khi gọi thêm LLM.
4. Chỉ mở rộng full-query và multi-city sau khi router vượt matched Random trên
   validation/test protocol đã khóa.

## 10. Publication gate

- City: `Tokyo` — hoàn thành bounded run.
- Test queries: `200` — chưa phải full-query.
- LLM: `qwen2:7b`.
- Validation-only threshold selection — đạt.
- Random permutations: `50` — đạt cho bounded diagnostic.
- Paired significance + Holm correction — đạt.
- Budget sweep — đạt.
- Oracle diagnostic — đạt.
- RQ8 bounded diagnostic — hoàn thành.
- Main Entropy/Margin claim — không được hỗ trợ.
- TIST2015 12-city gate — chưa hoàn thành.
