# RQ8 — Phân tích Uncertainty-aware LLM Routing

## 1. Câu hỏi nghiên cứu

RQ8 kiểm tra liệu có thể **giảm số lần gọi LLM nhưng vẫn giữ chất lượng dự đoán
gần với chính sách gọi LLM cho mọi query** hay không.

Thực nghiệm hiện tại là bounded experiment trên **200 test queries của
TIST2015–Tokyo**, dùng `qwen2:7b`. Threshold của Entropy và Margin được chọn
trên validation với call budget tối đa 25%; test không được dùng để điều chỉnh
router. Các policy sử dụng chung một Always-LLM cache để bảo đảm so sánh trên
cùng query và cùng LLM output.

## 2. Các chính sách routing

- **Never:** không gọi LLM; luôn dùng ranking của predictor.
- **Always:** gọi LLM cho mọi query và dùng LLM để rerank top-10 candidates.
- **Entropy:** gọi LLM khi entropy của predictor vượt threshold đã fit trên
  validation.
- **Margin:** gọi LLM khi chênh lệch xác suất giữa hai candidate đứng đầu thấp
  hơn threshold đã fit trên validation.
- **Random-budget-matched:** chọn ngẫu nhiên số query bằng đúng call budget thực
  tế của Entropy; đây là control để kiểm tra uncertainty có tốt hơn random hay
  không.

## 3. Kết quả test

| Router | R@1 | R@5 | R@10 | MRR | LLM call rate | Latency mean (s) | Latency p95 (s) | Tokens/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Never | 0.125000 | 0.250000 | 0.305000 | 0.186565 | 0.000000 | 0.000000 | 0.000000 | 0.00 |
| Always | 0.120000 | 0.280000 | 0.305000 | 0.193732 | 1.000000 | 2.822543 | 3.637879 | 955.88 |
| Entropy | 0.125000 | 0.250000 | 0.305000 | 0.186982 | 0.145000 | 0.390842 | 2.601479 | 141.40 |
| Margin | 0.125000 | 0.250000 | 0.305000 | 0.186565 | 0.000000 | 0.000000 | 0.000000 | 0.00 |
| Random-budget-matched | **0.130000** | **0.253333** | 0.305000 | 0.189885 | 0.145000 | 0.439303 | 2.617146 | 138.82 |

## 4. Phân tích chất lượng dự đoán

### 4.1. Always-LLM không cải thiện đồng đều mọi ranking metric

So với Never, Always làm Recall@5 tăng từ `0.250` lên `0.280` và MRR tăng từ
`0.186565` lên `0.193732`. Tuy nhiên Recall@1 giảm từ `0.125` xuống `0.120`,
còn Recall@10 giữ nguyên ở `0.305`.

Recall@10 không đổi là kết quả có thể dự kiến vì LLM chỉ rerank tập top-10 do
predictor cung cấp. Nếu ground-truth không nằm trong candidate set này, bước
rerank không thể đưa nó vào top-10. Vì vậy RQ8 chủ yếu đo khả năng sắp xếp lại
candidates, không đo khả năng tăng candidate recall.

### 4.2. Entropy giảm mạnh chi phí nhưng gần như không cải thiện quality

Entropy chỉ gọi LLM trên `14.5%` queries. So với Always, chính sách này:

- giảm **85.5%** số lần gọi LLM;
- giảm tokens/query từ `955.88` xuống `141.40`, tương đương khoảng **85.2%**;
- giảm latency trung bình từ `2.822543` giây xuống `0.390842` giây, tương đương
  khoảng **86.1%**.

Đổi lại, Entropy gần như trùng với Never: Recall@1, Recall@5 và Recall@10 không
đổi; MRR chỉ tăng `0.000417`. Do đó Entropy đạt hiệu quả chi phí cao, nhưng chưa
cho thấy nó chọn đúng các query mà LLM thực sự có thể cải thiện.

### 4.3. Margin bị validation loại bỏ

Margin có call rate bằng `0`, nên rút gọn hoàn toàn về Never. Theo objective và
budget hiện tại, validation không tìm thấy threshold Margin nào đem lại lợi ích
đủ để gọi LLM. Đây là negative result hợp lệ cho cách định nghĩa margin đang
dùng, không phải bằng chứng rằng mọi margin-based router đều không hiệu quả.

### 4.4. Random control tốt hơn Entropy tại cùng budget

Entropy và Random cùng có call rate `0.145`, nhưng Random đạt kết quả cao hơn:

| Metric | Entropy | Random | Random − Entropy |
|---|---:|---:|---:|
| Recall@1 | 0.125000 | 0.130000 | +0.005000 |
| Recall@5 | 0.250000 | 0.253333 | +0.003333 |
| Recall@10 | 0.305000 | 0.305000 | 0.000000 |
| MRR | 0.186982 | 0.189885 | +0.002903 |

Kết quả này đi ngược giả thuyết chính của RQ8: uncertainty score hiện tại chưa
định tuyến LLM tốt hơn lựa chọn ngẫu nhiên cùng chi phí. Do số lượng test query
nhỏ và chưa có paired significance test, chưa được khẳng định Random thực sự
tốt hơn Entropy; chỉ có thể kết luận rằng kết quả hiện tại **không cung cấp bằng
chứng ủng hộ Entropy router**.

## 5. Lưu ý về seed và tính độc lập

Never, Always, Entropy và Margin sử dụng cùng predictor, threshold và
Always-LLM cache nên chúng là deterministic trong ba lần evaluate. Ba seed
không phải ba LLM runs độc lập đối với các policy này. Seed chỉ thay đổi tập
query của Random-budget-matched.

Vì vậy không được dùng ba dòng seed lặp lại của các policy deterministic để
claim độ ổn định qua ba independent runs. Độ bất định phù hợp hơn cần được tính
bằng paired bootstrap trên cùng 200 queries; Random có thể được đánh giá qua
nhiều permutation seeds.

## 6. Kết luận RQ8 hiện tại

Claim an toàn cho bounded experiment:

> Trên 200 test queries của TIST2015–Tokyo, entropy routing giảm khoảng 85% chi
> phí LLM và giữ chất lượng gần Never-LLM. Tuy nhiên, nó không vượt random
> routing cùng call budget; do đó chưa có bằng chứng rằng uncertainty hiện tại
> xác định hiệu quả các query cần LLM.

Không nên claim rằng Entropy đạt quality gần Always theo mọi metric: Recall@5
giảm `0.030` và MRR giảm `0.006750` so với Always. Cũng không nên claim kết quả
này đại diện cho toàn bộ Tokyo test set hoặc đủ 12 thành phố.

## 7. Công việc cần bổ sung

1. Thực hiện paired bootstrap/permutation cho Entropy–Never,
   Entropy–Always, Entropy–Random và Always–Never.
2. Đánh giá quality–cost curve ở nhiều budget, ví dụ 10%, 25% và 50%.
3. Đo khả năng dự báo **LLM gain** thay vì chỉ dùng entropy tuyệt đối: router
   nên học query nào có xác suất được LLM sửa đúng.
4. Kiểm tra entropy sau calibration và thử các feature kết hợp như entropy,
   margin, trajectory length, candidate coverage và DBN disagreement.
5. Mở rộng số query và thành phố trước khi đưa ra publication claim.

## 8. Trạng thái publication gate

- City: `Tokyo` — hoàn thành bounded run.
- Test queries: `200` — bounded, chưa phải full-query.
- LLM backbone: `qwen2:7b`.
- Validation-only threshold selection — đạt.
- Matched Random control — đạt.
- Paired significance — chưa thực hiện.
- RQ8 bounded diagnostic — hoàn thành.
- RQ8 main claim — chưa được hỗ trợ.
- TIST2015 12-city gate — chưa hoàn thành.
