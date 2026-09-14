# Phase 2 — RQ8: Định tuyến LLM theo realized gain

## Mục tiêu nghiên cứu

Khắc phục kết quả âm của entropy/margin router Phase 1 bằng cách kiểm tra liệu bất định có dự đoán đúng query mà LLM thực sự cải thiện hay không, và liệu routing có tạo Pareto improvement giữa chất lượng và chi phí.

## Thiết kế thử nghiệm

- Dataset: TIST2015 12 thành phố, WWW2019-Shanghai và YJMob100K-Dataset1.
- Bounded matched test, mặc định `LLM_LIMIT=1000`; Always cache được tạo một lần rồi khóa.
- Baseline: never, always, entropy, margin và random-budget-matched.
- Threshold/budget chọn trên validation; test không dùng tuning.
- Deterministic policy chỉ báo cáo một run. Random control dùng 50 seed.
- Báo cáo R@1/R@5/R@10/MRR, call rate, latency mean/p95 và tokens/query.
- Bổ sung oracle realized-gain upper bound để đo headroom của router.

## Kết quả mong đợi và tiêu chí đạt

Router đạt khi vượt random-budget-matched có ý nghĩa ở MRR hoặc R@1 tại cùng call rate, đồng thời dùng ít nhất 50% số lời gọi/tokens của Always. Nếu oracle tốt nhưng entropy/margin không tốt, kết luận là routing signal chưa phù hợp, không phải LLM không có headroom.

## Bảng kết quả

| Dataset | Router | Runs | R@1 | R@5 | R@10 | MRR | Call rate | Latency p95 | Tokens/query |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TIST2015 macro | never/always/entropy/margin/random | 1/50 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| WWW2019 | never/always/entropy/margin/random | 1/50 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YJMob100K | never/always/entropy/margin/random | 1/50 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

| Dataset | Budget | Router | ΔMRR vs random | 95% CI | Holm p | Token saving vs always | Đạt |
|---|---:|---|---:|---:|---:|---:|---|
| TIST2015 macro | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| WWW2019 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YJMob100K | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Script: `phase2/scripts/rq8.sh`.
