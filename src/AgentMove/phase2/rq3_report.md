# Phase 2 — RQ3: Giá trị gia tăng của tri thức LLM

## Mục tiêu nghiên cứu

Xác định evidence có cấu trúc từ LLM có bổ sung thông tin ngoài prior dữ liệu và teacher định lượng hay không. Phase 2 không chỉ hỏi M4 có điểm cao nhất hay không, mà yêu cầu lợi ích của LLM tồn tại trên cả ba dataset và vượt được nhiễu do tập bounded nhỏ.

## Thiết kế thử nghiệm

- Dataset: TIST2015 12 thành phố, WWW2019-Shanghai và YJMob100K-Dataset1.
- Protocol: bounded matched last-query, mặc định `LLM_LIMIT=1000` cho validation và test.
- M1: data-only; M2: LLM-only; M3: quantitative-only; M4: quantitative + LLM.
- Weight chỉ chọn trên validation; test replay cùng immutable evidence cache.
- Không tạo nhiều seed giả cho cùng cache LLM. Báo cáo paired bootstrap theo query và Holm correction.
- So sánh chính: M2-vs-M1 và M4-vs-M3. M4-vs-M3 là phép đo trực tiếp giá trị gia tăng của LLM trên teacher định lượng.

## Kết quả mong đợi và tiêu chí đạt

RQ3 đạt khi M4 vượt M3 có ý nghĩa ở ít nhất một ranking metric và không làm giảm đáng kể các metric chính trên tối thiểu hai trong ba dataset. Đồng thời cache phải phủ đủ query, invalid-output rate được báo cáo và test không tham gia chọn weight.

## Bảng kết quả

| Dataset | Variant | Queries | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | Cache coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TIST2015 macro | M1-data-only | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TIST2015 macro | M3-quantitative | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TIST2015 macro | M4-both | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| WWW2019 | M1/M3/M4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YJMob100K | M1/M3/M4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

| So sánh | Dataset | Metric chính | Effect | 95% CI | Holm p | Đạt |
|---|---|---|---:|---:|---:|---|
| M4-both vs M3-quantitative | TIST2015 macro | MRR | TBD | TBD | TBD | TBD |
| M4-both vs M3-quantitative | WWW2019 | MRR | TBD | TBD | TBD | TBD |
| M4-both vs M3-quantitative | YJMob100K | MRR | TBD | TBD | TBD | TBD |

Script: `phase2/scripts/rq3.sh`.
