# Phase 2 — RQ4: Chắt lọc tri thức và dual-axis evolution

## Mục tiêu nghiên cứu

Kiểm tra khả năng khái quát của hai kết luận Phase 1: KD cải thiện CE và dual-axis có tạo lợi ích bổ sung ngoài KD hay không.

## Thiết kế thử nghiệm

- Dataset: TIST2015 12 thành phố, WWW2019-Shanghai và YJMob100K-Dataset1.
- Cùng preprocessing, candidate space, split và seed `42, 43, 44`.
- E0-ce, E1-kd và E5-dual là tập chính. Có thể đặt `RQ4_VARIANTS="E0-ce E1-kd E2-kd-traj E3-kd-vel E4-layer E6-temporal E5-dual"` để chạy ablation đầy đủ.
- Checkpoint chọn bằng validation; báo cáo trên full test last-query.
- Paired bootstrap dùng cùng query/seed; Holm correction áp dụng cho họ kiểm định đã khai báo.

## Kết quả mong đợi và tiêu chí đạt

- KD được xác nhận nếu E1 vượt E0 ổn định trên cả ba dataset và có ý nghĩa ở metric chính.
- Dual-axis được xác nhận mạnh nếu E5 vượt E1 có ý nghĩa trên tối thiểu hai dataset.
- Nếu E5 chỉ tăng trên TIST2015 nhưng ngang hoặc kém E1 ở YJMob100K, kết luận phải ghi là lợi ích phụ thuộc miền dữ liệu.

## Bảng kết quả

| Dataset | Variant | Seeds | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| TIST2015 macro | E0-ce | 42–44 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TIST2015 macro | E1-kd | 42–44 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TIST2015 macro | E5-dual | 42–44 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| WWW2019 | E0/E1/E5 | 42–44 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YJMob100K | E0/E1/E5 | 42–44 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

| So sánh | Dataset | Effect R@1 | Effect MRR | Holm p | Kết luận |
|---|---|---:|---:|---:|---|
| E1-kd vs E0-ce | TIST2015/WWW2019/YJMob100K | TBD | TBD | TBD | TBD |
| E5-dual vs E1-kd | TIST2015/WWW2019/YJMob100K | TBD | TBD | TBD | TBD |

Script: `phase2/scripts/rq4.sh`.
