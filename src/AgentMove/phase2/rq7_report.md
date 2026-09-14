# Phase 2 — RQ7: Belief memory tuần tự

## Mục tiêu nghiên cứu

Đánh giá liệu cập nhật belief theo lịch sử và transition có cải thiện ranking một cách ổn định, đồng thời xử lý hạn chế Phase 1 khi B1-history và B2-sequential bị chọn weight bằng 0.

## Thiết kế thử nghiệm

- Dataset: TIST2015 12 thành phố, WWW2019-Shanghai và YJMob100K-Dataset1.
- Frozen E5-dual checkpoint từ RQ4; không fine-tune bằng test.
- All-prefix evaluation, reset belief tại biên trajectory.
- Transition/prior chỉ fit train; fusion weight chọn trên validation.
- So sánh B0-static, B1-history, B2-sequential và B3-dbn.
- Ghi cả ranking và calibration; paired bootstrap theo cùng prefix/seed.
- Phase 2 phải lưu validation objective curve để giải thích tại sao một weight bằng 0 hoặc khác 0.

## Kết quả mong đợi và tiêu chí đạt

B3 đạt nếu cải thiện R@1/MRR so với B0 trên tối thiểu hai dataset mà không có leakage. Nếu NLL/Brier/ECE xấu đi, kết luận chỉ giới hạn ở ranking và yêu cầu calibration sau fusion. B1/B2 bằng B0 chỉ được xem là kết quả âm hợp lệ khi objective curve chứng minh validation chọn weight 0.

## Bảng kết quả

| Dataset | Variant | Weight validation | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| TIST2015 macro | B0-static | 0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TIST2015 macro | B3-dbn | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| WWW2019 | B0/B3 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YJMob100K | B0/B3 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

| So sánh | Dataset | Effect R@1 | Effect MRR | Effect NLL | Holm p | Kết luận |
|---|---|---:|---:|---:|---:|---|
| B3-dbn vs B0-static | TIST2015 macro | TBD | TBD | TBD | TBD | TBD |
| B3-dbn vs B0-static | WWW2019 | TBD | TBD | TBD | TBD | TBD |
| B3-dbn vs B0-static | YJMob100K | TBD | TBD | TBD | TBD | TBD |

Script: `phase2/scripts/rq7.sh`.
