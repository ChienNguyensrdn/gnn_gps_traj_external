# RQ10 — Độ bền theo kiến trúc teacher

> Teacher và student dùng cùng split, candidate set và seed; mọi lựa chọn checkpoint chỉ dùng validation. Kết quả dưới đây là trung bình của các seed 42, 43 và 44 trên test TIST2015–Tokyo.

## Câu hỏi nghiên cứu

RQ10 kiểm tra liệu lợi ích của distillation có phụ thuộc vào một kiến trúc teacher cụ thể hay không. Thí nghiệm giữ cố định kiến trúc student GRU, dữ liệu, candidate space, split và seed; yếu tố được thay đổi là teacher GRU hoặc Transformer.

- `none`: student chỉ học cross-entropy, không nhận tín hiệu distillation.
- `gru`: student nhận knowledge distillation và các tín hiệu biểu diễn từ GRU teacher.
- `transformer`: cùng student đó nhận các tín hiệu tương ứng từ Transformer teacher.

## Chất lượng teacher trên test

| Kiến trúc | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| gru | 0.143638 | 0.290502 | 0.344891 | 0.213183 | 8.799283 | 0.948259 | 0.035296 |
| transformer | 0.150417 | 0.302077 | 0.362089 | 0.222782 | 9.466769 | 0.949876 | 0.070451 |

Transformer teacher có ranking tốt hơn GRU teacher: R@1 tăng 0.006779, R@10 tăng 0.017198 và MRR tăng 0.009599. Tuy nhiên, xác suất của Transformer teacher được hiệu chỉnh kém hơn: NLL tăng 0.667486 và ECE tăng từ 0.035296 lên 0.070451. Vì bảng này chưa có paired test trực tiếp giữa hai teacher, các chênh lệch trên chỉ được diễn giải mô tả.

## Student sau distillation trên test

| Kiến trúc teacher | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0.133789 | 0.272528 | 0.325467 | 0.199728 | 9.210952 | 0.952485 | 0.030813 |
| gru | 0.147485 | 0.306734 | 0.370851 | 0.223206 | 7.575581 | 0.944471 | 0.029656 |
| transformer | 0.148623 | 0.308804 | 0.373680 | 0.224638 | 7.962563 | 0.949805 | 0.053708 |

### So với student không distillation

GRU distillation cải thiện tuyệt đối R@1 thêm 0.013696, R@5 thêm 0.034206, R@10 thêm 0.045384 và MRR thêm 0.023479. Tương ứng, mức tăng tương đối xấp xỉ 10.24%, 12.55%, 13.94% và 11.76%. NLL giảm 1.635370, tương đương khoảng 17.75%.

Transformer distillation cải thiện tuyệt đối R@1 thêm 0.014835, R@5 thêm 0.036276, R@10 thêm 0.048213 và MRR thêm 0.024910. Mức tăng tương đối xấp xỉ 11.09%, 13.31%, 14.81% và 12.47%. NLL giảm 1.248388, tương đương khoảng 13.55%.

Như vậy, cả hai teacher đều truyền được tín hiệu hữu ích sang cùng một student. Kết quả này cho thấy hiệu quả distillation không chỉ xuất hiện với GRU teacher.

### GRU teacher so với Transformer teacher

Student distillation từ Transformer nhỉnh hơn student distillation từ GRU về ranking: R@1 tăng 0.001138, R@5 tăng 0.002070, R@10 tăng 0.002829 và MRR tăng 0.001431. Tuy nhiên, các khác biệt ranking này không đạt ý nghĩa thống kê sau Holm correction.

Ngược lại, student từ GRU teacher có calibration tốt hơn rõ rệt. So với GRU, student từ Transformer có NLL cao hơn 0.386982, Brier cao hơn 0.005333 và ECE cao hơn 0.024052. Paired test xác nhận khác biệt NLL và Brier có ý nghĩa thống kê. Do đó, GRU là lựa chọn cân bằng hơn nếu hệ thống cần cả ranking và độ tin cậy xác suất.

## Paired significance

| So sánh student | Metric | Effect | 95% CI | Holm p | Significant |
|---|---|---:|---:|---:|---|
| gru-vs-none | recall@1 | 0.013696 | 0.011299–0.016146 | 0.00179982 | yes |
| gru-vs-none | recall@5 | 0.034206 | 0.031688–0.036794 | 0.00179982 | yes |
| gru-vs-none | recall@10 | 0.045384 | 0.042848–0.047920 | 0.00179982 | yes |
| gru-vs-none | mrr | 0.023479 | 0.021704–0.025241 | 0.00179982 | yes |
| gru-vs-none | nll | 1.635370 | 1.606007–1.664652 | 0.00179982 | yes |
| gru-vs-none | brier | 0.008013 | 0.006765–0.009233 | 0.00179982 | yes |
| transformer-vs-none | recall@1 | 0.014835 | 0.012230–0.017491 | 0.00179982 | yes |
| transformer-vs-none | recall@5 | 0.036276 | 0.033568–0.038967 | 0.00179982 | yes |
| transformer-vs-none | recall@10 | 0.048213 | 0.045487–0.050956 | 0.00179982 | yes |
| transformer-vs-none | mrr | 0.024910 | 0.022950–0.026874 | 0.00179982 | yes |
| transformer-vs-none | nll | 1.248388 | 1.216469–1.279938 | 0.00179982 | yes |
| transformer-vs-none | brier | 0.002680 | 0.001209–0.004150 | 0.0029997 | yes |
| transformer-vs-gru | recall@1 | 0.001138 | -0.001345–0.003588 | 0.370863 | no |
| transformer-vs-gru | recall@5 | 0.002070 | -0.000466–0.004640 | 0.338366 | no |
| transformer-vs-gru | recall@10 | 0.002829 | 0.000224–0.005330 | 0.114389 | no |
| transformer-vs-gru | mrr | 0.001431 | -0.000380–0.003177 | 0.338366 | no |
| transformer-vs-gru | nll | -0.386982 | -0.408788–-0.365434 | 0.00179982 | yes |
| transformer-vs-gru | brier | -0.005333 | -0.006658–-0.004000 | 0.00179982 | yes |

Positive effect nghĩa là student đứng trước tốt hơn; với NLL và Brier, dấu đã được đảo để giữ cùng cách diễn giải. Holm correction được áp dụng trên toàn bộ các phép kiểm định trong bảng. ECE không được paired test vì metric này không phân rã trực tiếp theo từng query trong artifact kiểm định.

## Kết luận RQ10

Kết quả ủng hộ kết luận rằng cơ chế distillation bền vững đối với hai kiến trúc teacher đã kiểm tra. Cả GRU và Transformer đều cải thiện có ý nghĩa toàn bộ metric ranking, NLL và Brier so với student chỉ học cross-entropy. Transformer cho ranking trung bình cao hơn một lượng nhỏ nhưng chưa có bằng chứng thống kê rằng nó vượt GRU về ranking. GRU tạo student có calibration tốt hơn đáng kể và hiện là lựa chọn thực dụng hơn.

Không được diễn giải kết quả không significant giữa hai student thành bằng chứng hai kiến trúc tương đương. Muốn khẳng định equivalence cần đặt trước biên tương đương và thực hiện equivalence hoặc non-inferiority test.

## Phạm vi và hạn chế

- Kết luận hiện chỉ áp dụng cho GRU và Transformer, trên TIST2015–Tokyo với seed 42–44.
- Đây không phải kết quả đa thành phố và không được suy diễn thành kết quả 12-city.
- PMT/UniTraj chưa được đưa vào bảng vì adapter preprocessing và candidate space chưa được xác minh.
- Báo cáo hiện chưa trình bày mean ± std theo seed, số tham số, epoch checkpoint được chọn và chi phí huấn luyện; nên bổ sung các thông tin này nếu dùng trong bản thảo chính.
- Hai teacher dùng cùng chiều biểu diễn nhưng không nhất thiết có cùng số tham số; vì vậy đây là kiểm tra robustness theo kiến trúc, không phải so sánh capacity-matched tuyệt đối.

## Publication gate

RQ10 đạt gate nội bộ cho thí nghiệm Tokyo: đủ checkpoint, test metrics và per-query predictions của seed 42–44, đồng thời paired significance đã áp dụng Holm correction. Gate cho tuyên bố đa kiến trúc rộng hoặc đa thành phố vẫn chưa hoàn thành.
