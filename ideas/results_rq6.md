# RQ6 — Dual-Axis Evolution

> Tên chuẩn hóa của báo cáo RQ6. Bảng representation, trajectory-length và
> paired significance đầy đủ nằm tại [result_rq6.md](result_rq6.md).

## Câu hỏi nghiên cứu

Tiến hóa biểu diễn theo layer hay theo thời gian quan trọng hơn, và kết hợp hai
trục có tạo lợi ích bổ sung hay không?

## Mục tiêu

Tách đóng góp layer/temporal evolution và liên hệ alignment representation với
ranking, calibration và độ dài trajectory.

## Tiêu chí đạt

Mọi biến thể phải dùng cùng split, candidates và seed; alignment/ngưỡng độ dài
chỉ fit validation. Claim E5 tốt hơn cần paired effect có ý nghĩa sau Holm;
CKA/cosine cao riêng lẻ không đủ chứng minh quality gain. RQ vẫn hoàn thành nếu
các khác biệt không significant, miễn đủ ablation và uncertainty analysis.

## Kết quả

Trong báo cáo Tokyo, E5 có mean ranking và CKA tốt nhất, còn E6 có NLL/ECE tốt
nhất; nhiều ranking difference giữa evolution variants chưa significant sau
Holm. Macro 12-city cho thấy E5 đạt R@1 `0,171903` và MRR `0,245829`, cao hơn
E6 lần lượt `0,002911` và `0,002122`; E6 vẫn có NLL thấp hơn rất nhẹ. Bằng
chứng ủng hộ dual-axis về ranking nhưng cần đọc cùng paired test.

## Phạm vi 12 thành phố

E5 đạt R@1 `0,171903`, R@5 `0,327474`, R@10 `0,384337`, MRR `0,245829`.
E6-temporal đạt tương ứng `0,168992/0,325905/0,382047/0,243706` và NLL
`6,671989`, thấp hơn nhẹ E5 (`6,673658`). Summary macro không xuất CKA,
cosine, trajectory-bin hoặc paired test đa thành phố; kết luận cơ chế vẫn phải
đọc cùng phân tích Tokyo.
