# RQ5 — Kiểm định ý nghĩa thống kê theo cặp

> Kiểm định order-corruption trên held-out test split của `TIST2015-Tokyo`. Mọi so sánh sử dụng đúng cùng query và cùng seed `42, 43, 44`.

## 1. Mục tiêu

Kiểm định này trả lời câu hỏi:

> Chất lượng của E5-dual khi giữ đúng thứ tự trajectory có thực sự tốt hơn khi đảo ngược hoặc hoán vị ngẫu nhiên thứ tự hay không?

Hai cặp được kiểm định:

- `correct-vs-reverse`: đúng thứ tự so với đảo ngược chuỗi;
- `correct-vs-random`: đúng thứ tự so với hoán vị ngẫu nhiên.

Giả thuyết không $H_0$: hiệu ứng trung bình theo cặp bằng 0. Giả thuyết đối $H_1$: hiệu ứng khác 0.

## 2. Phương pháp

Mỗi prediction artifact lưu kết quả theo từng query, gồm label, rank, top-1, reciprocal rank, true-label probability và Brier contribution. Trước khi kiểm định, pipeline xác nhận `query_index` và label của hai mô hình hoàn toàn thẳng hàng.

Quy trình thống kê:

1. Tính difference trên cùng query cho từng seed.
2. Macro-average effect giữa ba seed chung `42–44`.
3. Tính confidence interval 95% bằng paired bootstrap.
4. Tính p-value hai phía bằng sign-flip permutation test với 10.000 lần lặp.
5. Áp dụng Holm correction đồng thời cho toàn bộ 12 phép kiểm định.

Quy ước effect:

- Với Recall và MRR: `correct − corrupted`.
- Với NLL và Brier: `corrupted − correct`, vì giá trị nhỏ hơn tốt hơn.
- Vì vậy, **effect dương luôn có nghĩa correct order tốt hơn corrupted order**.

ECE không được kiểm định theo cặp vì đây là metric tổng hợp theo bin và không phân rã tự nhiên thành contribution độc lập của từng query.

## 3. Kết quả

| Comparison | Metric | Effect | Bootstrap 95% CI | Permutation p | Holm-adjusted p | Có ý nghĩa ở 0.05? |
|---|---|---:|---:|---:|---:|---|
| correct-vs-reverse | Recall@1 | 0.007297 | 0.005089–0.009539 | 9.999e-05 | 0.00119988 | Có |
| correct-vs-reverse | Recall@5 | 0.003105 | 0.000828–0.005261 | 0.00489951 | 0.00979902 | Có |
| correct-vs-reverse | Recall@10 | 0.002898 | 0.000673–0.005071 | 0.0107989 | 0.0107989 | Có |
| correct-vs-reverse | MRR | 0.006057 | 0.004552–0.007601 | 9.999e-05 | 0.00119988 | Có |
| correct-vs-reverse | NLL | 0.091404 | 0.078600–0.103978 | 9.999e-05 | 0.00119988 | Có |
| correct-vs-reverse | Brier | 0.006032 | 0.004954–0.007117 | 9.999e-05 | 0.00119988 | Có |
| correct-vs-random | Recall@1 | 0.012851 | 0.010557–0.015197 | 9.999e-05 | 0.00119988 | Có |
| correct-vs-random | Recall@5 | 0.010367 | 0.008125–0.012575 | 9.999e-05 | 0.00119988 | Có |
| correct-vs-random | Recall@10 | 0.009798 | 0.007607–0.012023 | 9.999e-05 | 0.00119988 | Có |
| correct-vs-random | MRR | 0.011993 | 0.010378–0.013621 | 9.999e-05 | 0.00119988 | Có |
| correct-vs-random | NLL | 0.094790 | 0.083066–0.106548 | 9.999e-05 | 0.00119988 | Có |
| correct-vs-random | Brier | 0.007881 | 0.006685–0.009063 | 9.999e-05 | 0.00119988 | Có |

Giá trị `9.999e-05` là p-value nhỏ nhất có thể quan sát với 10.000 permutation samples theo công thức hiệu chỉnh hữu hạn $(b+1)/(B+1)$; không nên diễn giải là p-value chính xác bằng 0.

## 4. Diễn giải

### 4.1. Correct vượt reverse

Toàn bộ sáu metric có CI 95% nằm hoàn toàn phía trên 0 và vẫn có ý nghĩa sau Holm correction. Correct order tăng Recall@1 theo cặp `0.007297`, tăng MRR `0.006057`, đồng thời giảm NLL tương đương `0.091404` và Brier `0.006032` so với reverse.

Kết quả xác nhận rằng hướng thời gian của trajectory có đóng góp cho dự đoán. Đảo ngược chuỗi không chỉ làm giảm ranking accuracy mà còn làm phân phối xác suất kém phù hợp hơn.

### 4.2. Correct vượt random

Random corruption tạo effect lớn hơn reverse trên phần lớn ranking metrics. Correct order tăng Recall@1 theo cặp `0.012851` và MRR `0.011993` so với random. Tất cả metric đều còn ý nghĩa sau Holm correction.

Điều này phù hợp với mức độ phá hủy cấu trúc: reverse vẫn giữ một thứ tự có hệ thống, còn random phá vỡ hầu hết quan hệ chuyển tiếp cục bộ.

### 4.3. Vì sao effect random khác chênh lệch trong bảng 10 seed?

Bảng aggregate chính của random sử dụng seed `42–51`, trong khi paired test bắt buộc dùng tập seed chung `42–44` với correct và reverse. Do đó:

- paired Recall@1 effect là `0.012851` trên seed `42–44`;
- không lấy trực tiếp chênh lệch giữa correct 3-seed mean và random 10-seed mean.

Đây là khác biệt đúng về protocol, không phải lỗi tính toán.

## 5. Kết luận RQ5

> Trên test split của TIST2015-Tokyo, E5-dual khi giữ đúng thứ tự trajectory vượt cả reverse và random corruption trên Recall@1/5/10, MRR, NLL và Brier. Tất cả 12 paired effects có bootstrap CI 95% không chứa 0 và vẫn có ý nghĩa sau Holm correction (`adjusted p < 0.05`). Kết quả xác nhận rằng BeliefMove-Evo khai thác cấu trúc tuần tự của trajectory thay vì chỉ dựa vào tập hợp POI đã quan sát.

## 6. Phạm vi được phép tuyên bố

Có thể tuyên bố:

- correct order tốt hơn reverse và random trên Tokyo;
- khác biệt có ý nghĩa thống kê sau Holm correction;
- mô hình khai thác thông tin tuần tự của trajectory;
- phá thứ tự làm giảm cả ranking performance và probabilistic quality.

Chưa nên tuyên bố:

- kết quả đại diện cho toàn bộ 12 thành phố TIST2015;
- temporal loss của E5 là nguyên nhân duy nhất của hiệu ứng;
- E5 nhạy với thứ tự hơn E4 nếu chưa chạy cùng corruption protocol cho E4;
- các query hoàn toàn độc lập, vì nhiều query có thể thuộc cùng user hoặc trajectory.

Nếu cần protocol chặt hơn cho publication, có thể bổ sung clustered bootstrap theo user/trajectory sau khi prediction artifact lưu thêm cluster identifier.

## Publication gate

- Test coverage correct/reverse/random: **đủ**.
- Paired alignment: **đạt**.
- Bootstrap CI 95%: **đạt**.
- Permutation test và Holm correction: **đạt**.
- Trạng thái RQ5 Tokyo: **ready**.
- Tổng quát hóa 12 thành phố: **chưa hoàn thành**.
