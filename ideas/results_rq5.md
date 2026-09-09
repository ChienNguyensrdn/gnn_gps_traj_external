# RQ5 — Ảnh hưởng của thứ tự trajectory

> Tên chuẩn hóa của báo cáo RQ5. Bảng kết quả và paired significance đầy đủ nằm
> tại [result_rq5.md](result_rq5.md).

## Câu hỏi nghiên cứu

Khi phá vỡ thứ tự thời gian của trajectory, chất lượng dự đoán vị trí kế tiếp có
suy giảm hay không?

## Mục tiêu

Kiểm chứng rằng mô hình sử dụng động lực tuần tự thay vì chỉ dựa vào tần suất
hoặc tập hợp POI đã quan sát.

## Tiêu chí đạt

`Correct` phải vượt `random` và `reverse` trên paired Recall/MRR, với bootstrap
CI không chứa 0 và Holm-adjusted p < 0,05. Corruption chỉ được đổi thứ tự input,
không đổi query, target, split hoặc candidate set. Negative result vẫn hoàn
thành RQ nhưng không hỗ trợ giả thuyết temporal order.

## Kết quả

Trên báo cáo Tokyo chi tiết, correct đạt R@1 `0,147140`, reverse `0,139843` và
random `0,133337`; correct vượt hai corruption có ý nghĩa trên các metric được
kiểm định. Macro 12-city tiếp tục cùng chiều: `0,171903`, `0,163801` và
`0,164812`. Kết quả hỗ trợ mô hình khai thác thứ tự thời gian.

## Phạm vi 12 thành phố

Macro correct đạt R@1/R@5/R@10/MRR
`0,171903/0,327474/0,384337/0,245829`; reverse đạt
`0,163801/0,323969/0,380686/0,239358`; random đạt
`0,164812/0,324336/0,381848/0,240639`. Correct đứng đầu cả bốn metric.
Paired significance Tokyo không tự động áp dụng cho macro; cần kiểm định phân
tầng theo city nếu muốn claim significance đa thành phố.
