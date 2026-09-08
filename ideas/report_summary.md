# Báo cáo tóm tắt kết quả TIST2015 — 12 thành phố

> Cập nhật từ snapshot tổng hợp trên server ngày 2026-09-09. Trạng thái toàn cục:
> **incomplete**. Các bảng neural/Bayesian bên dưới đã phủ đủ 12 thành phố và ba
> seed 42, 43, 44; phần LLM bounded và efficiency chưa qua publication gate.

## 1. Phạm vi và protocol

- Dataset: TIST2015 với đúng 12 thành phố: Tokyo, Nairobi, New York, Sydney,
  Cape Town, Paris, Beijing, Mumbai, San Francisco, London, São Paulo và Moscow.
- Neural sử dụng test `last-query`; Bayesian sử dụng test `all-prefix`. Không so
  trực tiếp trị tuyệt đối giữa hai protocol.
- Kết quả trong bảng là macro-average không trọng số qua 12 thành phố, sau đó
  tổng hợp qua ba seed.
- LLM: Qwen2:7b, giới hạn 200 query và chế độ `no-OSM`. Phần này không được gọi
  là full-query hoặc mô hình world-knowledge đầy đủ.

## 2. Kết quả chính — Distillation ablation (RQ4)

| Biến thể | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0-CE | 0,156010 | 0,304810 | 0,353479 | 0,225939 | 7,407897 | 0,945500 | 0,056975 |
| E1-KD | 0,167779 | 0,320729 | 0,377380 | 0,241204 | 6,710074 | 0,936330 | 0,048971 |
| E2-KD-Traj | 0,167959 | 0,322898 | 0,379605 | 0,242099 | 6,695095 | 0,935780 | 0,049009 |
| E3-KD-Vel | 0,169996 | 0,323083 | 0,379032 | 0,243411 | 6,698934 | 0,935271 | 0,048632 |
| E4-Layer | 0,171730 | 0,324089 | 0,380577 | 0,244642 | 6,694331 | 0,934733 | 0,048252 |
| **E5-Dual** | **0,171903** | **0,327474** | **0,384337** | **0,245829** | 6,673658 | **0,933304** | **0,047902** |

E5-Dual đạt chất lượng ranking tốt nhất. So với E0-CE, E5 tăng R@1
`0,015894` (khoảng 10,2% tương đối), R@5 `0,022663`, R@10 `0,030858` và
MRR `0,019890`; đồng thời giảm NLL `0,734239`, Brier `0,012196` và ECE
`0,009073`. So với E1-KD, mức tăng nhỏ hơn nhưng vẫn nhất quán: R@1
`+0,004124`, R@10 `+0,006957` và MRR `+0,004625`.

## 3. Vai trò của thứ tự thời gian (RQ5)

| Thứ tự | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| **Correct** | **0,171903** | **0,327474** | **0,384337** | **0,245829** |
| Random | 0,164812 | 0,324336 | 0,381848 | 0,240639 |
| Reverse | 0,163801 | 0,323969 | 0,380686 | 0,239358 |

Làm hỏng thứ tự trajectory làm giảm kết quả. So với correct, random giảm R@1
`0,007091`, còn reverse giảm `0,008102`. Reverse gây suy giảm mạnh nhất ở R@1
và MRR, ủng hộ giả thuyết rằng mô hình khai thác chiều tiến hóa thời gian thay
vì chỉ học một tập hợp điểm đến không có thứ tự.

## 4. Dual-axis evolution (RQ6)

E5-Dual đạt R@1 `0,171903`, R@10 `0,384337` và MRR `0,245829`, cao nhất
trong nhóm. E6-Temporal có NLL thấp hơn rất nhẹ (`6,671989` so với
`6,673658`) nhưng ranking thấp hơn E5. Điều này cho thấy kết hợp tín hiệu theo
layer và thời gian có lợi cho ranking; riêng temporal alignment có thể hữu ích
cho log-loss.

## 5. Belief memory (RQ7)

| Biến thể | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0-Static | 0,160954 | 0,330134 | 0,395361 | 0,241040 | **6,470550** | **0,943010** | **0,048827** |
| B3-DBN | **0,167185** | **0,341297** | **0,406880** | **0,249301** | 6,652804 | 0,967838 | 0,132316 |

B3-DBN tăng R@1 `0,006231`, R@5 `0,011163`, R@10 `0,011519` và MRR
`0,008261`. Đổi lại, NLL, Brier và đặc biệt ECE đều xấu hơn. Vì vậy DBN nên
được mô tả là cơ chế cải thiện ranking nhưng cần calibration trước khi xác suất
dự báo được dùng trong quyết định downstream.

## 6. Độ bền theo teacher (RQ10)

| Student | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Không distillation | 0,155951 | 0,304878 | 0,353451 | 0,225894 | 7,434454 | 0,945238 | 0,055881 |
| Distill từ GRU | **0,170549** | **0,325773** | **0,381963** | **0,244327** | **6,647197** | **0,934168** | **0,045113** |
| Distill từ Transformer | 0,167144 | 0,325721 | 0,379618 | 0,242336 | 6,703186 | 0,937478 | 0,046834 |

Cả hai teacher đều giúp student tốt hơn CE-only. Trong cấu hình hiện tại,
student distill từ GRU tốt hơn nhẹ so với student distill từ Transformer trên
hầu hết metric. Kết quả không ủng hộ giả định rằng teacher phức tạp hơn luôn tạo
student tốt hơn.

## 7. Tính ổn định địa lý

E5-Dual cải thiện R@1 so với E0-CE ở cả 12/12 thành phố. Mức tăng nằm trong
khoảng `+0,0089` tại London đến `+0,0256` tại Moscow. Tuy nhiên phương sai R@1
giữa thành phố của E5 là `0,002557`, tương ứng độ lệch chuẩn địa lý khoảng
`0,0506`. Do đó bài báo cần giữ cả macro-average và bảng per-city; chỉ báo cáo
macro mean sẽ che khuất khác biệt miền khá lớn.

## 8. Trạng thái publication gate

Snapshot còn **290** vấn đề:

- Tokyo đến São Paulo: mỗi thành phố có 6 cảnh báo RQ12 do GPU contention.
- Moscow thiếu 204 artifact RQ8, tương ứng các seed random-budget-matched
  58–91; các seed 42–57 đã có.
- Moscow thiếu 14 artifact RQ9, tương ứng 7 biến thể, mỗi biến thể cần metrics
  và predictions.
- RQ12 có tổng cộng 72 cảnh báo contention trên 12 thành phố.

Vì vậy:

- Có thể dùng các kết quả neural/Bayesian đã đủ 12 thành phố như kết quả tạm
  thời theo đúng scope.
- Chưa được gọi toàn bộ RQ1–RQ13 là hoàn thành.
- Chưa được dùng RQ12 làm benchmark publication.
- Kết quả LLM hiện vẫn là bounded `limit=200`, `no-OSM`.

## 9. Công việc còn lại

1. Resume RQ8 và RQ9 cho Moscow.
2. Chạy lại RQ12 khi GPU không có foreign process, với `FORCE=1`.
3. Chạy paired significance cross-city/query cho E5-vs-E0, E5-vs-E1 và
   correct-vs-random/reverse trước khi phát biểu về ý nghĩa thống kê.
4. Chạy lại `status`, sau đó chỉ chạy `aggregate` khi missing count bằng 0.
5. Giữ báo cáo `no-OSM` tách biệt với bất kỳ thí nghiệm full-OSM nào sau này.
