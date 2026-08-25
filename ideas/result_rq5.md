# RQ5 — Ảnh hưởng của thứ tự trajectory

> Báo cáo riêng cho thí nghiệm order-corruption trên `TIST2015-Tokyo`. Checkpoint được chọn bằng validation và chỉ đánh giá một lần trên test sau khi cấu hình đã đóng băng.

## 1. Câu hỏi nghiên cứu

RQ5 kiểm tra liệu BeliefMove-Evo có thực sự học quan hệ tuần tự trong trajectory hay chỉ dựa vào tập hợp các POI đã xuất hiện:

> Khi phá vỡ thứ tự thời gian của trajectory, chất lượng dự đoán vị trí tiếp theo có suy giảm hay không?

Nếu mô hình sử dụng thông tin tuần tự, `correct` phải tốt hơn `reverse` và `random`. Nếu kết quả gần như không đổi, mô hình có thể chủ yếu dựa vào tần suất hoặc sự hiện diện của POI thay vì động lực di chuyển.

## 2. Thiết lập thí nghiệm

Mô hình được sử dụng là `E5-dual`, cấu hình đầy đủ của distillation tiến hóa biểu diễn:

```text
CE + KD + trajectory + velocity + temporal evolution
```

Ba chế độ thứ tự:

| Chế độ | Biến đổi | Mục đích |
|---|---|---|
| `correct` | Giữ nguyên thứ tự thời gian | Đối chứng, phản ánh trajectory thật |
| `reverse` | Đảo ngược toàn bộ chuỗi input | Kiểm tra ảnh hưởng của hướng thời gian |
| `random` | Hoán vị các phần tử theo seed | Phá vỡ cấu trúc tuần tự mạnh nhất |

Các phép biến đổi chỉ thay đổi thứ tự input; không thay label, temporal split hoặc candidate vocabulary.

Phạm vi hiện tại:

- Dataset: `TIST2015-Tokyo`;
- split: temporal `validation` và held-out `test`;
- correct: seed `42, 43, 44`;
- reverse: seed `42, 43, 44`;
- random: seed `42–51`, tương ứng 10 permutation seeds.

## 3. Kết quả test chính

Số liệu là mean ± sample standard deviation giữa các seed. Ký hiệu ↑ là càng lớn càng tốt và ↓ là càng nhỏ càng tốt.

| Experiment | Seeds | Recall@1 ↑ | Recall@5 ↑ | Recall@10 ↑ | MRR ↑ | NLL ↓ | Brier ↓ | ECE ↓ | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **E5-dual-correct** | 42, 43, 44 | **0.147140 ± 0.002240** | **0.303595 ± 0.001035** | **0.367367 ± 0.000803** | **0.221656 ± 0.001334** | **7.528161 ± 0.012171** | **0.945082 ± 0.001529** | **0.029841 ± 0.003400** | ready |
| E5-dual-reverse | 42, 43, 44 | 0.139843 ± 0.002124 | 0.300490 ± 0.000598 | 0.364469 ± 0.001917 | 0.215599 ± 0.000839 | 7.619565 ± 0.011229 | 0.951115 ± 0.000579 | 0.030956 ± 0.002093 | ready |
| E5-dual-random | 42–51 | 0.133337 ± 0.001489 | 0.293449 ± 0.001387 | 0.356872 ± 0.001520 | 0.209137 ± 0.001138 | 7.647886 ± 0.027064 | 0.953999 ± 0.001249 | 0.034310 ± 0.003845 | ready |

## 4. Khoảng tin cậy bootstrap 95% trên test

| Experiment | Recall@1 | Recall@5 | Recall@10 | MRR | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| E5-dual-correct | 0.144587–0.148779 | 0.302577–0.304647 | 0.366539–0.368143 | 0.220205–0.222829 | 7.519984–7.542148 | 0.943489–0.946538 | 0.026094–0.032731 |
| E5-dual-reverse | 0.137549–0.141741 | 0.300145–0.301180 | 0.362296–0.365918 | 0.214635–0.216166 | 7.607132–7.628970 | 0.950694–0.951774 | 0.029116–0.033233 |
| E5-dual-random | 0.132483–0.134222 | 0.292579–0.294220 | 0.355982–0.357768 | 0.208462–0.209808 | 7.632620–7.664217 | 0.953297–0.954748 | 0.032099–0.036606 |

Các CI được bootstrap từ giá trị theo seed. Correct/reverse chỉ có ba seed; CI không thay thế paired significance test trên cùng query.

## 5. Kết quả validation tham khảo

Validation chỉ dùng để chọn checkpoint, không phải kết quả cuối của paper.

| Experiment | Seeds | Recall@1 | Recall@5 | Recall@10 | Gate |
|---|---|---:|---:|---:|---|
| E5-dual-correct | 42, 43, 44 | 0.159366 ± 0.002308 | 0.333297 ± 0.001550 | 0.401482 ± 0.002574 | not ready |
| E5-dual-reverse | 42, 43, 44 | 0.153380 ± 0.001297 | 0.326982 ± 0.002115 | 0.394145 ± 0.000276 | not ready |
| E5-dual-random | 42–51 | 0.139805 ± 0.001420 | 0.317981 ± 0.001973 | 0.387714 ± 0.002037 | not ready |

## 6. Mức suy giảm trên test khi phá thứ tự

### 6.1. Reverse so với correct

| Metric | Correct | Reverse | Giảm tuyệt đối | Giảm tương đối |
|---|---:|---:|---:|---:|
| Recall@1 | 0.147140 | 0.139843 | −0.007297 | **−4.96%** |
| Recall@5 | 0.303595 | 0.300490 | −0.003105 | **−1.02%** |
| Recall@10 | 0.367367 | 0.364469 | −0.002898 | **−0.79%** |
| MRR | 0.221656 | 0.215599 | −0.006057 | **−2.73%** |

Đảo ngược chuỗi làm giảm cả ba Recall. Ảnh hưởng lớn nhất xuất hiện ở Recall@1, cho thấy hướng thời gian đặc biệt quan trọng đối với lựa chọn vị trí đứng đầu.

### 6.2. Random so với correct

| Metric | Correct | Random | Giảm tuyệt đối | Giảm tương đối |
|---|---:|---:|---:|---:|
| Recall@1 | 0.147140 | 0.133337 | −0.013803 | **−9.38%** |
| Recall@5 | 0.303595 | 0.293449 | −0.010146 | **−3.34%** |
| Recall@10 | 0.367367 | 0.356872 | −0.010495 | **−2.86%** |
| MRR | 0.221656 | 0.209137 | −0.012519 | **−5.65%** |

Random permutation gây suy giảm mạnh hơn reverse trên mọi metric. Điều này phù hợp với kỳ vọng: reverse vẫn giữ một cấu trúc có hệ thống, trong khi random phá vỡ hầu hết quan hệ chuyển tiếp cục bộ.

## 7. Giải thích kết quả

### 7.1. Mô hình có sử dụng thứ tự thời gian

`correct` đạt kết quả cao nhất, `reverse` đứng thứ hai và `random` thấp nhất trên cả Recall@1, Recall@5 và Recall@10. Thứ hạng nhất quán này là bằng chứng rằng E5-dual không chỉ ghi nhớ tập hợp POI mà còn khai thác trình tự di chuyển.

### 7.2. Recall@1 nhạy nhất với order corruption

Trên test, Recall@1 giảm 4.96% khi đảo ngược và 9.38% khi hoán vị ngẫu nhiên. Mức giảm tương đối lớn hơn Recall@5/10 cho thấy thông tin thứ tự chủ yếu giúp mô hình xếp đúng POI lên vị trí đầu, thay vì chỉ đưa POI thật vào candidate list rộng hơn.

### 7.3. Calibration cũng suy giảm

Reverse và random đều làm NLL/Brier tăng. Random còn làm ECE tăng từ 0.029841 lên 0.034310. Như vậy, phá thứ tự không chỉ giảm ranking accuracy mà còn làm phân phối xác suất dự đoán kém phù hợp hơn.

### 7.4. Temporal evolution có tín hiệu thực nghiệm

E5-dual có temporal evolution loss. Việc chất lượng giảm khi phá thứ tự phù hợp với giả thuyết rằng mô hình học representation dynamics theo thời gian. Tuy nhiên, order corruption đánh giá toàn bộ mô hình E5 chứ chưa cô lập riêng temporal loss; muốn quy kết trực tiếp cho temporal component cần so sánh order sensitivity giữa E4-layer và E5-dual.

## 8. Những gì chưa được phép kết luận

- Chưa được tuyên bố hiệu ứng có ý nghĩa thống kê nếu chưa có paired test trên cùng query/seed.
- Correct/reverse có 3 seed trong khi random có 10 seed; so sánh mean hiện chưa hoàn toàn matched về số lần chạy.
- Kết quả mới thuộc Tokyo, chưa đại diện cho 12 thành phố TIST2015.
- Chưa thể khẳng định temporal loss là nguyên nhân duy nhất của độ nhạy thứ tự nếu chưa có đối chứng E4.

## 9. Công việc tiếp theo

1. Tính paired difference trên cùng query và ít nhất trên tập seed chung 42–44.
2. Nếu cần chứng minh vai trò riêng của temporal evolution, chạy order-corruption đối chứng cho E4-layer.
3. Lặp lại trên các thành phố còn lại trước khi đưa ra kết luận tổng quát cho TIST2015.

Test evaluator đã được sửa để đọc/kiểm tra `ORDER_MODE` từ checkpoint, áp dụng
đúng corruption lên test input và tự định tuyến reverse/random vào `rq5-test`.
Không cần train lại các checkpoint E5 hiện có.

## 10. Kết luận

> Trên test split của TIST2015-Tokyo, E5-dual đạt kết quả tốt nhất khi trajectory giữ đúng thứ tự thời gian. Đảo ngược chuỗi làm Recall@1 giảm 4.96%, trong khi hoán vị ngẫu nhiên làm giảm 9.38%; Recall@5, Recall@10, MRR và các chỉ số calibration cũng suy giảm nhất quán. Kết quả xác nhận ở mức mean qua nhiều seed rằng BeliefMove-Evo khai thác cấu trúc tuần tự của trajectory, đặc biệt trong việc xếp đúng POI ở vị trí đầu. Tuyên bố về ý nghĩa thống kê vẫn cần paired test trên cùng query.

## Publication gate

- Validation coverage: **đã đủ** cho correct, reverse và random.
- Test coverage: **đã đủ** cho correct, reverse và random.
- Trạng thái RQ5 Tokyo: **ready**.
- Trạng thái TIST2015 12 thành phố: **chưa hoàn thành**.
