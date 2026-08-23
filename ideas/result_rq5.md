# RQ5 — Ảnh hưởng của thứ tự trajectory

> Báo cáo riêng cho thí nghiệm order-corruption trên `TIST2015-Tokyo`. Số liệu hiện tại được tính trên **validation split**, không phải kết quả test cuối cùng.

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
- split: `validation`;
- correct: seed `42, 43, 44`;
- reverse: seed `42, 43, 44`;
- random: seed `42–51`, tương ứng 10 permutation seeds.

## 3. Kết quả validation

| Experiment | Seeds | Recall@1 | Recall@5 | Recall@10 | Gate |
|---|---|---:|---:|---:|---|
| E5-dual-correct | 42, 43, 44 | **0.159366 ± 0.002308** | **0.333297 ± 0.001550** | **0.401482 ± 0.002574** | not ready |
| E5-dual-reverse | 42, 43, 44 | 0.153380 ± 0.001297 | 0.326982 ± 0.002115 | 0.394145 ± 0.000276 | not ready |
| E5-dual-random | 42–51 | 0.139805 ± 0.001420 | 0.317981 ± 0.001973 | 0.387714 ± 0.002037 | not ready |

Số liệu được trình bày dưới dạng mean ± sample standard deviation giữa các seed.

## 4. Khoảng tin cậy bootstrap 95%

| Experiment | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|
| E5-dual-correct | 0.156702–0.160753 | 0.332348–0.335085 | 0.398927–0.404074 |
| E5-dual-reverse | 0.151883–0.154183 | 0.325120–0.329282 | 0.393890–0.394437 |
| E5-dual-random | 0.138929–0.140604 | 0.316820–0.319098 | 0.386498–0.388907 |

Các khoảng tin cậy được bootstrap từ các giá trị theo seed. Correct/reverse chỉ có ba seed nên CI chưa đủ mạnh để thay thế kiểm định paired trên từng query.

## 5. Mức suy giảm khi phá thứ tự

### 5.1. Reverse so với correct

| Metric | Correct | Reverse | Giảm tuyệt đối | Giảm tương đối |
|---|---:|---:|---:|---:|
| Recall@1 | 0.159366 | 0.153380 | −0.005986 | **−3.76%** |
| Recall@5 | 0.333297 | 0.326982 | −0.006315 | **−1.89%** |
| Recall@10 | 0.401482 | 0.394145 | −0.007337 | **−1.83%** |

Đảo ngược chuỗi làm giảm cả ba Recall. Ảnh hưởng lớn nhất xuất hiện ở Recall@1, cho thấy hướng thời gian đặc biệt quan trọng đối với lựa chọn vị trí đứng đầu.

### 5.2. Random so với correct

| Metric | Correct | Random | Giảm tuyệt đối | Giảm tương đối |
|---|---:|---:|---:|---:|
| Recall@1 | 0.159366 | 0.139805 | −0.019561 | **−12.27%** |
| Recall@5 | 0.333297 | 0.317981 | −0.015316 | **−4.60%** |
| Recall@10 | 0.401482 | 0.387714 | −0.013768 | **−3.43%** |

Random permutation gây suy giảm mạnh hơn reverse trên mọi metric. Điều này phù hợp với kỳ vọng: reverse vẫn giữ một cấu trúc có hệ thống, trong khi random phá vỡ hầu hết quan hệ chuyển tiếp cục bộ.

## 6. Giải thích kết quả

### 6.1. Mô hình có sử dụng thứ tự thời gian

`correct` đạt kết quả cao nhất, `reverse` đứng thứ hai và `random` thấp nhất trên cả Recall@1, Recall@5 và Recall@10. Thứ hạng nhất quán này là bằng chứng rằng E5-dual không chỉ ghi nhớ tập hợp POI mà còn khai thác trình tự di chuyển.

### 6.2. Recall@1 nhạy nhất với order corruption

Recall@1 giảm 3.76% khi đảo ngược và 12.27% khi hoán vị ngẫu nhiên. Mức giảm tương đối lớn hơn Recall@5/10 cho thấy thông tin thứ tự chủ yếu giúp mô hình xếp đúng POI lên vị trí đầu, thay vì chỉ đưa POI thật vào candidate list rộng hơn.

### 6.3. Temporal evolution có tín hiệu thực nghiệm sơ bộ

E5-dual có temporal evolution loss. Việc chất lượng giảm khi phá thứ tự phù hợp với giả thuyết rằng mô hình học representation dynamics theo thời gian. Tuy nhiên, order corruption đánh giá toàn bộ mô hình E5 chứ chưa cô lập riêng temporal loss; muốn quy kết trực tiếp cho temporal component cần so sánh order sensitivity giữa E4-layer và E5-dual.

## 7. Những gì chưa được phép kết luận

- Chưa được tuyên bố RQ5 hoàn thành cho publication vì toàn bộ số liệu hiện là validation.
- Chưa được tuyên bố hiệu ứng có ý nghĩa thống kê nếu chưa có paired test trên cùng query/seed.
- Correct/reverse có 3 seed trong khi random có 10 seed; so sánh mean hiện chưa hoàn toàn matched về số lần chạy.
- Kết quả mới thuộc Tokyo, chưa đại diện cho 12 thành phố TIST2015.
- Chưa thể khẳng định temporal loss là nguyên nhân duy nhất của độ nhạy thứ tự nếu chưa có đối chứng E4.

## 8. Công việc tiếp theo

1. Đánh giá checkpoint đã đóng băng trên test; không chọn lại epoch bằng test.
2. Ghi correct seed 42–44 làm đối chứng RQ5 test.
3. Ghi reverse seed 42–44 và random seed 42–51 trên test.
4. Tính paired difference trên cùng query và ít nhất trên tập seed chung 42–44.
5. Nếu cần chứng minh vai trò riêng của temporal evolution, chạy order-corruption đối chứng cho E4-layer.

Test evaluator đã được sửa để đọc/kiểm tra `ORDER_MODE` từ checkpoint, áp dụng
đúng corruption lên test input và tự định tuyến reverse/random vào `rq5-test`.
Không cần train lại các checkpoint E5 hiện có.

## 9. Kết luận tạm thời

> Trên validation split của TIST2015-Tokyo, E5-dual đạt chất lượng tốt nhất khi trajectory giữ đúng thứ tự thời gian. Đảo ngược chuỗi làm Recall@1 giảm 3.76%, trong khi hoán vị ngẫu nhiên làm giảm 12.27%; Recall@5 và Recall@10 cũng suy giảm nhất quán. Kết quả cung cấp bằng chứng sơ bộ rằng BeliefMove-Evo khai thác cấu trúc tuần tự của trajectory, đặc biệt trong việc xếp đúng POI ở vị trí đầu. Tuy nhiên, RQ5 vẫn chưa qua publication gate vì chưa có test evaluation và paired significance test.

## Publication gate

- Validation coverage: **đã đủ** cho correct, reverse và random.
- Test coverage: **chưa có**.
- Trạng thái RQ5: **not ready**.
