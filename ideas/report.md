# Báo cáo tổng hợp thực nghiệm trên ba bộ dữ liệu

## 1. Phạm vi và trạng thái

Báo cáo tổng hợp kết quả trên ba miền dữ liệu độc lập:

- **TIST2015:** 12 thành phố, gồm Tokyo, Nairobi, NewYork, Sydney, CapeTown, Paris, Beijing, Mumbai, SanFrancisco, London, SaoPaulo và Moscow.
- **WWW2019:** Shanghai-ISP.
- **YJMob100K:** Dataset1, biểu diễn vị trí bằng ô lưới 500 m.

Các kết quả đều sử dụng ba seed `42, 43, 44`. Tổng cộng **14/14 báo cáo nguồn đã sẵn sàng**, không còn artifact bị thiếu.

Ba gate tương ứng là:

- TIST2015: **ready-tist2015**;
- WWW2019: **ready-www2019**;
- YJMob100K: **ready-yjmob100k**.

Neural sử dụng protocol **last-query**, trong khi Bayesian sử dụng **all-prefix**. Do đó chỉ so sánh các biến thể trong cùng protocol, không so trực tiếp trị tuyệt đối giữa hai nhóm.

## 2. Các biến thể chính

- **E0-ce:** student chỉ học bằng cross-entropy với nhãn cứng.
- **E1-kd:** bổ sung knowledge distillation từ teacher định lượng.
- **E5-dual:** kết hợp knowledge distillation với tín hiệu tiến hóa theo thời gian và theo tầng biểu diễn.
- **B0-static:** dự đoán neural tĩnh, không cập nhật belief tuần tự.
- **B3-dbn:** cập nhật belief bằng prior và transition động theo lịch sử quan sát.

## 3. Kết quả TIST2015 trên 12 thành phố

### 3.1. Neural last-query — macro trung bình 12 thành phố

| Biến thể | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| E0-ce | 0.156010 | 0.304810 | 0.353479 | 0.225939 |
| E1-kd | 0.167779 | 0.320729 | 0.377380 | 0.241204 |
| E5-dual | **0.171903** | **0.327474** | **0.384337** | **0.245828** |

So với E0, E1 tăng tương đối khoảng **7,54% R@1**, **5,22% R@5**, **6,76% R@10** và **6,76% MRR**. Điều này cho thấy phân phối mềm từ teacher cung cấp tín hiệu học hữu ích hơn nhãn cứng đơn thuần.

E5 đạt macro tốt nhất trên cả bốn ranking metric. So với E1, E5 tăng khoảng **2,46% R@1**, **2,10% R@5**, **1,84% R@10** và **1,92% MRR**. Tuy nhiên, mức tăng E5 so với E1 không có ý nghĩa thống kê trên mọi metric và mọi thành phố. Vì vậy, kết luận phù hợp là dual-axis tạo lợi ích bổ sung có điều kiện, không phải luôn vượt KD.

### 3.2. Bayesian all-prefix — macro trung bình 12 thành phố

| Biến thể | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| B0-static | 0.160954 | 0.330134 | 0.395361 | 0.241040 |
| B3-dbn | **0.167185** | **0.341297** | **0.406880** | **0.249301** |

B3-dbn cải thiện tương đối khoảng **3,87% R@1**, **3,38% R@5**, **2,91% R@10** và **3,43% MRR** so với B0-static. Kết quả cho thấy cập nhật belief theo transition lịch sử có ích trên nhiều bối cảnh địa lý.

### 3.3. Độ phụ thuộc vào thứ tự thời gian

Frozen-checkpoint control giữ nguyên checkpoint E5-dual/correct và chỉ thay đổi thứ tự test thành `reverse` hoặc `random`. Correct order thường tốt hơn dữ liệu bị phá thứ tự, đặc biệt rõ tại Tokyo, NewYork và một số thành phố có nhiều trajectory dài. Ở các thành phố nhỏ hơn, hiệu ứng ranking có thể không đạt ý nghĩa sau Holm correction nhưng NLL/Brier vẫn thường suy giảm khi thứ tự bị phá.

Kết quả này xác nhận mô hình có sử dụng thông tin thời gian. Tuy nhiên, cường độ hiệu ứng phụ thuộc cấu trúc trajectory và kích thước mẫu của từng thành phố.

## 4. Kết quả WWW2019 — Shanghai-ISP

### 4.1. Neural last-query

| Biến thể | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| E0-ce | 0.124150 ± 0.006568 | 0.237448 ± 0.005500 | 0.280978 ± 0.006620 | 0.179870 ± 0.007604 |
| E1-kd | 0.133810 ± 0.000947 | 0.241383 ± 0.004559 | 0.280859 ± 0.007847 | 0.186597 ± 0.002905 |
| E5-dual | **0.136076 ± 0.005430** | **0.251282 ± 0.003792** | **0.285510 ± 0.005680** | **0.191104 ± 0.005391** |

E1 vượt E0 có ý nghĩa ở R@1, NLL và Brier. E5 vượt E1 có ý nghĩa ở R@5 và NLL; chênh lệch R@1, R@10 và MRR chưa đạt ý nghĩa sau hiệu chỉnh Holm. Vì vậy, WWW2019 xác nhận lợi ích bền vững của KD, còn lợi ích dual-axis ở mức chọn lọc theo metric.

### 4.2. Bayesian all-prefix

| Biến thể | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| B0-static | 0.125890 ± 0.002821 | 0.237013 ± 0.002609 | 0.273185 ± 0.004000 | 0.179791 ± 0.003417 |
| B3-dbn | **0.170055 ± 0.004250** | **0.309288 ± 0.005519** | **0.341395 ± 0.003657** | **0.234498 ± 0.004104** |

B3-dbn tạo mức tăng lớn: khoảng **35,08% R@1**, **30,49% R@5**, **24,97% R@10** và **30,43% MRR** so với B0-static. Đây là bằng chứng cross-dataset mạnh nhất cho cơ chế belief tuần tự.

### 4.3. Temporal-order control

Khi huấn luyện riêng trên dữ liệu correct/reverse/random, khác biệt không nhất quán. Frozen-checkpoint control là phép đo phù hợp hơn vì giữ nguyên mô hình và chỉ thay đổi test input. Trong control này, correct vượt reverse có ý nghĩa trên toàn bộ ranking metric; correct vượt random có ý nghĩa ở R@1 và MRR, nhưng không đồng thời trên R@5/R@10. Vì vậy, temporal mechanism được xác nhận, nhưng hiệu ứng với random corruption phụ thuộc metric.

## 5. Kết quả YJMob100K — Dataset1

### 5.1. Neural last-query

| Biến thể | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| E0-ce | 0.471781 ± 0.002855 | 0.695324 ± 0.001386 | 0.753256 ± 0.001683 | 0.572654 ± 0.001463 |
| E1-kd | **0.490048 ± 0.002986** | 0.712429 ± 0.001970 | 0.773871 ± 0.002187 | **0.591052 ± 0.001965** |
| E5-dual | 0.487083 ± 0.004520 | **0.712879 ± 0.001254** | **0.774583 ± 0.000673** | 0.589774 ± 0.002882 |

E1 cải thiện có ý nghĩa so với E0 trên toàn bộ ranking metric, NLL và Brier. E5 và E1 gần tương đương: E5 tăng rất nhỏ ở R@5/R@10 nhưng giảm nhẹ R@1/MRR; các khác biệt ranking đều không có ý nghĩa thống kê. Brier của E5 còn kém E1 có ý nghĩa. Do đó, trên YJMob100K nên chọn E1 nếu ưu tiên mô hình đơn giản và ổn định.

### 5.2. Bayesian all-prefix

| Biến thể | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| B0-static | 0.253281 ± 0.001712 | 0.455368 ± 0.000933 | 0.532750 ± 0.000516 | 0.348964 ± 0.001345 |
| B3-dbn | **0.261018 ± 0.001297** | **0.466883 ± 0.000038** | **0.542545 ± 0.000961** | **0.358624 ± 0.000622** |

B3-dbn tiếp tục vượt B0-static, với mức tăng tương đối khoảng **3,05% R@1**, **2,53% R@5**, **1,84% R@10** và **2,77% MRR**.

### 5.3. Temporal-order control

Frozen correct order vượt reverse và random có ý nghĩa trên toàn bộ R@1/R@5/R@10/MRR/NLL/Brier. Hiệu ứng đặc biệt lớn ở R@1: correct hơn reverse **0,147771** và hơn random **0,156217**. Đây là bằng chứng mạnh rằng mô hình YJMob100K học cấu trúc tuần tự thực sự, thay vì chỉ ghi nhớ tập vị trí của người dùng.

## 6. Kết luận xuyên dataset

### 6.1. Kết quả được xác nhận

1. **Knowledge distillation có hiệu quả bền vững.** E1-kd vượt E0-ce trên TIST2015 macro, WWW2019 và YJMob100K. Đây là kết luận neural mạnh nhất.
2. **Dynamic belief có khả năng khái quát tốt.** B3-dbn vượt B0-static trên cả ba dataset, đặc biệt mạnh trên Shanghai-ISP.
3. **Thứ tự thời gian là tín hiệu có thật.** Frozen-checkpoint corruption làm giảm chất lượng trên nhiều thành phố và đặc biệt rõ trên YJMob100K.
4. **Dual-axis không luôn vượt KD.** E5 tốt nhất theo macro TIST2015 và một số metric WWW2019, nhưng không cải thiện đáng kể so với E1 trên YJMob100K.

### 6.2. Cách trình bày trong bài chính thức

- Dùng E1-kd làm baseline distillation chính và E5-dual làm biến thể mở rộng.
- Tuyên bố E5 cải thiện macro TIST2015, nhưng ghi rõ lợi ích không đồng nhất giữa dataset.
- Dùng B3-dbn làm bằng chứng chính cho đóng góp belief tuần tự.
- Dùng frozen-checkpoint control, không dùng so sánh giữa các mô hình được huấn luyện bằng order khác nhau, để chứng minh tác dụng của temporal order.
- Báo cáo mean ± std theo ba seed và paired bootstrap CI kèm Holm correction.
- Giữ riêng neural last-query và Bayesian all-prefix trong mọi bảng và thảo luận.

## 7. Giới hạn

- TIST2015 cung cấp đa dạng địa lý qua 12 thành phố, nhưng WWW2019 và YJMob100K hiện chỉ có một cấu hình dữ liệu cho mỗi bộ.
- YJMob100K sử dụng ô lưới 500 m; không diễn giải kết quả như dự đoán POI địa lý chính xác và không dùng Haversine/OSM.
- Các kết quả LLM bounded chỉ phản ánh tập query giới hạn, không phải full-query evaluation.
- Holm correction trong các báo cáo nguồn được áp dụng theo nhóm phép kiểm định đã khai báo; không diễn giải một khoảng tin cậy chứa 0 là bằng chứng cải thiện.
- Không so sánh trực tiếp trị tuyệt đối giữa last-query và all-prefix vì số lượng và cấu trúc query khác nhau.

## 8. Trạng thái cuối

- Dữ liệu và artifact: **đầy đủ (14/14, thiếu 0)**.
- Kết quả đủ dùng cho phân tích cross-dataset: **đạt**.
- Tuyên bố KD và dynamic belief: **được hỗ trợ**.
- Tuyên bố E5 luôn vượt E1: **không được hỗ trợ**.
- Tuyên bố temporal order có ảnh hưởng nhân quả dưới frozen-checkpoint corruption: **được hỗ trợ**, với cường độ khác nhau theo dataset/thành phố.
