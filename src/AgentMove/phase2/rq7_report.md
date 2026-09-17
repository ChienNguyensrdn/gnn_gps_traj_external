# Phase 2 — RQ7: Belief memory tuần tự

## 1. Câu hỏi nghiên cứu

Việc cập nhật belief bằng lịch sử đã quan sát và xác suất chuyển tiếp có cải thiện dự đoán vị trí kế tiếp một cách ổn định trên nhiều miền dữ liệu hay không?

RQ7 kiểm tra bốn cơ chế:

1. **B0-static:** chỉ dùng phân phối của frozen E5-dual.
2. **B1-history:** bổ sung phân phối tần suất POI trong prefix hiện tại.
3. **B2-sequential:** dùng posterior của bước trước làm belief cho bước sau.
4. **B3-dbn:** bổ sung prior chuyển tiếp từ POI cuối prefix đến POI kế tiếp.

So sánh chính là B3-dbn với B0-static. Mục tiêu là xác định liệu transition prior có cải thiện ranking hay không và lợi ích đó phải đánh đổi bao nhiêu về calibration.

## 2. Mô hình belief memory

### 2.1. Phân phối nền

Với prefix quỹ đạo $x_{1:t}$, frozen E5-dual sinh logits $z_{t,c}$ cho mỗi POI ứng viên $c$. Phân phối nền là

$$
p_{0,t}(c)
=\operatorname{softmax}(z_t)_c
=\frac{\exp(z_{t,c})}{\sum_j\exp(z_{t,j})}.
$$

E5-dual được đóng băng trong toàn bộ RQ7. Không có tham số neural nào được cập nhật bằng validation hoặc test.

### 2.2. Phép hợp nhất belief

Gọi $e_t(c)$ là evidence của một biến thể và $w\geq 0$ là trọng số được chọn trên validation. Phân phối sau hợp nhất là

$$
q_t(c;w)
=\frac{p_{0,t}(c)e_t(c)^w}
{\sum_j p_{0,t}(j)e_t(j)^w}.
$$

Tương đương trong miền log:

$$
\log q_t(c;w)
=\log p_{0,t}(c)+w\log e_t(c)-\log Z_t.
$$

Khi $w=0$, $q_t=p_{0,t}$; evidence không ảnh hưởng đến dự đoán. Vì vậy, weight bằng 0 là một kết quả âm hợp lệ do validation lựa chọn, không phải lỗi thực thi.

### 2.3. Prior và transition chỉ được ước lượng từ train

Gọi $N_c$ là số lần POI $c$ xuất hiện trong train và $N_{i,c}$ là số chuyển tiếp từ POI $i$ sang $c$. Với smoothing $\alpha=1$, global prior được xác định bởi

$$
\pi(c)=\frac{N_c+\alpha}{\sum_j(N_j+\alpha)}.
$$

Các evidence của bốn biến thể là:

**B0-static** không thực hiện hợp nhất:

$$
q_t^{\mathrm{B0}}(c)=p_{0,t}(c).
$$

**B1-history** sử dụng tần suất POI trong prefix $x_{1:t}$:

$$
e_t^{\mathrm{hist}}(c)
=\frac{n_{1:t}(c)+\alpha\pi(c)}
{t+\alpha},
\qquad
q_t^{\mathrm{B1}}(c)=\operatorname{Fuse}
\left(p_{0,t},e_t^{\mathrm{hist}};w\right).
$$

Trong đó $n_{1:t}(c)$ là số lần POI $c$ xuất hiện trong prefix đã quan sát.

**B2-sequential** truyền posterior của query trước trong cùng trajectory:

$$
e_t^{\mathrm{seq}}(c)=
\begin{cases}
q_{t-1}^{\mathrm{B2}}(c), & t>1,\\
\pi(c), & t=1,
\end{cases}
$$

$$
q_t^{\mathrm{B2}}(c)=\operatorname{Fuse}
\left(p_{0,t},e_t^{\mathrm{seq}};w\right).
$$

Belief được reset tại biên trajectory, do đó không truyền thông tin giữa hai quỹ đạo khác nhau.

**B3-dbn** sử dụng transition prior điều kiện theo POI cuối prefix:

$$
e_t^{\mathrm{dbn}}(c)
=P(c\mid x_t)
=\frac{N_{x_t,c}+\alpha\pi(c)}
{\sum_jN_{x_t,j}+\alpha},
$$

$$
q_t^{\mathrm{B3}}(c)=\operatorname{Fuse}
\left(p_{0,t},e_t^{\mathrm{dbn}};w\right).
$$

B3 được gọi là DBN vì dự đoán hiện tại kết hợp xác suất neural với cạnh chuyển tiếp có điều kiện từ trạng thái quan sát gần nhất.

### 2.4. Chọn trọng số

Trọng số được tìm trên grid

$$
w\in\{0,0.25,0.5,0.75,1.0\}
$$

và chỉ sử dụng validation:

$$
w^*=\arg\max_w
\left(\operatorname{Recall@1}_{\mathrm{val}}(w)
+\operatorname{Recall@10}_{\mathrm{val}}(w)\right).
$$

Nếu nhiều weight có cùng objective, weight nhỏ hơn được ưu tiên. Test chỉ được đánh giá một lần với $w^*$ đã khóa.

## 3. Thiết kế thực nghiệm

- Dataset: TIST2015 macro 12 thành phố, WWW2019-Shanghai và YJMob100K-Dataset1.
- Seed neural: 42, 43 và 44.
- Backbone: frozen E5-dual/correct từ RQ4.
- Protocol all-prefix: mỗi prefix hợp lệ tạo một query dự đoán bước kế tiếp.
- Prior và transition chỉ được fit trên train; weight chỉ được chọn trên validation.
- Belief được reset ở biên trajectory; query chỉ sử dụng prefix đã quan sát.
- Paired bootstrap/sign-flip sử dụng cùng prefix và seed; Holm correction áp dụng chung.
- RQ7 dùng all-prefix nên không so trực tiếp trị tuyệt đối với RQ4 last-query.
- Với TIST2015, độ lệch chuẩn thể hiện biến thiên giữa 12 thành phố sau khi trung bình seed. Với WWW2019 và YJMob100K, độ lệch chuẩn thể hiện biến thiên giữa seed.

## 4. Kết quả tổng thể

| Dataset | Variant | Weight validation | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| TIST2015 macro (12 thành phố) | B0-static | 0 | 0.161012 ± 0.040820 | 0.330365 ± 0.060758 | 0.395597 ± 0.064729 | 0.241154 ± 0.048483 | **6.470186 ± 0.787828** | **0.942990 ± 0.025189** | **0.048865 ± 0.020591** |
| TIST2015 macro (12 thành phố) | B1-history | 0, 0.25 | 0.162550 ± 0.040398 | 0.335251 ± 0.058741 | 0.400626 ± 0.062819 | 0.243853 ± 0.047664 | 6.497756 ± 0.796906 | 0.953246 ± 0.033110 | 0.077771 ± 0.044464 |
| TIST2015 macro (12 thành phố) | B2-sequential | 0, 0.25, 0.5 | 0.160249 ± 0.039552 | 0.330579 ± 0.060148 | 0.395708 ± 0.064725 | 0.240779 ± 0.047770 | 6.534068 ± 0.784564 | 0.947524 ± 0.030622 | 0.059554 ± 0.026940 |
| TIST2015 macro (12 thành phố) | B3-dbn | 0.25 | **0.167215 ± 0.041271** | **0.341472 ± 0.060979** | **0.407096 ± 0.064773** | **0.249385 ± 0.049107** | 6.652262 ± 0.863169 | 0.967841 ± 0.028207 | 0.132365 ± 0.023120 |
| WWW2019-Shanghai | B0-static | 0 | 0.125890 ± 0.002821 | 0.237013 ± 0.002609 | 0.273185 ± 0.004000 | 0.179791 ± 0.003417 | **7.902174 ± 0.053170** | **0.970168 ± 0.001228** | **0.038188 ± 0.001712** |
| WWW2019-Shanghai | B1-history | 0.75, 1 | 0.125856 ± 0.002142 | **0.316932 ± 0.004841** | **0.355676 ± 0.005060** | 0.213018 ± 0.001624 | 8.826118 ± 0.487521 | 1.263000 ± 0.085223 | 0.456337 ± 0.093185 |
| WWW2019-Shanghai | B2-sequential | 0, 0.25 | 0.125508 ± 0.003005 | 0.242399 ± 0.005606 | 0.279683 ± 0.008371 | 0.181781 ± 0.003941 | 8.146263 ± 0.208141 | 0.977414 ± 0.007063 | 0.049140 ± 0.011361 |
| WWW2019-Shanghai | B3-dbn | 0.5, 0.75 | **0.170055 ± 0.004250** | 0.309288 ± 0.005519 | 0.341395 ± 0.003657 | **0.234498 ± 0.004104** | 8.025177 ± 0.352121 | 1.031688 ± 0.057974 | 0.212214 ± 0.094316 |
| YJMob100K-Dataset1 | B0-static | 0 | 0.253281 ± 0.001712 | 0.455368 ± 0.000933 | 0.532750 ± 0.000516 | 0.348964 ± 0.001345 | **4.806488 ± 0.003950** | **0.872399 ± 0.001566** | **0.013815 ± 0.001445** |
| YJMob100K-Dataset1 | B1-history | 0 | 0.253281 ± 0.001712 | 0.455368 ± 0.000933 | 0.532750 ± 0.000516 | 0.348964 ± 0.001345 | 4.806488 ± 0.003950 | 0.872399 ± 0.001566 | 0.013815 ± 0.001445 |
| YJMob100K-Dataset1 | B2-sequential | 0 | 0.253281 ± 0.001712 | 0.455368 ± 0.000933 | 0.532750 ± 0.000516 | 0.348964 ± 0.001345 | 4.806488 ± 0.003950 | 0.872399 ± 0.001566 | 0.013815 ± 0.001445 |
| YJMob100K-Dataset1 | B3-dbn | 0.25 | **0.261018 ± 0.001297** | **0.466883 ± 0.000038** | **0.542545 ± 0.000961** | **0.358624 ± 0.000622** | 4.954595 ± 0.013331 | 0.890353 ± 0.000539 | 0.136386 ± 0.001024 |

Chữ đậm biểu thị giá trị trung bình tốt nhất trong từng dataset. NLL, Brier và ECE càng thấp càng tốt. Danh sách weight của TIST2015 là tập các giá trị được chọn trên các thành phố và seed; không phải nhiều weight được dùng đồng thời trong một run.

## 5. Kiểm định B3-dbn so với B0-static

Positive effect luôn nghĩa là B3 tốt hơn B0. Với NLL và Brier, dấu đã được đảo trước khi kiểm định.

| Dataset | Metric | Effect | Bootstrap 95% CI | Holm p | Kết luận |
|---|---|---:|---:|---:|---|
| TIST2015 macro (12 thành phố) | Recall@1 | 0.006203 | 0.004915–0.007489 | 0.00179982 | Cải thiện có ý nghĩa |
| TIST2015 macro (12 thành phố) | MRR | 0.008231 | 0.007344–0.009128 | 0.00179982 | Cải thiện có ý nghĩa |
| TIST2015 macro (12 thành phố) | NLL | −0.182076 | −0.186616–−0.177473 | 0.00179982 | Suy giảm có ý nghĩa |
| TIST2015 macro (12 thành phố) | Brier | −0.024851 | −0.025756–−0.023958 | 0.00179982 | Suy giảm có ý nghĩa |
| WWW2019-Shanghai | Recall@1 | 0.044164 | 0.040377–0.047952 | 0.00179982 | Cải thiện có ý nghĩa |
| WWW2019-Shanghai | MRR | 0.054708 | 0.051734–0.057633 | 0.00179982 | Cải thiện có ý nghĩa |
| WWW2019-Shanghai | NLL | −0.123004 | −0.145927–−0.100332 | 0.00179982 | Suy giảm có ý nghĩa |
| WWW2019-Shanghai | Brier | −0.061521 | −0.065954–−0.057055 | 0.00179982 | Suy giảm có ý nghĩa |
| YJMob100K-Dataset1 | Recall@1 | 0.007737 | 0.007103–0.008397 | 0.00179982 | Cải thiện có ý nghĩa |
| YJMob100K-Dataset1 | MRR | 0.009660 | 0.009222–0.010113 | 0.00179982 | Cải thiện có ý nghĩa |
| YJMob100K-Dataset1 | NLL | −0.148107 | −0.151103–−0.145094 | 0.00179982 | Suy giảm có ý nghĩa |
| YJMob100K-Dataset1 | Brier | −0.017954 | −0.018459–−0.017453 | 0.00179982 | Suy giảm có ý nghĩa |

## 6. Phân tích kết quả

### 6.1. Transition prior cải thiện ranking trên cả ba dataset

B3-dbn cải thiện Recall@1 và MRR có ý nghĩa trên toàn bộ ba dataset. Mức tăng Recall@1 là 0.006203 trên TIST2015, 0.044164 trên WWW2019 và 0.007737 trên YJMob100K. Khoảng tin cậy của tất cả effect đều dương và Holm p đều nhỏ hơn 0.05.

Hiệu quả lớn nhất xuất hiện trên WWW2019: Recall@1 tăng từ 0.125890 lên 0.170055 và MRR tăng từ 0.179791 lên 0.234498. Điều này cho thấy transition từ POI cuối có tính dự báo mạnh hơn global prior trên dữ liệu Shanghai-ISP.

TIST2015 và YJMob100K có mức tăng nhỏ hơn nhưng ổn định. Vì B3 chỉ sử dụng transition fit từ train, weight chọn trên validation và prefix quá khứ, kết quả không chứa test leakage.

### 6.2. Ranking tăng nhưng calibration suy giảm

Trên cả ba dataset, B3 làm NLL và Brier xấu đi có ý nghĩa. ECE cũng tăng mạnh:

- TIST2015: 0.048865 lên 0.132365.
- WWW2019: 0.038188 lên 0.212214.
- YJMob100K: 0.013815 lên 0.136386.

Nguyên nhân là transition prior làm phân phối tập trung hơn vào các chuyển tiếp thường gặp. Cơ chế này có thể đẩy POI đúng lên thứ hạng cao hơn nhưng đồng thời tạo xác suất quá tự tin khi transition train không phù hợp query hiện tại. Vì weight được chọn bằng Recall@1 + Recall@10, validation objective không trực tiếp kiểm soát NLL, Brier hoặc ECE.

Do đó, B3 phù hợp khi mục tiêu chính là ranking, nhưng chưa phải một posterior được calibration tốt. Cần temperature scaling sau fusion hoặc chọn weight theo objective đa mục tiêu giữa ranking và calibration.

### 6.3. B1-history và B2-sequential không ổn định

Trên YJMob100K, validation chọn weight 0 cho cả B1 và B2 ở mọi seed, khiến chúng đúng bằng B0. Đây là bằng chứng âm: history histogram và posterior hồi quy không bổ sung tín hiệu đáng tin cậy cho miền này.

Trên TIST2015, weight của B1/B2 thay đổi giữa 0 và các giá trị dương; mức tăng trung bình nhỏ và calibration thường xấu hơn B0. Trên WWW2019, B1 cải thiện mạnh Recall@5/10 nhưng gần như không cải thiện Recall@1, đồng thời làm NLL, Brier và ECE suy giảm rất lớn. B2 chỉ tạo thay đổi nhỏ. Vì vậy, B1/B2 chưa cho thấy khả năng khái quát ổn định.

## 7. Trả lời RQ7

**Có, transition-aware belief cải thiện ranking ổn định.** B3-dbn vượt B0-static có ý nghĩa về Recall@1 và MRR trên cả TIST2015, WWW2019 và YJMob100K. Do đó, giả thuyết về lợi ích ranking của transition prior được xác nhận qua ba miền dữ liệu.

**Tuy nhiên, belief hiện tại chưa được calibration tốt.** NLL, Brier và ECE đều xấu đi trên cả ba dataset. Tuyên bố phù hợp là B3 cải thiện thứ hạng dự đoán, không phải cải thiện toàn diện chất lượng xác suất.

**B1-history và B2-sequential không được xác nhận.** Các cơ chế này phụ thuộc dataset, thường chọn weight bằng 0 hoặc gây suy giảm calibration.

## 8. Publication gate

- Đủ ba seed và ba dataset: **đạt**.
- TIST2015 macro đủ 12 thành phố: **đạt**.
- Prior/transition chỉ fit train: **đạt**.
- Weight chỉ chọn trên validation: **đạt**.
- Belief reset đúng biên trajectory: **đạt**.
- Paired prediction và Holm correction: **đạt**.
- Tuyên bố B3 cải thiện ranking qua ba dataset: **đạt**.
- Tuyên bố B3 cải thiện calibration: **không đạt**.
- Gate artifact: **ready**, không thiếu artifact.
