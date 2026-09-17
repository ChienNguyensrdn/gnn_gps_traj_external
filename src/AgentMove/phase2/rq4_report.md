# Phase 2 — RQ4: Chắt lọc tri thức và dual-axis evolution

## 1. Câu hỏi nghiên cứu

Knowledge distillation (KD) có cải thiện mô hình chỉ học bằng cross-entropy hay không, và dual-axis evolution có tạo thêm lợi ích ngoài KD trên nhiều miền dữ liệu hay không?

RQ4 tách hai giả thuyết:

1. **Đóng góp của KD:** so sánh E1-kd với E0-ce.
2. **Đóng góp bổ sung của dual-axis:** so sánh E5-dual với E1-kd.

### 1.1. Ký hiệu và hàm mất mát

Với một mẫu đầu vào quỹ đạo $x$, POI đích $y$, student $S$ và teacher $T$, ký hiệu $z^S,z^T$ là logits của hai mô hình. Phân phối dự đoán với temperature $\tau$ là

$$
p^S_{\tau}(c\mid x)=\frac{\exp(z^S_c/\tau)}{\sum_j\exp(z^S_j/\tau)},
\qquad
p^T_{\tau}(c\mid x)=\frac{\exp(z^T_c/\tau)}{\sum_j\exp(z^T_j/\tau)}.
$$

Mất mát cross-entropy sử dụng nhãn thật:

$$
\mathcal{L}_{\mathrm{CE}}=-\log p^S_{1}(y\mid x).
$$

Mất mát knowledge distillation buộc student mô phỏng toàn bộ phân phối mềm của teacher:

$$
\mathcal{L}_{\mathrm{KD}}
=\tau^2 D_{\mathrm{KL}}\!\left(p^T_{\tau}\,\|\,p^S_{\tau}\right).
$$

Hệ số $\tau^2$ giữ độ lớn gradient ổn định khi thay đổi temperature. Khác với CE chỉ truyền thông tin về POI đúng, KD còn truyền quan hệ tương đối giữa POI đúng và các ứng viên còn lại.

Gọi $h^{S,(l)}$ và $h^{T,(l)}$ là trạng thái ẩn tại tầng $l$. Do teacher và student có thể có không gian biểu diễn khác nhau, một phép chiếu học được $A_l$ ánh xạ trạng thái student sang không gian teacher. Mất mát khớp trạng thái theo chiều sâu là

$$
\mathcal{L}_{\mathrm{traj}}
=\frac{1}{L}\sum_{l=1}^{L}
\left\|A_lh^{S,(l)}-h^{T,(l)}\right\|_2^2.
$$

Mất mát vận tốc biểu diễn đo sự thay đổi giữa hai tầng liên tiếp:

$$
\mathcal{L}_{\mathrm{vel}}
=\frac{1}{L-1}\sum_{l=1}^{L-1}
\left\|
\left(A_{l+1}h^{S,(l+1)}-A_lh^{S,(l)}\right)
-\left(h^{T,(l+1)}-h^{T,(l)}\right)
\right\|_2^2.
$$

Gọi $u^S_t,u^T_t$ là trạng thái tuần tự tại bước thời gian $t$, và $m_t$ là mask loại bỏ padding. Thành phần theo chiều thời gian là

$$
\mathcal{L}_{\mathrm{temp}}
=\frac{1}{\sum_t m_t}
\sum_{t=2}^{n}m_t
\left\|
(u^S_t-u^S_{t-1})-(u^T_t-u^T_{t-1})
\right\|_2^2.
$$

Hai chiều tiến hóa trong E5 gồm: (i) **chiều sâu biểu diễn**, được mô hình hóa bởi $\mathcal{L}_{\mathrm{traj}}$ và $\mathcal{L}_{\mathrm{vel}}$; và (ii) **chiều thời gian**, được mô hình hóa bởi $\mathcal{L}_{\mathrm{temp}}$.

### 1.2. E0, E1 và E5 khác nhau như thế nào?

Hàm mục tiêu tổng quát là

$$
\mathcal{L}
=\mathcal{L}_{\mathrm{CE}}
+\lambda_{\mathrm{KD}}\mathcal{L}_{\mathrm{KD}}
+\lambda_{\mathrm{traj}}\mathcal{L}_{\mathrm{traj}}
+\lambda_{\mathrm{vel}}\mathcal{L}_{\mathrm{vel}}
+\lambda_{\mathrm{temp}}\mathcal{L}_{\mathrm{temp}}.
$$

| Biến thể | $\lambda_{\mathrm{KD}}$ | $\lambda_{\mathrm{traj}}$ | $\lambda_{\mathrm{vel}}$ | $\lambda_{\mathrm{temp}}$ | Ý nghĩa |
|---|---:|---:|---:|---:|---|
| E0-ce | 0 | 0 | 0 | 0 | Student chỉ học từ nhãn thật bằng CE. |
| E1-kd | 1 | 0 | 0 | 0 | Thêm phân phối mềm đầu ra của teacher. |
| E5-dual | 1 | 1 | 1 | 1 | Thêm KD, khớp trạng thái theo tầng, vận tốc giữa tầng và biến đổi theo thời gian. |

Do đó, hiệu $\mathrm{E1}-\mathrm{E0}$ đo lợi ích của distillation ở đầu ra, còn hiệu $\mathrm{E5}-\mathrm{E1}$ cô lập lợi ích bổ sung của việc truyền trạng thái theo hai chiều sâu–thời gian khi KD đã được giữ cố định.

## 2. Thiết kế thực nghiệm

- Ba bộ dữ liệu: TIST2015, WWW2019-Shanghai và YJMob100K-Dataset1.
- Các biến thể: E0-ce, E1-kd và E5-dual.
- Mỗi thí nghiệm sử dụng các seed 42, 43 và 44.
- Checkpoint được lựa chọn bằng validation; test không được dùng để tuning.
- Đánh giá theo protocol last-query, với cùng split và candidate space trong từng dataset.
- TIST2015 được tính macro trung bình đều trên 12 thành phố. Độ lệch chuẩn của TIST2015 thể hiện biến thiên giữa các thành phố sau khi trung bình seed.
- Độ lệch chuẩn của WWW2019 và YJMob100K thể hiện biến thiên giữa ba seed.
- Kiểm định paired bootstrap/sign-flip sử dụng cùng query và seed. Holm correction được áp dụng chung cho họ kiểm định.

## 3. Kết quả tổng thể

| Dataset | Variant | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TIST2015 macro (12 thành phố) | E0-ce | 0.155951 ± 0.050665 | 0.304878 ± 0.068963 | 0.353451 ± 0.069284 | 0.225894 ± 0.058201 | 7.434454 ± 1.299340 | 0.945238 ± 0.035238 | 0.055881 ± 0.017526 |
| TIST2015 macro (12 thành phố) | E1-kd | 0.167809 ± 0.051573 | 0.320899 ± 0.067519 | 0.377576 ± 0.074122 | 0.241300 ± 0.059604 | 6.709523 ± 0.911891 | 0.936317 ± 0.032913 | 0.049285 ± 0.018809 |
| TIST2015 macro (12 thành phố) | E5-dual | **0.171948 ± 0.052475** | **0.327696 ± 0.069439** | **0.384623 ± 0.073221** | **0.245935 ± 0.059480** | **6.673185 ± 0.917733** | **0.933293 ± 0.034057** | **0.047875 ± 0.019735** |
| WWW2019-Shanghai | E0-ce | 0.124150 ± 0.006568 | 0.237448 ± 0.005500 | 0.280978 ± 0.006620 | 0.179870 ± 0.007604 | 9.575063 ± 0.429188 | 1.023047 ± 0.012787 | 0.190014 ± 0.022756 |
| WWW2019-Shanghai | E1-kd | 0.133810 ± 0.000947 | 0.241383 ± 0.004559 | 0.280859 ± 0.007847 | 0.186597 ± 0.002905 | 8.007897 ± 0.056267 | **0.962484 ± 0.001908** | **0.030829 ± 0.005273** |
| WWW2019-Shanghai | E5-dual | **0.136076 ± 0.005430** | **0.251282 ± 0.003792** | **0.285510 ± 0.005680** | **0.191104 ± 0.005391** | **7.899574 ± 0.043841** | 0.963314 ± 0.000597 | 0.042302 ± 0.008042 |
| YJMob100K-Dataset1 | E0-ce | 0.471781 ± 0.002855 | 0.695324 ± 0.001386 | 0.753256 ± 0.001683 | 0.572654 ± 0.001463 | 3.544852 ± 0.032933 | 0.746958 ± 0.003052 | **0.148716 ± 0.005496** |
| YJMob100K-Dataset1 | E1-kd | **0.490048 ± 0.002986** | 0.712429 ± 0.001970 | 0.773871 ± 0.002187 | **0.591052 ± 0.001965** | **3.094815 ± 0.004344** | **0.729262 ± 0.003751** | 0.151829 ± 0.008342 |
| YJMob100K-Dataset1 | E5-dual | 0.487083 ± 0.004520 | **0.712879 ± 0.001254** | **0.774583 ± 0.000673** | 0.589774 ± 0.002882 | 3.098946 ± 0.012630 | 0.731955 ± 0.005276 | 0.152707 ± 0.003880 |

Chữ đậm biểu thị giá trị trung bình tốt nhất trong từng dataset. NLL, Brier và ECE càng thấp càng tốt.

## 4. Kiểm định giả thuyết trên Recall@1

Positive effect nghĩa là biến thể đứng trước tốt hơn. Holm correction được áp dụng trước khi xác định ý nghĩa thống kê.

| Dataset | So sánh | Effect | Bootstrap 95% CI | Holm p | Kết luận |
|---|---|---:|---:|---:|---|
| TIST2015 macro (12 thành phố) | E1-kd vs E0-ce | 0.011859 | 0.009030–0.014745 | 0.00359964 | Cải thiện có ý nghĩa |
| TIST2015 macro (12 thành phố) | E5-dual vs E1-kd | 0.004139 | 0.001536–0.006611 | 0.0149985 | Cải thiện có ý nghĩa |
| WWW2019-Shanghai | E1-kd vs E0-ce | 0.009660 | 0.003816–0.015385 | 0.0265973 | Cải thiện có ý nghĩa |
| WWW2019-Shanghai | E5-dual vs E1-kd | 0.002266 | −0.002504–0.007036 | 1.000000 | Chưa có ý nghĩa |
| YJMob100K-Dataset1 | E1-kd vs E0-ce | 0.018267 | 0.014519–0.022158 | 0.00359964 | Cải thiện có ý nghĩa |
| YJMob100K-Dataset1 | E5-dual vs E1-kd | −0.002965 | −0.006358–0.000474 | 0.830617 | Chưa có ý nghĩa |

## 5. Phân tích kết quả

### 5.1. KD tạo cải thiện nhất quán

E1-kd vượt E0-ce về Recall@1 trên cả ba bộ dữ liệu, với khoảng tin cậy không chứa 0 và Holm p nhỏ hơn 0.05. Mức tăng tuyệt đối lần lượt là 0.011859 trên TIST2015, 0.009660 trên WWW2019 và 0.018267 trên YJMob100K. Kết quả xác nhận phân phối mềm của teacher cung cấp tín hiệu học hữu ích hơn so với chỉ sử dụng nhãn cứng bằng cross-entropy.

KD cũng cải thiện MRR và phần lớn các ranking metric trung bình. NLL và Brier giảm rõ rệt, đặc biệt trên WWW2019 và YJMob100K. Vì vậy, KD là đóng góp ổn định nhất của RQ4 và có khả năng khái quát qua các miền dữ liệu khác nhau.

### 5.2. Dual-axis mang lại lợi ích phụ thuộc miền

Trên TIST2015, E5-dual đạt kết quả trung bình tốt nhất trên toàn bộ ranking và calibration metrics. So với E1-kd, Recall@1 tăng 0.004139 và có ý nghĩa sau Holm correction. Đây là bằng chứng rằng dual-axis có thể bổ sung thông tin hữu ích ngoài KD trên dữ liệu check-in đa thành phố.

Trên WWW2019, E5-dual có trung bình R@1, R@5, R@10, MRR và NLL tốt hơn E1-kd, nhưng mức tăng Recall@1 không có ý nghĩa thống kê. Brier và ECE cũng xấu hơn E1-kd. Do đó, kết quả chỉ cho thấy xu hướng cải thiện ranking, chưa đủ để xác nhận lợi ích dual-axis trên Shanghai.

Trên YJMob100K, E5-dual tăng nhẹ R@5 và R@10 nhưng giảm R@1 và MRR so với E1-kd. Effect Recall@1 là âm và không có ý nghĩa. NLL, Brier và ECE cũng không được cải thiện. Dual-axis vì vậy không tạo lợi ích rõ ràng trên dữ liệu mobility dạng cell-based này.

### 5.3. Đánh đổi calibration

TIST2015 cho thấy ranking và calibration cùng được cải thiện khi chuyển từ E0 sang E1 rồi E5. Trên WWW2019, E1-kd giảm ECE từ 0.190014 xuống 0.030829, trong khi E5 tăng ECE lên 0.042302. Trên YJMob100K, KD cải thiện ranking và proper scoring rules nhưng ECE tăng nhẹ. Điều này cho thấy chất lượng xếp hạng và calibration không luôn biến đổi cùng chiều; temperature scaling hoặc calibration sau huấn luyện vẫn cần được báo cáo riêng.

## 6. Trả lời RQ4

**Knowledge distillation được xác nhận:** E1-kd cải thiện Recall@1 có ý nghĩa trên cả TIST2015, WWW2019 và YJMob100K. Đây là kết quả có thể sử dụng làm tuyên bố chính trong bài báo.

**Dual-axis evolution chỉ được xác nhận theo miền:** E5-dual tạo cải thiện bổ sung có ý nghĩa trên TIST2015, nhưng chưa được xác nhận trên WWW2019 và YJMob100K. Vì vậy, không nên tuyên bố dual-axis cải thiện phổ quát; kết luận phù hợp là lợi ích của cơ chế này phụ thuộc đặc điểm dữ liệu.

## 7. Publication gate

- Đủ ba seed cho cả ba dataset: **đạt**.
- Đủ TIST2015 macro 12 thành phố: **đạt**.
- Paired prediction và kiểm định Holm: **đạt**.
- Tuyên bố KD khái quát qua ba dataset: **đạt**.
- Tuyên bố dual-axis khái quát qua ba dataset: **không đạt**.
- Gate artifact: **ready**, không thiếu artifact.
