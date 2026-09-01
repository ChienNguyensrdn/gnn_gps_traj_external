# RQ2 — Bayesian student data-only

> Mọi prior và transition chỉ được fit trên train; test dùng protocol `last-query matched`. Các baseline deterministic chỉ tính một run, không tạo pseudo-seed.

## 1. Câu hỏi nghiên cứu

RQ2 kiểm tra mức chất lượng có thể đạt được khi suy luận vị trí kế tiếp chỉ từ thống kê dữ liệu, không dùng LLM hoặc OSM. Bốn baseline data-only được so với quantitative teacher trên cùng preprocessing, temporal split, candidate space và test query của Tokyo.

- `unigram`: prior tần suất POI toàn cục.
- `markov-bigram`: xác suất chuyển tiếp bậc một từ POI gần nhất.
- `bn-data-only`: geometric fusion giữa empirical prior theo user và target-time.
- `dbn-data-only`: BN data-only có thêm first-order transition prior.
- `quantitative-teacher`: GRU teacher định lượng, chạy với seed 42, 43 và 44.

## 2. Kết quả test

| Variant | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| unigram | 0.024322 | 0.093097 | 0.121403 | 0.057474 | 8.753139 | 0.997667 | 0.002051 |
| markov-bigram | 0.095581 | 0.214552 | 0.264645 | 0.151951 | 9.857841 | 0.999326 | 0.093593 |
| bn-data-only | 0.084041 | 0.214190 | 0.288398 | 0.147744 | 9.348380 | 0.999483 | 0.083009 |
| dbn-data-only | 0.104171 | 0.245498 | 0.316860 | 0.171933 | 9.507008 | 0.999571 | 0.103411 |
| quantitative-teacher | 0.143638 ± 0.001971 | 0.290502 ± 0.002346 | 0.344891 ± 0.003259 | 0.213183 ± 0.001935 | 8.799284 ± 0.278926 | 0.948259 ± 0.002997 | 0.035296 ± 0.013116 |

## 3. Paired teacher comparisons

Positive effect nghĩa là quantitative teacher tốt hơn baseline đứng sau. Với NLL và Brier, dấu đã được quy đổi để giá trị dương luôn mang nghĩa teacher tốt hơn. Holm correction được áp dụng cho toàn bộ phép kiểm định.

| Comparison | Metric | Effect favoring teacher | 95% CI | Holm p | Significant |
|---|---|---:|---:|---:|---|
| quantitative-teacher-vs-unigram | recall@1 | 0.119316 | 0.116384–0.122283 | 0.00239976 | yes |
| quantitative-teacher-vs-unigram | recall@5 | 0.197406 | 0.193611–0.201183 | 0.00239976 | yes |
| quantitative-teacher-vs-unigram | recall@10 | 0.223487 | 0.219468–0.227472 | 0.00239976 | yes |
| quantitative-teacher-vs-unigram | mrr | 0.155709 | 0.152913–0.158490 | 0.00239976 | yes |
| quantitative-teacher-vs-unigram | nll | -0.046144 | -0.097842–0.005326 | 0.0791921 | no |
| quantitative-teacher-vs-unigram | brier | 0.049408 | 0.047472–0.051355 | 0.00239976 | yes |
| quantitative-teacher-vs-markov-bigram | recall@1 | 0.048058 | 0.045160–0.050956 | 0.00239976 | yes |
| quantitative-teacher-vs-markov-bigram | recall@5 | 0.075950 | 0.072380–0.079418 | 0.00239976 | yes |
| quantitative-teacher-vs-markov-bigram | recall@10 | 0.080246 | 0.076606–0.083954 | 0.00239976 | yes |
| quantitative-teacher-vs-markov-bigram | mrr | 0.061232 | 0.058630–0.063765 | 0.00239976 | yes |
| quantitative-teacher-vs-markov-bigram | nll | 1.058558 | 1.003188–1.114897 | 0.00239976 | yes |
| quantitative-teacher-vs-markov-bigram | brier | 0.051067 | 0.049145–0.053005 | 0.00239976 | yes |
| quantitative-teacher-vs-bn-data-only | recall@1 | 0.059598 | 0.056630–0.062530 | 0.00239976 | yes |
| quantitative-teacher-vs-bn-data-only | recall@5 | 0.076313 | 0.072966–0.079694 | 0.00239976 | yes |
| quantitative-teacher-vs-bn-data-only | recall@10 | 0.056493 | 0.053233–0.059822 | 0.00239976 | yes |
| quantitative-teacher-vs-bn-data-only | mrr | 0.065439 | 0.062959–0.067885 | 0.00239976 | yes |
| quantitative-teacher-vs-bn-data-only | nll | 0.549096 | 0.495466–0.604454 | 0.00239976 | yes |
| quantitative-teacher-vs-bn-data-only | brier | 0.051223 | 0.049271–0.053183 | 0.00239976 | yes |
| quantitative-teacher-vs-dbn-data-only | recall@1 | 0.039467 | 0.036656–0.042348 | 0.00239976 | yes |
| quantitative-teacher-vs-dbn-data-only | recall@5 | 0.045004 | 0.041848–0.048196 | 0.00239976 | yes |
| quantitative-teacher-vs-dbn-data-only | recall@10 | 0.028031 | 0.024874–0.031239 | 0.00239976 | yes |
| quantitative-teacher-vs-dbn-data-only | mrr | 0.041249 | 0.038962–0.043566 | 0.00239976 | yes |
| quantitative-teacher-vs-dbn-data-only | nll | 0.707723 | 0.652178–0.761981 | 0.00239976 | yes |
| quantitative-teacher-vs-dbn-data-only | brier | 0.051312 | 0.049358–0.053252 | 0.00239976 | yes |

## 4. Phân tích kết quả

### 4.1. DBN là baseline data-only tốt nhất về ranking

`dbn-data-only` đạt R@1 = 0,104171, R@5 = 0,245498, R@10 = 0,316860 và MRR = 0,171933, cao nhất trong bốn baseline chỉ dùng dữ liệu. So với `bn-data-only`, DBN tăng mô tả 0,020130 R@1, 0,031308 R@5, 0,028462 R@10 và 0,024189 MRR. Điều này cho thấy transition từ POI gần nhất bổ sung tín hiệu hữu ích ngoài user/time priors.

Aggregator đã được bổ sung paired test trực tiếp `dbn-data-only-vs-bn-data-only`. Bảng hiện tại vẫn chưa ghi kết quả kiểm định này vì cần chạy lại `aggregate` trên server chứa per-query predictions; cho tới lúc đó, mức tăng DBN–BN vẫn chỉ là chênh lệch mô tả.

### 4.2. Teacher vẫn vượt toàn bộ baseline data-only

Quantitative teacher đạt R@1 = 0,143638 và MRR = 0,213183. Ngay cả khi so với DBN mạnh nhất, teacher vẫn cao hơn 0,039467 R@1, 0,045004 R@5, 0,028031 R@10 và 0,041249 MRR. Tất cả chênh lệch ranking này có bootstrap 95% CI không cắt 0 và Holm-adjusted p = 0,00239976.

Kết quả trả lời phần chính của RQ2: empirical Bayesian priors có thể tạo một baseline có năng lực đáng kể, nhưng dữ liệu transition/user/time đơn thuần chưa thay thế được representation được học bởi quantitative teacher.

### 4.3. BN và Markov thể hiện các ưu thế khác nhau theo cutoff

Markov có R@1 = 0,095581, cao hơn BN = 0,084041, nhưng BN có R@10 = 0,288398, cao hơn Markov = 0,264645. Markov tập trung tốt hơn vào dự đoán đầu bảng, trong khi geometric user/time fusion mở rộng candidate coverage ở top-10. DBN kết hợp cả hai nguồn tín hiệu và đứng đầu nhóm data-only trên toàn bộ Recall@K/MRR.

### 4.4. Không dùng ECE thấp của unigram để kết luận mô hình tốt

Unigram có ECE rất thấp (0,002051) và NLL = 8,753139, nhưng ranking rất yếu: R@1 chỉ 0,024322 và MRR chỉ 0,057474. Một phân phối gần prior toàn cục có thể ít tự tin và tạo ECE thấp dù gần như không cá nhân hóa đúng vị trí kế tiếp. Brier của unigram cũng rất xấu (0,997667).

Do đó calibration phải được đọc đồng thời với Recall/MRR, NLL, Brier và reliability behavior. Không được claim unigram là baseline tốt nhất chỉ vì ECE hoặc NLL thấp. NLL giữa teacher và unigram cũng chưa khác biệt có ý nghĩa sau Holm correction (CI cắt 0; Holm p = 0,0791921).

### 4.5. Xác suất data-only cần được hiệu chỉnh tốt hơn

DBN cải thiện ranking nhưng có NLL = 9,507008, Brier = 0,999571 và ECE = 0,103411, đều xấu hơn teacher. Đây là trade-off rõ giữa candidate ranking và chất lượng xác suất. Nếu dùng DBN làm belief module hoặc prior cho downstream fusion, cần calibration trên validation thay vì dùng trực tiếp xác suất empirical chưa hiệu chỉnh.

## 5. Kết luận RQ2

- DBN là baseline data-only mạnh nhất về Recall@K và MRR trong bốn cấu hình đã chạy.
- First-order transition mang lại giá trị bổ sung so với user/time BN, nhưng chênh lệch DBN–BN hiện mới là mô tả vì chưa có paired test trực tiếp.
- Quantitative teacher vượt tất cả baseline data-only về mọi metric ranking với ý nghĩa thống kê sau Holm correction.
- Data-only Bayesian inference chưa thay thế được learned teacher, đặc biệt ở R@1 và MRR.
- ECE thấp đơn lẻ không đồng nghĩa với dự đoán tốt; unigram là ví dụ rõ nhất.

## 6. Protocol gate và giới hạn

- Gate: **ready-tokyo-matched-last-query**.
- Train chỉ được dùng để fit empirical prior/transition; test không dùng để tuning.
- Teacher dùng seed 42–44; baseline deterministic chỉ tính một run.
- Paired tests dùng cùng test query; Holm correction áp dụng cho toàn bộ teacher comparisons.
- Đây là categorical POI data-only experiment, không dùng LLM hoặc OSM.
- Kết luận hiện chỉ áp dụng cho TIST2015–Tokyo; chưa phải kết quả 12-city.
- Paired significance `dbn-data-only-vs-bn-data-only` đã có trong aggregator; cần chạy lại `aggregate` và cập nhật hàng kết quả trước khi khóa claim “DBN tốt hơn BN”.
