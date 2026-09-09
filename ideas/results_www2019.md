# WWW2019–Shanghai — Báo cáo kiểm chứng cross-dataset

> Nguồn số liệu: `www2019_summary.json`, protocol `cross-dataset confirmation`,
> seed 42, 43, 44. Gate kỹ thuật: **ready-www2019**, không thiếu artifact.

## 1. Mục tiêu

Thí nghiệm kiểm tra liệu các kết luận chính trên TIST2015 có chuyển sang miền
WWW2019–Shanghai-ISP hay không. Shanghai được báo cáo như một dataset độc lập,
không gộp vào macro 12 thành phố TIST2015.

Ba nội dung chính gồm:

1. Distillation có cải thiện student chỉ học cross-entropy hay không.
2. Dual-axis evolution và thứ tự trajectory có tạo lợi ích ổn định hay không.
3. Bayesian belief và LLM bounded tạo ra trade-off ranking–calibration–chi phí
   như thế nào.

## 2. Neural student — full test, last-query

| Variant | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0-ce | 0,124150 ± 0,006568 | 0,237448 ± 0,005500 | 0,280978 ± 0,006620 | 0,179870 ± 0,007604 | 9,575063 ± 0,429188 | 1,023047 ± 0,012787 | 0,190014 ± 0,022756 |
| E1-kd | 0,133810 ± 0,000947 | 0,241383 ± 0,004559 | 0,280859 ± 0,007847 | 0,186597 ± 0,002905 | 8,007897 ± 0,056267 | 0,962484 ± 0,001908 | 0,030829 ± 0,005273 |
| E5-dual | 0,136076 ± 0,005430 | 0,251282 ± 0,003792 | 0,285510 ± 0,005680 | 0,191104 ± 0,005391 | 7,899574 ± 0,043841 | 0,963314 ± 0,000597 | 0,042302 ± 0,008042 |

E1 cải thiện rõ E0, đặc biệt ở R@1 và calibration. E5 có mean ranking tốt nhất
và NLL thấp nhất. Tuy nhiên, lợi ích E5 so với E1 không đồng đều trên mọi metric.

### Paired significance được hỗ trợ

Holm correction được áp dụng chung cho 24 phép kiểm định.

| Comparison | Metric | Effect favoring first | 95% CI | Holm p | Kết luận |
|---|---|---:|---:|---:|---|
| E1-kd vs E0-ce | R@1 | 0,009660 | 0,003816–0,015504 | 0,022798 | Có ý nghĩa |
| E1-kd vs E0-ce | NLL | 1,567166 | 1,504273–1,628866 | 0,002400 | Có ý nghĩa |
| E1-kd vs E0-ce | Brier | 0,060563 | 0,055705–0,065402 | 0,002400 | Có ý nghĩa |
| E5-dual vs E1-kd | R@5 | 0,009899 | 0,004890–0,014908 | 0,005999 | Có ý nghĩa |
| E5-dual vs E1-kd | NLL | 0,108322 | 0,088082–0,128384 | 0,002400 | Có ý nghĩa |

Các khác biệt E1–E0 về R@5, R@10 và MRR; E5–E1 về R@1, R@10, MRR và Brier
không còn ý nghĩa sau Holm correction. Vì vậy, kết luận được phép dùng là
distillation chuyển miền thành công; incremental gain của dual-axis chỉ được
xác nhận cho R@5 và NLL.

## 3. Bayesian belief — full test, all-prefix

| Variant | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0-static | 0,125890 ± 0,002821 | 0,237013 ± 0,002609 | 0,273185 ± 0,004000 | 0,179791 ± 0,003417 | 7,902174 ± 0,053170 | 0,970168 ± 0,001228 | 0,038188 ± 0,001712 |
| B3-dbn | 0,170055 ± 0,004250 | 0,309288 ± 0,005519 | 0,341395 ± 0,003657 | 0,234498 ± 0,004104 | 8,025177 ± 0,352121 | 1,031688 ± 0,057974 | 0,212214 ± 0,094316 |

B3-DBN tạo mức tăng ranking lớn nhưng làm NLL, Brier và ECE xấu hơn. Đây là
trade-off giống TIST2015: belief tuần tự hữu ích cho xếp hạng nhưng cần bước
calibration sau fusion trước khi sử dụng xác suất dự đoán.

Không so trực tiếp trị tuyệt đối bảng này với bảng neural vì Bayesian dùng mọi
prefix, còn neural dùng last-query.

## 4. LLM bounded — 200 test query

| Queries | Acc@1 | Acc@5 | Acc@10 | MRR | Candidate recall | ECE↓ | NLL↓ | Invalid rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 0,195000 | 0,315000 | 0,345000 | 0,246903 | 0,340000 | 0,154816 | 19,259944 | 0,000000 |

Chi phí trung bình là `1.554,23` input token, `204,26` output token và `4` API
call trên mỗi query. LLM latency trung bình `3,0485` giây, median `2,8359` giây
và p95 `4,7103` giây.

Ranking trên subset bounded khá cao và không có output evidence bất hợp lệ,
nhưng NLL/ECE cho thấy xác suất dự đoán calibration kém. Không so trực tiếp
Acc/MRR này với full-test neural hoặc all-prefix Bayesian vì khác query scope.
Kết quả chỉ áp dụng cho Qwen2:7b, `limit=200`, `no-OSM`.

## 5. Vấn đề temporal order

### 5.1. Kết quả hiện tại

Trong paired test của protocol hiện tại:

- `correct-vs-reverse`: không metric ranking nào có ý nghĩa sau Holm.
- `correct-vs-random`: không metric ranking nào có ý nghĩa sau Holm.
- R@1 correct–random có effect `0,004413`, nhưng Holm p `0,798320`.
- NLL correct–random có effect `-0,082767`, CI
  `-0,099651–-0,065630`, Holm p `0,002400`: random tốt hơn correct về NLL.

Dấu âm của NLL không phải lỗi tính toán. Quy ước paired effect đã đảo dấu cho
NLL/Brier để giá trị dương luôn có nghĩa variant đứng trước tốt hơn.

### 5.2. Corruption đã thực sự thay đổi dữ liệu

Kiểm tra trực tiếp trên `2.795` test query cho thấy:

| Diagnostic | Kết quả |
|---|---:|
| History length ≤ 1 | 4,94% |
| Reverse làm thay đổi input | 95,06% |
| Random làm thay đổi input | 79,25% |

Do đó, kết quả không significant không thể giải thích chủ yếu bằng trajectory
quá ngắn hoặc corruption không tác động lên input.

### 5.3. Giới hạn của protocol hiện tại

Các mô hình `correct`, `reverse` và `random` được huấn luyện, chọn checkpoint và
đánh giá trong order mode tương ứng. Thí nghiệm hiện tại vì vậy đo:

> Mô hình có thể thích nghi khi được huấn luyện lại trên dữ liệu mất hoặc đảo
> thứ tự hay không?

Nó chưa đo trực tiếp:

> Prediction của một mô hình đã học trên chronology đúng có phụ thuộc vào thứ
> tự quan sát tại inference hay không?

Vì vậy chưa được kết luận “temporal order không quan trọng trên Shanghai”. Kết
luận chính xác là: **lợi ích của correct order chưa được xác nhận dưới protocol
retraining-under-corruption**.

### 5.4. Control cần bổ sung

Giữ nguyên checkpoint `E5-dual/correct` của từng seed và chỉ thay đổi test input:

1. correct checkpoint + correct test;
2. cùng checkpoint + reverse test;
3. cùng checkpoint + random test.

Control này không cần huấn luyện lại. Nếu quality giảm có ý nghĩa, mô hình có sử
dụng chronology nhưng có khả năng thích nghi khi retrain trên corrupted order.
Nếu vẫn không giảm, giả thuyết temporal-order không được hỗ trợ trên Shanghai.

Script đã được chuẩn bị qua action `www2019_pipeline.sh frozen-order`; các con số
vẫn để trống cho tới khi ba seed được đánh giá và aggregate lại.

## 6. Kết luận cross-dataset

- Response KD là kết luận chuyển miền chắc chắn nhất: E1 vượt E0 về R@1, NLL và
  Brier sau Holm correction.
- E5 đạt mean ranking tốt nhất, nhưng incremental gain so với E1 chỉ được xác
  nhận ở R@5 và NLL.
- DBN tăng ranking mạnh nhưng gây suy giảm calibration rõ rệt.
- LLM bounded có ranking tốt nhưng chi phí cao và xác suất calibration kém;
  chưa phải bằng chứng full-query.
- Temporal-order chưa được xác nhận bằng protocol retraining hiện tại; cần frozen
  checkpoint inference corruption trước khi đưa ra kết luận cơ chế.

## 7. Publication gate

- Neural E0/E1/E5, seed 42–44: **ready**.
- Bayesian B0/B3, seed 42–44: **ready**, nhưng protocol all-prefix báo cáo riêng.
- Paired tests và Holm correction: **ready**.
- LLM Qwen2:7b, 200 query, no-OSM: **ready-bounded**.
- Temporal-order retraining-under-corruption: **ready**, không hỗ trợ giả thuyết.
- Temporal-order frozen-checkpoint control: **script-ready, result-missing**.
- Gate kỹ thuật toàn pipeline: **ready-www2019**.
- Gate cho tuyên bố đầy đủ về temporal mechanism: **chưa hoàn thành**.
