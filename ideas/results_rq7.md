# RQ7 — Phân tích Belief Memory

## 1. Câu hỏi nghiên cứu

RQ7 trả lời câu hỏi: **việc duy trì belief theo chuỗi có cải thiện dự đoán địa
điểm kế tiếp so với suy luận độc lập tại từng bước hay không?**

Thực nghiệm được tiến hành trên **TIST2015–Tokyo**, sử dụng checkpoint
`E5-dual/correct` đã đóng băng với ba seed `42, 43, 44`. Mỗi trajectory được
đánh giá trên toàn bộ prefix theo thứ tự thời gian. Belief được reset khi bắt
đầu trajectory mới.

Transition và prior chỉ được ước lượng trên tập train. Trọng số kết hợp được
chọn trên validation; test không được dùng để điều chỉnh mô hình.

## 2. Các biến thể

- **B0-static:** E5-dual suy luận độc lập tại mỗi bước, không duy trì belief.
- **B1-history:** kết hợp dự đoán E5-dual với prior tần suất POI trong prefix
  đã quan sát.
- **B2-sequential:** posterior của bước trước được truyền sang bước tiếp theo
  như một sequential belief.
- **B3-dbn:** kết hợp E5-dual với prior chuyển trạng thái bậc một từ POI hiện
  tại, tương ứng với một Dynamic Bayesian Network đơn giản.

## 3. Kết quả tổng hợp trên test

| Biến thể | Weight chọn trên validation | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0-static | 0, 0, 0 | 0.141972 | 0.313228 | 0.383372 | 0.223297 | 7.130893 | 0.949882 | 0.042341 |
| B1-history | 0, 0, 0 | 0.141972 | 0.313228 | 0.383372 | 0.223297 | 7.130893 | 0.949882 | 0.042341 |
| B2-sequential | 0, 0, 0 | 0.141972 | 0.313228 | 0.383372 | 0.223297 | 7.130893 | 0.949882 | 0.042341 |
| B3-dbn | 0.25, 0.25, 0.25 | **0.148607** | **0.325217** | **0.396834** | **0.232260** | 7.487551 | 0.975386 | 0.142256 |

## 4. Phân tích kết quả

### 4.1. History và sequential belief không vượt baseline

`B1-history` và `B2-sequential` đều chọn weight bằng `0` ở cả ba seed. Vì vậy,
hai biến thể này rút gọn chính xác về `B0-static`; mọi effect đều bằng 0,
95% CI bằng `0–0` và Holm-adjusted p bằng `1`.

Đây là một **negative result hợp lệ**: với biểu diễn E5-dual và cách xây dựng
belief hiện tại, tần suất lịch sử và recursive posterior không cung cấp thêm
thông tin hữu ích trên validation. Kết quả này không chứng minh mọi dạng
sequential belief đều không hiệu quả; nó chỉ bác bỏ hai cơ chế cụ thể được kiểm
tra trong protocol này.

### 4.2. DBN cải thiện rõ rệt chất lượng xếp hạng

`B3-dbn` chọn cùng weight `0.25` ở cả ba seed, cho thấy lựa chọn hyperparameter
ổn định. So với `B0-static`, DBN đạt:

| Metric | B0-static | B3-dbn | Chênh lệch tuyệt đối | Cải thiện tương đối |
|---|---:|---:|---:|---:|
| Recall@1 | 0.141972 | 0.148607 | +0.006636 | +4.67% |
| Recall@5 | 0.313228 | 0.325217 | +0.011989 | +3.83% |
| Recall@10 | 0.383372 | 0.396834 | +0.013462 | +3.51% |
| MRR | 0.223297 | 0.232260 | +0.008963 | +4.01% |

Các cải thiện đều có bootstrap 95% CI không chứa 0 và Holm-adjusted
`p = 0.00239976`. Do đó, first-order transition prior mang thêm tín hiệu hữu ích
cho việc xếp hạng địa điểm kế tiếp ngoài thông tin đã có trong E5-dual.

### 4.3. DBN tạo ra trade-off calibration

Khả năng xếp hạng tốt hơn không đi kèm xác suất tốt hơn:

| Metric calibration | B0-static | B3-dbn | Thay đổi | Diễn giải |
|---|---:|---:|---:|---|
| NLL↓ | 7.130893 | 7.487551 | +0.356657 | Xấu hơn |
| Brier↓ | 0.949882 | 0.975386 | +0.025503 | Xấu hơn |
| ECE↓ | 0.042341 | 0.142256 | +0.099915 | Xấu hơn rõ rệt |

Theo quy ước paired test, effect dương nghĩa là biến thể đứng trước tốt hơn và
dấu của NLL/Brier đã được đảo. Vì vậy effect `-0.356657` cho NLL và `-0.025503`
cho Brier cho thấy DBN kém hơn baseline; cả hai khác biệt đều có ý nghĩa thống
kê sau Holm correction.

Nguyên nhân hợp lý là weight được chọn bằng mục tiêu
`Recall@1 + Recall@10`, không phải NLL hoặc ECE. Transition prior làm thay đổi
thứ hạng đúng hướng nhưng đồng thời làm phân phối xác suất quá sắc hoặc bị lệch.

## 5. Kết luận RQ7

Trên **TIST2015–Tokyo**, belief memory chỉ mang lại lợi ích khi được biểu diễn
bằng transition-aware DBN. First-order DBN cải thiện nhất quán và có ý nghĩa
thống kê toàn bộ ranking metrics, trong khi history-frequency và recursive
posterior memory bị validation loại bỏ. Tuy nhiên, DBN gây suy giảm đáng kể về
calibration, tạo ra trade-off giữa chất lượng xếp hạng và độ tin cậy xác suất.

Claim phù hợp:

> Trên TIST2015–Tokyo, first-order DBN fusion cải thiện có ý nghĩa thống kê khả
> năng xếp hạng next-location so với independent-step inference, nhưng làm suy
> giảm probabilistic calibration.

## 6. Giới hạn và công việc tiếp theo

- Kết quả hiện chỉ áp dụng cho Tokyo; chưa được phép suy diễn thành kết luận cho
  đủ 12 thành phố TIST2015.
- RQ7 dùng toàn bộ prefix, nên không so trực tiếp trị tuyệt đối với RQ4/RQ6 vốn
  chỉ đánh giá query cuối của mỗi trajectory.
- ECE không được đưa vào paired significance vì metric này không phân rã trực
  tiếp theo từng query.
- Calibration của DBN cần được xử lý bằng temperature scaling hoặc calibration
  sau fusion trong RQ11.
- Kết quả RQ7 cung cấp nền tảng cho RQ8: dùng uncertainty để quyết định khi nào
  cần gọi LLM thay vì gọi ở mọi query.

## 7. Trạng thái publication gate

- Dataset/city: `TIST2015–Tokyo`.
- Seeds: `42, 43, 44` — **đủ**.
- Evaluation split: test all-prefix — **đủ**.
- Per-query paired predictions — **đủ**.
- Validation-only hyperparameter selection — **đạt**.
- RQ7 Tokyo gate — **ready**.
- TIST2015 12-city gate — **chưa hoàn thành**.
