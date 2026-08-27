# RQ9 — Phân tích Semantic Knowledge Verification

## 1. Câu hỏi nghiên cứu

RQ9 kiểm tra personal memory và context/world knowledge có thực sự ảnh hưởng
đến dự đoán next-location hay không. Thực nghiệm sử dụng one-axis corruption:
memory variants luôn giữ context thật, còn context variants luôn giữ memory
thật.

Đây là bounded experiment trên 200 test queries của `TIST2015–Tokyo`, sử dụng
`qwen2:7b` và cùng Neural-CGM top-10 candidate set. Mỗi variant có LLM cache
riêng; corruption deterministic và test không được dùng để tuning.

## 2. Kết quả tổng thể

| Variant | R@1 | R@5 | R@10 | MRR | Invalid rate | Tokens/query | Latency mean (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| memory-true | 0.125000 | 0.255000 | 0.305000 | 0.191982 | 0.205000 | 1265.41 | 3.387925 |
| memory-shuffled | 0.130000 | 0.240000 | 0.305000 | 0.191349 | 0.175000 | 1253.49 | 3.303268 |
| memory-random-user | 0.130000 | 0.245000 | 0.305000 | 0.190641 | 0.215000 | 1092.34 | 3.099135 |
| memory-none | 0.115000 | 0.260000 | 0.305000 | 0.183730 | 0.080000 | 581.91 | 2.369035 |
| context-shuffled | **0.135000** | 0.255000 | 0.305000 | **0.199155** | 0.210000 | 1258.26 | 3.354803 |
| context-random-poi | 0.130000 | 0.250000 | 0.305000 | 0.192815 | 0.145000 | 1251.45 | 3.344594 |
| context-none | 0.130000 | 0.250000 | 0.305000 | 0.191982 | 0.095000 | 1160.17 | 3.075796 |

Recall@10 luôn bằng `0.305` vì mọi variant chỉ rerank cùng top-10 candidate
set. RQ9 vì vậy đo semantic reranking, không đo candidate recall.

## 3. Kết quả paired significance ban đầu

Không có corruption nào khác `memory-true` có ý nghĩa thống kê trên Recall@1,
Recall@5, Recall@10 hoặc MRR. Tất cả Holm-adjusted p đều bằng `1`.

Các xu hướng mean cũng không nhất quán:

- memory-none giảm Recall@1 `0.010` và MRR `0.008252`, nhưng lại tăng Recall@5
  `0.005`;
- memory-shuffled và memory-random-user tăng Recall@1 `0.005`;
- context-shuffled có Recall@1 và MRR cao hơn true;
- context-none có MRR giống hệt true.

Do đó chưa có bằng chứng rằng personal memory hoặc context/world prompt hiện
tại cải thiện ranking.

## 4. Invalid-output confound

Invalid-output rate khác nhau mạnh giữa các variant:

| Variant | Invalid rate |
|---|---:|
| memory-true | 20.5% |
| memory-shuffled | 17.5% |
| memory-random-user | 21.5% |
| memory-none | 8.0% |
| context-shuffled | 21.0% |
| context-random-poi | 14.5% |
| context-none | 9.5% |

Các prompt ngắn như memory-none/context-none tạo ít malformed/incomplete LLM
output hơn. Khi output invalid, pipeline điền phần ranking còn thiếu từ stage-1,
nên quality tổng thể trộn lẫn hai tác động:

1. tác động semantic của memory/context;
2. tác động format reliability và fallback.

Tokens/query của memory-none chỉ `581.91`, thấp hơn nhiều so với `1265.41` của
memory-true. Prompt length vì vậy là một confound thực sự, không chỉ là metric
hiệu năng phụ.

## 5. Jointly-valid paired analysis

Số query mà cả true và corruption đều sinh output hợp lệ nằm trong khoảng
`133–156`:

| Comparison | Jointly-valid queries | R@1 effect | R@5 effect | MRR effect | Kết luận sau Holm |
|---|---:|---:|---:|---:|---|
| true vs memory-shuffled | 143 | 0.000000 | +0.006993 | −0.000583 | Không significant |
| true vs memory-random-user | 133 | 0.000000 | +0.007519 | +0.003008 | Không significant |
| true vs memory-none | 151 | −0.006623 | −0.006623 | −0.005030 | Không significant |
| true vs context-shuffled | 149 | −0.013423 | 0.000000 | −0.009508 | Không significant |
| true vs context-random-poi | 149 | −0.006711 | +0.006711 | −0.002796 | Không significant |
| true vs context-none | 156 | −0.006410 | +0.006410 | −0.002671 | Không significant |

Tất cả jointly-valid Holm p đều bằng `1`. Việc loại bỏ output lỗi không làm
xuất hiện semantic gain bị che khuất. Negative result vì vậy mạnh hơn phân tích
all-query: với các output hợp lệ, true memory/context vẫn không vượt corruption.

## 6. Kiểm định invalid-output rate

Positive effect nghĩa là true có invalid rate thấp hơn corruption. Hai khác biệt
có ý nghĩa thống kê đều mang dấu âm:

| Comparison | Effect favoring true | 95% CI | Holm p | Kết luận |
|---|---:|---:|---:|---|
| true vs memory-shuffled | −0.030 | −0.090–0.030 | 1 | Không significant |
| true vs memory-random-user | +0.010 | −0.060–0.080 | 1 | Không significant |
| true vs memory-none | **−0.125** | **−0.185–−0.065** | **0.0019998** | memory-none ít lỗi hơn |
| true vs context-shuffled | +0.005 | −0.040–0.045 | 1 | Không significant |
| true vs context-random-poi | −0.060 | −0.115–−0.005 | 0.19558 | Không significant sau Holm |
| true vs context-none | **−0.110** | **−0.160–−0.065** | **0.00119988** | context-none ít lỗi hơn |

Loại bỏ memory giảm invalid rate 12.5 điểm phần trăm; loại bỏ context giảm 11
điểm phần trăm. Cả hai đều significant sau Holm correction. Điều này xác nhận
prompt length/complexity gây format-reliability cost thực sự.

## 7. Ranking sensitivity

| Corruption | Top-1 change rate | Ranking change rate | Mean Spearman top-10 |
|---|---:|---:|---:|
| memory-shuffled | 0.135 | 0.290 | 0.890667 |
| memory-random-user | 0.230 | 0.415 | 0.808909 |
| memory-none | **0.255** | **0.415** | 0.824485 |
| context-shuffled | 0.085 | 0.190 | 0.931394 |
| context-random-poi | 0.105 | 0.240 | 0.902545 |
| context-none | 0.110 | 0.250 | 0.905515 |

Memory corruption làm ranking thay đổi nhiều hơn context corruption. Như vậy
LLM không hoàn toàn bỏ qua semantic inputs: nó nhạy với thay đổi prompt. Tuy
nhiên, các thay đổi ranking không liên hệ ổn định với ground-truth quality, nên
sensitivity chưa chuyển thành useful semantic contribution.

## 8. Kết luận RQ9

Claim an toàn:

> Trong bounded experiment trên 200 queries của TIST2015–Tokyo, personal memory
> và context corruption làm thay đổi LLM ranking nhưng không gây thay đổi
> ground-truth quality có ý nghĩa thống kê, kể cả trên jointly-valid queries.
> Memory/context thật đồng thời làm tăng format failure so với prompt loại bỏ
> các thành phần này. Do đó semantic inputs hiện tại tạo sensitivity và chi phí,
> nhưng chưa chứng minh được predictive utility.

Không nên claim memory/context vô dụng nói chung, vì kết quả chỉ áp dụng cho
prompt, model, candidate set và bounded Tokyo protocol hiện tại.

## 9. Hướng phát triển

1. Dùng constrained JSON/schema decoding để giảm invalid-output rate.
2. Nén memory/context trước khi đưa vào prompt để kiểm soát token length.
3. So sánh true/corruption với cùng prompt length bằng padding hoặc matched
   retrieval count.
4. Học hoặc chọn memory/context evidence dựa trên validation gain thay vì đưa
   toàn bộ lịch sử dài vào prompt.
5. Chỉ mở rộng full-query/multi-city sau khi prompt rút gọn vừa giảm invalid
   rate vừa vượt corruption controls.

## 10. Publication gate

- City: `Tokyo` — bounded run hoàn thành.
- Test queries: `200` — chưa phải full-query.
- LLM: `qwen2:7b`.
- One-axis corruption — đạt.
- All-query paired test + Holm — đạt.
- Jointly-valid paired test + Holm — đạt.
- Invalid-rate paired test + Holm — đạt.
- Ranking sensitivity — đạt.
- Format reliability claim — được hỗ trợ cho memory-none/context-none.
- Main semantic contribution claim — chưa được hỗ trợ.
- TIST2015 12-city gate — chưa hoàn thành.
