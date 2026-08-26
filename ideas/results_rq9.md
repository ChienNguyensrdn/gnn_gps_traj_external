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

## 5. Phân tích bổ sung trong script đã điều chỉnh

Aggregator RQ9 hiện bổ sung bốn nhóm kiểm tra offline từ cache có sẵn:

- **All-query paired test:** kết quả primary trên toàn bộ 200 queries.
- **Jointly-valid paired test:** chỉ giữ query mà cả true và corruption đều có
  output hợp lệ.
- **Invalid-rate paired test:** kiểm tra chênh lệch format failure bằng paired
  bootstrap/permutation và Holm correction.
- **Ranking sensitivity:** top-1 change rate, full ranking change rate và mean
  Spearman agreement trên top-10.

Ngoài ra report JSON tách metrics của valid outputs và fallback outputs cho từng
variant. Các phân tích này không gọi lại LLM và không thay đổi cache gốc.

## 6. Cách diễn giải jointly-valid result

- Nếu jointly-valid vẫn không significant, negative result semantic được củng
  cố: memory/context hiện tại không được mô hình sử dụng hiệu quả.
- Nếu jointly-valid khác all-query, format reliability/fallback là nguyên nhân
  chính che khuất tác động semantic.
- Nếu ranking change rate cao nhưng quality không đổi, LLM nhạy với corruption
  nhưng thay đổi không theo hướng cải thiện ground-truth ranking.
- Nếu ranking change rate thấp, prompt semantic gần như bị LLM bỏ qua.

## 7. Kết luận hiện tại

Claim an toàn:

> Trong bounded experiment trên 200 queries của TIST2015–Tokyo, không có bằng
> chứng thống kê rằng personal memory hoặc context/world prompt hiện tại cải
> thiện next-location ranking. Kết quả bị confound bởi invalid-output rate phụ
> thuộc độ dài prompt; do đó jointly-valid analysis cần được ưu tiên khi diễn
> giải semantic contribution.

Không nên claim memory/context vô dụng nói chung, vì kết quả chỉ áp dụng cho
prompt, model, candidate set và bounded Tokyo protocol hiện tại.

## 8. Hướng phát triển

1. Dùng constrained JSON/schema decoding để giảm invalid-output rate.
2. Nén memory/context trước khi đưa vào prompt để kiểm soát token length.
3. So sánh true/corruption với cùng prompt length bằng padding hoặc matched
   retrieval count.
4. Chỉ mở rộng full-query/multi-city sau khi jointly-valid analysis xác nhận có
   semantic signal.

## 9. Publication gate

- City: `Tokyo` — bounded run hoàn thành.
- Test queries: `200` — chưa phải full-query.
- LLM: `qwen2:7b`.
- One-axis corruption — đạt.
- All-query paired test + Holm — đạt.
- Jointly-valid/invalid-rate/ranking-sensitivity scripts — đã chuẩn bị; cần chạy
  lại aggregate để điền kết quả.
- Main semantic contribution claim — chưa được hỗ trợ.
- TIST2015 12-city gate — chưa hoàn thành.
