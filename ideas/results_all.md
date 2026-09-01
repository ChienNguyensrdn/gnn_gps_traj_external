# BeliefMove-Evo — Tổng kết ý tưởng và RQ1–RQ13

## 1. Ý tưởng ngắn gọn

BeliefMove-Evo dự đoán địa điểm kế tiếp bằng cách kết hợp bốn loại tri thức:

1. **Mẫu di chuyển định lượng:** GRU/Transformer teacher học từ trajectory, user và thời gian.
2. **Student nhẹ:** học nhãn thật, soft output và sự tiến hóa biểu diễn của teacher.
3. **Bayesian belief:** bổ sung prior theo user, thời gian, lịch sử và transition giữa các POI.
4. **Tri thức ngữ nghĩa từ LLM:** tạo structured habit/semantic evidence và chỉ được gọi khi cần.

Mục tiêu không chỉ tăng Recall/MRR mà còn kiểm soát calibration, chi phí suy luận, độ bền với teacher khác nhau và robustness khi đầu vào thiếu hoặc nhiễu.

```text
Trajectory → quantitative teacher → distillation/evolution → lightweight student
                                      │
Data prior/transition → Bayesian belief
                                      │
LLM structured evidence → fusion/routing khi không chắc chắn → next POI
```

## 2. Protocol chung đã sử dụng

- Dataset: Foursquare TIST2015, temporal train/validation/test split.
- Candidate space, checkpoint, calibrator và cache được tách riêng theo thành phố.
- Checkpoint và fusion/calibration weight chỉ được chọn bằng validation.
- Test chỉ dùng để báo cáo cuối; paired tests dùng cùng query và cùng seed.
- Neural experiments dùng seed 42, 43, 44; deterministic baseline chỉ chạy một lần.
- Metrics: R@1/5/10, MRR, NLL, Brier, ECE; thêm latency, throughput, memory và representation similarity khi phù hợp.
- Multiple testing được hiệu chỉnh Holm; không suy diễn `not significant` thành equivalence.
- `last-query`, `all-prefix` và LLM `bounded` là ba protocol khác nhau, không so trực tiếp trị tuyệt đối.

## 3. Thực nghiệm và câu trả lời RQ1–RQ13

| RQ | Câu hỏi và thiết kế thực nghiệm | Kết quả ngắn gọn | Trạng thái hiện tại |
|---|---|---|---|
| **RQ1 — Baseline reproducibility** | Chạy GRU teacher, Transformer teacher và CE student trên cùng Tokyo full-test, seed 42–44; chạy Markov và AgentMove bounded 200 query trên 12 thành phố. | Quantitative baselines ổn định; Transformer ranking tốt nhất nhưng calibration kém GRU. AgentMove bounded có macro Acc@1/5 và MRR cao hơn Markov, nhưng hai phần dùng protocol khác nhau. | `ready-separated-protocols`; phần 12-city mới là bounded. |
| **RQ2 — Bayesian data-only** | So sánh unigram, Markov, BN user/time, DBN transition và quantitative teacher trên matched Tokyo last-query; paired bootstrap. | DBN tốt nhất trong nhóm data-only về ranking và vượt BN có ý nghĩa, nhưng NLL/Brier xấu hơn. Quantitative teacher vẫn vượt toàn bộ data-only baseline về Recall/MRR. | `ready-tokyo-matched-last-query`. |
| **RQ3 — LLM knowledge** | M1=BN, M2=BN+LLM, M3=BN+quantitative teacher, M4=BN+cả hai; weight fit validation; 200 matched test query. | Quantitative teacher tạo gain có ý nghĩa. LLM tạo gain mô tả nhưng M2 chưa vượt M1 và M4 chưa vượt M3 có ý nghĩa sau Holm correction. Chưa xác nhận incremental utility của LLM. | `ready-bounded-matched`, Tokyo, Qwen2:7b, no-OSM. |
| **RQ4 — Distillation ablation** | E0 CE-only; E1 KD; E2 KD+trajectory; E3 KD+velocity; E4 layer evolution; E5 dual evolution; seed 42–44. | Response KD tạo phần lớn gain. Trajectory/velocity riêng chưa vượt KD. E5 đạt mean tốt nhất, nhưng phần gain nhỏ so với KD cần đọc cùng paired tests. | Tokyo test `ready`. |
| **RQ5 — Temporal order** | Đánh giá frozen E5 với correct, reverse và random trajectory order; paired query/seed bootstrap và sign-flip test. | Correct order vượt reverse và random có ý nghĩa trên ranking và probabilistic metrics; random gây hại mạnh hơn. Mô hình thực sự khai thác thứ tự thời gian. | Tokyo `ready`. |
| **RQ6 — Dual-axis evolution** | So E1 KD, E2 trajectory, E3 velocity, E4 layer, E6 temporal và E5 dual; đo ranking, calibration, CKA, layer/temporal cosine và trajectory length. | E5 có mean ranking/CKA tốt nhất; E6 có NLL/ECE tốt nhất. Layer/temporal alignment cải thiện nhưng ranking gain giữa các evolution variant chưa significant sau Holm. | Tokyo `ready`. |
| **RQ7 — Belief memory** | So B0 static, B1 history, B2 sequential posterior và B3 transition-aware DBN trên all-prefix; weight chọn validation. | B1/B2 bị validation chọn weight 0. B3 cải thiện có ý nghĩa toàn bộ ranking metrics nhưng làm NLL/Brier/ECE xấu hơn. | Tokyo all-prefix `ready`. |
| **RQ8 — LLM routing** | Never, Always, entropy, margin và random budget-matched; threshold fit validation; đo accuracy, call rate, latency và tokens. | Entropy giảm khoảng 85% LLM calls nhưng không tăng quality so với Never và kém random ở R@5. Oracle cho thấy gain-aware routing còn tiềm năng. | Tokyo bounded 200 `ready-diagnostic`. |
| **RQ9 — Semantic verification** | Giữ một trục đúng và phá memory/context bằng shuffle, random donor hoặc remove; kiểm soát invalid output. | Semantic inputs làm ranking nhạy và tăng format failure, nhưng chưa tạo ground-truth gain có ý nghĩa kể cả trên jointly-valid queries. | Tokyo bounded 200 `ready-diagnostic`. |
| **RQ10 — Teacher robustness** | Distill cùng một GRU student từ GRU hoặc Transformer teacher, so với CE-only; cùng split/candidate/seeds. | Cả hai teacher cải thiện có ý nghĩa ranking, NLL và Brier. Transformer ranking nhỉnh hơn nhưng chưa significant; GRU cho calibration tốt hơn và thực dụng hơn. | Tokyo seed 42–44 `ready`. |
| **RQ11 — Calibration** | Fit temperature riêng theo NLL, Brier và ECE trên validation; đánh giá distillation last-query và Bayesian all-prefix riêng. | Không có một temperature tối ưu mọi metric. Objective-specific scaling cải thiện metric mục tiêu; distillation/DBN cải thiện proper scoring rules nhưng ECE phụ thuộc objective. | Tokyo seed 42–44 `ready`. |
| **RQ12 — Accuracy–efficiency** | Benchmark batch-1 và batch-256 với warm-up/CUDA sync; tách neural, Bayesian và recorded Ollama latency. | Distilled student là điểm vận hành tốt nhất về chất lượng–throughput. B3 có CPU fusion overhead lớn; LLM chậm hơn nhiều và chỉ nên gọi chọn lọc. | Tokyo/hardware-specific `ready`; LLM phần bounded. |
| **RQ13 — Missing/noisy input** | Frozen E5, seed 42–44; GPS dropout, time noise, position noise, missing/wrong user/time; paired dose-response. | Tương đối bền với GPS dropout/time noise nhẹ. Position remapping và sai user context gây hại mạnh nhất; user conditioning quan trọng hơn time conditioning. | Tokyo `ready`. |

## 4. Kết luận tổng thể hiện tại

- **Bằng chứng mạnh:** knowledge distillation cải thiện student; temporal order có vai trò; transition-aware DBN cải thiện ranking; distillation bền với GRU/Transformer teacher.
- **Trade-off chính:** ranking tốt hơn thường không đồng nghĩa calibration tốt hơn; DBN và evolution cần calibration theo objective.
- **Negative findings hữu ích:** entropy/margin routing chưa hơn random; LLM belief chưa tạo incremental gain có ý nghĩa; semantic prompt hiện làm tăng invalid output.
- **Điểm vận hành thực dụng:** GRU-distilled student cho throughput/calibration tốt; E5 phù hợp khi ưu tiên ranking; E6 phù hợp khi ưu tiên NLL/ECE.
- **Failure modes:** position/POI remapping, sai user context, GPU contention khi benchmark và cache LLM lệch protocol.
- Phần lớn kết luận hiện vẫn là **Tokyo-only**; chưa được viết thành claim tổng quát cho TIST2015.

## 5. Thiết kế thực nghiệm 12 thành phố

### 5.1. Thành phố và đơn vị độc lập

Chạy đúng thứ tự canonical:

```text
Tokyo, Nairobi, NewYork, Sydney, CapeTown, Paris,
Beijing, Mumbai, SanFrancisco, London, SaoPaulo, Moscow
```

Mỗi thành phố phải có độc lập:

- `candidate_ids.json` và candidate metadata;
- temporal train/validation/test split;
- teacher/student checkpoints theo seed;
- logits, calibrator, LLM cache và per-query predictions;
- thư mục kết quả chứa manifest config/commit/protocol.

Không nối candidate IDs, logits hoặc query giữa các thành phố.

### 5.2. Hai tầng chạy

**Tầng A — full neural/Bayesian:** chạy full-query cho RQ1, RQ2, RQ4–RQ7 và RQ10–RQ13 với seed 42, 43, 44. Baseline deterministic chạy một lần mỗi city. `last-query` và `all-prefix` lưu thành hai nhóm riêng.

**Tầng B — LLM bounded rồi mới mở rộng:** chạy RQ3, RQ8 và RQ9 với cùng limit 200 query/city, Qwen2:7b, `top-k=10`, `top-m=5` để tạo matched 12-city bounded result. Chỉ chuyển sang full-query sau khi kiểm tra chi phí, invalid rate và cache coverage. `no-OSM` và `full-OSM` phải nằm ở output khác nhau.

### 5.3. Trình tự chạy mỗi thành phố

```text
prepare/split
→ train quantitative teachers
→ train E0–E6 students
→ evaluate frozen checkpoints
→ fit calibration/fusion/routing trên validation
→ build immutable LLM evidence cache
→ evaluate test và lưu per-query predictions
→ paired significance
→ per-city summary
```

Chạy Tokyo smoke test trước. Sau khi smoke test hoàn tất, chạy 12 city tuần tự hoặc với mức song song đã kiểm tra tài nguyên; mọi script phải hỗ trợ resume và không ghi đè cache hoàn thành.

### 5.4. Ma trận tối thiểu

| Nhóm | Scope 12-city | Seeds | Query protocol |
|---|---|---:|---|
| RQ1 quantitative, RQ2 | Full | 42–44 cho neural; 1 deterministic | last-query |
| RQ4–RQ6 | Full | 42–44 | last-query |
| RQ7 | Full | 42–44 | all-prefix |
| RQ3, RQ8, RQ9 | Bounded 200/city trước | 1 deterministic LLM cache; random control nhiều seed khi cần | matched bounded |
| RQ10–RQ11 | Full | 42–44 | tách last-query/all-prefix |
| RQ12 | Full neural/Bayesian; bounded LLM | 42–44 | timing protocol riêng |
| RQ13 | Full | 42–44 | last-query perturbation |

### 5.5. Aggregation và kiểm định

Với mỗi variant:

1. Báo cáo per-city mean ± std qua seed.
2. Tính **macro mean không trọng số** trên đúng 12 city.
3. Báo cáo population variance của city R@1 để đo geographical bias.
4. Có thể thêm micro-average nhưng phải ghi nhãn riêng, không thay macro-average.
5. Paired test phải ghép cùng city, query và seed; bootstrap theo query trong city rồi macro qua seed/city.
6. Holm correction áp dụng trong từng family của mỗi RQ; ECE dùng bootstrap riêng hoặc reliability diagram.

### 5.6. Publication gate 12-city

Chỉ gắn nhãn `ready-12city` khi:

- đủ đúng 12 city và cùng config/split/top-k/top-m/limit;
- đủ seed đã khai báo và per-query predictions;
- không có smoke-test artifact trộn vào full result;
- calibration/routing weight chỉ fit validation;
- LLM cache coverage đầy đủ và invalid rate được báo cáo;
- OSM coverage đạt ít nhất 90% nếu gọi là `full`; nếu không phải ghi `no-OSM`;
- benchmark publication không có foreign GPU process;
- macro summary từ chối chạy nếu thiếu hoặc sai protocol ở bất kỳ city nào.

Nếu một thành phố thiếu, chỉ được báo cáo `partial 11/12`, không được tính và gọi là “12-city average”.

## 6. Các báo cáo chi tiết

- RQ1–RQ4: `results_rq1.md`, `results_rq2.md`, `results_rq3.md`, `results_rq4.md`.
- RQ5–RQ6: `result_rq5.md`, `result_rq5_significance.md`, `result_rq6.md`.
- RQ7–RQ13: `results_rq7.md` đến `results_rq13.md`.
