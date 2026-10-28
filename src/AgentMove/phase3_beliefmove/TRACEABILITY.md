# Đối chiếu tài liệu BeliefMove với protocol thực nghiệm

Nguồn khóa: `ideas/BeliefMove_y_tuong_va_huong_dan_cai_dat.docx`.
Không dùng kết quả Phase 2 để thay cho các thành phần dưới đây.

| Yêu cầu trong tài liệu | Hiện thực/contract | Trạng thái |
|---|---|---|
| LLM Mobility Teacher sinh activity, intent, destination, next-region, confidence | `hybrid/beliefmove/schema.py`, prompt JSON cố định, cache một request/query | pilot ready |
| Teacher là soft supervision, không phải ground truth | confidence-weighted expected soft counts; fallback có weight 0 | pilot ready |
| Bayesian knowledge distillation | `DataCPT`, `TeacherSoftCountCPT`, `BlendedCPT`; alpha chỉ chọn validation | primitive + pilot screening ready |
| Lightweight BN với activity–intent–destination–region | factorization khóa trong `protocol.json` | trainer M1–M5 pending |
| Belief tuần tự `P(Z_t | X_1:t)` | `SequentialBeliefState`, transition train-only, reset theo trajectory | primitive ready; confirmatory pending |
| LLM-on-demand theo uncertainty | entropy/margin primitives, call-rate cap chọn trên validation | primitive ready; online evaluator pending |
| M1 data-only, M2 + LLM distill, M3 + memory, M4 + adaptive, M5 full | định nghĩa bất biến trong `protocol.json` | confirmatory pending |
| Accuracy, NLL/Brier/ECE, latency/memory/call-rate | bảng và success rules trong `report_template.md`/`protocol.json` | aggregator pending |
| Không gọi LLM mọi bước | M4/M5 cap call rate 0.10; never/always/random controls | locked, evaluator pending |
| TIST2015, WWW2019 và YJMob100K | 12-city macro, Shanghai, anonymous-cell YJMob | prepare/audit scripts ready |

## Quyết định protocol quan trọng

- Bài toán là dự đoán next location từ prefix với **prediction timestamp đã
  biết** theo preprocessing AgentMove. Next POI, category và region luôn bị ẩn;
  mọi baseline phải nhận cùng timestamp.
- Region vocabulary chỉ fit từ train. TIST2015/WWW2019 dùng lưới metric cục bộ
  500 m; YJMob dùng cell 500 m ẩn danh và không được gọi là tọa độ GPS.
- Pilot 500 train/200 validation chỉ là kiểm tra go/no-go của structured
  teacher và region-CPT. Alpha được chọn trên một partition validation, rồi
  M2 phải giảm NLL nghiêm ngặt trên partition held-out rời nhau.
- Ba repeat cache 50-query phải hoàn toàn valid trước khi tính JSD. Report/lock
  luôn tái tính từ raw cache; summary viết tay hoặc stale không thể mở khóa.
- Test quantitative support bị chặn đến khi `protocol_lock.json` hợp lệ và hash
  của protocol hiện tại khớp lock.

## Ranh giới tuyên bố

Hiện chỉ được tuyên bố rằng protocol, data preparation, teacher pilot và các
primitive đã sẵn sàng để chạy. Chưa được tuyên bố M2/M3/M4/M5 cải thiện cho đến
khi các action confirmatory đang fail-closed được triển khai và chạy đủ ba
dataset family.
