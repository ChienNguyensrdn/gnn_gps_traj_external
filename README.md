# BeliefMove-Evo

BeliefMove-Evo là framework dự đoán next-location kết hợp:

- Neural-CGM quantitative teacher;
- lightweight mobility student;
- response/representation/velocity/temporal distillation;
- Bayesian belief và sequential belief;
- LLM semantic evidence;
- uncertainty-aware LLM routing.

Ý tưởng: [ideas/idea.md](ideas/idea.md)

Protocol: [ideas/pipline.md](ideas/pipline.md)

Kết quả: [ideas/results.md](ideas/results.md)

## 1. Làm việc từ AgentMove

```bash
cd src/AgentMove
```

Các thành phần chính:

```text
configs/beliefmove_evo/       cấu hình base, ablation, routing
hybrid/                       model, distillation, belief, metrics
scripts/                      setup, prepare, train, aggregate
data/                         raw và processed datasets
results/                      checkpoints, raw metrics, logs
```

## 2. Cài môi trường

Ubuntu:

```bash
chmod +x scripts/*.sh
./scripts/setup_ubuntu.sh
./scripts/beliefmove_evo.sh environment
```

Kiểm tra accelerator:

```bash
.venv/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`DEVICE=auto` ưu tiên CUDA → MPS → CPU.

## 3. Ollama

```bash
./scripts/install_ollama_ubuntu.sh
./scripts/start_ollama.sh
./scripts/test_ollama.sh qwen2:7b
```

Endpoint bắt buộc là `http://127.0.0.1:11434/v1`. Model mặc định là
`qwen2:7b`; `llama3.1:8b` là backbone open-weight thứ hai.

## 4. Chuẩn bị TIST2015

Pipeline sử dụng đúng 12 thành phố:

```text
Tokyo Nairobi NewYork Sydney CapeTown Paris
Beijing Mumbai SanFrancisco London SaoPaulo Moscow
```

```bash
./scripts/tist2015_pipeline.sh audit
./scripts/tist2015_pipeline.sh download  # chỉ khi audit báo raw=missing
./scripts/tist2015_pipeline.sh prepare
```

Mỗi city giữ temporal train/validation/test split và candidate vocabulary riêng.
Không thay preprocessing hoặc split giữa baseline và proposed method.

## 5. Train quantitative teacher

```bash
./scripts/tist2015_pipeline.sh train
```

Mặc định Neural-CGM:

| Tham số | Giá trị |
|---|---:|
| POI embedding | 64 |
| User embedding | 32 |
| Time embedding | 16 |
| GRU hidden | 128 |
| Epochs | 10 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Seed | 42 |

Checkpoint:

```text
data/hybrid/TIST2015/<CITY>/neural_cgm/best.pt
```

## 6. Train lightweight student

| Variant | Objective |
|---|---|
| `E0-ce` | CE only |
| `E1-kd` | CE + response KD |
| `E2-kd-traj` | CE + KD + trajectory |
| `E3-kd-vel` | CE + KD + velocity |
| `E4-layer` | CE + KD + trajectory + velocity |
| `E5-dual` | E4 + temporal evolution |
| `E6-temporal` | CE + KD + temporal evolution only |

### Smoke test

Luôn chạy smoke test trước full run:

```bash
CITY=Tokyo \
VARIANT=E0-ce \
EPOCHS=1 \
BATCH_SIZE=256 \
DEVICE=cuda \
TRAIN_LIMIT=2000 \
VALIDATION_LIMIT=500 \
./scripts/beliefmove_evo.sh train-student
```

Smoke checkpoint nằm trong `artifacts/smoke` và không được đưa vào kết quả
publication.

### Full E0-CE

```bash
CITY=Tokyo \
VARIANT=E0-ce \
EPOCHS=10 \
BATCH_SIZE=256 \
DEVICE=cuda \
./scripts/beliefmove_evo.sh train-student
```

Full run dùng toàn bộ prefix trajectory. Tokyo hiện có thể tạo khoảng 430.000
training examples; batch 256 tương ứng khoảng 1.680 steps/epoch.

E0-CE có `state_distillation=false`: không chạy teacher forward và không tính
KD/trajectory/velocity/temporal loss.

### Full dual-evolution

```bash
CITY=Tokyo \
VARIANT=E5-dual \
EPOCHS=10 \
BATCH_SIZE=256 \
DEVICE=cuda \
./scripts/beliefmove_evo.sh train-student
```

E5 chạy teacher và hidden-state distillation nên chậm và tốn VRAM hơn E0. Nếu
hết VRAM, giảm `BATCH_SIZE=128` hoặc `64`.

## 7. Theo dõi training

Dòng đầu mỗi run báo cấu hình thật:

```json
{
  "device": "cuda",
  "epochs": 10,
  "batch_size": 256,
  "train_examples": 430088,
  "validation_examples": 9132,
  "steps_per_epoch": 1681,
  "state_distillation": false
}
```

Mỗi epoch báo `train_loss`, validation `recall@1/5/10` và từng loss term. Model
tốt nhất được chọn bằng:

```text
validation recall@1 + validation recall@10
```

Nếu training loss tiếp tục giảm nhưng validation recall giảm nhiều epoch liên
tiếp, mô hình đang overfit. Không chọn epoch bằng test set.

Không nên `Ctrl+C` giữa run hiện tại vì checkpoint tốt nhất được ghi khi vòng
train kết thúc. Có thể giảm `EPOCHS` cho run sau khi validation đã plateau.

Checkpoint và history:

```text
results/beliefmove-evo/artifacts/full/<CITY>/<VARIANT>/<ORDER>/seed-<SEED>/best.pt
results/beliefmove-evo/artifacts/full/<CITY>/<VARIANT>/<ORDER>/seed-<SEED>/best.metrics.json
```

## 8. Order-corruption ablation

Chạy correct, reverse và 10 random permutation seeds:

```bash
CITY=Tokyo \
VARIANT=E5-dual \
BATCH_SIZE=256 \
DEVICE=cuda \
./scripts/beliefmove_evo.sh order-ablation
```

Transform chỉ đổi thứ tự input; không đổi label, split hoặc candidate set.

### Evaluate best checkpoint trên test split

Chỉ evaluate sau khi variant/epoch đã được chọn bằng validation:

```bash
for seed in 42 43 44; do
  for variant in E0-ce E1-kd E5-dual; do
    CITY=Tokyo SEED="$seed" VARIANT="$variant" BATCH_SIZE=256 DEVICE=cuda \
      ./scripts/beliefmove_evo.sh evaluate-student
  done
done
```

Test evaluator ghi Recall@1/5/10, MRR, NLL, Brier và ECE vào raw result có
`evaluation_split=test`. Không dùng test metrics để đổi hyperparameter.

Đánh giá RQ5 sau khi đã chọn checkpoint bằng validation. Evaluator kiểm tra
`ORDER_MODE` khớp metadata checkpoint và áp dụng cùng phép biến đổi lên test
input. Reverse/random tự động được ghi vào `rq5-test`:

```bash
for seed in 42 43 44; do
  CITY=Tokyo SEED="$seed" VARIANT=E5-dual ORDER_MODE=reverse \
  BATCH_SIZE=256 DEVICE=cuda ./scripts/beliefmove_evo.sh evaluate-student
done

for seed in 42 43 44 45 46 47 48 49 50 51; do
  CITY=Tokyo SEED="$seed" VARIANT=E5-dual ORDER_MODE=random \
  BATCH_SIZE=256 DEVICE=cuda ./scripts/beliefmove_evo.sh evaluate-student
done
```

Để ghi nhóm `correct` làm đối chứng riêng trong RQ5, dùng `EVALUATION_RQ=RQ5`:

```bash
for seed in 42 43 44; do
  CITY=Tokyo SEED="$seed" VARIANT=E5-dual ORDER_MODE=correct EVALUATION_RQ=RQ5 \
  BATCH_SIZE=256 DEVICE=cuda ./scripts/beliefmove_evo.sh evaluate-student
done
```

Mỗi evaluation đồng thời lưu `test.predictions.npz` theo từng query. Sau khi
đã chạy đủ correct/reverse/random, tính paired bootstrap, sign-flip permutation
test và Holm correction trên tập seed chung 42–44:

```bash
SIGNIFICANCE_ITERATIONS=10000 \
  ./scripts/beliefmove_evo.sh rq5-significance
```

Output:

```text
results/beliefmove-evo/aggregated/rq5_paired_significance.json
ideas/result_rq5_significance.md
```

Nếu các evaluation được tạo bằng phiên bản cũ và chưa có
`test.predictions.npz`, chỉ cần chạy lại `evaluate-student`; không train lại.

### RQ6 — Dual-Axis Evolution

RQ6 bổ sung temporal-only để tách đóng góp theo layer và theo thời gian. Chỉ
`E6-temporal` cần train mới:

```bash
for seed in 42 43 44; do
  CITY=Tokyo SEED="$seed" VARIANT=E6-temporal EPOCHS=10 \
  BATCH_SIZE=256 DEVICE=cuda ./scripts/beliefmove_evo.sh train-student
done
```

Đánh giá sáu variant bằng checkpoint đã chọn trên validation:

```bash
for seed in 42 43 44; do
  for variant in E1-kd E2-kd-traj E3-kd-vel E4-layer E6-temporal E5-dual; do
    CITY=Tokyo SEED="$seed" VARIANT="$variant" BATCH_SIZE=256 DEVICE=cuda \
      ./scripts/beliefmove_evo.sh evaluate-rq6
  done
done
```

Evaluator đo ranking/calibration, CKA, layer/temporal transition cosine và
short/medium/long. Ngưỡng độ dài là tertile fit trên validation rồi khóa cho
test. Tổng hợp và paired significance:

```bash
SIGNIFICANCE_ITERATIONS=10000 ./scripts/beliefmove_evo.sh aggregate-rq6
```

Output:

```text
results/beliefmove-evo/aggregated/rq6_summary.json
ideas/result_rq6.md
```

### RQ7 — Belief memory

RQ7 dùng checkpoint `E5-dual/correct` đã đóng băng và đánh giá bốn cơ chế trên
mọi prefix theo thứ tự thời gian của từng trajectory:

- `B0-static`: suy luận độc lập từ E5-dual.
- `B1-history`: thêm prior tần suất các POI đã quan sát trong prefix.
- `B2-sequential`: truyền posterior của bước trước sang bước kế tiếp.
- `B3-dbn`: kết hợp E5-dual với prior chuyển trạng thái bậc một từ POI hiện tại.

Transition/prior chỉ được fit từ train; trọng số fusion được chọn bằng validation.
Belief được reset ở ranh giới trajectory, sau đó test được đánh giá đúng một lần.
RQ7 dùng tất cả prefix nên số tuyệt đối không được so trực tiếp với RQ4/RQ6 vốn
chỉ dùng query cuối của mỗi trajectory.

```bash
cd src/AgentMove
for seed in 42 43 44; do
  CITY=Tokyo SEED="$seed" BATCH_SIZE=256 DEVICE=cuda \
    ./scripts/beliefmove_evo.sh evaluate-rq7
done
```

Kết quả mỗi seed nằm tại
`results/beliefmove-evo/artifacts/full/Tokyo/E5-dual/correct/seed-<SEED>/rq7/`.

Sau khi đủ ba seed, tổng hợp mean/std và paired significance (Holm correction):

```bash
CITY=Tokyo SIGNIFICANCE_ITERATIONS=10000 \
  ./scripts/beliefmove_evo.sh aggregate-rq7
```

Output là `results/beliefmove-evo/aggregated/rq7_summary.json` và
`ideas/result_rq7.md`.

## 9. Teacher cache

```bash
TYPE=llm \
INPUT=results/path/evidence_cache.jsonl \
OUTPUT=results/beliefmove-evo/teacher-cache/llm.jsonl \
./scripts/build_teacher_cache.sh
```

Với quantitative cache, dùng `TYPE=quantitative`. Cache là immutable và
content-addressed: cùng key nhưng khác content sẽ bị từ chối.

## 10. Bayesian belief và routing

- Static Bayesian network: `hybrid.bayesian_network`.
- Sequential update: `hybrid.sequential_belief`.
- Entropy/margin router: `hybrid.selective_llm`.
- Routing config: `configs/beliefmove_evo/routing.json`.

Calibration và router threshold chỉ fit trên validation. Không tune test set.

### RQ8 — Uncertainty-aware LLM routing

RQ8 dùng cùng Neural-CGM candidate space và một cache `Always-LLM` bất biến để
so sánh `Never`, `Always`, `Entropy`, `Margin` và `Random-budget-matched`.
Threshold Entropy/Margin chỉ được chọn trên validation với call budget mặc định
25%. Random dùng đúng call rate của Entropy trên test.

```bash
cd src/AgentMove
CITY=Tokyo RQ8_LIMIT=200 ./scripts/rq8_routing.sh audit

# Gọi Ollama và tạo cache; chạy lại cùng lệnh để resume.
CITY=Tokyo RQ8_LIMIT=200 OLLAMA_MODEL=qwen2:7b ./scripts/rq8_routing.sh collect

# Không gọi LLM; tạo 50 random permutations, deterministic policies giữ một run.
CITY=Tokyo RQ8_LIMIT=200 ./scripts/rq8_routing.sh evaluate-random

CITY=Tokyo RQ8_LIMIT=200 \
RQ8_SEEDS="$(seq -s ' ' 42 91)" SIGNIFICANCE_ITERATIONS=10000 \
  ./scripts/rq8_routing.sh aggregate
```

`collect` yêu cầu `evidence_cache.jsonl` và `calibration.json` của matched Hybrid
run. Chỉ rõ run khác bằng `HYBRID_RUN_DIR=/path/to/city/run`. Output tổng hợp là
`results/beliefmove-evo/aggregated/rq8_summary.json` và `ideas/results_rq8.md`.
Run giới hạn 200 query là bounded experiment, không phải full-query result.
Evaluator đồng thời sinh budget sweep 10/25/50%, oracle-gain upper bound và
paired significance với Holm correction; các bước này tái sử dụng cache, không
gọi lại Ollama.

### RQ9 — Semantic knowledge verification

RQ9 dùng matched one-axis corruption trên cùng Neural-CGM top-10 candidates.
Memory variants giữ context thật; context variants giữ memory thật. Mỗi variant
có LLM cache riêng để tránh dùng evidence thật cho prompt đã corruption.

```bash
cd src/AgentMove
CITY=Tokyo RQ9_LIMIT=200 OLLAMA_MODEL=qwen2:7b ./scripts/rq9_semantic.sh audit

# Gọi Ollama cho 7 variants; chạy lại cùng lệnh để resume.
CITY=Tokyo RQ9_LIMIT=200 OLLAMA_MODEL=qwen2:7b ./scripts/rq9_semantic.sh collect

SIGNIFICANCE_ITERATIONS=10000 CITY=Tokyo RQ9_LIMIT=200 \
  OLLAMA_MODEL=qwen2:7b ./scripts/rq9_semantic.sh aggregate
```

Các biến thể là `memory-true`, `memory-shuffled`, `memory-random-user`,
`memory-none`, `context-shuffled`, `context-random-poi`, `context-none`.
Aggregator thực hiện paired test giữa true và từng corruption, rồi áp dụng Holm
correction. Output: `results/beliefmove-evo/aggregated/rq9_summary.json` và
`ideas/results_rq9.md`. Run 200 query vẫn chỉ là bounded experiment.

### RQ10 — Độ bền theo kiến trúc teacher

RQ10 giữ cố định split, candidate set, seed và kiến trúc student, chỉ thay teacher
giữa GRU và Transformer. `none` là control chỉ học cross-entropy. Checkpoint tốt
nhất được chọn trên validation; test chỉ dùng để báo cáo cuối cùng.

```bash
cd src/AgentMove
CITY=Tokyo ./scripts/rq10_teacher_robustness.sh audit
CITY=Tokyo ./scripts/rq10_teacher_robustness.sh status

# Chạy đầy đủ teacher, student và test cho seed 42, 43, 44.
CITY=Tokyo DEVICE=cuda BATCH_SIZE=128 \
  ./scripts/rq10_teacher_robustness.sh run-seeds

CITY=Tokyo SIGNIFICANCE_ITERATIONS=10000 \
  ./scripts/rq10_teacher_robustness.sh aggregate
```

Có thể chạy/resume từng bước bằng `TEACHER=gru|transformer` với
`train-teacher`, hoặc `TEACHER=none|gru|transformer` với `train-student` và
`evaluate`. Output tổng hợp là
`results/beliefmove-evo/aggregated/rq10_summary.json` và
`ideas/results_rq10.md`. PMT/UniTraj chưa được tính là baseline hợp lệ cho đến
khi adapter preprocessing và candidate space được xác minh.
`status` liệt kê riêng checkpoint, test metrics và per-query predictions còn
thiếu. `aggregate` sẽ dừng sớm với hướng dẫn resume nếu publication gate chưa đủ.

### RQ11 — Calibration

RQ11 tách hai protocol để tránh so sánh sai: `distillation` dùng last-query của
RQ10; `bayesian` dùng all-prefix của RQ7. Bốn chiến lược gồm identity `T=1` và
temperature tối ưu riêng NLL, Brier, ECE; tất cả chỉ fit trên validation. Với
Bayesian, transition/prior vẫn chỉ fit train và B3 weight lấy từ validation RQ7.

```bash
cd src/AgentMove
CITY=Tokyo ./scripts/rq11_calibration.sh audit

CITY=Tokyo DEVICE=cuda BATCH_SIZE=128 \
  ./scripts/rq11_calibration.sh run-seeds

CITY=Tokyo SIGNIFICANCE_ITERATIONS=10000 ECE_BOOTSTRAP_ITERATIONS=1000 \
  ./scripts/rq11_calibration.sh aggregate
```

`status` kiểm tra đủ metrics và predictions identity/NLL/Brier/ECE cho seed
42–44. Aggregator báo trade-off đa mục tiêu, paired test cho tác động calibration
và so sánh trực tiếp GRU/Transformer–None, B3–B0, bootstrap CI cho ECE và hai
reliability diagram SVG. Temperature scaling không đổi ranking.
Output: `results/beliefmove-evo/aggregated/rq11_summary.json`,
`results/beliefmove-evo/aggregated/rq11_*_reliability.svg` và
`ideas/results_rq11.md`.

### RQ12 — Accuracy–Efficiency Trade-off

RQ12 benchmark neural last-query và Bayesian all-prefix bằng hai profile trên cùng
hardware/repeat: `batch-1` đo single-request latency trên mẫu xác định 2.000 query;
`batch-256` đo throughput toàn bộ test. Timing loại thời gian load checkpoint, CSV,
preprocessing và warm-up; CUDA được synchronize. Aggregate mặc định từ chối run
có tiến trình GPU ngoại lai. Chi phí LLM lấy từ live cache-generation RQ8 và luôn
gắn nhãn bounded, không giả vờ là cùng timing harness với PyTorch.

```bash
cd src/AgentMove
CITY=Tokyo DEVICE=cuda PROFILE=batch-1 ./scripts/rq12_efficiency.sh audit

# Smoke riêng, không thể lọt vào full aggregate.
CITY=Tokyo DEVICE=cuda PROFILE=batch-1 MAX_BATCHES=5 BENCHMARK_REPEATS=2 \
  ./scripts/rq12_efficiency.sh benchmark-neural

# Chạy đủ batch-1 và batch-256, mỗi profile cho seed 42–44.
CITY=Tokyo DEVICE=cuda BENCHMARK_REPEATS=5 \
  ./scripts/rq12_efficiency.sh run-profiles

CITY=Tokyo ./scripts/rq12_efficiency.sh status
CITY=Tokyo ./scripts/rq12_efficiency.sh aggregate
```

Output gồm raw timing/memory theo seed tại `results/beliefmove-evo/artifacts/full/`
và `results/beliefmove-evo/aggregated/rq12_summary.json`,
`ideas/results_rq12.md`. Có thể chỉ định RQ8 khác bằng `RQ8_SUMMARY=/path/to/rq8_summary.json`.
Report tách riêng hai profile và ghi timing/memory dưới dạng mean ± std. Trường
Bayesian `Post-processing/Fusion` bao gồm softmax/CPU transfer và belief fusion.
Nếu buộc phải dùng GPU đang có tải khác, `ALLOW_GPU_CONTENTION=1` cho phép aggregate
nhưng gate sẽ ghi rõ contention; không nên dùng kết quả đó để xuất bản. Offline
training/cache cost không có timer chuẩn từ đầu được ghi N/A, không nội suy.

### RQ13 — Robustness với GPS/context thiếu hoặc nhiễu

RQ13 giữ nguyên checkpoint E5-dual và test target, chỉ perturb phần context đã
quan sát. Protocol gồm GPS point dropout 25%/50%, timestamp noise 30/60 phút,
position noise 200/500 m, missing/wrong user context và missing/wrong temporal
context. Hai variant context tổng hợp vẫn được giữ để đo failure mode kết hợp.
Mỗi perturbation sinh per-query prediction để kiểm định paired với clean; ba cặp
mức nhiễu nhẹ/nặng còn được kiểm định dose-response trực tiếp, dùng Holm correction.

```bash
cd src/AgentMove
CITY=Tokyo SEED=42 ./scripts/rq13_robustness.sh audit

# Smoke một seed; có thể chạy lại để resume.
CITY=Tokyo SEED=42 DEVICE=cuda BATCH_SIZE=256 \
  ./scripts/rq13_robustness.sh evaluate-seed

# Full seed 42–44.
# Artifact cũ được giữ; sau cập nhật này chỉ bốn context one-axis mới phải chạy.
CITY=Tokyo DEVICE=cuda BATCH_SIZE=256 \
  ./scripts/rq13_robustness.sh run-seeds

CITY=Tokyo ./scripts/rq13_robustness.sh status
CITY=Tokyo SIGNIFICANCE_ITERATIONS=10000 \
  ./scripts/rq13_robustness.sh aggregate
```

Output: `results/beliefmove-evo/aggregated/rq13_summary.json` và
`ideas/results_rq13.md`. Position noise được ánh xạ về POI gần nhất nên đo độ bền
của categorical POI pipeline, không được diễn giải như raw-coordinate encoder.
Report ghi mean ± std qua seed 42–44 và paired significance đã hiệu chỉnh Holm.

### RQ1 — Baseline reproducibility (hệ BeliefMove-Evo)

RQ1 được kiểm tra hồi cứu và tách hai protocol: quantitative baselines trên Tokyo
full-test với seed 42–44; Markov/AgentMove trên TIST2015 bounded 12-city. Không tính
paired delta giữa hai protocol khác query/scope.

```bash
cd src/AgentMove
CITY=Tokyo ./scripts/rq1_reproducibility.sh audit

# Chỉ chạy nếu audit báo thiếu bounded baseline; AgentMove dùng Ollama và resume cache.
CITY=Tokyo RQ1_LIMIT=200 OLLAMA_MODEL=qwen2:7b \
  ./scripts/rq1_reproducibility.sh run-bounded

CITY=Tokyo ./scripts/rq1_reproducibility.sh aggregate
```

Output: `results/beliefmove-evo/aggregated/rq1_summary.json` và
`ideas/results_rq1.md`. Missing/incompatible bounded baseline tạo gate partial,
không được lấp bằng số liệu từ protocol khác.

## 11. Baselines

```bash
./scripts/run_baselines.sh audit
./scripts/run_baselines.sh llm-zs
./scripts/run_baselines.sh llm-mob
./scripts/run_baselines.sh agentmove
./scripts/run_baselines.sh hybrid
```

Các job Ollama có cache. Chạy lại cùng protocol để resume; không xóa evidence
cache của run chưa hoàn thành.

## 12. Tổng hợp kết quả

Full student run tự ghi raw JSON gồm RQ, experiment, seed, Git commit, dataset
hash, config và metrics.

```bash
./scripts/aggregate_beliefmove_results.sh
```

Output:

```text
results/beliefmove-evo/aggregated/summary.json
ideas/results.md
```

Aggregator tính mean, standard deviation và deterministic bootstrap 95% CI.
Nếu chưa có raw result, file kết quả ghi rõ chưa có dữ liệu; không tạo số giả.

## 13. Test

```bash
.venv/bin/python -m unittest discover -s tests -v
# hoặc
./scripts/beliefmove_evo.sh test
```

Test bao gồm timestamp parsing, preprocessing/split, representation leakage,
teacher cache, Bayesian normalization, distillation, routing và aggregation.

## 14. Publication gates

- Main TIST2015 result phải đủ đúng 12 city.
- Mỗi city có candidate space, checkpoint, calibrator và cache riêng.
- Main result dùng Neural-CGM; Markov chỉ dùng smoke/baseline.
- Giữ cùng top-k, top-m, model, split, seed protocol và limits giữa các city.
- Chỉ ghi “12-city macro average” khi đủ cả 12 city.
- Geographic bias là population variance của city Acc@1.
- OSM coverage dưới 90% phải gắn nhãn `no-OSM`, không gọi là full model.
- Không dùng smoke/limited run cho publication.
- Không xem LLM teacher là ground truth.
- Không claim module hữu ích nếu ablation/corruption không hỗ trợ.

## 15. Trình tự khuyến nghị

```text
environment
→ audit/download/prepare
→ quantitative teacher
→ Tokyo smoke test
→ E0 baseline
→ E1–E5 ablation
→ order corruption
→ Bayesian/sequential belief
→ uncertainty routing
→ multi-city run
→ aggregate
```
