# pipline.md — Kế hoạch thực nghiệm để Codex thực thi BeliefMove-Evo

> Tên file giữ theo yêu cầu: `pipline.md`.

# 0. Nguyên tắc cho Codex

1. Không sửa preprocessing baseline trước khi reproduce.
2. Không dùng test set để chọn threshold, calibration hoặc hyperparameter.
3. Không điền số giả vào `results.md`.
4. Mỗi experiment phải có config riêng.
5. Mỗi run phải lưu seed, git commit, dataset hash, config.
6. LLM outputs phải cache để tránh thay đổi giữa các ablation.
7. Không gọi LLM online trong experiment “Bayesian-only”.
8. Teacher belief không được coi là ground truth.
9. Mọi bảng phải truy xuất được từ raw metrics.
10. Không tối ưu trực tiếp trên test set.

# 1. Repository structure mục tiêu

```text
project/
├── idea.md
├── results.md
├── pipline.md
├── README.MD
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── datasets/
│   ├── teachers/
│   ├── students/
│   ├── bayes/
│   └── experiments/
├── src/beliefmove/
│   ├── data/
│   ├── representation.py
│   ├── teachers/
│   ├── students/
│   ├── distillation/
│   ├── bayes/
│   ├── routing/
│   ├── decoder.py
│   ├── metrics.py
│   └── evaluation.py
├── scripts/
│   ├── reproduce_baseline.py
│   ├── build_teacher_cache.py
│   ├── train_student.py
│   ├── train_bayesian.py
│   ├── evaluate.py
│   ├── run_rq.py
│   └── aggregate_results.py
├── tests/
├── artifacts/
└── results/
    ├── raw/
    ├── aggregated/
    ├── tables/
    └── figures/
```

# 2. Phase 0 — Environment

## Task

- xác định Python version;
- cài dependency;
- lock versions;
- kiểm tra CPU/CUDA/MPS;
- không ép `torch==2.1.0` nếu Python không hỗ trợ.

## Output

- `requirements.txt`;
- `environment_report.txt`;
- smoke test pass.

## Acceptance criteria

```bash
python -c "import torch; print(torch.__version__)"
pytest -q
```

# 3. Phase 1 — Reproduce baseline

1. Clone/reuse AgentMove.
2. Freeze preprocessing.
3. Freeze train/val/test split.
4. Chạy baseline.
5. Ghi metrics và config.

Output:

```text
results/raw/rq01/
artifacts/baseline/
```

# 4. Phase 2 — Mobility representation

Viết `representation.py`.

Feature tối thiểu:

- region/POI;
- time-of-day;
- day-of-week;
- speed nếu có;
- heading nếu có;
- stop duration nếu có;
- historical frequency nếu có.

Unit tests:

- deterministic encoding;
- shape consistency;
- no future leakage;
- missing handling.

# 5. Phase 3 — Teacher cache

## Quantitative teacher

Lưu logits + hidden states + label.

## LLM teacher

Lưu structured belief JSON.

Rule: teacher cache immutable trong ablation.

Acceptance:

- parse success ≥ 99% hoặc deterministic fallback;
- cache có hash/version;
- rerun không gọi lại LLM nếu cache hợp lệ.

# 6. Phase 4 — Data-only Bayesian baseline

Cài:

1. BN data-only;
2. optional DBN.

Không distillation.

Kiểm tra:

- probability normalization;
- CPT validity;
- no NaN;
- reproducible inference.

# 7. Phase 5 — LLM belief distillation

Chạy:

```text
M1 = data-only
M2 = data + LLM beliefs
M3 = data + quantitative teacher
M4 = data + both teachers
```

Nếu M2/M4 không tốt hơn M1, không claim LLM knowledge hữu ích.

# 8. Phase 6 — Representation evolution

Cài modules:

```text
src/beliefmove/distillation/kd.py
src/beliefmove/distillation/trajectory.py
src/beliefmove/distillation/velocity.py
src/beliefmove/distillation/temporal.py
```

Teacher/student states phải project về common latent space trước khi trừ.

# 9. Phase 7 — Evolution ablation

```text
E0 = CE
E1 = CE + KD
E2 = CE + KD + Traj
E3 = CE + KD + Vel
E4 = CE + KD + Traj + Vel
E5 = E4 + Temporal
```

Lưu Acc@1, MRR, CKA, transition cosine.

# 10. Phase 8 — Order corruption

Transforms:

```python
correct(sequence)
reverse(sequence)
random_permute(sequence, seed)
```

Rules:

- random ≥ 10 permutation seeds;
- không thay label;
- không thay split;
- không thay candidate vocabulary.

# 11. Phase 9 — Sequential belief

So sánh:

```text
independent BN
BN + history features
sequential belief
DBN
```

Unit test:

- belief sum = 1;
- update deterministic;
- past evidence ảnh hưởng current belief.

# 12. Phase 10 — Uncertainty router

Entropy:

```python
if entropy(belief) >= threshold:
    call_llm()
else:
    predict()
```

Margin:

```python
if top1_prob - top2_prob < threshold:
    call_llm()
```

Threshold fit validation only.

Output:

- Accuracy vs LLMCallRate;
- MRR vs latency.

# 13. Phase 11 — Semantic corruption

Memory:

```text
true
shuffled
random_user
none
```

Context:

```text
true
shuffled
random_poi
none
```

Giữ nguyên candidate set.

# 14. Phase 12 — Teacher robustness

Teacher:

- GRU;
- Transformer;
- Foundation Model nếu khả thi.

Student fixed.

# 15. Phase 13 — Calibration

Tính:

- NLL;
- Brier;
- ECE;
- reliability diagram.

Calibration fit validation only.

# 16. Phase 14 — Efficiency benchmark

Timing protocol:

- warm-up;
- synchronize GPU nếu CUDA;
- p50/p95;
- cùng hardware;
- cùng batch size;
- tách offline và online cost.

Báo riêng:

1. teacher offline;
2. LLM teacher-cache generation;
3. student online latency;
4. BBN latency;
5. LLM-on-demand latency.

# 17. Phase 15 — Robustness

```text
missing_points = [0.1, 0.3, 0.5]
gps_noise = [...]
timestamp_noise = [...]
context_missing = [true]
```

# 18. Metrics API

`metrics.py` phải có:

```python
accuracy_at_k(...)
mrr(...)
nll(...)
brier(...)
ece(...)
llm_call_rate(...)
latency_summary(...)
cka(...)
transition_cosine(...)
```

# 19. Raw result schema

```json
{
  "rq": "RQ4",
  "experiment": "kd_traj_vel",
  "seed": 42,
  "git_commit": "...",
  "dataset": "...",
  "config": "...",
  "metrics": {
    "acc1": 0.0,
    "mrr": 0.0,
    "ece": 0.0
  }
}
```

Không ghi số tay vào bảng nếu có thể aggregate tự động.

# 20. Aggregation

`aggregate_results.py`:

1. đọc raw JSON;
2. group experiment;
3. mean/std;
4. bootstrap CI;
5. sinh CSV/Markdown tables;
6. xuất `results/aggregated/`.

# 21. Statistical testing

Ưu tiên:

- paired bootstrap CI;
- Wilcoxon signed-rank khi phù hợp.

Báo mean difference, 95% CI, p-value, effect direction.

# 22. Leakage checklist

- [ ] Test labels không dùng training.
- [ ] Calibration không fit test.
- [ ] Router threshold không chọn từ test.
- [ ] Personal memory không chứa future point.
- [ ] Teacher cache không chứa test-ground-truth reasoning.
- [ ] Reverse/random chỉ thay input order.
- [ ] Corruption không thay candidate set.
- [ ] Same preprocessing across methods.

# 23. Execution order bắt buộc

```text
P0 environment
↓
P1 baseline
↓
P2 representation
↓
P3 teacher cache
↓
P4 BN data-only
↓
P5 LLM distillation
↓
P6 evolution modules
↓
P7 evolution ablation
↓
P8 order corruption
↓
P9 sequential belief
↓
P10 uncertainty router
↓
P11 semantic corruption
↓
P12 teacher robustness
↓
P13 calibration
↓
P14 efficiency
↓
P15 robustness
↓
final aggregation
```

Không chạy full model trước khi baseline và ablation ổn định.

# 24. Definition of Done

Một RQ hoàn thành khi:

1. config commit;
2. tests pass;
3. raw result tồn tại;
4. đủ seed;
5. aggregated table sinh thành công;
6. statistical test hoàn thành;
7. `results.md` cập nhật;
8. không còn TODO quan trọng.

# 25. Quy tắc cập nhật README

Sau mỗi phase, Codex phải cập nhật `README.MD` nếu:

- command thay đổi;
- dependency thay đổi;
- dataset path thay đổi;
- config mới được thêm;
- output artifact mới xuất hiện.
