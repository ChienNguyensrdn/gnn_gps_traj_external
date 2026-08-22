# BeliefMove-Evo

> **Trạng thái:** Core research pipeline đã triển khai; full experiments cần chạy theo thứ tự phase và publication gates bên dưới.

## 1. Mục tiêu

BeliefMove-Evo nghiên cứu next-location prediction bằng cách kết hợp:

- quantitative / trajectory teacher;
- LLM mobility teacher;
- representation-evolution distillation;
- lightweight student;
- Bayesian sequential belief;
- uncertainty-aware LLM-on-demand.

Chi tiết: [`ideas/idea.md`](ideas/idea.md)
Thực nghiệm: [`ideas/pipline.md`](ideas/pipline.md)
Kết quả sinh tự động: [`ideas/results.md`](ideas/results.md)

## 2. Repository status

| Component | Status |
|---|---|
| Baseline reproduction | Implemented; metrics pending run |
| Mobility representation | Implemented |
| Quantitative teacher | Neural-CGM implemented |
| LLM teacher cache | Immutable cache implemented |
| Bayesian data-only | Static BBN + sequential belief implemented |
| KD / trajectory / velocity / temporal | Implemented |
| Order corruption | Correct/reverse/random implemented |
| Uncertainty router | Entropy/margin implemented |
| Evaluation | Ranking, calibration, CKA, transition metrics implemented |
| Reproducibility scripts | Implemented |

## 3. Environment

```bash
cd src/AgentMove
./scripts/setup_ubuntu.sh
./scripts/beliefmove_evo.sh environment
```

### Check PyTorch/device

```bash
.venv/bin/python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("mps:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
PY
```

## 4. Dataset

Pipeline chính dùng Foursquare TIST2015 với 12 city canonical và temporal
train/validation/test split độc lập theo city. Raw files nằm tại
`src/AgentMove/data/dataset_tist2015/`. ISP-Shanghai dùng cho đối chứng bổ sung.

**Không thay đổi split giữa baseline và proposed methods.**

## 5. Data preparation

```bash
cd src/AgentMove
./scripts/tist2015_pipeline.sh audit
./scripts/tist2015_pipeline.sh download  # chỉ khi audit báo raw=missing
./scripts/tist2015_pipeline.sh prepare
```

## 6. Reproduce baseline

```bash
cd src/AgentMove
./scripts/run_baselines.sh audit
./scripts/run_baselines.sh llm-zs
./scripts/run_baselines.sh llm-mob
./scripts/run_baselines.sh agentmove
./scripts/run_baselines.sh hybrid
```

Expected:

```text
results/tist2015-*/
results/logs/baselines/
```

## 7. Build teacher cache

```bash
cd src/AgentMove
./scripts/tist2015_pipeline.sh train
```

```bash
TYPE=llm INPUT=results/path/evidence_cache.jsonl \
  ./scripts/build_teacher_cache.sh
```

Teacher outputs phải được cache để ablation reproducible.

## 8. Train Bayesian data-only model

Data-only Bayesian inference dùng `hybrid.bayesian_network`; sequential update
dùng `hybrid.sequential_belief`. Không gọi LLM trong Bayesian-only experiment.

## 9. Train lightweight student

```bash
cd src/AgentMove
CITY=Tokyo VARIANT=E0-ce ./scripts/beliefmove_evo.sh train-student
```

Mặc định là full run: 10 epochs, batch 64, toàn bộ prefix training và validation.
Trên Ubuntu, `DEVICE=auto` ưu tiên CUDA rồi mới fallback CPU. Smoke test nhanh,
được ghi vào thư mục `artifacts/smoke` và không đưa vào bảng publication:

```bash
CITY=Tokyo VARIANT=E0-ce EPOCHS=1 BATCH_SIZE=256 DEVICE=cuda \
TRAIN_LIMIT=2000 VALIDATION_LIMIT=500 \
  ./scripts/beliefmove_evo.sh train-student
```

Xem dòng cấu hình đầu run để kiểm tra `device`, `train_examples` và
`steps_per_epoch`. Nếu máy không có CUDA, dùng `DEVICE=cpu` và giảm limits.

```bash
CITY=Tokyo VARIANT=E1-kd ./scripts/beliefmove_evo.sh train-student
```

```bash
CITY=Tokyo VARIANT=E4-layer ./scripts/beliefmove_evo.sh train-student
```

```bash
CITY=Tokyo VARIANT=E5-dual ./scripts/beliefmove_evo.sh train-student
CITY=Tokyo VARIANT=E5-dual ./scripts/beliefmove_evo.sh order-ablation
```

## 10. Sequential belief

`SequentialBelief.step()` thực hiện transition prediction rồi Bayesian evidence
update; mọi bước kiểm tra normalization và NaN.

## 11. Uncertainty-aware inference

`SelectiveLLMPolicy` hỗ trợ entropy và top-2 margin. Threshold phải fit trên
validation; config tại `configs/beliefmove_evo/routing.json`.

Router threshold phải được chọn trên validation set.

## 12. Run RQ experiments

Các phase chính:

```bash
cd src/AgentMove
./scripts/beliefmove_evo.sh audit
./scripts/beliefmove_evo.sh environment
./scripts/beliefmove_evo.sh train-teacher
CITY=Tokyo VARIANT=E5-dual ./scripts/beliefmove_evo.sh train-student
./scripts/beliefmove_evo.sh aggregate
```

## 13. Aggregate results

```bash
cd src/AgentMove
./scripts/aggregate_beliefmove_results.sh
```

Script đọc `results/beliefmove-evo/raw/**/*.json`, kiểm tra schema/provenance,
group theo RQ/experiment/dataset, tính mean/std và ghi đồng thời
`results/beliefmove-evo/aggregated/summary.json` và `ideas/results.md`. Nếu chưa
có raw metrics, file kết quả ghi rõ chưa có dữ liệu và không tạo số giả.

## 14. Tests

```bash
cd src/AgentMove
.venv/bin/python -m unittest discover -s tests -v
# hoặc
./scripts/beliefmove_evo.sh test
```

Tests tối thiểu:

- preprocessing;
- split integrity;
- teacher cache;
- belief normalization;
- uncertainty;
- metrics.

## 15. Reproducibility

Mỗi run phải log:

- seed;
- git commit;
- dataset hash/version;
- config path;
- Python version;
- Torch version;
- device;
- raw metrics.

## 16. Metrics

Prediction:

- Acc@1;
- Acc@5;
- Acc@10;
- MRR.

Probabilistic quality:

- NLL;
- Brier Score;
- ECE.

Efficiency:

- p50 latency;
- p95 latency;
- peak memory;
- token/query;
- LLMCallRate.

Representation analysis:

- CKA;
- transition cosine similarity.

## 17. Important rules

1. Không tune test set.
2. Không xem LLM teacher là ground truth.
3. Không gọi LLM mọi query trong adaptive-routing experiment.
4. Không thay preprocessing giữa baseline và proposed.
5. Không điền fabricated results.
6. Không claim semantic module hữu ích nếu corruption/ablation không hỗ trợ.
7. Không claim foundation-model gain nếu chưa có direct comparison.

## 18. Results

Xem [`ideas/results.md`](ideas/results.md).

## 19. Roadmap

```text
Baseline
→ representation
→ teacher cache
→ BN data-only
→ LLM distillation
→ representation evolution
→ order corruption
→ sequential belief
→ uncertainty routing
→ semantic verification
→ calibration
→ efficiency
→ robustness
```

## 20. Citation

```bibtex
@article{beliefmove_evo_2026,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {2026}
}
```

## 21. License

TODO

## 22. Contact

TODO
