# RQ13 — Robustness với đầu vào thiếu hoặc nhiễu

> Chưa có kết quả aggregate. File này sẽ được ghi đè từ raw JSON, không điền số liệu thủ công.

Chạy từ `src/AgentMove`:

```bash
CITY=Tokyo DEVICE=cuda BATCH_SIZE=256 ./scripts/rq13_robustness.sh run-seeds
CITY=Tokyo SIGNIFICANCE_ITERATIONS=10000 ./scripts/rq13_robustness.sh aggregate
```

Publication gate yêu cầu đủ metrics và paired predictions của chín variant cho seed 42–44.
