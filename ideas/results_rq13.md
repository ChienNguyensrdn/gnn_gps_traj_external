# RQ13 — Robustness với đầu vào thiếu hoặc nhiễu

> Kết quả 9 variant ban đầu đã có trên server. Protocol vừa bổ sung bốn context
> one-axis và dose-response tests; file này sẽ được ghi đè từ raw JSON sau khi chạy
> bổ sung, không chép tay số liệu chưa đủ gate.

Chạy từ `src/AgentMove`:

```bash
CITY=Tokyo DEVICE=cuda BATCH_SIZE=256 ./scripts/rq13_robustness.sh run-seeds
CITY=Tokyo SIGNIFICANCE_ITERATIONS=10000 ./scripts/rq13_robustness.sh aggregate
```

Publication gate yêu cầu đủ metrics và paired predictions của 13 variant cho seed
42–44. Script resume sẽ giữ chín variant đã chạy và chỉ tạo artifact còn thiếu.
