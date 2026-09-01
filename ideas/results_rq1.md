# RQ1 — Baseline reproducibility

> Chưa aggregate trên server. Báo cáo sẽ tách Tokyo matched full-test khỏi TIST2015 bounded 12-city và không so trực tiếp hai protocol.

```bash
cd src/AgentMove
CITY=Tokyo ./scripts/rq1_reproducibility.sh audit
CITY=Tokyo ./scripts/rq1_reproducibility.sh aggregate
```

Nếu baseline bounded còn thiếu, xem audit rồi chạy `run-bounded`. Lệnh này có thể gọi Ollama cho AgentMove và sẽ resume cache hiện có.
