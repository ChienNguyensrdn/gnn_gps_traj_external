# BeliefMove-Evo — Phase 2

Phase 2 chỉ tập trung vào bốn câu hỏi có giá trị tiếp tục: RQ3, RQ4, RQ7 và RQ8. Mọi run mới được ghi dưới `results/phase2/`; dữ liệu đã chuẩn bị và teacher Phase 1 được giữ nguyên.

## Phạm vi dữ liệu

- TIST2015: macro 12 thành phố.
- WWW2019: Shanghai-ISP.
- YJMob100K: Dataset1/YJMob.

Mặc định neural chạy seed `42 43 44`. RQ3 dùng một cache LLM bất biến, không pseudo-replicate. RQ8 dùng deterministic policy một lần và 50 seed chỉ cho random-budget-matched.

## Chạy

```bash
cd src/AgentMove

# Kiểm tra input/artifact
./phase2/scripts/run_phase2.sh audit

# RQ4 rồi RQ7; không gọi LLM
RQ_SEEDS="42 43 44" DEVICE=cuda BATCH_SIZE=128 \
  ./phase2/scripts/run_phase2.sh neural

# RQ3 và RQ8 bounded; script tạo/reuse cache đúng limit qua Ollama
LLM_LIMIT=1000 OLLAMA_MODEL=qwen2:7b \
  ./phase2/scripts/run_phase2.sh llm
```

Có thể giới hạn dataset khi debug:

```bash
PHASE2_DATASETS=www2019 ./phase2/scripts/rq4.sh run
PHASE2_DATASETS=yjmob100k ./phase2/scripts/rq7.sh run
TIST_CITIES="Tokyo NewYork" ./phase2/scripts/rq3.sh audit
```

Không so trực tiếp neural last-query với Bayesian all-prefix. Kết quả LLM `LLM_LIMIT=1000` vẫn là bounded experiment và phải được ghi nhãn như vậy.
