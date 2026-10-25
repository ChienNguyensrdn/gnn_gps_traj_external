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

# RQ3 và RQ8 bounded theo protocol dùng trong paper
LLM_LIMIT=200 OLLAMA_MODEL=qwen2:7b \
  ./phase2/scripts/run_phase2.sh llm

# Xem tiến độ cache LLM của từng dataset/thành phố
LLM_LIMIT=200 OLLAMA_MODEL=qwen2:7b \
  ./phase2/scripts/run_phase2.sh llm-status

# Theo dõi liên tục từ terminal khác
watch -n 10 'LLM_LIMIT=200 OLLAMA_MODEL=qwen2:7b ./phase2/scripts/run_phase2.sh llm-status'

# Xuất báo cáo bounded RQ3/RQ8 sau khi limit=200 hoàn tất
LLM_LIMIT=200 OLLAMA_MODEL=qwen2:7b SIGNIFICANCE_ITERATIONS=10000 \
  ./phase2/scripts/run_phase2.sh llm-report

# Tổng hợp RQ4 và RQ7, ghi số liệu thật vào hai report Markdown
SIGNIFICANCE_ITERATIONS=10000 ./phase2/scripts/run_phase2.sh report

# Kiểm tra gate của cả bốn RQ trước khi xuất bảng cho paper
LLM_LIMIT=200 ./phase2/scripts/publication_paper.sh audit

# Sinh lại sections/generated_phase2_tables.tex từ JSON đã khóa
PAPER_DIR=/path/to/IEEE-Transactions_Latex LLM_LIMIT=200 \
  ./phase2/scripts/publication_paper.sh export

# Biên dịch paper sau khi export
PAPER_DIR=/path/to/IEEE-Transactions_Latex \
  ./phase2/scripts/publication_paper.sh build

# Hoặc aggregate + audit + export + build trong một lệnh
PAPER_DIR=/path/to/IEEE-Transactions_Latex LLM_LIMIT=200 \
  ./phase2/scripts/publication_paper.sh all
```

Có thể tạo riêng bằng `./phase2/scripts/aggregate.sh rq4` hoặc
`./phase2/scripts/aggregate.sh rq7`. JSON trung gian nằm trong
`results/phase2/aggregated/`; hai báo cáo được ghi tại
`phase2/rq4_report.md` và `phase2/rq7_report.md`.

Có thể giới hạn dataset khi debug:

```bash
PHASE2_DATASETS=www2019 ./phase2/scripts/rq4.sh run
PHASE2_DATASETS=yjmob100k ./phase2/scripts/rq7.sh run
TIST_CITIES="Tokyo NewYork" ./phase2/scripts/rq3.sh audit
```

`publication_paper.sh` dừng ngay nếu một JSON chưa có gate `ready`/`ready-bounded`, còn artifact thiếu, hoặc thiếu một trong ba dataset. Không so trực tiếp neural last-query với Bayesian all-prefix. Kết quả LLM `LLM_LIMIT=200` vẫn là bounded experiment và phải được ghi nhãn như vậy.
