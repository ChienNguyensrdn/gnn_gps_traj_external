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

# Đối chứng query-time LLM sạch: Q0 quantitative vs Q1 Direct-1Call.
# Bước prepare chỉ fit temperature trên validation và KHÔNG gọi LLM.
LLM_LIMIT=200 ./phase2/scripts/run_phase2.sh direct-llm-prepare

# CẢNH BÁO CHI PHÍ: tối đa 5.600 request cho 14 đơn vị dữ liệu,
# gồm validation + test. Collection tạo Always response bank; mỗi query chưa
# cache trả đúng một physical request, bất kể policy sẽ được replay sau đó.
LLM_LIMIT=200 OLLAMA_MODEL=qwen2:7b \
  ./phase2/scripts/run_phase2.sh direct-llm

# Xem tiến độ hoặc chỉ chạy pilot riêng (không trộn với limit=200).
LLM_LIMIT=200 ./phase2/scripts/run_phase2.sh direct-llm-status
PHASE2_DATASETS=www2019 LLM_LIMIT=10 OLLAMA_MODEL=qwen2:7b \
  ./phase2/scripts/run_phase2.sh direct-llm

# Audit riêng response bank validation/test, chưa yêu cầu routing output.
LLM_LIMIT=200 ./phase2/scripts/direct_query_llm.sh audit-cache

# Audit đầy đủ response bank và 50 replay seed (mặc định 42--91).
LLM_LIMIT=200 ./phase2/scripts/run_phase2.sh direct-llm-audit

# Tổng hợp Q1-vs-Q0 và entropy-vs-random bằng user-cluster inference.
LLM_LIMIT=200 SIGNIFICANCE_ITERATIONS=10000 \
  ./phase2/scripts/run_phase2.sh direct-llm-report

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

Đối chứng Direct-1Call nằm độc lập dưới
`results/phase2/<dataset>/artifacts/full/<unit>/direct-query-llm/`. Q1 nhận
đúng frozen quantitative top-10 và xác suất của Q0, nhưng không được đọc
structured-evidence cache hay rationale do LLM sinh trước. Mỗi uncached query
chỉ có một physical request: Tenacity và retry mặc định của OpenAI SDK đều bị
tắt. Output sai JSON/thiếu candidate fallback toàn bộ về thứ tự Q0 nhưng vẫn
tính một request.

Collection luôn gọi LLM cho từng query validation/test chưa có cache để tạo
**Always response bank bất biến**; vì vậy giai đoạn thu thập đã trả một physical
call cho mỗi query. Lệnh `route` không gọi LLM mà thực hiện
**counterfactual deployment replay** trên response bank này. Các số
`never=0`, call rate của entropy/margin/random, token và latency được báo cáo là
**ngân sách triển khai mô phỏng**: chi phí sẽ phát sinh nếu policy quyết định
trước khi gọi ở deployment. Chúng không phải số call thực tế đã tránh được khi
collection và không được mô tả là một thí nghiệm route-before-call thực tế.
Comparator này cũng không bao gồm pipeline hai tầng cũ.

Ngay cả với `DIRECT_WORLD_MODE=internal_only`, prompt vẫn sử dụng metadata
category/latitude/longitude trong `candidate_metadata`. Metadata hiện được tạo
từ prepared/full corpus thay vì train-only; do đó Direct-1Call và routing phải
được ghi nhãn **bounded, transductive diagnostic**, không phải inductive
deployment result.

## Gate trước khi nộp bài

- Đồng bộ raw Phase-2 artifacts/aggregate JSON từ server; không xuất bảng từ các JSON local đang có gate `incomplete`.
- Tạo lại `candidate_metadata.json` chỉ từ train rồi chạy lại RQ3/RQ8. Bản chuẩn bị dữ liệu hiện tại lấy metadata từ toàn bộ prepared corpus, nên kết quả LLM hiện chỉ là transductive diagnostic.
- Chạy sensitivity test theo cụm user cho RQ4/RQ3/RQ8 và theo user → trajectory cho RQ7; paired test hiện tại resample ở query level.
- Ghi run manifest gồm batch size, commit, teacher checksum, Ollama version/model digest/quantization/context.
- Tổng hợp invalid-output/retry rate của cache LLM và retained/OOV query rate cho từng dataset.
