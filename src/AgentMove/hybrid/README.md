# Hybrid GPS Trajectory — RQ1–RQ4

Module này triển khai pipeline ba tầng cho bài báo:

```text
Trajectory → CGM/top-k prior → LLM evidence → Bayesian Network → posterior
```

- Stage 1: Markov CGM hiện có hoặc logits GETNext sau khi huấn luyện.
- Stage 2: Ollama trích xuất `habit_score` và `semantic_score`.
- Stage 3: Bayesian Network tường minh với `L → H`, `L → S`.
- Output: metrics, ablation và artefact trả lời RQ1–RQ4.

Các lệnh dưới đây chạy từ thư mục `src/AgentMove`.

## 0. Khởi tạo môi trường

```bash
cd /Users/chiennguyen/Documents/Codex/Hybrid-GPS-Traj/src/AgentMove
source .venv/bin/activate
python --version
python -m unittest discover -s tests -v
```

## 1. Kiểm tra Ollama port 11434

Ollama native API:

```bash
curl http://127.0.0.1:11434/api/version
curl http://127.0.0.1:11434/api/tags
```

Hybrid sử dụng OpenAI-compatible endpoint:

```bash
export OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
export OLLAMA_API_KEY=ollama
```

`OLLAMA_API_KEY=ollama` chỉ là placeholder, không phải secret. Ollama local mặc
định không yêu cầu API key.

Các model hiện có trên máy:

```text
qwen2:7b
qwen2.5-coder:7b
```

Nên dùng `qwen2:7b` cho mobility reasoning. Kiểm tra:

```bash
OLLAMA_MODEL=qwen2:7b ./scripts/hybrid_pipeline.sh ollama-test
```

Nếu server chưa chạy:

```bash
./scripts/start_ollama.sh
```

## 2. ISP-Shanghai: extract dữ liệu

```bash
DATASET=isp \
CITY=Shanghai \
./scripts/hybrid_pipeline.sh extract
```

Output:

```text
data/input_trajectories/Shanghai_filtered.csv
```

## 3. ISP-Shanghai: OSM enrichment tùy chọn

Không có Nominatim local thì bỏ qua OSM:

```bash
export USE_OSM=0
```

Nếu Nominatim chạy tại port `8080`:

```bash
DATASET=isp \
CITY=Shanghai \
USE_OSM=1 \
NOMINATIM_URL=http://127.0.0.1:8080 \
./scripts/hybrid_pipeline.sh osm
```

Output:

```text
data/input_trajectories_clean/Shanghai_filtered.csv
data/nominatim/Shanghai_hybrid.jsonl
```

OSM cache được ghi tăng dần; chạy lại sẽ tiếp tục các POI còn thiếu.

## 4. ISP-Shanghai: tạo split và CGM artefacts

Không dùng OSM:

```bash
DATASET=isp \
CITY=Shanghai \
USE_OSM=0 \
./scripts/hybrid_pipeline.sh prepare
```

Có dùng OSM:

```bash
DATASET=isp \
CITY=Shanghai \
USE_OSM=1 \
./scripts/hybrid_pipeline.sh prepare
```

Output:

```text
data/hybrid/Shanghai/
├── candidate_ids.json
├── candidate_metadata.json
├── validation_logits.npy
├── test_logits.npy
├── validation_metadata.jsonl
├── test_metadata.jsonl
├── validation.jsonl
├── test.jsonl
├── dataset_statistics.json
└── getnext/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

Logits hiện được sinh bởi smoothed first-order Markov CGM. Đây là Markov
baseline và đủ để chạy end-to-end. Đối với kết quả chính thức, huấn luyện
GETNext bằng các CSV trong `getnext/`, sau đó thay hai file logits.

## 5. Chạy smoke test 10 validation + 10 test queries

```bash
DATASET=isp \
CITY=Shanghai \
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1 \
OLLAMA_MODEL=qwen2:7b \
TOP_K=10 \
TOP_M=5 \
LLM_BATCH_SIZE=3 \
LLM_RETRIES=2 \
LLM_MISSING_POLICY=neutral \
VALIDATION_LIMIT=10 \
TEST_LIMIT=10 \
./scripts/hybrid_pipeline.sh run
```

### Preset nhẹ cho Mac GPU/Apple Silicon

Preset mặc định chỉ chạy 5 validation + 5 test queries, `top-k=5`, và xử lý
từng candidate một để giảm RAM/VRAM:

```bash
./scripts/run_hybrid_mac_small.sh
```

Chạy cực nhỏ để kiểm tra nhanh:

```bash
VALIDATION_LIMIT=2 TEST_LIMIT=2 TOP_K=3 \
./scripts/run_hybrid_mac_small.sh
```

Tăng dần sau khi chạy ổn:

```bash
VALIDATION_LIMIT=20 TEST_LIMIT=20 TOP_K=5 \
./scripts/run_hybrid_mac_small.sh
```

Nếu Qwen thường bỏ candidate trong JSON, dùng chế độ ổn định hơn nhưng chậm:

```bash
DATASET=isp \
CITY=Shanghai \
OLLAMA_MODEL=qwen2:7b \
LLM_BATCH_SIZE=1 \
LLM_RETRIES=3 \
LLM_MISSING_POLICY=neutral \
VALIDATION_LIMIT=10 \
TEST_LIMIT=10 \
./scripts/hybrid_pipeline.sh run
```

### Chính sách evidence bị thiếu

Mặc định:

```bash
LLM_MISSING_POLICY=neutral
```

Sau retry, candidate vẫn thiếu evidence sẽ được cache với:

```json
{"valid": false, "habit_score": 0.5, "semantic_score": 0.5}
```

Trong Bayesian Network, candidate này nhận:

```text
P(H|L) = 1
P(S|L) = 1
```

Nó giữ nguyên CGM prior, không bị thưởng hoặc phạt bởi evidence không tồn tại.
Metric `invalid_evidence_rate` phải được báo cáo cùng kết quả.

Chế độ debug nghiêm ngặt, dừng ngay khi thiếu evidence:

```bash
LLM_MISSING_POLICY=error
```

## 6. Kiểm tra output smoke test

```text
results/hybrid/isp-Shanghai-ollama-qwen2-7b/
├── evidence_cache.jsonl
├── calibration.json
├── bbn_structure.json
├── rq1_main_results.json
├── rq1_full_vs_stage1_bootstrap.json
├── rq2_ablation.json
├── rq3_efficiency.json
├── rq4_calibration_and_generalization.json
├── full/
│   ├── predictions.jsonl
│   └── metrics.json
└── <ablation>/
    ├── predictions.jsonl
    └── metrics.json
```

Xem metric:

```bash
python -m json.tool \
  results/hybrid/isp-Shanghai-ollama-qwen2-7b/full/metrics.json
```

Xem BN:

```bash
python -m json.tool \
  results/hybrid/isp-Shanghai-ollama-qwen2-7b/bbn_structure.json
```

## 7. Resume sau khi lỗi hoặc dừng chương trình

Evidence được append ngay vào:

```text
results/hybrid/isp-Shanghai-ollama-qwen2-7b/evidence_cache.jsonl
```

Chạy lại đúng lệnh cũ. Candidate đã hoàn thành sẽ được đọc từ cache và không
gọi Ollama lần nữa.

Pipeline kiểm tra cổng Ollama trước khi chạy. Nếu cổng `11434` mất kết nối giữa
chừng, tiến trình trả mã `75`, tự khởi động Ollama và chạy lại tối đa 3 lần.
Lượt chạy lại tiếp tục từ cache, không tính lỗi kết nối thành evidence neutral.
Có thể đổi số lần tự khởi động lại:

```bash
OLLAMA_RESTART_ATTEMPTS=5 ./scripts/run_hybrid_80pct.sh
```

Nếu vẫn dừng, kiểm tra server và log rồi chạy lại cùng lệnh:

```bash
curl -fsS http://127.0.0.1:11434/api/version
tail -n 100 /tmp/hybrid-ollama.log
```

Không xóa `evidence_cache.jsonl` nếu muốn resume. File `evidence_cache.json`
nếu còn từ phiên bản cũ không được pipeline mới sử dụng.

## 8. Chạy toàn bộ ISP-Shanghai

### Stage 1 Neural CGM trên Shanghai-50

Sau khi train/export `data/hybrid/Shanghai/neural_cgm/best.pt`, chạy pipeline
với logits neural bằng:

```bash
./scripts/run_shanghai_neural_50pct_rqs.sh
```

Runner này dùng output riêng và tạo hai cache evidence: cache embedding-memory
cho `full`, và cache frequency-memory cho ablation `no_embedding_memory`.
Không dùng cache Markov cũ vì top-10 candidates đã thay đổi.

### Pipeline RQ1–RQ4 trên mẫu 50% tái lập (khuyến nghị)

Pipeline này chọn đúng 50% query bằng SHA-256 của `seed:query_id`, không lấy
nửa đầu file. Nó dùng output/cache riêng theo seed và model, chạy toàn bộ
variant rồi sinh `RQ_REPORT.md`:

```bash
cd /Users/chiennguyen/Documents/Codex/Hybrid-GPS-Traj/src/AgentMove
./scripts/run_shanghai_50pct_rqs.sh
```

Cấu hình ổn định hơn cho Mac:

```bash
LLM_BATCH_SIZE=1 \
LLM_RETRIES=3 \
OLLAMA_RESTART_ATTEMPTS=5 \
./scripts/run_shanghai_50pct_rqs.sh
```

Artefact nằm tại:

```text
data/hybrid/Shanghai/sample-50-seed42/sample_manifest.json
results/hybrid/shanghai-50-seed42/qwen2-7b/RQ_REPORT.md
```

Tạo lại report mà không gọi Ollama:

```bash
REPORT_ONLY=1 ./scripts/run_shanghai_50pct_rqs.sh
```

Đổi seed sẽ tạo một protocol/cache riêng:

```bash
SAMPLE_SEED=2026 ./scripts/run_shanghai_50pct_rqs.sh
```

Run này đo đủ pipeline cho RQ1–RQ4 nhưng phạm vi kết luận là Shanghai: RQ1
thiếu baseline độc lập nếu chưa chạy GETNext/AgentMove; RQ3 thiếu đối chứng
LLM-only; RQ4 chưa chứng minh cross-city. Report ghi rõ các giới hạn này để
không diễn giải quá mức.

### Preset chạy khoảng 50% dữ liệu

Preset giữ nguyên `TOP_K=10`, `TOP_M=5` như full run nhưng chỉ chạy:

```text
validation: 521 / 1.042
test:       1.202 / 2.403
```

```bash
./scripts/run_hybrid_half.sh
```

Chạy ổn định hơn với batch nhỏ:

```bash
LLM_BATCH_SIZE=1 LLM_RETRIES=3 ./scripts/run_hybrid_half.sh
```

Preset này dùng cùng output/cache với full run. Sau khi chạy 50%, chạy full sẽ
tiếp tục những query còn thiếu thay vì thực hiện lại evidence đã có.

### Preset chạy khoảng 80% dữ liệu

```text
validation: 834 / 1.042
test:       1.923 / 2.403
```

```bash
./scripts/run_hybrid_80pct.sh
```

Script này tự kiểm tra/khởi động Ollama, xác nhận `qwen2:7b` đã được cài, và tự
resume tối đa 3 lần nếu Ollama bị ngắt. Trên Mac có RAM/GPU hạn chế, nên giữ
`OLLAMA_NUM_PARALLEL=1`; nếu model trả JSON thiếu thường xuyên, dùng batch 1:

Nếu đã chạy preset 50%, preset 80% dùng chung cache và chỉ bổ sung phần từ 50%
đến 80%. Có thể dùng cấu hình ổn định hơn:

```bash
LLM_BATCH_SIZE=1 LLM_RETRIES=3 ./scripts/run_hybrid_80pct.sh
```

Bỏ `VALIDATION_LIMIT` và `TEST_LIMIT`:

```bash
DATASET=isp \
CITY=Shanghai \
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1 \
OLLAMA_MODEL=qwen2:7b \
TOP_K=10 \
TOP_M=5 \
LLM_BATCH_SIZE=3 \
LLM_RETRIES=2 \
LLM_MISSING_POLICY=neutral \
./scripts/hybrid_pipeline.sh run
```

Kiểm tra command được tạo mà không gọi Ollama:

```bash
DATASET=isp CITY=Shanghai OLLAMA_MODEL=qwen2:7b \
HYBRID_DRY_RUN=1 ./scripts/hybrid_pipeline.sh run
```

## 9. LLM-only baseline và dữ liệu bảng paper

LLM-ZS reproduction chỉ dùng history/context, không dùng CGM logits hoặc BN. Theo protocol
AgentMove, runner sắp xếp user/session theo ID, lấy một session cho mỗi user, chọn 200 user đầu,
yêu cầu đúng 5 dự đoán và báo cáo Acc@1, Acc@5, NDCG@5:

```bash
./scripts/run_llm_zs_agentmove_200.sh
```

Smoke test không dùng chung cache với run chính:

```bash
TEST_LIMIT=10 \
OUTPUT_DIR=results/llm-only/smoke-agentmove-10/qwen2-7b \
./scripts/run_llm_zs_agentmove_200.sh
```

Kết quả 1.202 query trước đây là extended evaluation, không phải reproduction protocol của
AgentMove. Kết quả Qwen2 cũng phải ghi rõ base LLM khác GPT-4o-mini trong paper.

Chạy Hybrid trên đúng cùng 200 user/session (validation Shanghai-50 vẫn dùng cho calibration):

```bash
./scripts/run_hybrid_agentmove_200.sh
```

Runner tái sử dụng các evidence row phù hợp từ Shanghai-50 rồi chỉ gọi Ollama cho query còn
thiếu. Output nằm tại `results/hybrid/agentmove-faithful-shanghai-200/qwen2-7b/` và metrics
có đủ Acc@1, Acc@5, NDCG@5 để so sánh với LLM-ZS trên cùng test IDs.

### Paper-faithful Hybrid v2

V2 sửa ba sai lệch của prototype: history-augmented CGM được tune trên validation,
likelihood-ratio calibration đúng công thức Bayes, và một compact LLM request cho toàn bộ
top-k. Runner có protocol gate cho candidate recall và OSM coverage.

```bash
# 1) Xem CGM/OSM đã đủ điều kiện chưa (không gọi LLM)
./scripts/run_hybrid_paper_v2.sh audit

# 2) Khi OSM coverage chưa đạt, bật Nominatim rồi enrich metadata
NOMINATIM_URL=http://127.0.0.1:8080 ./scripts/run_hybrid_paper_v2.sh osm

# 3) Audit lại; chỉ khi pass mới chạy Ollama
./scripts/run_hybrid_paper_v2.sh audit
./scripts/run_hybrid_paper_v2.sh run
```

Không hạ `MIN_OSM_COVERAGE` cho số liệu paper. Output v2 nằm riêng tại
`results/hybrid/paper-v2-agentmove-200/qwen2-7b/`; cache prototype cũ không được tái sử dụng
vì prompt, candidate set và ý nghĩa likelihood đã thay đổi.

Chạy toàn bộ tuần tự bằng một lệnh (Ollama luôn ở `127.0.0.1:11434`):

```bash
./scripts/run_hybrid_paper_v2_full.sh
```

Lần đầu script dùng Nominatim HTTPS công khai và giữ tốc độ 1.1 giây/request; OSM responses
được cache tại `data/nominatim/`, nên chạy lại sẽ resume. Có thể đặt `NOMINATIM_URL` thành
dịch vụ Nominatim riêng nếu đã triển khai local.

Exact RQ2 `w/o BBN` trên cùng Hybrid v2 test-200 và evidence cache:

```bash
./scripts/run_rq2_no_bbn_paper_v2.sh
```

Runner giữ Stage-1 top-10, OSM, structured personal memory và $(h_i,s_i)$; chỉ thay Stage 3
BBN bằng một direct free-text ranking call. Output resume tại
`results/rq2/paper-v2-agentmove-200/qwen2-7b/no-bbn-free-text/`.

Exact RQ2 `w/o OSM` (baseline phải có OSM coverage $\geq 90\%$):

```bash
./scripts/run_rq2_no_osm_paper_v2.sh
```

Runner giữ nguyên CGM, structured memory, compact extraction, likelihood-ratio calibration
và BBN; chỉ xóa address/OSM metadata khỏi LLM prompt và dùng internal knowledge.

### Exact RQ2 `w/o Personal Memory (embedding)`

Runner sau giữ nguyên Neural-CGM top-10, OSM, Qwen2:7b, calibration và BBN.
Nó chỉ thay embedding-memory retrieval bằng frequency-memory retrieval, đồng
thời tái sử dụng cache embedding của full run để fit cùng calibrator:

```bash
cd /Users/chiennguyen/Documents/Codex/Hybrid-GPS-Traj/src/AgentMove
./scripts/run_rq2_no_embedding_memory_paper_v2.sh
```

Ollama luôn dùng `127.0.0.1:11434`. Có thể chạy lại cùng lệnh để resume cache.
Kết quả nằm tại:

```text
results/rq2/paper-v2-agentmove-200/qwen2-7b/no-embedding-memory/no_embedding_memory/metrics.json
```

### Table 1/3: LLM-Mob và AgentMove original trên Shanghai-200

Runner dùng code/prompt gốc của repo, chọn một session cho mỗi user, chạy tuần
tự để đo latency chính xác và resume từng JSON đã hoàn thành. Ollama luôn dùng
`127.0.0.1:11434`:

```bash
cd /Users/chiennguyen/Documents/Codex/Hybrid-GPS-Traj/src/AgentMove
./scripts/run_original_baselines_shanghai_200.sh llmmob
./scripts/run_original_baselines_shanghai_200.sh agentmove
```

Hoặc chạy cả hai tuần tự:

```bash
./scripts/run_original_baselines_shanghai_200.sh all
```

`LLM-Mob` chỉ thực hiện call dự đoán mà prompt thật sự sử dụng. `AgentMove`
giữ hai world-model calls và một final-prediction call. Mỗi output lưu
`call_stats`; metrics gồm accuracy, MRR, token và tổng LLM latency/query.

Smoke test 10 query:

```bash
TEST_LIMIT=10 OUTPUT_DIR=results/llm-only/smoke-qwen2-7b \
./scripts/run_llm_only_shanghai_50pct.sh
```

Kết quả append/resume tại
`results/llm-only/shanghai-50-seed42/qwen2-7b/predictions.jsonl`.
Sau khi run hoàn tất, tái sinh dữ liệu LaTeX cho Tables 1--4:

```bash
./scripts/generate_paper_tables.sh
```

Các fragment và manifest nằm trong `../../paper/generated/`. Ô chưa có đúng
protocol được để trống; số từ proxy không được tự động điền vào hàng paper.

## 10. Foursquare TIST2015

Raw data phải nằm tại:

```text
data/dataset_tist2015/
├── dataset_TIST2015_Checkins.txt
├── dataset_TIST2015_POIs.txt
└── dataset_TIST2015_Cities.txt
```

### AgentMove original modules trên TIST2015 (200 query/thành phố)

Run hiện tại dùng dữ liệu TIST2015 chưa enrich OSM, vì vậy kết quả phải ghi là
`AgentMove (no-OSM matched)`, không phải bản full. Ollama luôn ở port `11434`.
Script có thể resume theo từng file prediction:

```bash
cd /Users/chiennguyen/Documents/Codex/Hybrid-GPS-Traj/src/AgentMove

# Smoke test một thành phố trước
./scripts/run_tist2015_agentmove_200.sh Tokyo

# Chạy các thành phố chưa hoàn tất rồi tổng hợp Table II
./scripts/run_tist2015_agentmove_200.sh pending
./scripts/run_tist2015_agentmove_200.sh aggregate
```

Kiểm tra tiến độ:

```bash
./scripts/run_tist2015_agentmove_200.sh audit
```

Prompt gốc AgentMove chỉ trả về 5 POI, nên Acc@10 được để trống. Không được
sao chép Acc@5 sang Acc@10. Muốn đo Acc@10 phải tạo một protocol top-10 riêng
và chạy lại toàn bộ baseline bằng output/cache tách biệt.

Ví dụ với Tokyo:

```bash
DATASET=tist2015 CITY=Tokyo ./scripts/hybrid_pipeline.sh extract
DATASET=tist2015 CITY=Tokyo USE_OSM=0 ./scripts/hybrid_pipeline.sh prepare
```

Smoke test:

```bash
DATASET=tist2015 \
CITY=Tokyo \
OLLAMA_MODEL=qwen2:7b \
LLM_BATCH_SIZE=3 \
LLM_RETRIES=2 \
LLM_MISSING_POLICY=neutral \
VALIDATION_LIMIT=10 \
TEST_LIMIT=10 \
./scripts/hybrid_pipeline.sh run
```

Full run:

```bash
DATASET=tist2015 \
CITY=Tokyo \
OLLAMA_MODEL=qwen2:7b \
LLM_MISSING_POLICY=neutral \
./scripts/hybrid_pipeline.sh run
```

Lặp lại cho 12 thành phố:

```text
Tokyo, Nairobi, NewYork, Sydney, CapeTown, Paris,
Beijing, Mumbai, SanFrancisco, London, SaoPaulo, Moscow
```

Mỗi city có POI class space riêng. Không ghép logits; tổng hợp metrics cuối của
từng city để trả lời generalization trong RQ4.

## 10. Chạy tự động toàn pipeline

ISP smoke test không OSM:

```bash
DATASET=isp \
CITY=Shanghai \
USE_OSM=0 \
OLLAMA_MODEL=qwen2:7b \
LLM_MISSING_POLICY=neutral \
VALIDATION_LIMIT=10 \
TEST_LIMIT=10 \
./scripts/hybrid_pipeline.sh all
```

Các subcommand hỗ trợ:

```bash
./scripts/hybrid_pipeline.sh help
./scripts/hybrid_pipeline.sh extract
./scripts/hybrid_pipeline.sh osm
./scripts/hybrid_pipeline.sh prepare
./scripts/hybrid_pipeline.sh ollama-test
./scripts/hybrid_pipeline.sh run
./scripts/hybrid_pipeline.sh all
```

## 11. Bayesian Network Stage 3

Triển khai tại `hybrid/bayesian_network.py`:

```text
L: latent next-location candidate
H: observed personal-habit evidence
S: observed urban-semantic evidence
L → H
L → S
```

CGM probability `q_i` là prior:

```text
P(L=c_i) = q_i
P(L|H,S) ∝ P(L;Q) × P(H|L) × P(S|L)
```

Inference chạy trong log-space. Mỗi prediction lưu:

```text
log_prior
log_habit
log_semantic
log_joint_unnormalized
log_normalizer
log_posterior
```

## 12. Ablation mapping cho RQ2

```text
full
no_temperature
no_world
no_embedding_memory
no_bbn
no_link_calibration
stage1_only
stage1_uncalibrated
top1_only
```

## 13. Troubleshooting

### RQ2 chính xác: free-text reranking và no OSM (Shanghai-50)

Reranking trực tiếp bằng LLM, giữ nguyên Neural-CGM top-k, evidence cache và embedding memory,
nhưng bỏ Bayesian Network:

```bash
./scripts/run_rq2_free_text_shanghai_50pct.sh
```

Có thể smoke test trước bằng `TEST_LIMIT=10`. Runner có resume qua
`results/rq2/.../free-text-rerank/predictions.jsonl`.

Ablation bỏ OSM nhưng vẫn dùng tri thức nội tại của LLM:

```bash
./scripts/run_rq2_no_osm_shanghai_50pct.sh
```

Runner no-OSM yêu cầu metadata của baseline full có ít nhất 90% địa chỉ OSM. Nếu coverage thấp,
runner chủ động dừng: cần enrich OSM và chạy lại baseline full trước, nếu không hai cấu hình
`full` và `no OSM` thực chất dùng cùng dữ liệu và kết quả RQ2 không hợp lệ. Có thể chỉnh ngưỡng
bằng `MIN_OSM_COVERAGE`, nhưng không nên hạ ngưỡng khi lấy số liệu cho paper.

### Ollama không kết nối

```bash
curl http://127.0.0.1:11434/api/version
./scripts/start_ollama.sh
```

### Model không tồn tại

```bash
ollama list
ollama pull qwen2:7b
```

Tên `OLLAMA_MODEL` phải khớp chính xác output của `ollama list`.

### LLM evidence incomplete

Pipeline hiện không dừng ở chế độ `neutral`. Chạy:

```bash
LLM_BATCH_SIZE=1 \
LLM_RETRIES=3 \
LLM_MISSING_POLICY=neutral \
DATASET=isp CITY=Shanghai OLLAMA_MODEL=qwen2:7b \
./scripts/hybrid_pipeline.sh run
```

### Chỉ muốn kiểm tra code, không gọi LLM

```bash
python -m hybrid.cli \
  --validation data/hybrid/Shanghai/validation.jsonl \
  --test data/hybrid/Shanghai/test.jsonl \
  --output-dir results/hybrid/Shanghai-heuristic \
  --extractor heuristic \
  --validation-limit 10 \
  --test-limit 10
```

Heuristic chỉ dùng cho smoke test, không dùng làm kết quả paper.

## 14. Unit tests

```bash
python -m compileall -q hybrid tests
python -m unittest discover -s tests -v
```

Tất cả tests phải kết thúc bằng `OK`.

## 15. TIST2015 LLM-only cho Table II

Baseline này là LLM-ZS history-only theo prompt AgentMove, dùng đúng test split và giới hạn
200 query giống runner Hybrid của từng thành phố. Ollama luôn dùng port `11434`.

Smoke test một thành phố rồi chạy tuần tự 12 thành phố:

```bash
./scripts/run_tist2015_llm_only_200.sh Tokyo
./scripts/run_tist2015_llm_only_200.sh all
```

Kiểm tra hoặc tổng hợp lại mà không gọi LLM:

```bash
./scripts/run_tist2015_llm_only_200.sh audit
./scripts/run_tist2015_llm_only_200.sh aggregate
```

Chạy lại cùng lệnh sẽ tiếp tục từ `predictions.jsonl`. Macro average chỉ được xem là kết quả
12 thành phố khi audit xác nhận đủ cả 12; kết quả thiếu thành phố được đánh dấu tạm thời.
Do prompt gốc chỉ trả năm location IDs, Table II để trống Acc@10 cho baseline này.

### Hoàn thành các hàng TIST2015 của Table II

Master runner chỉ chạy các city còn thiếu và giữ nguyên cache đã có:

```bash
./scripts/complete_tist2015_table2.sh audit
./scripts/complete_tist2015_table2.sh run-hybrid-pending
./scripts/complete_tist2015_table2.sh run-llmzs-pending
./scripts/complete_tist2015_table2.sh aggregate
```

Hoặc chạy tuần tự toàn bộ phần còn thiếu rồi tổng hợp:

```bash
./scripts/complete_tist2015_table2.sh all
```

Kết quả tổng hợp nằm dưới `results/tist2015-table2/<model>/limit-200/`. Kết quả Hybrid
TIST2015 hiện là no-OSM ablation; ngay cả khi đủ 12 city cũng không được đổi nhãn thành
`Ours (full)` cho đến khi đạt OSM coverage theo publication gate.

Sau khi LLM-ZS đủ 12 city, chạy baseline LLM-Mob bằng cùng split và cache engine:

```bash
./scripts/run_tist2015_llm_mob_200.sh audit
./scripts/run_tist2015_llm_mob_200.sh Tokyo
./scripts/run_tist2015_llm_mob_200.sh pending
./scripts/run_tist2015_llm_mob_200.sh aggregate
```

LLM-Mob được lưu riêng dưới `results/tist2015-llm-only/<model>/limit-200/llm-mob/`.
Duration không tồn tại trong JSONL TIST2015 nên prompt ghi rõ trường này là unavailable;
không được mô tả kết quả này là reproduction có duration.

## 16. Markov/Bi-gram TIST2015 cho Table II

Baseline đọc Markov logits đã tạo trong bước `prepare`, xếp hạng trên toàn bộ
candidate space và đánh giá tối đa 200 test query đầu tiên của cùng temporal
split. Chạy hoàn toàn trên CPU, không cần Ollama:

```bash
cd /Users/chiennguyen/Documents/Codex/Hybrid-GPS-Traj/src/AgentMove
./scripts/run_tist2015_markov_200.sh Tokyo
./scripts/run_tist2015_markov_200.sh pending
./scripts/run_tist2015_markov_200.sh aggregate
```

Kiểm tra trạng thái bằng `./scripts/run_tist2015_markov_200.sh audit`. Kết quả
tổng hợp nằm tại `results/tist2015-markov/limit-200/tist2015_markov_summary.json`;
fragment Table II nằm trong `tist2015_markov_table2_cells.tex` cùng thư mục.
