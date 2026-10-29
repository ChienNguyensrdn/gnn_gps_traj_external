# BeliefMove — protocol bám sát tài liệu gốc

Thư mục này triển khai và khóa protocol cho kiến trúc trong
`ideas/BeliefMove_y_tuong_va_huong_dan_cai_dat.docx`:

```text
GPS/cell trajectory
  -> train-only spatial representation
  -> offline structured LLM teacher beliefs
  -> Bayesian student learned from data + soft teacher counts
  -> sequential belief state
  -> uncertainty router
  -> optional structured-LLM belief update
  -> region-to-POI/GPS decoder
```

Đây là namespace mới. Kết quả `phase2/rq3_report.md` là **legacy
inference-time fusion control**, không phải bằng chứng cho LLM-to-Bayesian
knowledge distillation và không được gộp với kết quả ở đây.

Ma trận đối chiếu từng yêu cầu của tài liệu với code/gate nằm tại
[`TRACEABILITY.md`](TRACEABILITY.md).

## Trạng thái triển khai

- Đã có: audit dữ liệu, causal all-prefix preparation, region vocabulary
  train-only, target/query separation, structured-belief schema, immutable
  one-request teacher cache, CPT data/teacher primitives, sequential belief,
  uncertainty và decoder, smoke/pilot cache collector, category-to-ontology
  labels, evaluator `pilot_evaluation.json`, pilot-report validator và
  protocol-lock validator, cùng exporter/binder support top-k theo frozen
  quantitative checkpoint cho train/validation.
- Teacher-response contract v2 dùng strict JSON Schema với đúng support đã
  khóa. LLM trả về trọng số tương đối hữu hạn, không âm và có tổng dương;
  collector mới là biên tin cậy thực hiện chuẩn hóa L1 xác định. Mỗi record
  lưu version/hash của response schema, normalization contract và response
  format để không thể resume nhầm cache v1.
- Collector có recovery fail-closed cho đúng một trường hợp gián đoạn: request
  đã được ghi bền vững vào journal nhưng chưa có belief tương ứng. Recovery
  không gọi LLM, không sửa attempt cũ và chỉ append một fallback xác định với
  outcome `indeterminate`; request vẫn bị tính là đã tiêu thụ ngân sách.
- Chưa có: bộ fit/evaluate hoàn chỉnh cho M1–M5, quantitative-teacher
  distillation vào student, test-panel bank, controls và aggregator confirmatory. Vì vậy
  `teacher-offline` và các action confirmatory vẫn bị khóa fail-closed. Không
  action nào âm thầm gọi lại pipeline Phase 2.
- Chưa có lượt LLM/GPU tốn chi phí nào được chạy trong lần chuẩn bị này.

Vì vậy bước chạy hợp lệ hiện tại dừng ở `prepare`, `smoke`, `pilot`,
`recover-dangling`, `pilot-report`, `lock` và `status`. `lock` chỉ thành công
nếu mọi pilot gate đạt; nếu pilot âm tính thì protocol dừng đúng theo stop
rule.

Pilot chỉ là **region-CPT screening** trên quan hệ `previous_region ->
next_region`, dùng để quyết định teacher có đủ chất lượng để đầu tư bước sau
hay không. Nó không phải kết quả M2 và không được dùng để tuyên bố rằng toàn bộ
BN activity--intent--destination--region đã cải thiện; tuyên bố đó chỉ hợp lệ
sau evaluator confirmatory M1--M5.

## 1. Các biến thể bắt buộc

| Biến thể | Data CPT | LLM teacher CPT | Belief memory | LLM on-demand | Quantitative likelihood + calibration |
|---|---:|---:|---:|---:|---:|
| M1-data-bn | yes | no | no | no | no |
| M2-llm-distilled | yes | yes | no | no | no |
| M3-belief-memory | yes | yes | yes | no | no |
| M4-adaptive-reasoning | yes | yes | yes | yes | no |
| M5-beliefmove-full | yes | yes | yes | yes | offline quantitative-teacher distillation + calibration |

M2 phải học từ **expected soft counts/CPT của teacher**, không được thay bằng
việc cộng score LLM lúc test. M4/M5 phải nhận một structured belief mới rồi
cập nhật posterior; LLM không chỉ sắp xếp lại top-k POI. Quantitative teacher
của M5 chỉ được dùng offline; deployment vẫn là Bayesian student nhẹ.

Các phép toán chính được khóa như sau:

$$
\theta^{*}
=
\alpha\theta_{\mathrm{data}}
+
(1-\alpha)\theta_{\mathrm{LLM}},
\qquad
\alpha\in\{0,0.25,0.5,0.75,1\},
$$

$$
b_t(z)
=
\eta\,P(e_t\mid z)
\sum_{z'}P(z\mid z')b_{t-1}(z'),
$$

và khi router gọi LLM:

$$
b_t^{+}(z)
=
\eta\,b_t(z)q_{\mathrm{LLM}}(z\mid X_{1:t})^{\gamma}.
$$

M5 bổ sung distillation từ quantitative teacher bằng KL trên phân phối POI,
nhưng không chạy teacher định lượng trong deployment.

Ontology khóa gồm 6 activity (`residential`, `work-education`,
`food-retail-service`, `leisure-social`, `transport-outdoor`, `other`) và 5
intent (`same-poi`, `within-zone`, `return-known-poi`,
`new-poi-known-zone`, `new-zone`). Destination dùng 6 loại tương ứng
(`home-residential`, `work-education`, `food-retail-service`,
`leisure-social`, `transport-outdoor`, `other`); next-region dùng vocabulary
không gian train-only.

Biến trạng thái là $Z_t=(A_t,I_t,D_t,R_t)$ và BN/DBN được khóa theo phân rã:

$$
P(A_t\mid s_t,s_{t-1},A_{t-1})
P(I_t\mid A_t,R_{t-1},f_t)
P(D_t\mid A_t,I_t)
P(R_t\mid D_t,R_{t-1}),
$$

trong đó đầu vào của activity còn gồm time-slot dự đoán $s_t$ và time-slot
quan sát cuối $s_{t-1}$; $f_t$ là các cờ quay lại POI/vùng đã thấy trong prefix.
Để giữ đúng preprocessing/evaluation của AgentMove, timestamp dự đoán là một
covariate ngoại sinh đã biết. POI, category và vùng của sự kiện kế tiếp luôn bị
ẩn. Không artifact nào được phép dùng thêm thông tin từ hàng đích.
Activity/destination hard label phải đến từ một mapping category-to-ontology đã
khóa; intent hard label được suy ra từ prefix và target chỉ trong bước fit. Soft
count của teacher được nhân với confidence; fallback có confidence 0 nên không
đóng góp teacher count.

## 2. Representation và ontology

- TIST2015 và WWW2019: tạo lưới không gian 500 m chỉ từ tọa độ train.
- YJMob100K: giữ nguyên cell 500 m ẩn danh; không gọi chúng là GPS.
- Validation/test chỉ được ánh xạ vào vocabulary đã khóa; vùng ngoài vocabulary
  đi vào `__UNK_REGION__`.
- Smoke/pilot dùng tối đa 20 vùng train phổ biến cộng
  `__UNK_REGION__`/`__OTHER__`. Confirmatory bắt buộc thay support này bằng các
  vùng top-k từ quantitative teacher đã đóng băng; nếu chưa có support đó thì
  không được tạo kết quả confirmatory.
- TIST2015/WWW2019 dùng local-equirectangular projection với reference latitude
  chỉ fit từ train; cell có cạnh đúng 500 m trong mặt phẳng metric cục bộ và
  toàn bộ projection/vocabulary được hash trong manifest.
- Ground-truth next region được lưu tách riêng để đánh giá và tuyệt đối không đi
  vào prompt.

Ontology activity/intent phải cố định trước khi gọi LLM. Teacher output gồm
`activity`, `intent`, `destination`, `next_region`, `confidence`; mọi phân phối
thô phải có đúng support đã khóa, hữu hạn, không âm và có tổng dương. Collector
chuẩn hóa L1 từng phân phối; belief canonical sau chuẩn hóa mới phải có tổng
bằng 1 (sai số tối đa $10^{-6}$). Không sửa hoặc tái sử dụng cache v1 để tạo
cache v2.

## 3. Split và chống leakage

| Thành phần | Split được phép |
|---|---|
| region vocabulary, data CPT, transition | train |
| offline teacher beliefs | train, validation |
| chọn alpha, temperature, router threshold | validation |
| báo cáo cuối | test |
| online teacher intervention | chỉ query test mà router đã chọn |

Cache teacher là bất biến và phải chứa hash của dataset, source query, ontology,
prompt và model. Model/prompt/ontology có namespace riêng; khi đổi limit, seed
hoặc tập query phải dùng `cache-tag`/output namespace mới. Manifest xung đột sẽ
dừng thay vì ghi đè cache cũ.

## 4. Trình tự chạy

Các lệnh dưới đây không tự gọi LLM trong `audit` hoặc `prepare`.

```bash
cd src/AgentMove

# 1) Kiểm tra dữ liệu và checkpoint định lượng đã đóng băng
./phase3_beliefmove/scripts/run.sh audit

# 2) Tạo region vocabulary, causal-prefix queries và manifest; không gọi LLM
./phase3_beliefmove/scripts/run.sh prepare

# 3) Smoke: 20 train + 20 validation cho Tokyo/Shanghai/YJMob; không đọc test.
#    Bắt buộc dùng full model digest của đúng Ollama server. Sau ba unit,
#    action này tự sinh smoke_report.json và fail-closed nếu một unit không đạt.
OLLAMA_MODEL=qwen2.5:7b \
OLLAMA_MODEL_DIGEST=<full-sha256> \
  ./phase3_beliefmove/scripts/run.sh smoke

# Có thể tái kiểm tra report mà không gọi LLM.
OLLAMA_MODEL=qwen2.5:7b \
OLLAMA_MODEL_DIGEST=<full-sha256> \
  ./phase3_beliefmove/scripts/run.sh smoke-report

# 4) Pilot chỉ dùng train/validation; mỗi unit gồm base 500/200 và 3 repeat
#    validation 50-query. Alpha chọn trên một nửa validation cố định và chỉ
#    được chấm trên nửa held-out rời nhau.
OLLAMA_MODEL=qwen2.5:7b \
OLLAMA_MODEL_DIGEST=<full-sha256> \
  ./phase3_beliefmove/scripts/run.sh pilot
./phase3_beliefmove/scripts/run.sh pilot-report

# 5) Chỉ khóa protocol nếu report tái tính được pilot từ raw cache và mọi gate
#    dương; JSON summary viết tay hoặc stale không thể mở khóa.
./phase3_beliefmove/scripts/run.sh lock

# 6) Sau lock, export frozen quantitative support cho train/validation rồi
#    bind top-k vùng riêng của từng query. Hai lệnh này không gọi LLM/test.
./phase3_beliefmove/scripts/run.sh quantitative-support-audit
DEVICE=cuda ./phase3_beliefmove/scripts/run.sh quantitative-support

# 7) Confirmatory offline teacher: 1000 train + 250 validation mỗi unit.
OLLAMA_MODEL=qwen2.5:7b ./phase3_beliefmove/scripts/run.sh teacher-offline

# 8) Frozen quantitative teacher, train M1-M5 và đánh giá M1-M3 full-test.
./phase3_beliefmove/scripts/run.sh quantitative-teacher
./phase3_beliefmove/scripts/run.sh train

# 9) Chỉ sau lock + offline go/no-go: test bank bounded 500 query/unit.
OLLAMA_MODEL=qwen2.5:7b ./phase3_beliefmove/scripts/run.sh test-bank
./phase3_beliefmove/scripts/run.sh evaluate
./phase3_beliefmove/scripts/run.sh controls

# 10) Tổng hợp bảng/CI/gate từ raw JSON; không nhập số tay.
SIGNIFICANCE_ITERATIONS=10000 \
  ./phase3_beliefmove/scripts/run.sh aggregate
./phase3_beliefmove/scripts/run.sh report
```

`audit` với `gate=ready` chỉ xác nhận rằng dữ liệu nguồn và checkpoint định
lượng tồn tại; nó chưa có nghĩa là unit đã được prepare. Một unit chỉ hoàn
thành bước này khi có dòng cuối `"action":"prepare","gate":"ready"` và
`status` trả về `"prepared":true`. Trong lúc đọc dữ liệu lớn, `prepare` in
tiến độ sau mỗi 100.000 dòng và sau khi dựng xong prefix của từng split.

Nếu một lần chạy bị ngắt, chỉ tiếp tục unit chưa hoàn thành thay vì đọc lại các
unit trước đó. Ví dụ:

```bash
BELIEFMOVE_DATASETS=tist2015 TIST_CITIES=Moscow \
  ./phase3_beliefmove/scripts/run.sh status
BELIEFMOVE_DATASETS=tist2015 TIST_CITIES=Moscow \
  ./phase3_beliefmove/scripts/run.sh prepare
```

Artifact hoàn chỉnh được ghi nguyên tử và cache đã hoàn tất là bất biến, nên
chạy lại cùng unit sẽ tái sử dụng kết quả hợp lệ. Teacher journal là append-only;
trường hợp process dừng giữa durable attempt và durable belief phải dùng quy
trình recovery riêng ngay dưới đây.

### Phục hồi attempt bị gián đoạn

Nếu collector dừng sau khi đã ghi một physical attempt nhưng trước khi ghi
belief, chạy lại `pilot` sẽ chủ động dừng với thông báo
`attempts without beliefs`. Không xóa/sửa `attempts.jsonl`, không sao chép
belief và không gọi lại query: output của request cũ không còn đủ provenance để
khôi phục, còn retry có thể tạo physical request thứ hai cho cùng query.

Chỉ trong trường hợp này, dùng recovery tường minh trên đúng dataset/unit bị
ảnh hưởng:

```bash
BELIEFMOVE_DATASETS=www2019 \
OLLAMA_MODEL=qwen2.5:7b \
OLLAMA_MODEL_DIGEST=<full-sha256> \
  ./scripts/beliefmove_faithful.sh recover-dangling

# Sau recovery, resume collector/evaluator bình thường.
BELIEFMOVE_DATASETS="www2019 yjmob100k" \
OLLAMA_MODEL=qwen2.5:7b \
OLLAMA_MODEL_DIGEST=<full-sha256> \
  ./scripts/beliefmove_faithful.sh pilot
./scripts/beliefmove_faithful.sh pilot-report
```

Một recovery thành công phải in JSON với `gate=ready`, `llm_calls=0`,
`request_budget_already_consumed` bằng số dangling mới được materialize và
`transport_errors_added=0`. `new_recoveries` cho biết số fallback mới append;
`recovered_total` giữ tổng số recovery đã audit được. Chạy lại đúng lệnh sau
recovery là idempotent: không gọi LLM và không append record thứ hai.

`recover-dangling` chỉ áp dụng cho base cache `pilot`, không gọi Ollama và
không sửa journal attempt. Với mỗi ID dangling đã được kiểm chứng, nó append
một belief fallback xác định, ghi `request_outcome=indeterminate` và metadata
recovery. Đây **không phải** transport error đã xác nhận: transport-error rate
không tăng. Ngược lại, physical-request budget được coi là đã tiêu thụ,
fallback count/rate tăng, confidence của teacher bằng 0 và record không đóng
góp soft count. Recovery không biến cache thành kết quả dương; các gate chất
lượng teacher và M2 vẫn được đánh giá nguyên trạng.

Do collector giữ lock và xử lý tuần tự từng query, một lần dừng chỉ có thể để
lại tối đa **một** attempt chưa có belief. Nếu audit thấy từ hai dangling
attempt trở lên, `recover-dangling` dừng fail-closed vì đó là dấu hiệu journal
bị hỏng hoặc bị chỉnh ngoài protocol.

Shell wrapper yêu cầu chọn đúng **một** pilot unit cho mỗi lần recovery và sẽ
thoát trước khi sửa artifact nếu cấu hình chọn 0 hoặc nhiều hơn 1 unit. Với
TIST2015 cần đặt cả `BELIEFMOVE_DATASETS=tist2015` và `TIST_CITIES=Tokyo`.

Lệnh này không được dùng cho cache thiếu hoàn toàn, response sai schema, model
digest sai hoặc manifest xung đột. Nếu không có đúng quan hệ “durable attempt,
missing belief”, phải dừng và audit thay vì sửa journal thủ công.

Có thể giới hạn khi debug:

```bash
BELIEFMOVE_DATASETS=www2019 ./phase3_beliefmove/scripts/run.sh prepare
BELIEFMOVE_DATASETS=tist2015 TIST_CITIES="Tokyo NewYork" \
  ./phase3_beliefmove/scripts/run.sh audit
```

## 5. Artifacts

```text
results/beliefmove-faithful/v2/<dataset>/<unit>/
  manifests/
  representation/
  teacher/<model-slug>/<model-digest>/<prompt-contract-hash>/<cache-tag>/<selection-hash>/
  models/<variant>/seed-<seed>/
  online/<variant>/seed-<seed>/
  raw/
results/beliefmove-faithful/v2/aggregated/
```

Mỗi run manifest phải ghi seed, git commit, dataset hash, split hashes, config
hash, teacher model digest, prompt/ontology hash, physical request count và
hardware. Teacher cache v2 còn phải ghi `prompt_version`,
`response_schema_version`, `normalization_version`, `prompt_contract_hash` và
`response_format_hash`; raw response, canonical belief và metadata chuẩn hóa
được giữ riêng để audit. Belief được recovery phải giữ riêng provenance của
attempt ban đầu, outcome `indeterminate`, recovery reason/version và cờ budget
đã tiêu thụ; report không được gộp nó với output LLM hợp lệ hay transport error
đã xác nhận.

Mọi cache gọi LLM (`smoke`, `pilot`, repeat và `official`) phải truyền full
model digest 64-hex lấy từ chính Ollama server đang phục vụ model:

```bash
OLLAMA_MODEL=qwen2.5:7b \
OLLAMA_MODEL_DIGEST=<sha256-từ-http://127.0.0.1:11434/api/tags> \
  ./phase3_beliefmove/scripts/run.sh teacher-offline
```

## 6. Tiêu chí kết luận

1. **LLM distillation:** M2 cải thiện NLL so với M1 và không giảm quá 0.005 MRR
   ở ít nhất hai dataset family.
2. **Belief memory:** M3 vượt M2 trên all-prefix MRR ở ít nhất hai dataset.
3. **Adaptive reasoning:** M4 vượt M3 và random-budget-matched trên MRR tại
   call rate không quá 10% ở ít nhất hai dataset family.
4. **Full system:** M5 vượt M1 trên cả ba family, cách Always-LLM không quá
   0.01 MRR, dùng không quá 15% token/query và 25% p95 latency của Always-LLM,
   không quá 10% số call, và không làm ECE xấu quá 0.01.

Nếu một gate không đạt, báo kết quả âm và dừng mở rộng chi phí; không đổi câu
hỏi sau khi xem test.
