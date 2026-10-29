# BeliefMove faithful experiment report

> Không điền số bằng tay. File này phải được sinh lại từ raw JSON.

## Protocol identity

| Field | Value |
|---|---|
| Git commit | TODO |
| Protocol hash | TODO |
| Dataset/split hashes | TODO |
| Teacher model/digest | TODO |
| Prompt/ontology hash | TODO |
| Response schema version / response-format hash | TODO |
| Normalization version | TODO |
| Artifact namespace | `results/beliefmove-faithful/v2/` |
| Seeds | 42, 43, 44 |
| Estimand | Next location conditional on observed prefix and known prediction timestamp |

## Teacher-cache audit

| Dataset | Unit | Split | Requested | Cached | Valid | Fallback | Indeterminate recovery | Confirmed transport error | Physical requests consumed |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|

## M1–M5 quality

| Dataset | Variant | Queries | R@1 | R@5 | R@10 | MRR | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Efficiency

| Dataset | Variant | LLM call rate | Requests/query | Tokens/query | p50 latency | p95 latency | Peak RSS |
|---|---|---:|---:|---:|---:|---:|---:|

## Confirmatory tests

| Dataset | Comparison | Metric | Effect | Cluster 95% CI | Holm p | Decision |
|---|---|---|---:|---:|---:|---|

## Decision gates

| Claim | Pre-registered rule | Observed decision |
|---|---|---|
| LLM structured-belief distillation | M2 lowers NLL and loses no more than 0.005 MRR vs M1 on at least 2 datasets | TODO |
| Sequential memory | M3 > M2 on at least 2 datasets | TODO |
| Adaptive routing | M4 > M3 and matched random at call rate <= 0.10 on at least 2 datasets | TODO |
| Accuracy–cost trade-off | M5 within 0.01 MRR of Always; call rate <= 0.10; tokens/query <= 15%; p95 latency <= 25% of Always | TODO |

## Limitations

- YJMob100K coordinates are anonymous cells, not GPS.
- Teacher beliefs are soft supervision, not semantic ground truth.
- The task conditions on the timestamp of the next event; every baseline must
  receive this same covariate, while next POI/category/region remain hidden.
- Any bounded teacher cache must be labelled bounded and cannot be generalized to
  full-query performance.
- A recovered dangling attempt is an indeterminate request outcome, not a valid
  LLM response or a confirmed transport error. It consumes one physical-request
  budget unit, increments fallback accounting and contributes zero teacher
  soft-count weight.
