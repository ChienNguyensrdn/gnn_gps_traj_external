# Đề xuất mở rộng: Representation-Evolution Distillation cho Framework Neuro-Symbolic-Probabilistic dự đoán vị trí tiếp theo

## 1. Mục tiêu của tài liệu

Tài liệu này đề xuất một hướng mở rộng cho bài báo:

**“A Hybrid Neuro-Symbolic-Probabilistic Framework: Combining a Trajectory Foundation Model, a Bayesian Belief Network, and an LLM Agent for Next-Location Prediction on GPS Trajectory Data.”**

Mục tiêu không phải là sao chép trực tiếp cơ chế của UniSTD, mà là **chuyển hóa tư tưởng “representation evolution distillation” sang bài toán mobility trajectory**, nơi tồn tại đồng thời hai dạng tiến hóa:

1. **Tiến hóa theo chiều sâu mạng** (*representation evolution across layers*);
2. **Tiến hóa theo thời gian của quỹ đạo di chuyển** (*mobility evolution across trajectory steps*).

Ý tưởng trung tâm là thay đổi câu hỏi nghiên cứu từ:

> “Mô hình teacher dự đoán vị trí tiếp theo nào?”

sang:

> “Teacher hình thành tri thức dự đoán như thế nào qua các tầng biểu diễn và qua chuỗi hành vi di chuyển, và phần tri thức nào thực sự cần được chuyển giao sang mô hình triển khai nhẹ?”

---

# 2. Framework hiện tại và khoảng trống cần giải quyết

## 2.1. Framework hiện tại

Framework hiện tại gồm ba giai đoạn:

### Stage 1 — Candidate Generation Model (CGM)

Một mô hình định lượng tạo top-k ứng viên:

$$
f_{\theta}(H_u)
\rightarrow
\{(c_1,q_1),\ldots,(c_k,q_k)\},
$$

trong đó:

- $H_u$: lịch sử di chuyển của người dùng;
- $c_i$: ứng viên vị trí;
- $q_i$: xác suất đã calibration của ứng viên.

### Stage 2 — LLM Evidence Extractor

LLM không trực tiếp xếp hạng cuối cùng mà tạo hai nguồn bằng chứng:

$$
g_{\mathrm{LLM}}(c_i,M_u,W)
\rightarrow
(h_i,s_i),
$$

trong đó:

- $h_i$: mức phù hợp với thói quen cá nhân;
- $s_i$: mức phù hợp với ngữ cảnh đô thị/thế giới.

### Stage 3 — Bayesian Belief Network

BBN hợp nhất prior và semantic evidence:

$$
P(L=c_i\mid Q,H,S)
\propto
q_i
P(H\mid L=c_i)
P(S\mid L=c_i).
$$

Prediction cuối:

$$
\hat{p}_{n+1}
=
\arg\max_{c_i}
P(L=c_i\mid Q,H,S).
$$

---

## 2.2. Vấn đề quan sát được từ kết quả hiện tại

Kết quả hiện tại cho thấy framework có ưu điểm về calibration, diễn giải và giảm chi phí so với AgentMove nhiều lần gọi LLM, nhưng vẫn tồn tại một số điểm nghiên cứu quan trọng:

1. **Full model chưa vượt Stage 1 một cách nhất quán về Acc@1**.
2. OSM-based world knowledge chưa cho thấy gain có ý nghĩa thống kê rõ ràng.
3. Embedding-based personal memory chưa cho thấy gain có ý nghĩa thống kê rõ ràng.
4. LLM evidence có thể chưa được chuyển hóa thành một representation space đủ ổn định.
5. Cần làm rõ:
   - tri thức nào thật sự đến từ teacher;
   - tri thức nào đến từ input engineering;
   - thứ tự trajectory có vai trò gì;
   - representation evolution có mang thông tin thêm ngoài final logits hay không.

---

# 3. Ý tưởng mới đề xuất

## 3.1. Từ “Trajectory Foundation Model làm predictor” sang “Trajectory Foundation Model làm teacher”

Thay vì sử dụng một foundation model lớn trực tiếp trong deployment, ta sử dụng nó như **teacher offline**.

Teacher:

$$
f_T(H_u)
\rightarrow
\left\{
H_T^{(1)},
H_T^{(2)},
\ldots,
H_T^{(L)},
Z_T
\right\}.
$$

Student nhẹ:

$$
f_S(H_u)
\rightarrow
\left\{
H_S^{(1)},
H_S^{(2)},
\ldots,
H_S^{(M)},
Z_S
\right\}.
$$

Mục tiêu không chỉ là:

$$
Z_T \rightarrow Z_S,
$$

mà là:

$$
\boxed{
\text{final prediction}
+
\text{intermediate states}
+
\text{representation transitions}
}
$$

được chuyển từ teacher sang student.

---

# 4. Cơ sở lý thuyết

## 4.1. Response-level Knowledge Distillation

Teacher và student tạo distribution:

$$
P_T^{(\tau)}
=
\operatorname{Softmax}(Z_T/\tau),
$$

$$
P_S^{(\tau)}
=
\operatorname{Softmax}(Z_S/\tau).
$$

Loss chưng cất:

$$
L_{\mathrm{KD}}
=
\tau^2
D_{\mathrm{KL}}
\left(
P_T^{(\tau)}
\Vert
P_S^{(\tau)}
\right).
$$

Mục tiêu supervised:

$$
L_{\mathrm{CE}}
=
-
\sum_{i}
\sum_{c}
Y_{ic}\log P_{S,ic}.
$$

Response-level objective:

$$
L_{\mathrm{resp}}
=
L_{\mathrm{CE}}
+
\lambda_{\mathrm{KD}}L_{\mathrm{KD}}.
$$

### Hạn chế

Response distillation chỉ trả lời:

> Teacher cuối cùng tin ứng viên nào?

Nó không trả lời:

> Teacher biến đổi representation của lịch sử di chuyển như thế nào trước khi đưa ra prediction?

---

# 5. Representation Trajectory Distillation

## 5.1. Không gian chung giữa teacher và student

Không nên trực tiếp trừ hai hidden representation có dimension khác nhau.

Đề xuất dùng projection head:

$$
R_T^{(m)}
=
P_T^{(m)}\left(H_T^{(m)}\right),
$$

$$
R_S^{(m)}
=
P_S^{(m)}\left(H_S^{(m)}\right),
$$

với:

$$
R_T^{(m)},
R_S^{(m)}
\in
\mathbb{R}^{N\times d_*}.
$$

Trong đó $$d_*$$ là latent dimension chung.

---

## 5.2. Trajectory Distillation

Trajectory loss căn chỉnh trạng thái representation:

$$
L_{\mathrm{traj}}
=
\frac{1}{M}
\sum_{m=1}^{M}
\left\|
R_S^{(m)}
-
R_T^{(m)}
\right\|_F^2.
$$

Trực giác:

$$
\boxed{
\text{Student cần hình thành representation phù hợp tại từng stage}
}
$$

---

# 6. Velocity Distillation

Chỉ khớp trạng thái tĩnh chưa đủ.

Ta định nghĩa representation transition:

$$
\Delta R_T^{(m)}
=
R_T^{(m+1)}
-
R_T^{(m)},
$$

$$
\Delta R_S^{(m)}
=
R_S^{(m+1)}
-
R_S^{(m)}.
$$

Velocity loss:

$$
L_{\mathrm{vel}}
=
\frac{1}{M-1}
\sum_{m=1}^{M-1}
\left\|
\Delta R_S^{(m)}
-
\Delta R_T^{(m)}
\right\|_F^2.
$$

Trajectory distillation trả lời:

> Representation đang ở đâu?

Velocity distillation trả lời:

> Representation đang thay đổi theo hướng nào?

---

# 7. Điểm mở rộng quan trọng cho mobility: Temporal Evolution Distillation

UniSTD chủ yếu xét evolution theo network depth.

Trong mobility prediction, ta có thêm chiều thời gian:

$$
(p_1,t_1)
\rightarrow
(p_2,t_2)
\rightarrow
\cdots
\rightarrow
(p_n,t_n).
$$

Do đó representation có thể được ký hiệu:

$$
H_{t}^{(\ell)},
$$

trong đó:

- $t$: trajectory step;
- $\ell$: network layer.

Ta có hai dạng evolution.

## 7.1. Layer-wise evolution

$$
\Delta_{\ell}H_t^{(\ell)}
=
H_t^{(\ell+1)}
-
H_t^{(\ell)}.
$$

## 7.2. Temporal evolution

$$
\Delta_tH_t^{(\ell)}
=
H_{t+1}^{(\ell)}
-
H_t^{(\ell)}.
$$

Có thể định nghĩa temporal distillation:

$$
L_{\mathrm{temp}}
=
\frac{1}{T-1}
\sum_{t=1}^{T-1}
\left\|
\Delta_tR_{S,t}
-
\Delta_tR_{T,t}
\right\|_F^2.
$$

### Ý nghĩa

Student không chỉ học:

$$
\text{“next location là gì?”}
$$

mà còn học:

$$
\boxed{
\text{“hành vi mobility đang tiến hóa theo pattern nào?”}
}
$$

Đây là điểm có thể tạo novelty riêng so với UniSTD.

---

# 8. Multi-view Mobility Modeling

Đề xuất xây dựng nhiều view thay vì chỉ một representation.

## 8.1. Raw mobility view

$$
X_r
=
[
\text{POI},
\text{GPS},
\text{time},
\text{weekday}
].
$$

## 8.2. Personal behavior view

$$
X_p
=
[
\text{retrieved personal history},
\text{habit patterns}
].
$$

## 8.3. World/context view

$$
X_w
=
[
\text{POI type},
\text{urban function},
\text{OSM/context}
].
$$

## 8.4. Fused view

$$
X_f
=
[X_r,X_p,X_w].
$$

Một shared evidence model có thể tạo:

$$
Z_r=g_{\phi}(X_r),
$$

$$
Z_p=g_{\phi}(X_p),
$$

$$
Z_w=g_{\phi}(X_w),
$$

$$
Z_f=g_{\phi}(X_f).
$$

Consistency loss:

$$
L_{\mathrm{cons}}
=
D_{\mathrm{KL}}(P_r\Vert P_f)
+
D_{\mathrm{KL}}(P_p\Vert P_f)
+
D_{\mathrm{KL}}(P_w\Vert P_f).
$$

### Chú ý lý thuyết

Không nên trực tiếp buộc likelihood cuối của BBN hoàn toàn giống nhau vì BBN hiện giả định:

$$
Q\perp H\perp S\mid L.
$$

Consistency nên dùng ở **representation/evidence-extractor level**.

Nếu muốn model dependency rõ hơn, có thể nghiên cứu:

- Tree-Augmented Naive Bayes;
- conditional Bayesian network;
- learned dependency structure.

---

# 9. Tổng hàm mục tiêu đề xuất

Một phiên bản đầy đủ:

$$
L
=
L_{\mathrm{CE}}
+
\lambda_{\mathrm{KD}}L_{\mathrm{KD}}
+
\lambda_{\mathrm{traj}}L_{\mathrm{traj}}
+
\lambda_{\mathrm{vel}}L_{\mathrm{vel}}
+
\lambda_{\mathrm{temp}}L_{\mathrm{temp}}
+
\lambda_{\mathrm{cons}}L_{\mathrm{cons}}.
$$

Không nhất thiết bật tất cả loss ngay từ đầu.

Lộ trình hợp lý:

### Version A

$$
L_{\mathrm{CE}}
+
L_{\mathrm{KD}}.
$$

### Version B

$$
L_{\mathrm{CE}}
+
L_{\mathrm{KD}}
+
L_{\mathrm{traj}}.
$$

### Version C

$$
L_{\mathrm{CE}}
+
L_{\mathrm{KD}}
+
L_{\mathrm{traj}}
+
L_{\mathrm{vel}}.
$$

### Version D

$$
L_{\mathrm{CE}}
+
L_{\mathrm{KD}}
+
L_{\mathrm{traj}}
+
L_{\mathrm{vel}}
+
L_{\mathrm{temp}}.
$$

### Full

$$
+
L_{\mathrm{cons}}.
$$

---

# 10. Kiến trúc tổng thể đề xuất

```text
                ┌────────────────────────────┐
                │ Trajectory Foundation Model│
                │          Teacher           │
                └──────────────┬─────────────┘
                               │
              logits + hidden states + evolution
                               │
                               ▼
                ┌────────────────────────────┐
                │ Lightweight Mobility Student│
                │      Candidate Generator    │
                └──────────────┬─────────────┘
                               │
                         calibrated q_i
                               │
                               ▼
                ┌────────────────────────────┐
                │   LLM Semantic Teacher     │
                │ personal + world evidence  │
                └──────────────┬─────────────┘
                               │
                            h_i, s_i
                               │
                               ▼
                ┌────────────────────────────┐
                │ Calibrated Bayesian Network │
                └──────────────┬─────────────┘
                               │
                               ▼
                 posterior + faithful explanation
```

Một extension mạnh hơn cho deployment:

```text
LLM Semantic Teacher
        │
        ▼
Lightweight Evidence Student
        │
        ▼
Bayesian Network
```

Khi đó LLM có thể chỉ cần gọi trong các trường hợp uncertainty cao.

---

# 11. Selective LLM Invocation

Gọi LLM khi posterior không chắc chắn.

Entropy:

$$
H(P)
=
-
\sum_i
P_i\log P_i.
$$

Nếu:

$$
H(P)>\gamma,
$$

thì gọi LLM.

Hoặc sử dụng top-2 margin:

$$
P_{(1)}-P_{(2)}<\delta.
$$

Như vậy:

```text
confident query
    → lightweight student + BBN

ambiguous query
    → lightweight student + LLM + BBN
```

Đây là hướng quan trọng để cải thiện accuracy–latency trade-off.

---

# 12. Bộ câu hỏi nghiên cứu đề xuất

# RQ1 — Hiệu quả tổng thể của framework

## Câu hỏi

**RQ1: Framework neuro-symbolic-probabilistic có cải thiện next-location prediction so với quantitative-only, LLM-only và các hybrid baseline hay không?**

## Giả thuyết

$$
H_1:
\text{Hybrid}
>
\text{LLM-only}
$$

về Acc@1/MRR và:

$$
\text{Hybrid}
\approx
\text{strong quantitative baseline}
$$

hoặc tốt hơn trong một số setting.

## Thực nghiệm

So sánh:

- Markov/Bi-gram;
- GETNext;
- Stage-1 CGM;
- LLM-Mob;
- LLM-ZS;
- AgentMove;
- NextLocLLM;
- TrajAgent;
- DBN;
- Proposed full model.

## Datasets

- ISP-Shanghai;
- Foursquare TIST2015;
- YJMob100K nếu hoàn thành.

## Metrics

- Acc@1;
- Acc@5;
- Acc@10;
- MRR;
- ECE.

---

# RQ2 — Đóng góp của Bayesian Fusion

## Câu hỏi

**RQ2: Bayesian fusion có tạo ra quyết định ổn định, calibrated và interpretable hơn free-text LLM re-ranking hay không?**

## Thực nghiệm

So sánh:

```text
CGM only
CGM + LLM free-text re-ranking
CGM + raw LLM scores
CGM + calibrated LLM scores
CGM + calibrated LLM scores + BBN
```

## Metrics

- Acc@1;
- MRR;
- ECE;
- malformed-output rate;
- posterior validity;
- explanation consistency.

## Kiểm định

Bootstrap paired confidence interval.

---

# RQ3 — Representation-Evolution Distillation có thực sự hữu ích?

## Câu hỏi

**RQ3: Chuyển giao representation evolution của trajectory teacher có mang thêm tri thức ngoài final soft targets hay không?**

## Thực nghiệm ablation

So sánh:

```text
Student + CE
Student + CE + KD
Student + CE + KD + Trajectory
Student + CE + KD + Velocity
Student + CE + KD + Trajectory + Velocity
Full model
```

## Hypothesis

Nếu:

$$
Acc_{\mathrm{KD+traj+vel}}
>
Acc_{\mathrm{KD}},
$$

thì intermediate states và transitions chứa transferable knowledge ngoài final logits.

## Metrics

- Acc@1;
- Acc@5;
- MRR;
- CKA;
- representation-distance;
- transition cosine similarity.

---

# RQ4 — Thứ tự mobility trajectory có phải là một nguồn tri thức?

## Câu hỏi

**RQ4: Gain có đến từ ordered mobility dynamics hay chỉ từ tập hợp các visited locations?**

## Thực nghiệm order corruption

### Correct

$$
p_1
\rightarrow
p_2
\rightarrow
\cdots
\rightarrow
p_n.
$$

### Reverse

$$
p_n
\rightarrow
p_{n-1}
\rightarrow
\cdots
\rightarrow
p_1.
$$

### Random

$$
\pi(p_1,\ldots,p_n).
$$

Phải giữ nguyên:

- model architecture;
- labels;
- optimization;
- user split;
- candidate set;
- hyperparameters.

Random order nên chạy nhiều permutation:

$$
K_{\mathrm{perm}}
\geq
10.
$$

## Metrics

- Acc@1;
- MRR;
- CKA;
- temporal-transition similarity;
- confidence calibration.

## Kết luận mong đợi

Nếu:

$$
Correct > Reverse \approx Random,
$$

thì ordered mobility dynamics chứa non-trivial predictive knowledge.

---

# RQ5 — Dual Evolution: layer-wise hay temporal evolution quan trọng hơn?

## Câu hỏi

**RQ5: Gain của representation-evolution distillation đến chủ yếu từ evolution qua network layers hay evolution qua trajectory time steps?**

## Thực nghiệm

So sánh:

```text
KD only
KD + layer trajectory
KD + layer velocity
KD + temporal evolution
KD + layer + temporal
Full
```

## Metrics

- Acc@1;
- MRR;
- per-user improvement;
- short-history vs long-history performance;
- urban-core vs suburban trajectory performance.

## Phân tích

Có thể phân nhóm theo trajectory length:

$$
T<5,
$$

$$
5\leq T<10,
$$

$$
T\geq10.
$$

Nếu temporal loss chỉ tốt với long trajectory thì đó là một insight khoa học quan trọng.

---

# RQ6 — Semantic evidence có thật sự được sử dụng?

## Câu hỏi

**RQ6: Personal memory và world knowledge có thực sự cung cấp causal/predictive evidence hay chỉ tạo ra scores tương quan yếu với prior?**

## Thực nghiệm semantic corruption

### Personal memory

```text
True user memory
Shuffled history
Random user's memory
No memory
```

### World knowledge

```text
True POI context
Shuffled POI context
Random POI context
No OSM/context
```

## Đo lường

- Acc@1;
- MRR;
- ECE;
- posterior KL shift;
- log-odds contribution;
- candidate rank change.

### Bayesian contribution

Với posterior:

$$
\log P(L=c_i\mid E)
=
\log q_i
+
\log P(H\mid c_i)
+
\log P(S\mid c_i)
-
\log Z,
$$

đo riêng:

$$
\Delta_H
=
\log P(H_{\mathrm{true}}\mid c_i)
-
\log P(H_{\mathrm{corrupt}}\mid c_i),
$$

$$
\Delta_S
=
\log P(S_{\mathrm{true}}\mid c_i)
-
\log P(S_{\mathrm{corrupt}}\mid c_i).
$$

Nếu semantic corruption làm giảm performance và làm posterior thay đổi đúng hướng, ta có bằng chứng mạnh hơn rằng semantic module thực sự được khai thác.

---

# RQ7 — Khả năng tổng quát hóa qua teacher backbone

## Câu hỏi

**RQ7: Representation-evolution distillation có phụ thuộc vào một trajectory teacher cụ thể hay không?**

## Teacher candidates

Có thể dùng:

- GRU/Transformer teacher mạnh hơn;
- PMT;
- UniTraj;
- trajectory Transformer khác.

## Student

Giữ cố định cùng một lightweight student.

## So sánh

Với mỗi teacher:

```text
Teacher
Student CE
Student CE+KD
Student Full Distillation
```

## Metrics

- Acc@1;
- MRR;
- calibration;
- distillation gain;
- training cost.

## Kết luận

Nếu gain tồn tại trên nhiều teacher:

$$
\boxed{
\text{framework is teacher-agnostic}
}
$$

sẽ được hỗ trợ.

---

# RQ8 — Calibration và uncertainty

## Câu hỏi

**RQ8: Representation distillation và Bayesian fusion có cải thiện calibration hay chỉ cải thiện ranking accuracy?**

## Metrics

- ECE;
- NLL;
- Brier Score;
- Reliability Diagram;
- selective risk.

## Thực nghiệm

So sánh:

```text
Teacher
Student CE
Student KD
Student KD+Evolution
Bayesian fusion
```

Đồng thời thực hiện temperature scaling.

## Selective prediction

Với confidence threshold:

$$
P_{\max}>\eta,
$$

đo:

- coverage;
- accuracy;
- selective risk.

---

# RQ9 — Accuracy–Latency Trade-off

## Câu hỏi

**RQ9: Distillation có thể giữ knowledge của teacher trong khi giảm inference latency hay không?**

## So sánh

```text
Trajectory Foundation Model
Large Transformer
Lightweight student
Student + LLM
Student + selective LLM
Full always-LLM framework
```

## Metrics

- latency/query;
- p50 latency;
- p95 latency;
- tokens/query;
- energy nếu đo được;
- peak memory;
- Acc@1;
- MRR.

## Hiển thị

Accuracy–Latency Pareto plot.

Mục tiêu:

$$
\boxed{
\text{high accuracy}
+
\text{low online cost}
}
$$

---

# RQ10 — Khi nào nên gọi LLM?

## Câu hỏi

**RQ10: Uncertainty-aware LLM invocation có giữ được accuracy trong khi giảm số lần gọi LLM hay không?**

## Chính sách

### Entropy-based

$$
H(P)>\gamma.
$$

### Margin-based

$$
P_{(1)}-P_{(2)}<\delta.
$$

### Random-call baseline

Cùng số lượng query gọi LLM nhưng chọn ngẫu nhiên.

## Metrics

- Acc@1;
- MRR;
- % queries calling LLM;
- tokens/query;
- latency;
- cost.

## Curve

Vẽ:

$$
\text{Accuracy}
\quad \text{vs} \quad
\% \text{LLM calls}.
$$

---

# RQ11 — Faithful Explanation

## Câu hỏi

**RQ11: Explanation từ Bayesian decomposition có phản ánh đúng computation tạo ra prediction hay không?**

## Counterfactual evidence removal

Loại từng term:

$$
\log q_i,
$$

$$
\log P(H\mid c_i),
$$

$$
\log P(S\mid c_i).
$$

Sau đó kiểm tra:

- posterior change;
- rank change;
- prediction flip.

## Metric đề xuất

Counterfactual Consistency:

$$
CC
=
\frac{
\#\text{cases where reported evidence contribution matches posterior change}
}{
N
}.
$$

So sánh với free-text LLM explanation.

---

# 13. Experimental Matrix tổng thể

| Nhóm thí nghiệm | Mục tiêu | Datasets chính | Metrics |
|---|---|---|---|
| Main comparison | RQ1 | Shanghai, TIST2015 | Acc@k, MRR |
| BBN vs LLM reranking | RQ2 | Shanghai | Acc@1, MRR, ECE |
| Distillation ablation | RQ3 | Shanghai + TIST | Acc@1, MRR, CKA |
| Order corruption | RQ4 | Shanghai + 2–3 cities | Acc@1, MRR |
| Layer vs temporal evolution | RQ5 | Shanghai/TIST | Acc@1, MRR |
| Semantic corruption | RQ6 | Shanghai | posterior shift, Acc@1 |
| Teacher generalization | RQ7 | Shanghai + TIST | Acc@1, gain |
| Calibration | RQ8 | Shanghai + TIST | ECE, NLL |
| Efficiency | RQ9 | Shanghai + OGB-sized/large mobility set if available | latency, memory |
| Selective LLM | RQ10 | Shanghai + TIST | accuracy/cost |
| Explanation faithfulness | RQ11 | Shanghai subset | counterfactual consistency |

---

# 14. Thứ tự thực nghiệm nên triển khai

Không nên triển khai toàn bộ ngay từ đầu.

## Phase 1 — Chứng minh Representation Evolution

Ưu tiên:

1. CE;
2. CE + KD;
3. + Trajectory;
4. + Velocity;
5. Correct/Reverse/Random.

Nếu không có gain ở phase này, không nên mở rộng framework.

---

## Phase 2 — Mobility-specific novelty

Thêm:

$$
L_{\mathrm{temp}}.
$$

So sánh:

```text
layer evolution
temporal evolution
dual evolution
```

Đây có thể là contribution khác biệt chính của paper.

---

## Phase 3 — Semantic verification

Làm:

```text
True memory
Shuffled memory
Random-user memory

True OSM
Shuffled OSM
Random POI context
```

Mục tiêu là kiểm chứng module LLM/BBN thực sự dùng semantic evidence.

---

## Phase 4 — Deployment

Thêm:

- teacher backbone sensitivity;
- selective LLM;
- latency;
- token count;
- calibration;
- explanation faithfulness.

---

# 15. Ablation tối thiểu bắt buộc

Một bảng ablation nên có:

| Variant | CE | KD | Traj | Vel | Temp | Cons | BBN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Student | ✓ |  |  |  |  |  |  |
| KD | ✓ | ✓ |  |  |  |  |  |
| KD+Traj | ✓ | ✓ | ✓ |  |  |  |  |
| KD+Vel | ✓ | ✓ |  | ✓ |  |  |  |
| KD+Traj+Vel | ✓ | ✓ | ✓ | ✓ |  |  |  |
| Dual Evolution | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |
| Full | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

---

# 16. Kiểm định thống kê

Không chỉ báo mean.

Nên báo:

$$
\mathrm{mean}\pm\mathrm{std}.
$$

Với paired evaluation, dùng:

- paired bootstrap;
- Wilcoxon signed-rank;
- paired t-test nếu phân phối phù hợp.

Confidence interval:

$$
95\%\ CI.
$$

Đối với multi-city:

- macro average;
- micro average;
- per-city variance;
- confidence interval qua cities.

---

# 17. Kiểm tra leakage

Các experiment distillation phải đảm bảo:

1. Teacher không nhìn test label.
2. Calibration chỉ fit trên validation.
3. LLM evidence calibration không dùng test.
4. Personal memory retrieval không chứa future trajectory point.
5. Reverse/random trajectory experiment chỉ thay đổi thứ tự input, không thay label/split.
6. Semantic corruption không vô tình làm thay candidate set.

---

# 18. Những claim có thể đưa ra nếu kết quả hỗ trợ

## Claim 1

> Final logits không phải là nguồn tri thức duy nhất; intermediate representation states và transition dynamics chứa transferable knowledge cho next-location prediction.

## Claim 2

> Mobility prediction cần mô hình hóa cả representation evolution theo chiều sâu mạng và behavioral evolution theo thời gian.

## Claim 3

> Semantic evidence chỉ có giá trị khi có thể chứng minh được bằng corruption/counterfactual experiments rằng posterior phụ thuộc có hệ thống vào nguồn evidence thực.

## Claim 4

> Foundation-model knowledge có thể được chuyển sang lightweight student nhằm đạt trade-off tốt hơn giữa accuracy và deployment latency.

## Claim 5

> Bayesian fusion cung cấp probability semantics và explanation faithfulness rõ ràng hơn free-text LLM re-ranking.

---

# 19. Những claim chưa nên đưa ra nếu chưa có bằng chứng

Không nên khẳng định:

- OSM luôn cải thiện accuracy;
- embedding personal memory luôn tốt hơn frequency-based memory;
- full hybrid luôn vượt Stage-1;
- BBN luôn vượt LLM re-ranking về mọi metric;
- graph/world semantic evidence chắc chắn độc lập có điều kiện;
- Foundation Model chắc chắn tốt hơn GRU nếu chưa chạy comparison.

---

# 20. Novelty dự kiến của phiên bản mở rộng

Có thể đóng gói novelty thành bốn lớp:

## Contribution 1 — Dual-Axis Representation Evolution Distillation

Không chỉ distill theo network layers mà còn theo mobility time.

$$
\boxed{
\text{Depth evolution}
+
\text{Temporal evolution}
}
$$

## Contribution 2 — Neuro-Symbolic-Probabilistic Fusion

Student cung cấp quantitative prior.

LLM/semantic model cung cấp structured evidence.

BBN thực hiện explicit probabilistic fusion.

## Contribution 3 — Knowledge Verification

Không chỉ chứng minh accuracy gain mà kiểm tra:

- order corruption;
- semantic corruption;
- posterior counterfactuals.

## Contribution 4 — Efficient Deployment

Teacher chỉ dùng offline.

LLM có thể dùng selectively.

Student + BBN đảm nhiệm phần lớn online inference.

---

# 21. Tên phương pháp gợi ý

Một số tên có thể cân nhắc:

### Dual-EvoTraj

**Dual-Evolution Trajectory Distillation for Neuro-Symbolic-Probabilistic Next-Location Prediction**

### TrajEvo-Bayes

**Trajectory Evolution Distillation with Bayesian Semantic Fusion**

### EvoMob-Hybrid

**Evolution-Aware Mobility Distillation with LLM and Bayesian Evidence Fusion**

### TRED-BBN

**Trajectory Representation Evolution Distillation with Bayesian Belief Fusion**

---

# 22. Research story đề xuất

Câu chuyện khoa học có thể được trình bày theo chuỗi:

```text
Trajectory Foundation Model mạnh
            ↓
final output chưa mô tả toàn bộ knowledge
            ↓
representation states + transitions chứa thêm thông tin
            ↓
mobility còn có temporal evolution riêng
            ↓
distill depth + time dynamics sang lightweight student
            ↓
student tạo calibrated candidate prior
            ↓
LLM chỉ tạo structured semantic evidence
            ↓
BBN fuse evidence bằng probability semantics rõ ràng
            ↓
selective LLM giúp giảm online cost
            ↓
corruption experiments kiểm chứng knowledge thật sự được sử dụng
```

---

# 23. Kết luận định hướng

Điểm đáng khai thác nhất từ UniSTD không phải là một loss cụ thể, mà là một cách đặt câu hỏi nghiên cứu:

> **Tri thức nào đang được chuyển giao, và tri thức đó tiến hóa như thế nào trước khi tạo thành prediction?**

Trong bài toán next-location prediction, câu hỏi này có thể mở rộng mạnh hơn vì tồn tại đồng thời:

$$
\boxed{
\text{representation evolution across layers}
}
$$

và:

$$
\boxed{
\text{mobility evolution across time}
}
$$

Do đó hướng có tiềm năng nhất là:

$$
\boxed{
\text{Trajectory Foundation Teacher}
\rightarrow
\text{Dual-Evolution Distillation}
\rightarrow
\text{Lightweight Student}
\rightarrow
\text{LLM Semantic Evidence}
\rightarrow
\text{Calibrated Bayesian Fusion}
}
$$

Nếu các RQ về order corruption, semantic corruption, dual evolution và efficiency đều được xác nhận thực nghiệm, phiên bản mở rộng sẽ không còn chỉ là một pipeline tích hợp nhiều thành phần, mà trở thành một framework có **research hypothesis rõ ràng về quá trình hình thành, chuyển giao và hợp nhất tri thức trong mobility prediction**.
