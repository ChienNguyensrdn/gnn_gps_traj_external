# idea.md — BeliefMove-Evo: Khung Neuro-Symbolic-Probabilistic với Distillation tiến hóa biểu diễn

## 0. Mục tiêu tài liệu

Tài liệu này là **đặc tả ý tưởng nghiên cứu** để Codex và nhóm nghiên cứu hiểu đúng bài toán trước khi viết code.

Mục tiêu chính là phát triển BeliefMove theo hướng:

> **LLM Mobility Teacher + Quantitative/Trajectory Teacher → Distillation → Lightweight Bayesian/Neural Student → Sequential Belief → Uncertainty-aware LLM-on-Demand**

Trong đó, phần mở rộng mới lấy cảm hứng từ tư tưởng **representation-evolution distillation**: student không chỉ học output cuối của teacher mà còn học **trạng thái biểu diễn** và **quá trình biến đổi biểu diễn**.

---

# 1. Bài toán nghiên cứu

Cho trajectory của người dùng:

$$
H_u = \{(p_1,t_1),(p_2,t_2),\ldots,(p_n,t_n)\},
$$

mục tiêu là dự đoán vị trí tiếp theo:

$$
\hat p_{n+1}.
$$

Không ép hệ thống dự đoán trực tiếp latitude/longitude ngay từ đầu. Phiên bản đầu ưu tiên:

$$
\text{GPS} \rightarrow \text{Region/Road/POI state} \rightarrow \text{Next Region/Road} \rightarrow \text{GPS}.
$$

Điều này giúp Bayesian component làm việc trên không gian trạng thái rời rạc hoặc bán rời rạc, ổn định hơn.

---

# 2. Vấn đề khoa học

Các hướng hiện tại thường rơi vào một trong ba nhóm:

1. **Quantitative-only**: mạnh về pattern nhưng khó khai thác semantic knowledge.
2. **LLM-only**: có reasoning nhưng tốn chi phí, khó calibration, không mạnh về numerical mobility dynamics.
3. **Hybrid re-ranking**: quantitative model sinh candidate, LLM chấm lại, nhưng LLM vẫn được gọi thường xuyên và semantic evidence chưa chắc tạo gain đáng tin cậy.

BeliefMove-Evo đặt câu hỏi khác:

> Có thể dùng teacher mạnh để chuyển tri thức mobility vào một student nhẹ, rồi chỉ gọi LLM khi belief state không chắc chắn hay không?

---

# 3. Kiến trúc tổng thể

## 3.1. Offline phase

```text
GPS trajectories
      │
      ├──────────────► Quantitative / Trajectory Teacher
      │                         │
      │                         ├── logits
      │                         ├── hidden states
      │                         └── representation evolution
      │
      └──────────────► LLM Mobility Teacher
                                │
                                ├── intent belief
                                ├── destination belief
                                ├── next-region belief
                                └── confidence / semantic evidence

                     ↓ distillation

               Lightweight Student
                     │
                     ├── mobility representation
                     ├── candidate prior
                     └── compact semantic belief

                     ↓

              Bayesian Belief Model
```

## 3.2. Online phase

```text
Current trajectory
      ↓
Mobility representation
      ↓
Lightweight student
      ↓
Bayesian belief update
      ↓
Uncertainty estimation
      ├── low  → Bayesian prediction
      └── high → LLM on-demand
                    ↓
               new evidence
                    ↓
               belief update
                    ↓
              final prediction
```

---

# 4. Vai trò của từng thành phần

## 4.1. Quantitative / Trajectory Teacher

Teacher có thể là:

- GRU/Transformer mạnh;
- PMT;
- UniTraj;
- trajectory foundation model khác.

Teacher tạo:

$$
\{H_T^{(1)},H_T^{(2)},\ldots,H_T^{(L)},Z_T\}.
$$

Mục tiêu: cung cấp **soft predictive knowledge + representation dynamics**.

## 4.2. LLM Mobility Teacher

LLM không được xem là ground truth.

LLM tạo structured beliefs, ví dụ:

```json
{
  "intent": {"commute": 0.65, "shopping": 0.20, "other": 0.15},
  "destination": {"home": 0.55, "work": 0.30, "other": 0.15},
  "next_region": {"r12": 0.60, "r08": 0.25, "r03": 0.15},
  "confidence": 0.72
}
```

LLM có hai vai trò:

1. **Offline teacher** để tạo teacher-belief dataset.
2. **Online expert on-demand** khi uncertainty cao.

## 4.3. Lightweight Student

Student phải rẻ hơn teacher.

Student học từ ba nguồn:

1. ground truth;
2. quantitative-teacher soft outputs;
3. LLM teacher structured beliefs.

Không được thiết kế student quá lớn đến mức phá mục tiêu efficiency.

---

# 5. Cơ sở lý thuyết về Distillation

## 5.1. Supervised loss

$$
L_{\mathrm{CE}}
=
-\sum_i \sum_c Y_{ic}\log P_{S,ic}.
$$

## 5.2. Response distillation

$$
P_T^{(\tau)}=\operatorname{Softmax}(Z_T/\tau),
$$

$$
P_S^{(\tau)}=\operatorname{Softmax}(Z_S/\tau),
$$

$$
L_{\mathrm{KD}}
=
\tau^2
D_{\mathrm{KL}}
\left(
P_T^{(\tau)} \Vert P_S^{(\tau)}
\right).
$$

## 5.3. Representation trajectory distillation

Không dùng truncate/zero-padding trực tiếp giữa teacher và student.

Dùng learnable projection:

$$
R_T^{(m)} = P_T^{(m)}(H_T^{(m)}),
$$

$$
R_S^{(m)} = P_S^{(m)}(H_S^{(m)}).
$$

Loss:

$$
L_{\mathrm{traj}}
=
\frac{1}{M}
\sum_{m=1}^{M}
\left\|
R_S^{(m)}-R_T^{(m)}
\right\|_F^2.
$$

## 5.4. Velocity distillation

$$
\Delta R_T^{(m)}=R_T^{(m+1)}-R_T^{(m)},
$$

$$
\Delta R_S^{(m)}=R_S^{(m+1)}-R_S^{(m)}.
$$

$$
L_{\mathrm{vel}}
=
\frac{1}{M-1}
\sum_{m=1}^{M-1}
\left\|
\Delta R_S^{(m)}-\Delta R_T^{(m)}
\right\|_F^2.
$$

### Diễn giải

- trajectory loss: student cần hình thành **state nào**;
- velocity loss: student cần biến đổi representation theo **hướng nào**.

---

# 6. Mobility-specific extension: Dual-Axis Evolution

Đây là phần novelty đề xuất, không phải sao chép UniSTD.

Representation của trajectory có thể ký hiệu:

$$
H_t^{(\ell)},
$$

trong đó:

- $t$: mobility time step;
- $\ell$: network layer.

Ta có hai hướng tiến hóa.

## 6.1. Depth evolution

$$
\Delta_{\ell}H_t^{(\ell)}
=
H_t^{(\ell+1)}-H_t^{(\ell)}.
$$

## 6.2. Temporal evolution

$$
\Delta_tH_t^{(\ell)}
=
H_{t+1}^{(\ell)}-H_t^{(\ell)}.
$$

Temporal distillation:

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

### Research hypothesis

Nếu mobility order chứa tri thức dự đoán thật sự, thì:

$$
\text{Correct Order}
>
\text{Reverse Order / Random Order}.
$$

---

# 7. Bayesian Mobility Student / Belief Engine

Biến latent có thể gồm:

$$
Z_t = \{
Activity_t,
Intent_t,
Destination_t,
NextRegion_t
\}.
$$

Một graph khởi đầu:

```text
Time ─────► Activity ─────► Intent ─────► Destination ─────► NextRegion
  │             ▲              ▲                 ▲
  │             │              │                 │
  └────► History / CurrentRoad / Speed / Heading
```

Graph này là **biến thực nghiệm**, không phải cấu trúc đúng mặc định.

---

# 8. Sequential belief

Belief state:

$$
B_t=P(Z_t\mid X_{1:t}).
$$

Nếu static BN không đủ:

$$
P(Z_{t+1}\mid Z_t)
$$

và:

$$
P(E_{t+1}\mid Z_{t+1})
$$

được dùng trong Dynamic Bayesian Network.

---

# 9. Data-derived belief và teacher-induced belief

Không xem teacher là ground truth tuyệt đối.

Có thể thử:

$$
\mathrm{CPT}^{*}
=
\alpha \mathrm{CPT}_{data}
+
(1-\alpha)\mathrm{CPT}_{teacher},
$$

với:

$$
\alpha\in\{0,0.25,0.5,0.75,1\}.
$$

Hyperparameter phải chọn trên validation set.

---

# 10. Uncertainty-aware LLM-on-Demand

Entropy:

$$
H(B_t)
=
-\sum_z B_t(z)\log B_t(z).
$$

Nếu:

$$
H(B_t)<\gamma,
$$

thì dùng Bayesian prediction.

Nếu:

$$
H(B_t)\geq\gamma,
$$

thì gọi LLM.

Có thể so sánh margin:

$$
P_{(1)}-P_{(2)}<\delta.
$$

Mục tiêu:

$$
\boxed{
\text{Accuracy}
+
\text{Calibration}
+
\text{Latency}
+
\text{LLMCallRate}
}
$$

---

# 11. Tổng objective đề xuất

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
\lambda_{\mathrm{sem}}L_{\mathrm{sem}}.
$$

Không bật tất cả ngay; phải tăng dần qua ablation.

---

# 12. Research Questions

## RQ1 — Baseline reproducibility

**Câu hỏi:** Có reproduce được AgentMove / quantitative baseline trên cùng preprocessing và split hay không?

**Thực nghiệm:** freeze preprocessing; freeze split; chạy baseline nhiều seed; log config/commit.

**Tiêu chí:** chưa triển khai BeliefMove nếu baseline chưa ổn định.

## RQ2 — Bayesian student data-only

**Câu hỏi:** Bayesian model dùng dữ liệu thật tạo baseline cạnh tranh tới đâu?

**Thực nghiệm:** Markov/Bi-gram, BN data-only, DBN data-only, quantitative teacher.

**Metrics:** Acc@1/5/10, MRR, NLL, Brier, ECE.

## RQ3 — LLM knowledge distillation

**Câu hỏi:** Structured mobility beliefs từ LLM có giúp student tốt hơn data-only hay không?

**Thực nghiệm:**

```text
M1 = BN data-only
M2 = BN + LLM beliefs
M3 = BN + quantitative teacher
M4 = BN + both teachers
```

**Kiểm định:** paired bootstrap 95% CI.

## RQ4 — Representation-evolution distillation

**Câu hỏi:** Intermediate states và representation transitions có mang transferable knowledge ngoài final logits hay không?

**Thực nghiệm:** CE; CE+KD; +Traj; +Vel; +Traj+Vel.

**Metrics:** Acc@1, MRR, CKA, transition cosine similarity.

## RQ5 — Mobility order corruption

**Câu hỏi:** Gain có đến từ ordered mobility dynamics hay chỉ từ tập hợp visited locations?

**Thực nghiệm:** Correct, Reverse, Random order. Random phải nhiều permutation.

## RQ6 — Dual-Axis Evolution

**Câu hỏi:** Layer-wise evolution hay temporal evolution quan trọng hơn?

**Thực nghiệm:** KD only, layer trajectory, layer velocity, temporal evolution, dual evolution.

Phân nhóm thêm short/medium/long trajectory.

## RQ7 — Belief memory

**Câu hỏi:** Sequential belief có cải thiện prediction so với independent-step inference không?

**Thực nghiệm:** static BN, BN+history, sequential belief, DBN.

## RQ8 — Uncertainty-aware LLM routing

**Câu hỏi:** Có thể giảm số lần gọi LLM mà giữ quality gần full LLM không?

**Thực nghiệm:** Never, Always, Entropy, Margin, Random router.

**Metrics:** Accuracy, MRR, LLMCallRate, latency, token/query.

## RQ9 — Semantic knowledge verification

**Câu hỏi:** Personal memory và context/world knowledge có thật sự ảnh hưởng prediction không?

**Memory:** true, shuffled, random-user, none.

**Context:** true, shuffled, random-POI, none.

## RQ10 — Teacher robustness

**Câu hỏi:** Framework có phụ thuộc teacher backbone cụ thể không?

**Teacher:** GRU, Transformer, PMT/UniTraj nếu khả thi. Student giữ cố định.

## RQ11 — Calibration

**Câu hỏi:** Distillation và Bayesian update có cải thiện calibration không?

**Metrics:** ECE, NLL, Brier, reliability diagram.

## RQ12 — Efficiency

**Câu hỏi:** BeliefMove-Evo có accuracy–efficiency trade-off tốt hơn always-LLM và heavy teacher không?

**Metrics:** p50/p95 latency, token/query, peak memory, LLMCallRate.

## RQ13 — Robustness

**Câu hỏi:** Framework phản ứng thế nào khi GPS/context thiếu hoặc nhiễu?

**Perturbations:** missing GPS points, timestamp noise, position noise, missing/wrong context.

---

# 13. Claims chỉ được phép đưa ra nếu có bằng chứng

Có thể claim nếu kết quả hỗ trợ:

1. representation evolution chứa transferable mobility knowledge;
2. temporal-order preservation quan trọng;
3. lightweight student đạt trade-off tốt hơn;
4. uncertainty routing giảm LLM calls;
5. Bayesian belief cải thiện calibration / sequential reasoning.

Không được mặc định claim:

- LLM luôn tốt hơn data-only;
- OSM luôn giúp accuracy;
- teacher belief là ground truth;
- full model luôn thắng mọi baseline;
- semantic context luôn độc lập có điều kiện.

---

# 14. Tiêu chí thành công khoa học

1. Full/student model > data-only baseline có ý nghĩa thống kê.
2. KD+evolution > KD-only.
3. Correct order > corrupted order.
4. Sequential belief > static prediction.
5. Entropy router giảm đáng kể LLMCallRate với accuracy drop nhỏ.
6. Calibration không suy giảm nghiêm trọng.
7. Kết quả ổn định trên ít nhất 2 dataset / nhiều city.
8. Ablation tách được đóng góp từng module.

---

# 15. Tên tạm

**BeliefMove-Evo**

> Evolution-Aware Neuro-Symbolic-Probabilistic Mobility Prediction with Distilled Teachers, Sequential Bayesian Beliefs, and Uncertainty-Aware LLM Intervention
