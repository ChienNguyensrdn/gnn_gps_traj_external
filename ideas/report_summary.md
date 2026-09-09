# Báo cáo tổng hợp BeliefMove-Evo trên TIST2015 — 12 thành phố

> Cập nhật từ kết quả server ngày 2026-09-09. Gate: **ready-internal-gpu-contention**.
> Tất cả 12 thành phố đều có 92 nhóm kết quả và không còn artifact bị thiếu.
> Riêng RQ12 được chấp nhận cho phân tích nội bộ vì GPU có tiến trình cạnh tranh;
> số liệu hiệu năng chưa đủ điều kiện dùng trong publication.

## 1. Phạm vi thực nghiệm

- Dataset: TIST2015 trên Tokyo, Nairobi, New York, Sydney, Cape Town, Paris,
  Beijing, Mumbai, San Francisco, London, São Paulo và Moscow.
- Mô hình stochastic dùng seed 42, 43 và 44.
- Neural sử dụng `last-query`; Bayesian sử dụng `all-prefix`.
- Kết quả chính là macro-average không trọng số qua đúng 12 thành phố, sau đó
  tổng hợp qua seed.
- RQ3, RQ8 và RQ9 dùng Qwen2:7b, `limit=200`, `no-OSM`; đây là bounded result,
  không phải full-query hoặc full-world-knowledge.
- Không so trực tiếp trị tuyệt đối giữa các query protocol khác nhau.

## 2. Kết luận chính

1. **E5-Dual là biến thể distillation tốt nhất:** R@1 `0,171903`, R@10
   `0,384337`, MRR `0,245829`, đồng thời ổn định qua ba seed.
2. **Thứ tự thời gian có ích:** correct order tốt hơn random và reverse.
3. **DBN cải thiện ranking nhưng làm calibration xấu:** B3 tăng Recall/MRR,
   nhưng NLL, Brier và ECE đều tăng.
4. **Distillation bền vững với hai teacher:** GRU và Transformer đều giúp
   student tốt hơn CE-only; student nhận teacher GRU tốt nhất hiện tại.
5. **LLM chưa tạo lợi ích ổn định:** trong RQ8, `never` tốt hơn các router dùng
   LLM về R@1/MRR; RQ9 không cho thấy context đúng luôn tốt hơn context nhiễu.

## 3. RQ2 — Bayesian data-only và quantitative teacher

| Biến thể | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| DBN data-only | 0,154823 | 0,298252 | 0,358989 | 0,224754 | 8,491321 | 0,999051 | 0,153880 |
| Quantitative teacher | **0,171751 ± 0,002154** | **0,324055 ± 0,002071** | **0,377889 ± 0,001171** | **0,244027 ± 0,002008** | **7,093902 ± 0,097125** | **0,938170 ± 0,003423** | **0,056555 ± 0,007519** |

Teacher định lượng vượt DBN data-only trên cả ranking và calibration. Điều này
cho thấy prior/transition thống kê đơn thuần chưa thay thế được representation
học từ trajectory. DBN data-only là deterministic nên chỉ có một run.

## 4. RQ3 — LLM knowledge distillation bounded

| Biến thể | R@1 | R@5 | R@10 | MRR | NLL↓ |
|---|---:|---:|---:|---:|---:|
| M1 data-only | 0,022079 | 0,059487 | 0,084025 | 0,045216 | 8,402382 |
| M2 LLM-only | 0,032912 | 0,071571 | 0,094441 | 0,056014 | 8,371884 |
| M3 quantitative | 0,135096 | **0,271209** | 0,320330 | 0,201125 | 6,799041 |
| **M4 both** | **0,135929** | **0,271209** | **0,324913** | **0,202728** | **6,783468** |

LLM-only tốt hơn data-only nhưng còn xa quantitative teacher. Kết hợp LLM với
quantitative signal chỉ tăng nhẹ so với M3: R@1 `+0,000833`, R@10
`+0,004583`, MRR `+0,001603`; R@5 không đổi. Vì mỗi biến thể chỉ có một
bounded run, đây là bằng chứng thăm dò chứ chưa phải kết luận tổng quát.

## 5. RQ4 — Distillation ablation

| Biến thể | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0-CE | 0,156010 ± 0,003212 | 0,304810 ± 0,001440 | 0,353479 ± 0,003175 | 0,225939 ± 0,001327 | 7,407897 | 0,945500 | 0,056975 |
| E1-KD | 0,167779 ± 0,001362 | 0,320729 ± 0,001699 | 0,377380 ± 0,002057 | 0,241204 ± 0,000401 | 6,710074 | 0,936330 | 0,048971 |
| E2-KD-Traj | 0,167959 ± 0,003202 | 0,322898 ± 0,000915 | 0,379605 ± 0,001087 | 0,242099 ± 0,001252 | 6,695095 | 0,935780 | 0,049009 |
| E3-KD-Vel | 0,169996 ± 0,002341 | 0,323083 ± 0,002324 | 0,379032 ± 0,003040 | 0,243411 ± 0,001450 | 6,698934 | 0,935271 | 0,048632 |
| E4-Layer | 0,171730 ± 0,001667 | 0,324089 ± 0,000622 | 0,380577 ± 0,002676 | 0,244642 ± 0,000951 | 6,694331 | 0,934733 | 0,048252 |
| **E5-Dual** | **0,171903 ± 0,000780** | **0,327474 ± 0,001340** | **0,384337 ± 0,001949** | **0,245829 ± 0,000469** | **6,673658** | **0,933304** | **0,047902** |

So với E0-CE, E5-Dual tăng R@1 `0,015894` (10,2% tương đối), R@5
`0,022664`, R@10 `0,030858`, MRR `0,019890`; NLL giảm `0,734239`.
So với E1-KD, E5 vẫn tăng R@1 `0,004124`, R@10 `0,006957` và MRR
`0,004625`. KD cơ bản mang lại phần lớn cải thiện; dual-axis alignment bổ sung
một mức tăng nhỏ nhưng nhất quán.

## 6. RQ5 — Vai trò của thứ tự trajectory

| Thứ tự | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| **Correct** | **0,171903** | **0,327474** | **0,384337** | **0,245829** |
| Random | 0,164812 | 0,324336 | 0,381848 | 0,240639 |
| Reverse | 0,163801 | 0,323969 | 0,380686 | 0,239358 |

Random làm giảm R@1 `0,007091`, reverse làm giảm `0,008102` so với correct.
Reverse cũng làm giảm MRR `0,006471`. Kết quả ủng hộ việc mô hình học chiều
tiến hóa thời gian, không chỉ học một tập điểm đã ghé qua.

## 7. RQ6 — Dual-axis evolution

E5-Dual đạt ranking tốt nhất. E6-Temporal có NLL thấp hơn E5 rất nhẹ
(`6,671989` so với `6,673658`) nhưng R@1 thấp hơn `0,002911`, R@5 thấp hơn
`0,001569`, MRR thấp hơn `0,002122`. Temporal alignment riêng có lợi cho
log-loss, còn kết hợp hai trục phù hợp hơn với mục tiêu xếp hạng.

## 8. RQ7/RQ11 — Belief memory và calibration

| Biến thể | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0-Static | 0,160954 | 0,330134 | 0,395361 | 0,241040 | **6,470550** | **0,943010** | **0,048827** |
| B3-DBN | **0,167185** | **0,341297** | **0,406880** | **0,249301** | 6,652804 | 0,967838 | 0,132316 |

B3-DBN tăng R@1 `0,006231`, R@5 `0,011163`, R@10 `0,011519`, MRR
`0,008261`. Tuy nhiên, ECE tăng `0,083489`; B3 bị over-confidence rõ rệt.

Snapshot 12-city chỉ xuất các dòng identity/reference của RQ11. Bảng này chưa
thay thế báo cáo calibration đa mục tiêu chi tiết và không được mô tả là kết quả
sau temperature scaling.

## 9. RQ8 — Uncertainty-aware LLM routing bounded

| Policy | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| **Never** | **0,180472** | **0,335458** | 0,393452 | **0,254299** |
| Always | 0,151603 | 0,330875 | 0,393452 | 0,236354 |
| Entropy | 0,177972 | 0,335082 | 0,393452 | 0,252666 |
| Margin | 0,179679 | **0,335458** | 0,393452 | 0,253807 |
| Random budget-matched | 0,177431 | 0,335417 | 0,393452 | 0,252029 |

Gọi LLM cho mọi query làm giảm R@1 `0,028869` và MRR `0,017945` so với
never. Entropy và margin hạn chế tác hại nhưng vẫn không vượt never. R@10 giống
nhau ở mọi policy, cho thấy reranker chỉ đổi thứ tự trong cùng candidate top-10,
không cải thiện candidate recall. Kết quả chưa chứng minh routing LLM tạo lợi
ích chất lượng; giá trị hiện tại chủ yếu là kiểm soát chi phí và tác hại.

## 10. RQ9 — Semantic knowledge verification bounded

| Biến thể | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| Memory true | 0,179432 | 0,338292 | 0,393452 | 0,256099 |
| Memory shuffled | 0,179432 | 0,337500 | 0,393452 | 0,255517 |
| Memory random-user | 0,166140 | 0,323709 | 0,393452 | 0,241119 |
| Memory none | 0,169602 | 0,333333 | 0,393452 | 0,247643 |
| Context shuffled | 0,178599 | **0,338791** | 0,393452 | 0,255843 |
| Context random-POI | **0,183723** | 0,335375 | 0,393452 | **0,257776** |
| Context none | 0,179057 | 0,334084 | 0,393452 | 0,254183 |

Memory đúng tốt hơn random-user và none, nhưng gần như ngang memory shuffled.
Ở trục context, random-POI lại có R@1/MRR cao nhất. RQ9 mới cung cấp bằng chứng
hạn chế cho personal memory, chưa xác minh semantic context đúng là nguyên nhân
tạo cải thiện. Cần paired significance và limit lớn hơn cho kết luận nhân quả.

## 11. RQ10 — Độ bền theo kiến trúc teacher

| Student | R@1 | R@5 | R@10 | MRR | NLL↓ | ECE↓ |
|---|---:|---:|---:|---:|---:|---:|
| CE-only | 0,155951 | 0,304878 | 0,353451 | 0,225894 | 7,434454 | 0,055881 |
| **Distill từ GRU** | **0,170549** | **0,325773** | **0,381963** | **0,244327** | **6,647197** | **0,045113** |
| Distill từ Transformer | 0,167144 | 0,325721 | 0,379618 | 0,242336 | 6,703186 | 0,046834 |

Cả hai teacher đều cải thiện student. GRU teacher tạo student tốt hơn nhẹ so
với Transformer teacher; teacher phức tạp hơn không mặc nhiên tốt hơn.

## 12. RQ12 — Accuracy–efficiency trade-off nội bộ

| Profile | Student GRU R@1 | Throughput |
|---|---:|---:|
| Batch 1 | 0,170549 ± 0,001154 | 988,26 ± 16,47 query/s |
| Batch 256 | 0,170549 ± 0,001154 | 148.798,18 ± 1.430,97 query/s |

Batching làm tăng throughput quan sát khoảng 150 lần. Tuy nhiên, mỗi thành phố
có 6 marker `GPU_CONTENTION`, tổng cộng 72 marker. Các số này chỉ dùng cho lập
kế hoạch nội bộ, không dùng để tuyên bố speedup trong bài báo. Muốn công bố cần
benchmark lại trên GPU độc quyền với cùng hardware và timing harness.

## 13. RQ13 — Robustness

Summary 12-city mới xuất hàng `clean`, trùng E5-Dual: R@1
`0,171903 ± 0,000780`, R@10 `0,384337 ± 0,001949`, MRR
`0,245829 ± 0,000469`. Các artifact missing/noisy input đã qua gate nhưng chưa
được xuất thành macro row. Report này vì thế chưa đủ để kết luận độ bền theo
từng mức nhiễu; cần mở rộng aggregator hoặc dùng báo cáo RQ13 chi tiết.

## 14. Tính ổn định địa lý

E5-Dual cải thiện R@1 so với E0-CE ở cả 12/12 thành phố trong dữ liệu đã kiểm
tra. Mức tăng khoảng `+0,0089` đến `+0,0256`. Phương sai R@1 trung bình giữa
thành phố của E5 là `0,002557`, tương ứng độ lệch chuẩn địa lý khoảng `0,0506`.
Khác biệt miền vẫn lớn; bài báo cần giữ macro-average, city variance và bảng
per-city.

## 15. Trạng thái cuối

| Hạng mục | Trạng thái |
|---|---|
| Đủ 12 thành phố | Có |
| Blocking missing artifact | 0 |
| Seed neural 42–44 | Đủ |
| LLM bounded limit 200 | Đủ |
| RQ12 GPU sạch | Không |
| OSM world knowledge | Không (`no-OSM`) |
| Gate nội bộ | **ready-internal-gpu-contention** |
| Publication eligible toàn bộ | **Không** |

Kết quả đủ để phân tích nội bộ và viết phần chất lượng mô hình 12-city. Trước
publication vẫn cần benchmark lại RQ12 không contention, giữ nhãn bounded cho
RQ3/RQ8/RQ9, bổ sung paired significance 12-city cho các so sánh chính và
không gọi cấu hình `no-OSM` là mô hình world-knowledge đầy đủ.
