# RQ13 — Robustness với đầu vào thiếu hoặc nhiễu

> Frozen E5-dual được đánh giá trên cùng test query và seed 42, 43, 44. Chỉ context quan sát bị perturb; checkpoint, target và label không đổi. Các metrics là mean ± std qua seed.

## 1. Câu hỏi nghiên cứu

RQ13 kiểm tra mức độ ổn định của BeliefMove-Evo khi dữ liệu GPS, timestamp, vị trí, user context hoặc temporal context bị thiếu/sai. Mọi so sánh dùng cùng query và cùng seed để đo trực tiếp mức suy giảm so với `clean`.

## 2. Kết quả test

| Variant | Changed queries | R@1 | R@5 | R@10 | MRR | NLL↓ | Brier↓ | ECE↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.000000 ± 0.000000 | 0.147140 ± 0.002240 | 0.303595 ± 0.001035 | 0.367367 ± 0.000803 | 0.221656 ± 0.001334 | 7.528161 ± 0.012171 | 0.945082 ± 0.001529 | 0.029841 ± 0.003400 |
| gps-drop-25 | 0.571448 ± 0.003184 | 0.144432 ± 0.003123 | 0.300300 ± 0.000967 | 0.363469 ± 0.002018 | 0.218809 ± 0.001928 | 7.564862 ± 0.014275 | 0.946471 ± 0.001632 | 0.029678 ± 0.004064 |
| gps-drop-50 | 0.813565 ± 0.000920 | 0.141189 ± 0.002748 | 0.296902 ± 0.000521 | 0.358828 ± 0.000730 | 0.215074 ± 0.001840 | 7.621460 ± 0.014503 | 0.948333 ± 0.001350 | 0.029287 ± 0.002763 |
| time-noise-30m | 0.989064 ± 0.000768 | 0.142672 ± 0.002107 | 0.301870 ± 0.000606 | 0.365711 ± 0.001460 | 0.218476 ± 0.001412 | 7.544027 ± 0.010388 | 0.947876 ± 0.001401 | 0.032420 ± 0.003496 |
| time-noise-60m | 0.998034 ± 0.000237 | 0.141534 ± 0.002466 | 0.301163 ± 0.000233 | 0.364814 ± 0.000949 | 0.217329 ± 0.001640 | 7.559471 ± 0.014928 | 0.949300 ± 0.001810 | 0.032142 ± 0.003687 |
| position-noise-200m | 0.995136 ± 0.000442 | 0.098272 ± 0.002241 | 0.229749 ± 0.004806 | 0.290037 ± 0.004867 | 0.162565 ± 0.003055 | 8.077444 ± 0.050302 | 0.973191 ± 0.000869 | 0.016907 ± 0.002480 |
| position-noise-500m | 0.998637 ± 0.000294 | 0.092769 ± 0.001690 | 0.220331 ± 0.000589 | 0.280359 ± 0.000706 | 0.155179 ± 0.001687 | 8.166101 ± 0.014017 | 0.977119 ± 0.000849 | 0.017223 ± 0.000179 |
| context-missing-user | 1.000000 ± 0.000000 | 0.051766 ± 0.006223 | 0.130615 ± 0.005651 | 0.171445 ± 0.006345 | 0.092823 ± 0.005213 | 10.483165 ± 1.268530 | 1.000232 ± 0.008617 | 0.053233 ± 0.034507 |
| context-missing-time | 0.982819 ± 0.000000 | 0.119920 ± 0.001794 | 0.269199 ± 0.003274 | 0.327434 ± 0.002096 | 0.190957 ± 0.001927 | 8.652827 ± 0.023710 | 0.975477 ± 0.000439 | 0.075355 ± 0.001966 |
| context-wrong-user | 1.000000 ± 0.000000 | 0.051697 ± 0.001831 | 0.127700 ± 0.001402 | 0.167822 ± 0.000424 | 0.091939 ± 0.001460 | 9.918428 ± 0.035547 | 0.999470 ± 0.000662 | 0.037611 ± 0.000906 |
| context-wrong-time | 1.000000 ± 0.000000 | 0.111864 ± 0.000775 | 0.259953 ± 0.002123 | 0.322035 ± 0.000661 | 0.182698 ± 0.000855 | 8.646413 ± 0.026802 | 0.982956 ± 0.001293 | 0.078591 ± 0.002287 |
| context-missing | 1.000000 ± 0.000000 | 0.049955 ± 0.005364 | 0.124871 ± 0.007753 | 0.164114 ± 0.009547 | 0.088671 ± 0.004331 | 11.833064 ± 1.487442 | 1.007070 ± 0.011652 | 0.075367 ± 0.031905 |
| context-wrong | 1.000000 ± 0.000000 | 0.049696 ± 0.002363 | 0.120713 ± 0.001739 | 0.158473 ± 0.001428 | 0.087264 ± 0.002134 | 11.037231 ± 0.033808 | 1.005686 ± 0.000820 | 0.054863 ± 0.002660 |

## 3. Tóm tắt paired significance

Positive effect nghĩa là variant đứng trước tốt hơn; với NLL/Brier, dấu đã được đảo để quy ước “cao hơn là tốt hơn”. Holm correction được áp dụng đồng thời cho toàn bộ 90 phép kiểm định.

### Clean so với perturbation

| Perturbation | ΔR@1 favoring clean | ΔR@10 favoring clean | ΔMRR favoring clean | ΔNLL favoring clean | Holm-significant |
|---|---:|---:|---:|---:|---|
| gps-drop-25 | 0.002708 | 0.003898 | 0.002847 | 0.036702 | tất cả metrics |
| gps-drop-50 | 0.005951 | 0.008539 | 0.006582 | 0.093299 | tất cả metrics |
| time-noise-30m | 0.004468 | 0.001656 | 0.003180 | 0.015867 | tất cả metrics |
| time-noise-60m | 0.005606 | 0.002553 | 0.004327 | 0.031310 | tất cả metrics |
| position-noise-200m | 0.048868 | 0.077330 | 0.059091 | 0.549283 | tất cả metrics |
| position-noise-500m | 0.054371 | 0.087008 | 0.066477 | 0.637941 | tất cả metrics |
| context-missing-user | 0.095374 | 0.195922 | 0.128833 | 2.955004 | tất cả metrics |
| context-missing-time | 0.027220 | 0.039933 | 0.030699 | 1.124666 | tất cả metrics |
| context-wrong-user | 0.095443 | 0.199545 | 0.129718 | 2.390268 | tất cả metrics |
| context-wrong-time | 0.035276 | 0.045332 | 0.038958 | 1.118252 | tất cả metrics |
| context-missing | 0.097185 | 0.203253 | 0.132985 | 4.304903 | tất cả metrics |
| context-wrong | 0.097444 | 0.208894 | 0.134392 | 3.509069 | tất cả metrics |

Mỗi dòng trên có bootstrap 95% CI không cắt 0 và Holm-adjusted p < 0,05 cho các metrics được ghi. Bảng đầy đủ theo từng metric được lưu trong `results/beliefmove-evo/aggregated/rq13_summary.json`.

### Dose-response trực tiếp

| So sánh nhẹ–nặng | Metric | Effect favoring mức nhẹ | 95% CI | Holm p | Kết luận |
|---|---|---:|---:|---:|---|
| gps-drop-25 vs gps-drop-50 | R@1 | 0.003243 | 0.001880–0.004606 | 0.0089991 | significant |
| gps-drop-25 vs gps-drop-50 | R@10 | 0.004640 | 0.003260–0.006003 | 0.0089991 | significant |
| gps-drop-25 vs gps-drop-50 | MRR | 0.003736 | 0.002828–0.004637 | 0.0089991 | significant |
| gps-drop-25 vs gps-drop-50 | NLL | 0.056598 | 0.050498–0.062555 | 0.0089991 | significant |
| time-noise-30m vs time-noise-60m | R@1 | 0.001138 | -0.000017–0.002277 | 0.159584 | không significant |
| time-noise-30m vs time-noise-60m | R@10 | 0.000897 | -0.000121–0.001932 | 0.165583 | không significant |
| time-noise-30m vs time-noise-60m | MRR | 0.001147 | 0.000457–0.001820 | 0.0089991 | significant |
| time-noise-30m vs time-noise-60m | NLL | 0.015443 | 0.012438–0.018551 | 0.0089991 | significant |
| position-noise-200m vs position-noise-500m | R@1 | 0.005503 | 0.003226–0.007780 | 0.0089991 | significant |
| position-noise-200m vs position-noise-500m | R@10 | 0.009677 | 0.007021–0.012385 | 0.0089991 | significant |
| position-noise-200m vs position-noise-500m | MRR | 0.007386 | 0.005588–0.009183 | 0.0089991 | significant |
| position-noise-200m vs position-noise-500m | NLL | 0.088658 | 0.073195–0.104458 | 0.0089991 | significant |

GPS dropout và position noise có dose-response rõ trên toàn bộ metrics. Với timestamp noise, mức 60 phút xấu hơn 30 phút ở MRR, NLL và Brier, nhưng khác biệt R@1/R@5/R@10 chưa có ý nghĩa sau Holm correction.

## 4. Phân tích kết quả

### 4.1. Mô hình tương đối bền với thiếu một phần GPS trajectory

Khi bỏ 25% điểm GPS, R@1 chỉ giảm 0,002708 và R@10 giảm 0,003898. Bỏ 50% làm mức giảm tăng lên 0,005951 và 0,008539. Các khác biệt nhỏ về độ lớn tuyệt đối nhưng ổn định trên cùng query và có ý nghĩa thống kê. Việc luôn giữ điểm cuối context cho thấy mô hình vẫn tận dụng tốt recent-location signal khi phần lịch sử xa hơn bị thiếu.

Không nên diễn giải kết quả này là mô hình bền với việc mất toàn bộ GPS: protocol chỉ bỏ một phần prefix và luôn giữ quan sát gần nhất.

### 4.2. Timestamp noise ảnh hưởng ranking tổng thể nhiều hơn Recall@K

Cả nhiễu 30 và 60 phút đều làm clean tốt hơn có ý nghĩa. Tuy nhiên, so sánh trực tiếp 30–60 phút không khác biệt có ý nghĩa ở Recall@K, trong khi MRR, NLL và Brier xấu đi. Điều này cho thấy tăng nhiễu thời gian chủ yếu dịch chuyển thứ hạng và xác suất bên trong top-K thay vì luôn làm target rời khỏi top-K.

### 4.3. Position noise là failure mode nghiêm trọng

Position noise 200 m làm R@1 giảm gần 0,049; 500 m làm giảm hơn 0,054. R@10 giảm tương ứng khoảng 0,077 và 0,087. Dose-response 200–500 m có ý nghĩa trên toàn bộ ranking/probabilistic metrics.

Đây là robustness của categorical POI pipeline sau nearest-POI remapping. Nó không chứng minh raw GPS encoder nhạy ở cùng mức độ, vì mô hình hiện nhận POI ID thay vì tọa độ liên tục. Changed-query rate gần 1 cho thấy phép remapping thay đổi hầu hết query; vì vậy đây là một perturbation mạnh dù bán kính được gọi là 200/500 m.

### 4.4. User context quan trọng hơn temporal context

Tách one-axis cho thấy missing/wrong user làm R@1 giảm khoảng 0,095, trong khi missing time giảm 0,027 và wrong time giảm 0,035. Như vậy phần lớn suy giảm của context tổng hợp đến từ user conditioning.

`context-missing-user` có độ lệch chuẩn NLL lớn (±1,268530), phản ánh phản ứng khác nhau giữa các checkpoint seed khi gặp unknown-user embedding. Đây là dấu hiệu cần thận trọng nếu triển khai cho cold-start user: chất lượng không chỉ thấp mà còn kém ổn định giữa seed.

Temporal context vẫn có đóng góp đáng kể. Wrong time gây hại mạnh hơn missing time ở accuracy/MRR, cho thấy thông tin thời gian sai có thể nguy hiểm hơn việc dùng sentinel thiếu thời gian. Tuy nhiên, time slot 0 là một slot hợp lệ chứ chưa phải learned missing-time token, nên `context-missing-time` chỉ là proxy tương thích kiến trúc hiện tại.

### 4.5. Không diễn giải ECE tách khỏi accuracy, NLL và Brier

ECE của position-noise thấp hơn clean dù accuracy, NLL và Brier đều xấu đi mạnh. Điều này có thể xảy ra khi mô hình giảm confidence cùng lúc với accuracy. Vì vậy không được claim position noise cải thiện calibration; đánh giá calibration phải đọc đồng thời reliability, NLL và Brier.

## 5. Kết luận RQ13

- E5-dual tương đối bền với GPS point dropout và timestamp noise nhẹ, nhưng mọi perturbation vẫn gây suy giảm có ý nghĩa so với clean.
- GPS dropout và position noise thể hiện dose-response rõ ràng; timestamp dose-response chỉ rõ ở MRR/NLL/Brier, chưa rõ ở Recall@K.
- Position/POI remapping và sai user context là hai failure mode mạnh nhất.
- User conditioning đóng góp lớn hơn target-time conditioning trong kiến trúc hiện tại.
- Missing/wrong context tổng hợp không còn bị diễn giải như một nguyên nhân đơn lẻ nhờ bốn one-axis controls.

## 6. Protocol gate và giới hạn

- Đã đủ 13 variant, seed 42–44, raw metrics và paired predictions.
- Checkpoint đóng băng; không fine-tune hoặc chọn perturbation bằng test.
- Paired bootstrap/sign-flip dùng cùng query và cùng seed; Holm correction áp dụng cho toàn bộ phép kiểm định.
- Target POI/label không đổi trong mọi perturbation.
- `position-noise` dùng nearest-POI remapping trong test coordinates, không phải raw-coordinate encoder.
- `context-missing-time` dùng slot 0 làm proxy; mô hình chưa có learned missing-time token.
- Kết luận hiện chỉ áp dụng cho TIST2015–Tokyo và E5-dual; chưa phải kết quả 12-city hoặc cross-backbone.
