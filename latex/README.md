# Bản thảo Springer — BeliefMove-Evo

Tệp chính: `main.tex`. Bản thảo dùng lớp Springer LNCS (`llncs.cls`) có sẵn trong TeX Live và XeLaTeX để hiển thị tiếng Việt.

Biên dịch:

```bash
cd latex
latexmk -xelatex main.tex
```

Xóa tệp trung gian:

```bash
latexmk -C
```

Nguồn số liệu được tổng hợp từ `ideas/results_rq1.md` đến `ideas/results_rq13.md`, `ideas/report_summary.md` và `ideas/results_www2019.md`. Không sửa số liệu trực tiếp trong PDF; cập nhật báo cáo nguồn rồi đồng bộ lại các bảng trong `main.tex`.
