.PHONY: all clean

all:
	python3 build_rq_reports.py
	latexmk -g -xelatex main.tex

clean:
	latexmk -C
