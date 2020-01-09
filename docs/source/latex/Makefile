# -*- mode: makefile; -*-
files_tex  = $(wildcard *.tex)
files_pdf  = $(files_tex:.tex=.pdf)

%.pdf: %.tex
	@echo "NOTICE: Exporting $< to pdf..."
	@pdflatex -shell-escape $<

all: pdf

pdf: $(files_pdf)

clean:
	@rm -rf *.pdf *.toc *.fdb* *~ *.log *.aux *.out _minted*

.PHONY: all pdf clean
