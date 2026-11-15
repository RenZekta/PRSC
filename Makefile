.PHONY: test validate-example paper clean

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

validate-example:
	PYTHONPATH=src python -m prsc.validate examples/knowledge-gram.example.json

paper:
	cd paper && pandoc -f markdown+tex_math_single_backslash -s paper.md -o paper.tex --resource-path=. -V fontsize=10pt -V geometry:margin=0.85in -V colorlinks=true
	cd paper && pandoc -f markdown+tex_math_single_backslash -s paper.md -o paper.pdf --pdf-engine=xelatex --resource-path=. -V fontsize=10pt -V geometry:margin=0.85in -V colorlinks=true

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
