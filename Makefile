.PHONY: test lint type deps-check check eval install run

PYTHON = $(shell if [ -f venv/bin/python ]; then echo "venv/bin/python"; else echo "python3"; fi)

install:
	$(PYTHON) -m pip install -c constraints.txt -r requirements.txt
	$(PYTHON) -m pip install -c constraints.txt -r requirements-dev.txt

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

type:
	$(PYTHON) -m mypy .

deps-check:
	$(PYTHON) -m pip check

check: deps-check lint type test

eval:
	@echo "Ensuring Ollama is running..."
	@curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1 || (echo "Local Ollama engine not found. Booting 'ollama serve' in background..." && ollama serve >/dev/null 2>&1 & sleep 3)
	@echo "Running local OCR evaluations using the default model (gemma4)..."
	$(PYTHON) evaluate.py --model gemma4

run:
	$(PYTHON) -m streamlit run app.py
