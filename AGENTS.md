# Repository Guidelines

## Project Structure & Module Organization
Application entry points live in `app.py` (Streamlit UI) and `cli.py` (headless runs). Processing primitives sit under `core/` (`pipeline.py`, `json_extract.py`, `image_utils.py`, `templates.py`) with model bindings in `adapters/ollama_adapter.py`. UI helpers reside in `ui/export.py`, shared typings in `utils/typing.py`, and reusable assets in `assets/`. Test data and reference documents live in `samples/`, while automated tests live in `tests/`.

## Build, Test, and Development Commands
Create a virtual environment and install runtime deps:
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
Install tooling for contribution work:
```bash
pip install -r requirements-dev.txt
```
Run the Streamlit app locally with `streamlit run app.py`. Start the CLI batch processor via `python cli.py --help` to explore available flags — prime examples include `--pdf-pages` to fan out multi-page PDFs and `--pdf-scale` for render DPI. Execute automated tests using `pytest`.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation and keep functions small and composable, mirroring the current module layout. Use descriptive, snake_case names for functions and variables; reserve PascalCase for classes and TypedDicts. Maintain type hints and docstrings where already present, and prefer helper functions within `core/` over duplicating logic in `app.py`. When formatting sizable changes, run `ruff format` or `black` if already configured in your environment; otherwise stay consistent with existing whitespace.

## Testing Guidelines
Place new unit tests in `tests/`, mirroring the module under test (`test_json_extract.py`, `test_pdf_convert.py` provide patterns). Name tests with `test_<feature>` for pytest discovery. Ensure new OCR or extraction behaviors include assertions for both success paths and failure handling. Run `pytest -q` before submitting; target full pass on macOS with Ollama available, and guard tests so they gracefully skip when optional services are missing.

## Commit & Pull Request Guidelines
Match the existing history by writing imperative, concise subject lines (e.g., “Add PDF scale control”). Include contextual details in the body when touching multiple modules. Reference related issues and add screenshots or CLI transcripts when UI or output changes. PRs should summarize intent, list validation steps (`pytest`, manual Streamlit run), and call out any Ollama model prerequisites so reviewers can reproduce results quickly.
