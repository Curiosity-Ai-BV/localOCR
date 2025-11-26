## Thank you to everyone for starring my repo! I'll do my best to extend the functionality regularly and fix things if people find problems.

# Curiosity AI Scans

A streamlined Streamlit app that uses local AI vision models (via Ollama) to analyze images and PDFs. Upload multiple files, choose a model, get detailed descriptions or extract structured fields, and export the results to CSV.

## What’s New (Single‑file, High‑quality Refresh)

- Robust JSON extraction from model outputs (fenced blocks, brace scanning, and heuristics)
- Advanced model options in the sidebar (temperature, top‑p, max tokens, context length)
- Optional system prompt to steer model behavior
- Adjustable image resize and JPEG quality for better performance and output
- PDF render scale control for PDFs (affects page → image resolution)
- Model availability check with actionable guidance when models are missing
- Clearer errors and progress feedback
- Processing time display per item and batch (see how settings affect latency)
- Minimalist, modern UI refresh with compact mode and a cleaner export panel

Additionally, each item now shows the actual model input size (WxH) and the encoded JPEG size in KB, so you can confirm preprocessing is applied before inference.

## What this application does

- Upload multiple images (JPG, PNG) and PDF documents
- Choose Gemma 3 12B, Llama 3.2 Vision, Granite 3.2 Vision, or your own local model
- Get detailed descriptions or extract custom fields (invoice no., dates, amounts, etc.)
- Process PDF files page-by-page or as a single document
- Export results as standard CSV and structured CSV (for extraction mode)

The app uses Streamlit for the interface, Ollama for local model serving, Pillow for image processing, and PyMuPDF for PDF pages. Everything remains in a single file for simplicity while meeting high code-quality standards.

## Installation and setup

### Step 1: Install Ollama

#### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### macOS
```bash
brew install ollama
# Or download from https://ollama.com/download
```

#### Windows
1. Download the installer from https://ollama.com/download
2. Run the installer and follow the instructions

### Step 2: Pull a vision model

```bash
# Gemma 3 Vision
ollama pull gemma3:12b

# Llama 3.2 Vision
ollama pull llama3.2-vision

# Granite 3.2 Vision (smaller footprint)
ollama pull granite3.2-vision

# DeepSeek OCR
ollama pull deepseek-ocr
```

Pull one or more — the app works with whichever you have installed.

### Step 3: Python environment

Use Python 3.9–3.12 for best compatibility.

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# macOS/Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# Windows (CMD)
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

## Running the application

1. Start Ollama if not already running
   ```bash
   ollama serve
   ```
   - Windows: Ollama typically runs as a service after installation. If you get connection errors, run the command above in a new terminal.
2. Launch the app
   ```bash
   streamlit run app.py
   ```
3. Open your browser to http://localhost:8501 if it doesn’t auto‑open.

## CLI usage (headless)

You can now process files without the UI:

```bash
python cli.py \
  --model gemma3:12b \
  --mode extract \
  --fields "Invoice number, Date, Total amount" \
  --pdf-pages \
  --pdf-scale 1.5 \
  --templates templates.json \
  --schema schema.json \
  --max-concurrency 2 \
  --rate-limit 0.5 \
  --out-results results.csv \
  --out-structured structured.csv \
  samples/invoice1.pdf samples/receipt.png
```

- `--mode description|extract`: general description vs extracting specific fields
- `--fields`: comma‑separated field names (for extract mode)
- `--pdf-pages`: split PDFs so each page is processed and exported separately
- `--pdf-scale`: adjust render DPI before OCR (higher values sharpen text but take longer)
- `--max-concurrency`: number of files to process in parallel
- `--rate-limit`: requests per second (0 = unlimited)
- Also available: temperature, top‑p, tokens, context, max image size, JPEG quality, and PDF scale

Templates JSON example:

```json
{
  "description": "Describe the image focusing on text and layout.",
  "extraction": "Extract these fields from the image: {fields}. Return strict JSON."
}
```

Schema JSON example:

```json
{
  "fields": ["Invoice number", "Date", "Company name", "Total amount"]
}
```

## Features

- Multiple file uploads (images and PDFs)
- General description or custom field extraction
- Advanced Model Options:
  - Temperature, top‑p, max tokens (`num_predict`), context length (`num_ctx`)
  - System prompt (optional)
- Adjustable image resize and JPEG quality
- PDF render scale (pre‑rendering DPI via scale multiplier)
- PDF processing per page or first page only
- CSV export for both general and structured results
- Processing time shown under each item and as a batch summary
- Appearance controls: compact results view and show/hide thumbnails
- Headless CLI with optional concurrency and rate limiting

## Design Language

The app follows a minimalist, contemporary design that emphasizes clarity and progressive disclosure. Primary actions use a single accent color; advanced settings live in collapsible panels; results are easy to scan with soft dividers and compact metadata.

- Two‑pane layout: inputs in the sidebar, results in the main area
- Accent color for primary actions only; otherwise neutral surfaces
- Card‑like result grouping with clear captions for time, size, dimensions
- JSON details shown inside a collapsible expander to reduce noise
- Optional compact mode and ability to hide thumbnails

See design.md for the full design language.

## Project structure

After modularization, the repo is organized as:

- `core/` — image/PDF processing and extraction pipeline
- `adapters/` — external service adapters (Ollama)
- `ui/` — Streamlit UI helpers (export panel)
- `utils/` — shared types and small utilities
- `cli.py` — Headless batch processor
- `tests/` — Unit tests for JSON extraction and PDF conversion

Run tests with pytest:

```bash
pytest -q
```

## Advanced model options

Open the “Advanced Model Options” expander in the sidebar to configure:

- System prompt: steer the model with an instruction
- Temperature and top‑p: control creativity and sampling
- Max tokens (`num_predict`): cap the number of generated tokens
- Context length (`num_ctx`): increase when prompts + images are large
- Max image dimension and JPEG quality: balance speed and fidelity
- PDF render scale: changes the PDF page rasterization resolution before resizing

### Appearance

- Compact results view: condenses spacing and uses smaller thumbnails
- Show images: toggle thumbnails on/off in results

## Performance tips

- The largest impact on latency typically comes from generation length. Reduce “Max tokens (`num_predict`)” for faster responses.
- For PDFs, lowering the PDF render scale can significantly reduce pixels processed.
- Lower “Max image dimension (px)” reduces pixels; quality mostly affects encoded file size and decode cost (smaller effect than pixels or tokens).
- If running on CPU, expect slower times. GPU acceleration (where available) and quantized models often help.

## Troubleshooting

- Ollama not running: start with `ollama serve`
- Model not found: pull it with `ollama pull <model_name>`. The app tries to detect installed models and will proceed even if it can’t confirm; failures will include a concrete error message.
- PDF support missing: install PyMuPDF — `pip install pymupdf`
- Python compatibility: prefer Python 3.9–3.12
- Long or complex prompts: if hitting context limits, increase `num_ctx`

---

Made with ❤️ by Adrian with GPT-5-Codex — [ad1x.com](https://ad1x.com)
