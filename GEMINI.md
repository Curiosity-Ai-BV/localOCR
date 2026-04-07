# Curiosity AI Scans (localOCR)

This project is a local, private AI vision tool that leverages Ollama to analyze images and PDFs. It provides both a web-based UI (Streamlit) and a command-line interface for batch processing.

## Project Overview

*   **Goal:** To provide a secure, local way to extract information from visual documents and images without sending data to the cloud.
*   **Core Technology:** Uses local vision models (like Gemma 3 12B, Llama 3.2 Vision) running via Ollama.
*   **Key Features:**
    *   **General Description:** Generates textual descriptions of images.
    *   **Structured Extraction:** Extracts specific fields (e.g., Invoice Number, Date, Total) into JSON/CSV formats.
    *   **PDF Support:** Renders and processes PDF pages individually or as a document.
    *   **Dual Interface:** Interactive Streamlit UI and headless CLI.

## Architecture

The codebase is modularized into the following components:

*   **`app.py`:** The main entry point for the Streamlit web interface. Handles UI layout, state management, and user interaction.
*   **`cli.py`:** The command-line interface for headless batch processing. Supports concurrency and rate limiting.
*   **`core/`:** Contains the business logic.
    *   `pipeline.py`: Orchestrates the image/PDF processing flow (resize -> encode -> infer).
    *   `image_utils.py`: Handles image resizing and base64 encoding.
    *   `pdf_utils.py`: Manages PDF loading and page rendering using PyMuPDF.
    *   `json_extract.py`: Logic for parsing structured data from model outputs.
    *   `templates.py`: Manages prompt templates.
*   **`adapters/`:** Interfaces with external services.
    *   `ollama_adapter.py`: Handles communication with the local Ollama instance (model listing, chat API).
*   **`ui/`:** Helper modules for the Streamlit UI (e.g., export functionality).
*   **`utils/`:** Shared types and small utility functions.

## Setup and Usage

### Prerequisites

1.  **Python:** 3.9 - 3.12
2.  **Ollama:** Must be installed and running (`ollama serve`).
3.  **Models:** Vision models must be pulled (e.g., `ollama pull gemma3:12b`).

### Installation

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Running the UI

```bash
streamlit run app.py
```
Access the app at `http://localhost:8501`.

### Running the CLI

```bash
python cli.py --model gemma3:12b --mode extract --fields "Date, Total" samples/receipt.png
```

## Development Conventions

*   **Type Hinting:** The project uses Python type hints extensively (`typing` module).
*   **Code Style:** Follows standard Python conventions (PEP 8).
*   **Testing:** Unit tests are located in `tests/` and can be run with `pytest`.
*   **Modular Design:** Logic should remain separated from the presentation layer (UI/CLI). Core logic resides in `core/`.

## Key Dependencies

*   `streamlit`: Web framework.
*   `ollama`: Python client for Ollama.
*   `pillow` (PIL): Image processing.
*   `pymupdf`: PDF rendering.
