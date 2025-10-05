from __future__ import annotations

import base64
import csv
import io
import json
import re
from datetime import datetime
import time
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import ollama
import streamlit as st
from PIL import Image

# Set page config (must be first Streamlit command)
st.set_page_config(
    page_title="Curiosity AI Scans",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Try to import PyMuPDF for PDF support
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    st.warning("PDF support requires PyMuPDF. Install it with: pip install pymupdf")

###############################################################################
# Constants and Types
###############################################################################

APP_TITLE = "Curiosity AI Scans"
DEFAULT_MAX_IMAGE_SIZE = 1920
DEFAULT_JPEG_QUALITY = 90
DEFAULT_PDF_SCALE = 1.5

JSONDict = Dict[str, Any]


###############################################################################
# Helper functions
###############################################################################

def resize_image(image: Image.Image, max_size: int = DEFAULT_MAX_IMAGE_SIZE) -> Image.Image:
    """Resize an image while maintaining aspect ratio.

    Args:
        image: PIL Image.
        max_size: Maximum width or height in pixels.

    Returns:
        Resized PIL Image (or the original if no resize needed).
    """
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image

    if width > height:
        new_width = max_size
        new_height = max(1, int(height * (max_size / max(width, 1))))
    else:
        new_height = max_size
        new_width = max(1, int(width * (max_size / max(height, 1))))

    return image.resize((new_width, new_height), Image.LANCZOS)


def image_to_base64(image: Image.Image, quality: int = DEFAULT_JPEG_QUALITY) -> Tuple[str, int]:
    """Convert PIL Image to base64 encoded JPEG string.

    Returns (base64_string, encoded_byte_size). Ensures RGB mode and applies JPEG quality.
    """
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG", quality=int(quality), optimize=True)
    raw = img_byte_arr.getvalue()
    return base64.b64encode(raw).decode("utf-8"), len(raw)


def ensure_model_available(model: str) -> Tuple[bool, Optional[str]]:
    """Best‑effort check for model availability in Ollama.

    Uses `ollama.show(model)` first (most accurate), then falls back to
    `ollama.list()` with tolerant matching (case‑insensitive and tag‑agnostic).
    Never raises; returns (available, warning_message_if_any).
    """
    try:
        # Most reliable: resolves aliases and tags if present
        _ = ollama.show(model)
        return True, None
    except Exception:
        pass

    try:
        listing = ollama.list()
        raw_models = listing.get("models", [])

        def norm(s: str) -> str:
            return s.strip().lower()

        def base(s: str) -> str:
            # split on last ':' only (repo paths may contain ':')
            parts = s.rsplit(":", 1)
            return parts[0] if len(parts) == 2 else s

        target = norm(model)
        target_base = base(target)

        candidates: List[str] = []
        for m in raw_models:
            for key in ("name", "model"):
                val = m.get(key)
                if isinstance(val, str):
                    candidates.append(norm(val))

        # Exact or base/tag‑agnostic matches
        for c in candidates:
            if c == target:
                return True, None
            if base(c) == target_base:
                return True, None
            if c.startswith(target) or target.startswith(c):
                return True, None

        # Not found in list, but we won't hard‑block; just warn
        return False, "Model not detected from Ollama list; proceeding anyway."
    except Exception as e:
        # If list fails (daemon down etc.), be permissive
        return True, f"Could not verify model availability: {e}"


def query_ollama(
    prompt: str,
    image_base64: str,
    model: str,
    *,
    options: Optional[JSONDict] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Query Ollama with an image and prompt, returning model content.

    Raises RuntimeError on failure.
    """
    messages: List[JSONDict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": prompt,
        "images": [image_base64],
    })

    try:
        response = ollama.chat(model=model, messages=messages, options=options or {})
        content = response.get("message", {}).get("content", "")
        if not isinstance(content, str):
            raise RuntimeError("Unexpected response content type from model")
        return content
    except Exception as e:
        raise RuntimeError(f"Ollama chat failed: {e}")


def get_available_models(defaults: Optional[List[str]] = None) -> List[str]:
    """Return a list of available model names, preferring locally installed.

    If listing fails, fall back to provided defaults. Duplicates are removed,
    preserving the order of installed first, then defaults.
    """
    defaults = defaults or []
    ordered: List[str] = []
    seen: set[str] = set()
    # Try to list installed models
    try:
        listing = ollama.list()
        for m in listing.get("models", []):
            name = m.get("name") or m.get("model")
            if isinstance(name, str) and name not in seen:
                ordered.append(name)
                seen.add(name)
    except Exception:
        pass
    # Add defaults after
    for d in defaults:
        if d not in seen:
            ordered.append(d)
            seen.add(d)
    return ordered or defaults


def _extract_json_from_fenced_blocks(text: str) -> List[JSONDict]:
    """Extract JSON objects from ```json fenced code blocks."""
    results: List[JSONDict] = []
    for match in re.finditer(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE):
        candidate = match.group(1)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                results.append(obj)
        except Exception:
            continue
    return results


def _extract_json_by_brace_scanning(text: str) -> List[JSONDict]:
    """Extract JSON objects by scanning for balanced braces.

    Note: This is a heuristic and does not fully parse JSON strings with nested
    braces inside quoted strings. It works well for typical LLM outputs.
    """
    results: List[JSONDict] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{" and depth == 0:
            start = i
            depth = 1
        elif ch == "{" and depth > 0:
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                snippet = text[start : i + 1]
                try:
                    obj = json.loads(snippet)
                    if isinstance(obj, dict):
                        results.append(obj)
                except Exception:
                    pass
                start = -1
    return results


def _extract_field_heuristics(text: str, fields: Iterable[str]) -> JSONDict:
    """Heuristic key/value extraction for requested fields from plain text."""
    data: JSONDict = {}
    for field in fields:
        f = field.strip()
        if not f:
            continue
        # Simple patterns: Field: "value" | Field: value | Field = value
        pattern = rf"(?i)\b{re.escape(f)}\b\s*[:=\-]\s*(\".*?\"|'.*?'|[\w\-./$%,]+)"
        m = re.search(pattern, text)
        if m:
            val = m.group(1)
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            data[f] = val
    return data


def extract_structured_data(content: str, fields: Optional[List[str]]) -> JSONDict:
    """Extract structured data from model content.

    Tries fenced blocks, then brace scanning, then heuristics if fields given.
    """
    fields = fields or []
    # 1) Fenced blocks
    for obj in _extract_json_from_fenced_blocks(content):
        return obj
    # 2) Brace scan
    objs = _extract_json_by_brace_scanning(content)
    if objs:
        # Prefer an object that contains any field requested
        if fields:
            for o in objs:
                if any(f in o for f in fields):
                    return o
        return objs[0]
    # 3) Heuristics
    if fields:
        return _extract_field_heuristics(content, fields)
    return {}


def process_image(
    image: Image.Image,
    filename: str,
    fields: Optional[List[str]] = None,
    *,
    model: str,
    system_prompt: Optional[str],
    options: Optional[JSONDict],
    max_image_size: int,
    jpeg_quality: int,
) -> Tuple[JSONDict, str, Optional[JSONDict]]:
    """Process a single image with optional field extraction.

    Returns a tuple of (result_dict, raw_content, structured_dict_or_none).
    The result dict includes a 'duration_sec' key for UI timing display.
    """
    t0 = time.perf_counter()
    prepared = resize_image(image, max_image_size)
    img_base64, encoded_size = image_to_base64(prepared, jpeg_quality)

    if not fields:
        prompt = "Describe what you see in this image in detail."
        content = query_ollama(prompt, img_base64, model, options=options, system_prompt=system_prompt)
        elapsed = time.perf_counter() - t0
        return {
            "filename": filename,
            "description": content,
            "duration_sec": round(elapsed, 3),
            "input_width": prepared.size[0],
            "input_height": prepared.size[1],
            "encoded_bytes": int(encoded_size),
        }, content, None
    else:
        fields_str = ", ".join(fields)
        prompt = (
            f"Extract the following information from this image: {fields_str}. "
            f"Return the results in JSON format with these exact field names."
        )
        content = query_ollama(prompt, img_base64, model, options=options, system_prompt=system_prompt)
        structured_data: JSONDict = {"filename": filename}
        parsed = extract_structured_data(content, fields)
        structured_data.update(parsed)
        elapsed = time.perf_counter() - t0
        return {
            "filename": filename,
            "extraction": content,
            "duration_sec": round(elapsed, 3),
            "input_width": prepared.size[0],
            "input_height": prepared.size[1],
            "encoded_bytes": int(encoded_size),
        }, content, structured_data


def process_pdf(
    file_bytes: bytes,
    filename: str,
    fields: Optional[List[str]] = None,
    process_pages_separately: bool = True,
    *,
    model: str,
    system_prompt: Optional[str],
    options: Optional[JSONDict],
    max_image_size: int,
    jpeg_quality: int,
    pdf_scale: float,
) -> Generator[Tuple[Optional[int], Optional[int], Optional[Image.Image], str, str, Optional[JSONDict], Optional[float], Optional[Tuple[int,int]], Optional[int]], None, None]:
    """Process a PDF file using PyMuPDF, yielding page-level results."""
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        page_count = len(pdf_document)

        if process_pages_separately:
            for page_num in range(page_count):
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(pdf_scale, pdf_scale))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_filename = f"{filename} (Page {page_num + 1})"

                result, content, structured_data = process_image(
                    img,
                    page_filename,
                    fields,
                    model=model,
                    system_prompt=system_prompt,
                    options=options,
                    max_image_size=max_image_size,
                    jpeg_quality=jpeg_quality,
                )
                elapsed = result.get("duration_sec") if isinstance(result, dict) else None
                dims = (result.get("input_width"), result.get("input_height")) if isinstance(result, dict) else None
                size_bytes = result.get("encoded_bytes") if isinstance(result, dict) else None
                yield page_num, page_count, img, page_filename, content, structured_data, elapsed, dims, size_bytes
        else:
            page = pdf_document[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(pdf_scale, pdf_scale))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            result, content, structured_data = process_image(
                img,
                filename,
                fields,
                model=model,
                system_prompt=system_prompt,
                options=options,
                max_image_size=max_image_size,
                jpeg_quality=jpeg_quality,
            )
            elapsed = result.get("duration_sec") if isinstance(result, dict) else None
            dims = (result.get("input_width"), result.get("input_height")) if isinstance(result, dict) else None
            size_bytes = result.get("encoded_bytes") if isinstance(result, dict) else None
            yield 0, page_count, img, filename, content, structured_data, elapsed, dims, size_bytes
    except Exception as e:
        yield None, None, None, filename, f"Error processing PDF: {str(e)}", None, None, None, None


def create_download_buttons(results: List[JSONDict], structured_results: List[JSONDict], extraction_mode: str) -> None:
    """Create and display download buttons for results."""
    st.header("Download Results")

    csv_data = io.StringIO()
    csv_writer = csv.writer(csv_data)

    if extraction_mode == "General description" or not structured_results:
        csv_writer.writerow(["Filename", "Description"])
        for result in results:
            csv_writer.writerow([result["filename"], result.get("description", result.get("extraction", ""))])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"image_analysis_{timestamp}.csv"

        st.success("All files have been processed successfully!")
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data.getvalue(),
            file_name=csv_filename,
            mime="text/csv",
            use_container_width=True,
        )

    if extraction_mode == "Custom field extraction" and structured_results:
        all_fields = set(["filename"])
        for result in structured_results:
            all_fields.update(result.keys())

        field_list = sorted(list(all_fields))
        structured_csv = io.StringIO()
        structured_writer = csv.writer(structured_csv)
        structured_writer.writerow(field_list)

        for result in structured_results:
            row = [result.get(field, "") for field in field_list]
            structured_writer.writerow(row)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        structured_filename = f"structured_data_{timestamp}.csv"

        st.success("Structured data extracted successfully!")
        st.download_button(
            label="📥 Download Structured Data as CSV",
            data=structured_csv.getvalue(),
            file_name=structured_filename,
            mime="text/csv",
            use_container_width=True,
        )

# --- MAIN APP UI ---

# Display the app title
st.title(APP_TITLE)

# Initialize session state for storing results
if 'results' not in st.session_state:
    st.session_state.results = []
if 'structured_results' not in st.session_state:
    st.session_state.structured_results = []

# Create a sidebar for the file upload functionality
with st.sidebar:
    st.header("Upload Files")
    uploaded_files = st.file_uploader(
        "Choose images or PDFs", 
        accept_multiple_files=True, 
        type=['png', 'jpg', 'jpeg', 'pdf']
    )
    
    # Model selection
    st.header("Model Settings")
    default_models = [
        "gemma3:12b",
        "llama3.2-vision",
        "granite3.2-vision",
        "MHKetbi/Unsloth_gemma3-12b-it:latest",
    ]
    model_options = get_available_models(default_models)
    selected_model = st.selectbox(
        "Choose vision model:",
        model_options,
        help="Select which AI model to use for image analysis",
    )

    # Advanced model options
    with st.expander("Advanced Model Options", expanded=False):
        system_prompt = st.text_area(
            "System prompt (optional)",
            value="",
            help="Steer the model's behavior with a system instruction.",
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.05)
        num_predict = st.slider("Max tokens to generate", 64, 4096, 512, 64)
        num_ctx = st.slider("Context length (num_ctx)", 1024, 8192, 4096, 256)

        st.caption("Image settings")
        max_image_size = st.slider("Max image dimension (px)", 256, 4096, DEFAULT_MAX_IMAGE_SIZE, 64)
        jpeg_quality = st.slider("JPEG quality", 60, 100, DEFAULT_JPEG_QUALITY, 1)
        pdf_scale = st.slider("PDF render scale", 0.5, 3.0, DEFAULT_PDF_SCALE, 0.1, help="Affects PDF → image resolution before model input")

    options = {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "num_predict": int(num_predict),
        "num_ctx": int(num_ctx),
    }
    
    extraction_mode = "General description"
    pdf_process_mode = "Process each page separately"
    fields = None
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} files")
        
        # Add option for structured data extraction
        st.header("Data Extraction Options")
        extraction_mode = st.radio(
            "Choose extraction mode:",
            ["General description", "Custom field extraction"]
        )
        
        # If custom extraction is selected, show field input
        if extraction_mode == "Custom field extraction":
            custom_fields = st.text_area(
                "Enter fields to extract (comma separated):", 
                value="Invoice number, Date, Company name, Total amount"
            )
            fields = [field.strip() for field in custom_fields.split(",")]
            
            # Option to process PDF pages separately or as a whole
            if any(file.name.lower().endswith('.pdf') for file in uploaded_files):
                pdf_process_mode = st.radio(
                    "How to process PDF files:",
                    ["Process each page separately", "Process entire PDF as one document"]
                )
        
        # Process button in sidebar
        process_button = st.button("Process Files")
    else:
        st.info("Please upload images or PDF files to analyze")
        process_button = False

# Main app logic
if uploaded_files and process_button:
    st.header("Processing Results")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Clear previous results when starting a new batch
    st.session_state.results = []
    st.session_state.structured_results = []

    # Verify model availability
    available, availability_note = ensure_model_available(selected_model)
    if availability_note:
        st.warning(availability_note)
    if not available:
        st.info(
            f"Could not confirm model '{selected_model}' locally. Attempting anyway. "
            f"If you see failures, run: `ollama pull {selected_model}`."
        )
    
    # Count total items to process (including PDF pages)
    total_items = 0
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer after reading
        
        if uploaded_file.name.lower().endswith('.pdf') and PDF_SUPPORT:
            if pdf_process_mode == "Process each page separately":
                try:
                    pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
                    total_items += len(pdf_document)
                except Exception as e:
                    st.error(f"Error checking PDF {uploaded_file.name}: {e}")
                    total_items += 1
            else:
                total_items += 1
        else:
            total_items += 1
    
    processed_count = 0
    durations: List[float] = []
    batch_start = time.perf_counter()
    
    # Process each file
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        
        # Handle PDF files
        if uploaded_file.name.lower().endswith('.pdf'):
            if not PDF_SUPPORT:
                st.error(f"Cannot process PDF file {uploaded_file.name}. Please install PyMuPDF library.")
                processed_count += 1
                progress_bar.progress(processed_count / total_items)
                continue
                
            try:
                process_separately = pdf_process_mode == "Process each page separately"
                
                for page_info in process_pdf(
                    file_bytes,
                    uploaded_file.name,
                    fields,
                    process_separately,
                    model=selected_model,
                    system_prompt=system_prompt or None,
                    options=options,
                    max_image_size=max_image_size,
                    jpeg_quality=jpeg_quality,
                    pdf_scale=pdf_scale,
                ):
                    page_num, page_count, image, page_filename, content, structured_data, elapsed_sec, dims, size_bytes = page_info
                    
                    if page_num is None:  # Error case
                        st.error(content)
                        continue
                    
                    status_text.text(f"Processing {page_filename} ({page_num+1}/{page_count})")
                    
                    # Add to session state
                    result = {'filename': page_filename, 'description': content}
                    if isinstance(elapsed_sec, (int, float)):
                        result['duration_sec'] = round(float(elapsed_sec), 3)
                        durations.append(float(elapsed_sec))
                    st.session_state.results.append(result)
                    
                    if structured_data and len(structured_data) > 1:
                        st.session_state.structured_results.append(structured_data)
                    
                    # Display the processed image and its results
                    st.subheader(page_filename)
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(image, width=250)
                        if page_count > 1 and not process_separately:
                            st.info(f"PDF has {page_count} pages. Showing first page only.")
                    with col2:
                        st.write(content)
                        meta_bits = []
                        if isinstance(elapsed_sec, (int, float)):
                            meta_bits.append(f"⏱ {float(elapsed_sec):.2f} s")
                        if dims and all(isinstance(x, (int, float)) for x in dims):
                            meta_bits.append(f"{int(dims[0])}×{int(dims[1])} px")
                        if isinstance(size_bytes, int):
                            meta_bits.append(f"{size_bytes/1024:.1f} KB")
                        if meta_bits:
                            st.caption(" • ".join(meta_bits))
                        if structured_data and len(structured_data) > 1:
                            st.success("Successfully extracted structured data")
                            st.json(structured_data)
                    
                    st.divider()
                    
                    processed_count += 1
                    progress_bar.progress(min(processed_count / total_items, 1.0))
                    
            except Exception as e:
                st.error(f"Error processing PDF {uploaded_file.name}: {e}")
                processed_count += 1
                progress_bar.progress(processed_count / total_items)
        
        else:
            # Process regular image file
            status_text.text(f"Processing image {uploaded_file.name}")
            
            try:
                image = Image.open(uploaded_file)
                result, content, structured_data = process_image(
                    image,
                    uploaded_file.name,
                    fields,
                    model=selected_model,
                    system_prompt=system_prompt or None,
                    options=options,
                    max_image_size=max_image_size,
                    jpeg_quality=jpeg_quality,
                )
                st.session_state.results.append(result)
                if isinstance(result, dict) and isinstance(result.get('duration_sec'), (int, float)):
                    durations.append(float(result['duration_sec']))
                
                if structured_data and len(structured_data) > 1:
                    st.session_state.structured_results.append(structured_data)
                
                # Display the processed image and its results
                st.subheader(f"Image: {uploaded_file.name}")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image, width=250)
                with col2:
                    st.write(content)
                    meta_bits = []
                    if isinstance(result.get('duration_sec'), (int, float)):
                        meta_bits.append(f"⏱ {float(result['duration_sec']):.2f} s")
                    if isinstance(result.get('input_width'), int) and isinstance(result.get('input_height'), int):
                        meta_bits.append(f"{result['input_width']}×{result['input_height']} px")
                    if isinstance(result.get('encoded_bytes'), int):
                        meta_bits.append(f"{result['encoded_bytes']/1024:.1f} KB")
                    if meta_bits:
                        st.caption(" • ".join(meta_bits))
                    if structured_data and len(structured_data) > 1:
                        st.success("Successfully extracted structured data")
                        st.json(structured_data)
                
                st.divider()
                
            except Exception as e:
                st.error(f"Error processing image {uploaded_file.name}: {e}")
            
            processed_count += 1
            progress_bar.progress(processed_count / total_items)
    
    batch_elapsed = time.perf_counter() - batch_start
    status_text.text("Processing complete!")
    if durations:
        avg = sum(durations) / max(len(durations), 1)
        st.info(f"⏱ Total processing time: {batch_elapsed:.2f} s  |  Avg per item: {avg:.2f} s")
    
    # Create download buttons
    if st.session_state.results:
        create_download_buttons(
            st.session_state.results, 
            st.session_state.structured_results, 
            extraction_mode
        )

# Display instructions when no files are processed yet
if not uploaded_files:
    st.info("👈 Upload files using the sidebar to get started")
    st.write("""
    ## How to use this app:
    1. Upload one or more images or PDF files using the sidebar on the left
    2. Select which vision model to use for analysis
    3. Choose between general description or custom field extraction
    4. If using custom extraction, specify the fields you want to extract
    5. For PDFs, choose whether to process each page separately or the entire document
    6. Click the 'Process Files' button to analyze them
    7. View the results for each image or PDF page
    8. Download results as a CSV file
    
    This app uses either the Gemma 3 12B vision model or Llama 3.2 Vision model to analyze images and PDFs.
    """)

# Add a footer with attribution
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; margin-top: 20px; opacity: 0.7;">
        Made with ❤️ by Adrian with GPT-5 - <a href="https://ad1x.com" target="_blank">ad1x.com</a>
    </div>
    """, 
    unsafe_allow_html=True
)
