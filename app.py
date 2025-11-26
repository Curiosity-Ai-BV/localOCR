from __future__ import annotations

import io
import json
import time
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import streamlit as st
from PIL import Image

# Set page config (must be first Streamlit command)
st.set_page_config(
    page_title="Curiosity AI Scans",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal theme layer (accent, spacing, subtle surfaces)
st.markdown(
    """
    <style>
    :root { --accent: #6C5CE7; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }
    [data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,0.15); }
    .stButton>button {
        background: var(--accent); color: #fff; border: 1px solid rgba(0,0,0,0.05);
        border-radius: 10px; padding: 0.5rem 0.9rem; font-weight: 600;
    }
    .stButton>button:hover { opacity: .95; }
    .download-panel {
        border: 1px dashed rgba(127,127,127,0.25); border-radius: 12px;
        padding: 0.9rem 1rem; background: rgba(127,127,127,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

from core.pdf_utils import get_pdf_page_count, is_pdf_supported

PDF_SUPPORT = is_pdf_supported()
if not PDF_SUPPORT:
    st.warning("PDF support requires PyMuPDF. Install it with: pip install pymupdf")

###############################################################################
# Constants and Types
###############################################################################

APP_TITLE = "Curiosity AI Scans"
DEFAULT_MAX_IMAGE_SIZE = 1920
DEFAULT_JPEG_QUALITY = 90
DEFAULT_PDF_SCALE = 1.5

JSONDict = Dict[str, Any]


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_results(placeholder, entries: List[Dict[str, Any]], show_images: bool, compact_view: bool) -> None:
    """Render stored OCR results inside the provided placeholder."""
    with placeholder.container():
        st.subheader("Results")
        if not entries:
            st.info("No results yet. Upload files and run a scan.")
            return

        for entry in entries:
            st.markdown(f"#### {entry['filename']}")
            col1, col2 = st.columns([1, 2])
            image_bytes = entry.get("image_bytes")
            if show_images and image_bytes:
                col1.image(
                    image_bytes,
                    width=(180 if compact_view else 250),
                )
            elif show_images:
                col1.empty()
            else:
                col1.empty()

            with col2:
                st.markdown(entry["content"])
                meta_bits: List[str] = []
                duration = entry.get("duration_sec")
                if isinstance(duration, (int, float)):
                    meta_bits.append(f"⏱ {float(duration):.2f} s")
                dims = entry.get("dimensions")
                if isinstance(dims, tuple) and len(dims) == 2 and all(isinstance(x, (int, float)) for x in dims):
                    meta_bits.append(f"{int(dims[0])}×{int(dims[1])} px")
                encoded_bytes = entry.get("encoded_bytes")
                if isinstance(encoded_bytes, (int, float)):
                    meta_bits.append(f"{float(encoded_bytes) / 1024:.1f} KB")
                if meta_bits:
                    st.caption(" • ".join(meta_bits))
            page_note = entry.get("page_note")
            if isinstance(page_note, str) and page_note.strip():
                st.info(page_note.strip())
            structured = entry.get("structured_data")
            if isinstance(structured, dict) and len(structured) > 1:
                with st.expander("Structured JSON"):
                    st.json(structured)
            st.divider()


from adapters.ollama_adapter import ensure_model_available, get_available_models
from core.pipeline import process_image, process_pdf
from ui.export import create_download_buttons
from core.templates import set_templates

# --- MAIN APP UI ---

# Display the app title + caption
st.title(APP_TITLE)
st.caption("Local, private, minimalist vision scanning")

# Initialize session state for storing results
if 'results' not in st.session_state:
    st.session_state.results = []
if 'structured_results' not in st.session_state:
    st.session_state.structured_results = []
if 'display_entries' not in st.session_state:
    st.session_state.display_entries = []

# Create a sidebar for the file upload functionality
with st.sidebar:
    st.subheader("Files")
    uploaded_files = st.file_uploader(
        "Choose images or PDFs", 
        accept_multiple_files=True, 
        type=['png', 'jpg', 'jpeg', 'pdf']
    )
    
    # Model selection
    st.subheader("Model")
    default_models = [
        "gemma3:12b",
        "llama3.2-vision",
        "granite3.2-vision",
        "deepseek-ocr",
        "MHKetbi/Unsloth_gemma3-12b-it:latest",
    ]
    model_options = [
        m for m in get_available_models(default_models)
        if "gpt-oss" not in str(m).lower()
    ]
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

    with st.expander("Appearance", expanded=False):
        compact_view = st.checkbox("Compact results view", value=False)
        show_images = st.checkbox("Show images", value=True)

    # Optional templates and schema paste boxes
    schema_fields: Optional[List[str]] = None
    with st.expander("Templates & Schema", expanded=False):
        templates_json_text = st.text_area(
            "Templates JSON (optional)",
            value="",
            help="JSON with 'description' and 'extraction' keys to override prompts.",
            height=120,
        )
        if templates_json_text.strip():
            try:
                data = json.loads(templates_json_text)
                if isinstance(data, dict):
                    set_templates(
                        description=data.get("description"),
                        extraction=data.get("extraction"),
                    )
                    st.success("Templates loaded")
                else:
                    st.error("Templates must be a JSON object")
            except Exception as e:
                st.error(f"Invalid templates JSON: {e}")

        schema_json_text = st.text_area(
            "Schema JSON (optional)",
            value="",
            help="JSON with a 'fields' array, e.g., {\"fields\":[\"Invoice number\",...]}",
            height=120,
        )
        if schema_json_text.strip():
            try:
                data = json.loads(schema_json_text)
                if isinstance(data, dict) and isinstance(data.get("fields"), list):
                    schema_fields = [str(x).strip() for x in data.get("fields", []) if str(x).strip()]
                    if schema_fields:
                        st.success(f"Loaded {len(schema_fields)} schema field(s)")
                        st.code(", ".join(schema_fields))
                else:
                    st.error("Schema must be an object with a 'fields' array")
            except Exception as e:
                st.error(f"Invalid schema JSON: {e}")

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
        has_pdf_upload = any(file.name.lower().endswith(".pdf") for file in uploaded_files)
        
        # Add option for structured data extraction
        st.subheader("Extraction")
        extraction_mode = st.radio(
            "Choose extraction mode:",
            ["General description", "Custom field extraction"]
        )
        
        # If custom extraction is selected, show field input
        if extraction_mode == "Custom field extraction":
            if schema_fields:
                st.caption("Using fields from schema JSON")
                fields = schema_fields
            else:
                custom_fields = st.text_area(
                    "Enter fields to extract (comma separated):", 
                    value="Invoice number, Date, Company name, Total amount"
                )
                fields = [field.strip() for field in custom_fields.split(",") if field.strip()]

        # Option to process PDF pages separately or as a whole when PDFs are present
        if has_pdf_upload:
            pdf_process_mode = st.radio(
                "How to process PDF files:",
                ["Process each page separately", "Process entire PDF as one document"],
                key="pdf_process_mode",
                help="Choose whether to run OCR on every page or treat the PDF as a single document.",
            )
        
        # Process button in sidebar
        process_button = st.button("Run Scan")
    else:
        st.info("Please upload images or PDF files to analyze")
        process_button = False

results_placeholder = st.empty()

# Main app logic
if uploaded_files and process_button:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Clear previous results when starting a new batch
    st.session_state.results = []
    st.session_state.structured_results = []
    st.session_state.display_entries = []

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
                    page_total = get_pdf_page_count(file_bytes)
                    total_items += page_total
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
                        duration_clean = round(float(elapsed_sec), 3)
                        result['duration_sec'] = duration_clean
                        durations.append(float(elapsed_sec))
                    st.session_state.results.append(result)

                    if structured_data and len(structured_data) > 1:
                        st.session_state.structured_results.append(structured_data)

                    clean_dims: Optional[Tuple[int, int]] = None
                    if isinstance(dims, tuple) and len(dims) == 2 and all(isinstance(x, (int, float)) for x in dims):
                        clean_dims = (int(dims[0]), int(dims[1]))

                    entry: Dict[str, Any] = {
                        "filename": page_filename,
                        "content": content,
                        "duration_sec": result.get("duration_sec"),
                        "dimensions": clean_dims,
                        "encoded_bytes": size_bytes,
                        "structured_data": structured_data if isinstance(structured_data, dict) else None,
                        "image_bytes": _image_to_png_bytes(image) if isinstance(image, Image.Image) else None,
                        "page_note": None,
                    }
                    if page_count and page_count > 1 and not process_separately:
                        entry["page_note"] = f"PDF has {page_count} pages. Showing first page only."

                    st.session_state.display_entries.append(entry)
                    render_results(results_placeholder, st.session_state.display_entries, show_images, compact_view)

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

                entry: Dict[str, Any] = {
                    "filename": f"Image: {uploaded_file.name}",
                    "content": content,
                    "duration_sec": result.get('duration_sec'),
                    "dimensions": (
                        (int(result.get('input_width')), int(result.get('input_height')))
                        if isinstance(result.get('input_width'), int) and isinstance(result.get('input_height'), int)
                        else None
                    ),
                    "encoded_bytes": result.get('encoded_bytes'),
                    "structured_data": structured_data if isinstance(structured_data, dict) else None,
                    "image_bytes": _image_to_png_bytes(image) if isinstance(image, Image.Image) else None,
                    "page_note": None,
                }

                st.session_state.display_entries.append(entry)
                render_results(results_placeholder, st.session_state.display_entries, show_images, compact_view)
                
            except Exception as e:
                st.error(f"Error processing image {uploaded_file.name}: {e}")
            
            processed_count += 1
            progress_bar.progress(processed_count / total_items)
    
    batch_elapsed = time.perf_counter() - batch_start
    status_text.text("Processing complete!")
    if durations:
        avg = sum(durations) / max(len(durations), 1)
        st.info(f"⏱ Total processing time: {batch_elapsed:.2f} s  |  Avg per item: {avg:.2f} s")
else:
    if st.session_state.display_entries:
        render_results(results_placeholder, st.session_state.display_entries, show_images, compact_view)
# Create download buttons outside processing flow so reruns keep data available
if st.session_state.results:
    create_download_buttons(
        st.session_state.results,
        st.session_state.structured_results,
        extraction_mode,
    )

# Display instructions when no files are processed yet
if not uploaded_files and not st.session_state.results:
    st.info("👈 Add files on the left to get started")
    st.write("""
    ## How to use this app:
    1. Upload one or more images or PDF files using the sidebar on the left
    2. Select which vision model to use for analysis
    3. Choose between general description or custom field extraction
    4. If using custom extraction, specify the fields you want to extract
    5. For PDFs, choose whether to process each page separately or the entire document
    6. Click 'Run Scan' to analyze them
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
