"""Sidebar UI component."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import streamlit as st

from core.prompts import PromptConfig
from core.profiles import get_profile
from core.settings import Settings
from .setup_status import (
    PDF_PER_PAGE_MODE,
    build_readiness_items,
    get_pdf_mode_help,
    get_pdf_mode_options,
    render_setup_status,
)

PROFILE_OPTIONS = ("generic", "invoice", "receipt", "table")
BACKEND_OPTIONS = ("ollama", "docling", "hybrid", "auto")
PREPROCESS_PROFILE_DEFAULT = "profile-default"
PREPROCESS_OPTIONS = (
    PREPROCESS_PROFILE_DEFAULT,
    "none",
    "document-clean",
    "high-accuracy-scan",
)
DEFAULT_MODELS = [
    "gemma4:latest",
    "gemma4",
    "gemma4:e4b",
    "gemma4:e2b",
    "gemma4:26b",
    "gemma4:31b",
    "gemma3:12b",
    "llama3.2-vision",
    "granite3.2-vision",
    "deepseek-ocr",
    "MHKetbi/Unsloth_gemma3-12b-it:latest",
]
GENERIC_CUSTOM_FIELDS = "Invoice number, Date, Company name, Total amount"


@dataclass
class SidebarState:
    uploaded_files: List[Any] = field(default_factory=list)
    selected_model: str = ""
    system_prompt: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)
    settings: Settings = field(default_factory=Settings)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    fields: Optional[List[str]] = None
    profile_id: str = "generic"
    ocr_backend: str = "ollama"
    preprocess: Optional[str] = None
    extraction_mode: str = "General description"
    pdf_process_mode: str = PDF_PER_PAGE_MODE
    show_images: bool = True
    compact_view: bool = False
    process_button: bool = False


def _load_local_model_inventory(base_settings: Settings) -> tuple[List[str], bool]:
    from adapters.ollama_adapter import list_models_with_status

    inventory = list_models_with_status(settings=base_settings)
    model_names = [
        str(model.get("name") or model.get("model") or "")
        for model in inventory.models
        if model.get("name") or model.get("model")
    ]
    return model_names, inventory.reachable


def _available_model_options(default_models: List[str], base_settings: Settings) -> List[str]:
    from adapters.ollama_adapter import get_available_models

    return [
        m for m in get_available_models(default_models, settings=base_settings)
        if "gpt-oss" not in str(m).lower()
    ]


def _preprocess_label(option: str) -> str:
    if option == PREPROCESS_PROFILE_DEFAULT:
        return "Profile default"
    return option


def _custom_field_default(profile_id: str) -> str:
    try:
        profile_fields = get_profile(profile_id).fields
    except ValueError:
        profile_fields = []
    if profile_fields:
        return ", ".join(profile_fields)
    return GENERIC_CUSTOM_FIELDS


def render_sidebar(base_settings: Settings, *, pdf_supported: bool = True) -> SidebarState:
    state = SidebarState(settings=base_settings)
    with st.sidebar:
        st.subheader("Files")
        state.uploaded_files = st.file_uploader(
            "Upload images or PDFs",
            accept_multiple_files=True,
            type=["png", "jpg", "jpeg", "pdf"],
        ) or []

        st.subheader("Processing")
        state.profile_id = st.selectbox(
            "Profile",
            PROFILE_OPTIONS,
            index=PROFILE_OPTIONS.index(state.profile_id),
            help="Choose built-in extraction defaults for common document types.",
        )
        state.ocr_backend = st.selectbox(
            "Backend mode",
            BACKEND_OPTIONS,
            index=BACKEND_OPTIONS.index(state.ocr_backend),
            help="Choose the OCR engine route. Docling runs without Ollama for text-only OCR.",
        )
        preprocess_index = (
            0
            if state.preprocess is None
            else PREPROCESS_OPTIONS.index(state.preprocess)
        )
        selected_preprocess = st.selectbox(
            "Preprocessing",
            PREPROCESS_OPTIONS,
            index=preprocess_index,
            format_func=_preprocess_label,
            help="Prepare scans before OCR without changing the source file.",
        )
        state.preprocess = (
            None
            if selected_preprocess == PREPROCESS_PROFILE_DEFAULT
            else selected_preprocess
        )

        st.subheader("Model")
        if state.ocr_backend == "docling":
            local_models: List[str] = []
            ollama_available = False
            model_options = list(DEFAULT_MODELS)
        else:
            local_models, ollama_available = _load_local_model_inventory(base_settings)
            model_options = _available_model_options(DEFAULT_MODELS, base_settings)
        state.selected_model = st.selectbox(
            "Vision model",
            model_options,
            help="Select which AI model to use for image analysis",
        )
        if state.ocr_backend == "docling":
            st.caption("Docling mode skips Ollama model checks.")
        else:
            render_setup_status(
                build_readiness_items(
                    ollama_available=ollama_available,
                    model_names=local_models,
                    selected_model=state.selected_model,
                    pdf_supported=pdf_supported,
                )
            )

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
            max_image_size = st.slider(
                "Max image dimension (px)", 256, 4096, base_settings.max_image_size, 64
            )
            jpeg_quality = st.slider(
                "JPEG quality", 60, 100, base_settings.jpeg_quality, 1
            )
            pdf_scale = st.slider(
                "PDF render scale",
                0.5,
                3.0,
                float(base_settings.pdf_scale),
                0.1,
                help="Higher = more detail but slower; 1.5 is a good default.",
            )

        with st.expander("Appearance", expanded=False):
            state.compact_view = st.checkbox("Compact results view", value=False)
            state.show_images = st.checkbox("Show images", value=True)

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
                        state.prompts = PromptConfig.from_dict(data)
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

        state.system_prompt = system_prompt or None
        state.options = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "num_predict": int(num_predict),
            "num_ctx": int(num_ctx),
        }
        state.settings = base_settings.merged(
            max_image_size=max_image_size,
            jpeg_quality=jpeg_quality,
            pdf_scale=pdf_scale,
        )

        if state.uploaded_files:
            file_label = "file" if len(state.uploaded_files) == 1 else "files"
            st.caption(f"{len(state.uploaded_files)} {file_label} ready")
            has_pdf = any(f.name.lower().endswith(".pdf") for f in state.uploaded_files)

            st.subheader("Extraction")
            state.extraction_mode = st.radio(
                "Extraction mode",
                ["General description", "Custom field extraction"],
            )

            if state.extraction_mode == "Custom field extraction":
                if schema_fields:
                    st.caption("Using fields from schema JSON")
                    state.fields = schema_fields
                else:
                    custom_fields = st.text_area(
                        "Enter fields to extract (comma separated):",
                        value=_custom_field_default(state.profile_id),
                    )
                    state.fields = [f.strip() for f in custom_fields.split(",") if f.strip()]

            if has_pdf:
                state.pdf_process_mode = st.radio(
                    "PDF processing",
                    get_pdf_mode_options(),
                    key="pdf_process_mode",
                    help=get_pdf_mode_help(),
                )

            state.process_button = st.button("Run scan", type="primary", use_container_width=True)
        else:
            st.info("Please upload images or PDF files to analyze")
            state.process_button = False

    return state
