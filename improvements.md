# Curiosity AI Scans — Improvement Plan

This plan outlines targeted improvements to functionality, reliability, and code quality while keeping the application as a single file for now. The goals are to deliver a more robust user experience, make the code easier to maintain, and prepare for a future modular split without breaking behavior.

## Objectives

- Increase reliability and correctness (especially JSON extraction and Ollama calls)
- Improve UX configuration and feedback without clutter
- Raise code quality (typing, docs, structure, error handling) while staying single‑file
- Maintain backward compatibility with current workflows

## Current State — Observations

- Solid Streamlit UI with multiple file types and PDF support
- Simple sequential processing with progress UI
- Naive JSON extraction can miss valid payloads or parse incorrect text
- No configurable model options (temperature, context length)
- Minimal error handling for model availability and request failures
- Image handling always JPEG, fixed max size/quality, no controls
- Functions lack type hints and structured results; CSV export is fine but rigid

## Functional Improvements (Now)

- Robust JSON extraction
  - Parse fenced ```json blocks first
  - Fallback to scanning for balanced braces and validating with `json.loads`
  - Final fallback to heuristic key-value extraction for requested fields

- Model options and prompts
  - Add advanced options: `temperature`, `top_p`, `max_tokens`, `num_ctx`
  - Optional system prompt to steer model behavior
  - Validate and pass options to `ollama.chat`

- Model availability check
  - Check `ollama.list()` for selected model; surface actionable warning if missing with pull command

- Image handling controls
  - Configurable resize max dimension and JPEG quality
  - Safe conversion from `RGBA`/`P` to `RGB` before JPEG encoding

- Error handling and feedback
  - Clear error messages for Ollama failures and PDF issues
  - Maintain progress updates and never crash the batch on a single failure

## Code Quality Improvements (Now)

- Add type hints, docstrings, and constants for defaults
- Introduce small data structures (simple typed dicts) to maintain compatibility
- Centralize Ollama call with error handling and options
- Extract helper utilities for image conversions and JSON parsing
- Keep functions small, pure where possible; avoid UI state in helpers

## UX Improvements (Now)

- Sidebar section “Advanced Model Options” (collapsed by default)
- Sliders for temperature, max tokens, context length, resize size, and JPEG quality
- System prompt text area
- Keep the main flow unchanged to avoid confusion

## Performance Considerations

- Resize images client-side before sending to the model
- Allow adjusting render scale for PDFs (still default to 1.5x)
- Avoid heavy caching due to varying prompts and models

## Export Improvements

- Preserve current CSV export
- When structured extraction is used, export a normalized CSV with all discovered fields

## Future Roadmap (After single-file phase)

- Modularize into packages: `ui/`, `core/`, `adapters/`, `utils/`
- Add tests for JSON extraction and PDF conversion helpers
- Add CLI mode for headless batch processing
- Add async/concurrent processing with rate controls and queueing
- Pluggable prompts/templates and field schemas

## Acceptance Criteria for this iteration

- App remains single file and runs with `streamlit run app.py`
- Advanced options available and passed to Ollama
- JSON extraction succeeds with fenced and unfenced blocks
- Reasonable errors when model is missing or request fails
- Code is typed, documented, and organized for maintainability

