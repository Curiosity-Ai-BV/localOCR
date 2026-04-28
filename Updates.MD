## 2026-04-28

- Refreshed the Streamlit UI with a shared theme layer, accessible focus states, 44px controls, reduced-motion safeguards, result cards, status badges, and a compact empty state.
- Polished sidebar wording, setup readiness badges, and the primary scan action while preserving the existing upload, OCR, preview, structured JSON, and export flows.
- Added focused UI regression tests for theme safeguards and result metadata/status formatting.

## 2026-04-28

- Completed the mypy cleanup for PDF rendering, image resizing, Ollama request payloads, pipeline result metadata, and CLI result modes.
- Added a `make type` target and wired it into `make check` so CI runs mypy with the existing check workflow.
- Added `types-requests` to the development dependency set and constraints so evaluation imports are typed reproducibly.

## 2026-04-28

- Added `constraints.txt` as the reproducible install layer for the verified runtime, development, and stable transitive dependency versions.
- Updated CI to run `make check` across Python 3.10, 3.11, 3.12, and 3.13 while keeping the Ollama evaluation on a single Python 3.11 job.
- Routed CI and `make install` dependency installation through `constraints.txt`, and documented the constrained install path in the README.

## 2026-04-28

- Wired runtime `Settings` into Ollama adapter model listing and request execution.
- Added injectable model-list TTL support through `Settings.model_list_ttl` while preserving existing default cache behavior.
- Added timeout-aware adapter paths for Ollama `list`, `show`, chat, and streaming chat calls via explicit `timeout` or `Settings.request_timeout`.
- Passed `BatchConfig.settings` through the default pipeline inference path so Streamlit and CLI batch runs use the configured adapter timeout.
- Added regression tests for settings-driven TTL, timeout propagation, and pipeline settings forwarding without requiring a live Ollama service.

## 2026-04-28

- Added CLI agent mode with `--json` and `--quiet` for a single machine-readable stdout summary.
- Preserved deterministic CLI result ordering while reporting missing input paths as explicit error rows in JSON and CSV outputs.
- Added focused CLI tests for quiet JSON output, missing-input rows, and all-missing nonzero exit codes.

## 2026-04-28

- Replaced the misleading PDF sidebar option with truthful first-page wording while preserving the existing processing behavior.
- Added a compact sidebar setup check for Ollama reachability, local vision models, and PDF support.
- Replaced the long empty-state instruction block with a short upload/model/run prompt.
- Added focused helper tests for the PDF copy contract and setup readiness summary.
