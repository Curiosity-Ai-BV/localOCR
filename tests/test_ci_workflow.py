from pathlib import Path


def test_github_actions_evaluation_keeps_live_ollama_gate() -> None:
    workflow = Path(".github/workflows/eval.yml").read_text(encoding="utf-8")

    assert "curl -fsSL https://ollama.com/install.sh | sh" in workflow
    assert "ollama serve" in workflow
    assert "ollama pull granite3.2-vision" in workflow
    assert "LOCALOCR_REQUEST_TIMEOUT: 360" in workflow
    assert "python evaluate.py --fail-on-errors" in workflow
    assert "--allow-mock" not in workflow
