from pathlib import Path


def test_github_actions_evaluation_uses_service_free_mock_mode() -> None:
    workflow = Path(".github/workflows/eval.yml").read_text(encoding="utf-8")

    assert "python evaluate.py --allow-mock" in workflow
    assert "ollama serve" not in workflow
    assert "ollama pull" not in workflow
    assert "ollama.com/install.sh" not in workflow
