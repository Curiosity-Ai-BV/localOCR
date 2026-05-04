from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import requests

try:
    from cli import _make_inference, RateLimiter
    from core.export import write_evidence
    from core.extraction import build_field_evidence
    from core.models import OCRBlock, Result
    from core.pipeline import BatchConfig, run_batch
    from core.preprocessing import expected_preprocess_steps
except ImportError:
    print("Error: Must run evaluate.py from project root where cli.py exists.")
    sys.exit(1)


JSONDict = Dict[str, Any]
GroundTruth = Dict[str, Dict[str, Any]]
GroundTruthEntry = tuple[str, Dict[str, Any]]
CountStats = Dict[str, int]

METRICS_SCHEMA_VERSION = "localocr.eval.metrics.v1"
OLLAMA_REQUIRED_BACKENDS = {"ollama", "hybrid", "auto"}
EMPTY_GROUND_TRUTH_REASON = "empty_ground_truth"


def is_ollama_running() -> bool:
    try:
        requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    normalized = str(text).lower()
    return re.sub(r"[^a-z0-9]", "", normalized)


def calculate_accuracy(extracted: Any, ground_truth: Any) -> bool:
    e_norm = normalize_text(extracted)
    g_norm = normalize_text(ground_truth)
    if not g_norm:
        return False
    if not e_norm:
        return False
    return g_norm in e_norm


def _has_empty_ground_truth(value: Any) -> bool:
    return not normalize_text(value)


def create_chart(metrics: Dict[str, float], output_path: str) -> None:
    fields = list(metrics.keys())
    accuracies = [metrics[f] * 100 for f in fields]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(fields, accuracies, color=["#4C72B0", "#55A868", "#C44E52"])

    plt.title("OCR Retrieval Accuracy by Field", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 110)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 2,
            f"{yval:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Chart saved to {output_path}")


def update_readme(metrics: Dict[str, float], chart_path: str) -> None:
    readme_path = Path("README.MD")
    if not readme_path.exists():
        return

    content = readme_path.read_text(encoding="utf-8")

    markdown_results = "## Automated Evaluation Results\n\n"
    markdown_results += (
        "This project includes an automated evaluation pipeline validating "
        "extraction accuracy against a ground truth dataset of invoices.\n\n"
    )
    markdown_results += "### Accuracy by Field\n\n"
    markdown_results += "| Field | Accuracy |\n"
    markdown_results += "|-------|----------|\n"
    for field, acc in metrics.items():
        markdown_results += f"| {field} | {acc * 100:.1f}% |\n"
    markdown_results += "\n"
    markdown_results += f"![Evaluation Chart]({chart_path})\n\n"
    markdown_results += (
        "*(Run `python evaluate.py --update-readme` to regenerate these "
        "README metrics and chart)*\n\n"
    )

    start_marker = "## Automated Evaluation Results"
    if start_marker in content:
        new_content = re.sub(
            r"## Automated Evaluation Results.*?(?=\n## |\Z)",
            markdown_results.strip(),
            content,
            flags=re.DOTALL,
        )
        readme_path.write_text(new_content, encoding="utf-8")
    else:
        readme_path.write_text(content + "\n" + markdown_results, encoding="utf-8")


def _backend_may_need_ollama(backend: str) -> bool:
    return backend.strip().lower() in OLLAMA_REQUIRED_BACKENDS


def _load_ground_truth(path: Path) -> GroundTruth:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Ground truth must be a non-empty JSON object.")

    ground_truth: GroundTruth = {}
    for filename, fields in raw.items():
        if not isinstance(fields, dict):
            raise ValueError(f"Ground truth row for {filename!r} must be an object.")
        ground_truth[str(filename)] = {
            str(field): value
            for field, value in fields.items()
        }
    return ground_truth


def _fields_from_ground_truth(ground_truth: GroundTruth) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for expected in ground_truth.values():
        for field in expected:
            if field in seen:
                continue
            seen.add(field)
            fields.append(field)
    return fields


def _select_entries(ground_truth: GroundTruth, max_images: int) -> list[GroundTruthEntry]:
    entries = list(ground_truth.items())
    if max_images > 0:
        return entries[:max_images]
    return entries


def _mock_field_matches(filename: str, field: str, index: int) -> bool:
    digest = hashlib.sha256(f"{filename}:{field}:{index}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10 != 0


def generate_mock_results(
    ground_truth: GroundTruth,
    *,
    model: str,
    ocr_backend: str,
    profile_id: str,
    preprocess: Optional[str],
    max_images: int = 0,
) -> list[Result]:
    """Return deterministic Result objects for service-free evaluation runs."""
    print("Warning: Ollama is not running. Using mock results for demonstration.")
    fields = _fields_from_ground_truth(ground_truth)
    cfg = BatchConfig(
        model=model,
        fields=fields,
        ocr_backend=ocr_backend,
        profile_id=profile_id,
        preprocess=preprocess,
    )
    preprocess_steps = expected_preprocess_steps(cfg.preprocess or "none")
    results: list[Result] = []

    for row_index, (filename, expected) in enumerate(_select_entries(ground_truth, max_images)):
        extracted: JSONDict = {}
        for field_index, (field, value) in enumerate(expected.items()):
            if _mock_field_matches(filename, field, row_index + field_index):
                extracted[field] = value
            else:
                extracted[field] = "MISSING"

        raw_content = json.dumps(extracted, ensure_ascii=False, sort_keys=True)
        ocr_text = "; ".join(
            f"{field}: {expected[field]}"
            for field in sorted(expected)
        )
        results.append(
            Result(
                source=filename,
                mode="extract",
                text=raw_content,
                raw=raw_content,
                fields=extracted,
                ocr_text=ocr_text,
                ocr_blocks=[OCRBlock(text=ocr_text, kind="mock")],
                field_evidence=build_field_evidence(
                    requested_fields=expected.keys(),
                    parsed_fields=extracted,
                    raw_content=ocr_text,
                    engine=cfg.ocr_backend,
                ),
                engine=cfg.ocr_backend,
                profile_id=cfg.profile_id,
                preprocess_steps=preprocess_steps,
                backend_note="deterministic mock evaluation result",
                latency_ms=0,
            )
        )
    return results


def _missing_file_result(filename: str, path: Path, cfg: BatchConfig) -> Result:
    return Result(
        source=filename,
        mode="extract",
        text="",
        fields={},
        error=f"Input file not found: {path}",
        engine=cfg.ocr_backend,
        profile_id=cfg.profile_id,
        preprocess_steps=expected_preprocess_steps(cfg.preprocess or "none"),
    )


def _run_pipeline_evaluation(
    entries: Iterable[GroundTruthEntry],
    dataset_dir: Path,
    cfg: BatchConfig,
) -> list[Result]:
    paths: list[str] = []
    missing_results: list[Result] = []
    for filename, _ in entries:
        path = dataset_dir / filename
        if path.exists():
            paths.append(str(path))
        else:
            print(f"Warning: Dataset file {path} not found; counting as failed.")
            missing_results.append(_missing_file_result(filename, path, cfg))

    results = list(run_batch(paths, cfg)) if paths else []
    results.extend(missing_results)
    return results


def _index_results(results: Iterable[Result]) -> dict[str, Result]:
    indexed: dict[str, Result] = {}
    for result in results:
        indexed[result.source] = result
        indexed[os.path.basename(result.source)] = result
    return indexed


def _empty_count_stats() -> CountStats:
    return {"correct": 0, "total": 0}


def _add_count(stats: CountStats, match: bool) -> None:
    stats["total"] += 1
    if match:
        stats["correct"] += 1


def _finalize_count_stats(stats: CountStats) -> JSONDict:
    total = stats["total"]
    correct = stats["correct"]
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def _finalize_group_metrics(
    groups: dict[str, dict[str, CountStats]],
) -> dict[str, dict[str, JSONDict]]:
    return {
        group_name: {
            key: _finalize_count_stats(stats)
            for key, stats in sorted(group_values.items())
        }
        for group_name, group_values in groups.items()
    }


def _write_detailed_csv(rows: list[JSONDict], path: str) -> None:
    fieldnames = [
        "Filename",
        "Field",
        "Expected",
        "Extracted",
        "Match",
        "Ignored",
        "Ignore Reason",
        "Engine",
        "Profile",
        "Preprocess",
        "Error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Filename": row["source"],
                    "Field": row["field"],
                    "Expected": row["expected"],
                    "Extracted": row["extracted"],
                    "Match": "" if row["match"] is None else str(row["match"]),
                    "Ignored": str(row["ignored"]),
                    "Ignore Reason": row["ignore_reason"] or "",
                    "Engine": row["engine"],
                    "Profile": row["profile_id"],
                    "Preprocess": row["preprocess"],
                    "Error": row["error"] or "",
                }
            )


def _build_metrics_payload(
    *,
    entries: list[GroundTruthEntry],
    results: list[Result],
    cfg: BatchConfig,
    dataset_dir: Path,
    ground_truth_file: str,
    max_images: int,
    mock: bool,
    evidence_path: Optional[str],
) -> JSONDict:
    result_by_source = _index_results(results)
    fields = _fields_from_ground_truth(dict(entries))
    field_metrics = {field: _empty_count_stats() for field in fields}
    aggregate = _empty_count_stats()
    group_counts: dict[str, dict[str, CountStats]] = {
        "backend": {},
        "profile": {},
        "preprocess": {},
    }
    detailed_rows: list[JSONDict] = []
    preprocess = cfg.preprocess or "none"
    ignored_total = 0
    ignored_by_reason: dict[str, int] = {}
    ignored_by_field: dict[str, int] = {}

    for filename, expected in entries:
        result = result_by_source.get(filename) or Result(
            source=filename,
            mode="extract",
            text="",
            fields={},
            error="No result returned for dataset file.",
            engine=cfg.ocr_backend,
            profile_id=cfg.profile_id,
            preprocess_steps=expected_preprocess_steps(preprocess),
        )
        backend_key = result.engine or "unknown"
        profile_key = result.profile_id or "unknown"

        for field, true_val in expected.items():
            extracted_val = result.fields.get(field, "") if result.fields else ""
            if _has_empty_ground_truth(true_val):
                ignored_total += 1
                ignored_by_reason[EMPTY_GROUND_TRUTH_REASON] = (
                    ignored_by_reason.get(EMPTY_GROUND_TRUTH_REASON, 0) + 1
                )
                ignored_by_field[field] = ignored_by_field.get(field, 0) + 1
                detailed_rows.append(
                    {
                        "source": filename,
                        "result_source": result.source,
                        "field": field,
                        "expected": true_val,
                        "extracted": extracted_val,
                        "match": None,
                        "ignored": True,
                        "ignore_reason": EMPTY_GROUND_TRUTH_REASON,
                        "engine": result.engine,
                        "profile_id": result.profile_id,
                        "preprocess": preprocess,
                        "preprocess_steps": list(result.preprocess_steps),
                        "backend_note": result.backend_note,
                        "error": result.error,
                    }
                )
                continue

            is_match = calculate_accuracy(extracted_val, true_val)
            _add_count(field_metrics.setdefault(field, _empty_count_stats()), is_match)
            _add_count(aggregate, is_match)
            for group_name, key in (
                ("backend", backend_key),
                ("profile", profile_key),
                ("preprocess", preprocess),
            ):
                group = group_counts[group_name].setdefault(key, _empty_count_stats())
                _add_count(group, is_match)

            detailed_rows.append(
                {
                    "source": filename,
                    "result_source": result.source,
                    "field": field,
                    "expected": true_val,
                    "extracted": extracted_val,
                    "match": is_match,
                    "ignored": False,
                    "ignore_reason": None,
                    "engine": result.engine,
                    "profile_id": result.profile_id,
                    "preprocess": preprocess,
                    "preprocess_steps": list(result.preprocess_steps),
                    "backend_note": result.backend_note,
                    "error": result.error,
                }
            )

    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "run": {
            "dataset": str(dataset_dir),
            "ground_truth": ground_truth_file,
            "model": cfg.model,
            "ocr_backend": cfg.ocr_backend,
            "profile_id": cfg.profile_id,
            "preprocess": preprocess,
            "mock": mock,
            "max_images": max_images,
            "evaluated_files": len(entries),
            "total_results": len(results),
            "error_results": sum(1 for result in results if result.error),
            "evidence_path": evidence_path,
        },
        "field_accuracy": {
            field: _finalize_count_stats(stats)
            for field, stats in field_metrics.items()
        },
        "aggregate": _finalize_count_stats(aggregate),
        "group_metrics": _finalize_group_metrics(group_counts),
        "ignored": {
            "total": ignored_total,
            "by_reason": dict(sorted(ignored_by_reason.items())),
            "by_field": dict(sorted(ignored_by_field.items())),
        },
        "detailed_rows": detailed_rows,
    }


def _write_metrics_json(payload: JSONDict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Metrics JSON saved to {path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate localOCR against ground truth.")
    parser.add_argument(
        "--dataset",
        default="eval_dataset",
        help="Path to the dataset directory containing images.",
    )
    parser.add_argument(
        "--ground-truth",
        default="ground_truth.json",
        help="Name of the ground truth file inside the dataset folder.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("EVAL_MODEL", "gemma3:12b"),
        help="Model to use (defaults to EVAL_MODEL env or gemma3:12b).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=int(os.environ.get("EVAL_MAX_IMAGES", "0")),
        help="Max images to evaluate. 0 means all.",
    )
    parser.add_argument(
        "--ocr-backend",
        choices=["ollama", "docling", "hybrid", "auto"],
        default="ollama",
        help="OCR backend path to evaluate.",
    )
    parser.add_argument(
        "--profile",
        choices=["generic", "invoice", "receipt", "table"],
        default="generic",
        help="Document profile to evaluate.",
    )
    parser.add_argument(
        "--preprocess",
        choices=["none", "document-clean", "high-accuracy-scan"],
        default=None,
        help="Preprocessing preset. Omit to use the selected profile default.",
    )
    parser.add_argument(
        "--out-evidence",
        default="",
        help="Optional path to write detailed OCR evidence JSON.",
    )
    parser.add_argument(
        "--metrics-json",
        default="",
        help="Optional path to write deterministic evaluation metrics JSON.",
    )
    parser.add_argument(
        "--chart-output",
        default="eval_results.png",
        help="Path for the field accuracy chart.",
    )
    parser.add_argument(
        "--detailed-csv",
        default="eval_detailed_results.csv",
        help="Path for the detailed field comparison CSV.",
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Fall back to deterministic mock results if Ollama is not running.",
    )
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Exit non-zero when any evaluated file returns an OCR error.",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Opt in to updating README.MD with the computed chart and metrics.",
    )
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="Deprecated compatibility flag. README updates are opt-in by default.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset)
    gt_path = dataset_dir / args.ground_truth
    if not gt_path.exists():
        print(f"Ground truth file {gt_path} not found.", file=sys.stderr)
        return 1

    try:
        ground_truth = _load_ground_truth(gt_path)
    except ValueError as exc:
        print(f"Invalid ground truth file {gt_path}: {exc}", file=sys.stderr)
        return 1

    fields_to_extract = _fields_from_ground_truth(ground_truth)
    entries = _select_entries(ground_truth, args.max_images)
    if not entries:
        print("No ground truth rows selected for evaluation.", file=sys.stderr)
        return 1

    options = {"temperature": 0.0, "top_p": 1.0}
    system_prompt = None
    cfg = BatchConfig(
        model=args.model,
        fields=fields_to_extract,
        system_prompt=system_prompt,
        options=options,
        pdf_pages_separately=False,
        ocr_backend=args.ocr_backend,
        profile_id=args.profile,
        preprocess=args.preprocess,
    )

    use_mock = False
    if _backend_may_need_ollama(args.ocr_backend) and not is_ollama_running():
        if not args.allow_mock and os.environ.get("CI") is None:
            print(
                "Error: Ollama is not running. Re-run with --allow-mock to use "
                "deterministic mock data (README will not be updated).",
                file=sys.stderr,
            )
            return 1
        use_mock = True

    if use_mock:
        results = generate_mock_results(
            ground_truth,
            model=args.model,
            ocr_backend=args.ocr_backend,
            profile_id=args.profile,
            preprocess=args.preprocess,
            max_images=args.max_images,
        )
        cfg = BatchConfig(
            model=args.model,
            fields=fields_to_extract,
            system_prompt=system_prompt,
            options=options,
            pdf_pages_separately=False,
            ocr_backend=args.ocr_backend,
            profile_id=args.profile,
            preprocess=args.preprocess,
        )
    else:
        limiter = RateLimiter(None)
        cfg.inference = _make_inference(options, system_prompt, limiter)
        results = _run_pipeline_evaluation(entries, dataset_dir, cfg)

    evidence_path = args.out_evidence or None
    if evidence_path:
        write_evidence(results, evidence_path)
        print(f"Evidence JSON saved to {evidence_path}")

    metrics_payload = _build_metrics_payload(
        entries=entries,
        results=results,
        cfg=cfg,
        dataset_dir=dataset_dir,
        ground_truth_file=args.ground_truth,
        max_images=args.max_images,
        mock=use_mock,
        evidence_path=evidence_path,
    )
    final_metrics = {
        field: stats["accuracy"]
        for field, stats in metrics_payload["field_accuracy"].items()
        if stats["total"] > 0
    }

    for field, accuracy in final_metrics.items():
        print(f"Accuracy for {field}: {accuracy * 100:.1f}%")

    create_chart(final_metrics, args.chart_output)

    detailed_rows = metrics_payload["detailed_rows"]
    _write_detailed_csv(detailed_rows, args.detailed_csv)
    print(f"Detailed comparison saved to {args.detailed_csv}")

    if args.metrics_json:
        _write_metrics_json(metrics_payload, args.metrics_json)

    if args.update_readme and not args.no_readme and not use_mock:
        update_readme(final_metrics, args.chart_output)
    else:
        print("Skipping README update.")

    error_results = metrics_payload["run"]["error_results"]
    if args.fail_on_errors and error_results:
        print(
            f"Evaluation failed: {error_results} result(s) returned OCR errors.",
            file=sys.stderr,
        )
        return 1

    print("Evaluation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
