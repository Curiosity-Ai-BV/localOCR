from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from adapters.ollama_adapter import ensure_model_available, query_ollama
from core.pdf_utils import PDFNotSupportedError, ensure_pdf_support, iter_pdf_pages
from core.pipeline import process_image
from core.templates import load_templates_file


class RateLimiter:
    def __init__(self, per_second: Optional[float] = None):
        self.per_second = per_second
        self._last = 0.0

    def wait(self) -> None:
        if not self.per_second or self.per_second <= 0:
            return
        now = time.monotonic()
        min_interval = 1.0 / float(self.per_second)
        delta = now - self._last
        if delta < min_interval:
            time.sleep(min_interval - delta)
        self._last = time.monotonic()


def _make_inference(options: Dict, system_prompt: Optional[str], limiter: RateLimiter) -> Callable[[str, str, str], str]:
    def _inner(prompt: str, image_b64: str, model: str) -> str:
        limiter.wait()
        return query_ollama(prompt, image_b64, model, options=options, system_prompt=system_prompt)
    return _inner


def _process_file(
    path: str,
    *,
    fields: Optional[List[str]],
    model: str,
    system_prompt: Optional[str],
    options: Dict,
    max_image_size: int,
    jpeg_quality: int,
    pdf_scale: float,
    pdf_pages: bool,
    inference: Callable[[str, str, str], str],
) -> Tuple[List[Dict], List[Dict]]:
    results: List[Dict] = []
    structured: List[Dict] = []
    filename = os.path.basename(path)
    try:
        if path.lower().endswith(".pdf"):
            ensure_pdf_support()
            with open(path, "rb") as f:
                file_bytes = f.read()

            def _handle_page(page_img: Image.Image, page_name: str) -> None:
                result, content, structured_data = process_image(
                    page_img,
                    page_name,
                    fields,
                    model=model,
                    system_prompt=system_prompt,
                    options=options,
                    max_image_size=max_image_size,
                    jpeg_quality=jpeg_quality,
                    inference=inference,
                )
                entry: Dict = {
                    "filename": page_name,
                    "description": content if fields is None else result.get("extraction", content),
                }
                if isinstance(result.get("duration_sec"), (int, float)):
                    entry["duration_sec"] = float(result["duration_sec"])
                results.append(entry)
                if structured_data and len(structured_data) > 1:
                    structured.append(structured_data)

            if pdf_pages:
                for page_index, _, img in iter_pdf_pages(file_bytes, scale=pdf_scale):
                    page_name = f"{filename} (Page {page_index + 1})"
                    _handle_page(img, page_name)
            else:
                first_entry = next(iter_pdf_pages(file_bytes, scale=pdf_scale), None)
                if first_entry is None:
                    raise RuntimeError("PDF contains no renderable pages.")
                _, _, img = first_entry
                _handle_page(img, filename)
        else:
            image = Image.open(path)
            result, content, structured_data = process_image(
                image,
                filename,
                fields,
                model=model,
                system_prompt=system_prompt,
                options=options,
                max_image_size=max_image_size,
                jpeg_quality=jpeg_quality,
                inference=inference,
            )
            results.append(result)
            if structured_data and len(structured_data) > 1:
                structured.append(structured_data)
    except PDFNotSupportedError as e:
        results.append({"filename": filename, "description": f"Error: {e}"})
    except Exception as e:
        results.append({"filename": filename, "description": f"Error: {e}"})
    return results, structured


def write_csv(results: List[Dict], path: str) -> None:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Filename", "Description"])
    for r in results:
        w.writerow([r.get("filename", ""), r.get("description", r.get("extraction", ""))])
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(buf.getvalue())


def write_structured(structured: List[Dict], path: str) -> None:
    if not structured:
        return
    all_fields = set(["filename"])
    for r in structured:
        all_fields.update(r.keys())
    field_list = sorted(all_fields)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(field_list)
    for r in structured:
        w.writerow([r.get(f, "") for f in field_list])
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(buf.getvalue())


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Headless OCR/Vision batch scans via Ollama")
    p.add_argument("files", nargs="+", help="Image or PDF files to process")
    p.add_argument("--model", default="gemma3:12b", help="Model name")
    p.add_argument("--mode", choices=["description", "extract"], default="description")
    p.add_argument("--fields", default="", help="Comma-separated fields for extract mode")
    p.add_argument("--system-prompt", default="", help="Optional system prompt")
    p.add_argument("--templates", default="", help="Path to JSON templates with 'description' and 'extraction'")
    p.add_argument("--schema", default="", help="Path to JSON file with {\"fields\": [..]} for extract mode")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--num-predict", type=int, default=512)
    p.add_argument("--num-ctx", type=int, default=4096)
    p.add_argument("--max-image-size", type=int, default=1920)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--pdf-scale", type=float, default=1.5)
    p.add_argument("--pdf-pages", action="store_true", help="Process each PDF page separately")
    p.add_argument("--out-results", default="results.csv")
    p.add_argument("--out-structured", default="structured.csv")
    p.add_argument("--max-concurrency", type=int, default=1)
    p.add_argument("--rate-limit", type=float, default=0.0, help="Requests per second (0 = unlimited)")

    args = p.parse_args(argv)

    files = [f for f in args.files if os.path.exists(f)]
    if not files:
        print("No valid files provided.", file=sys.stderr)
        return 2

    options = {
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "num_predict": int(args.num_predict),
        "num_ctx": int(args.num_ctx),
    }

    if args.templates:
        try:
            load_templates_file(args.templates)
        except Exception as e:
            print(f"[!] Failed to load templates: {e}")

    fields = None
    if args.mode == "extract":
        if args.schema:
            try:
                import json
                with open(args.schema, "r", encoding="utf-8") as f:
                    data = json.load(f)
                fields = [str(x) for x in data.get("fields", []) if str(x).strip()]
            except Exception as e:
                print(f"[!] Failed to load schema: {e}")
        if not fields:
            fields = [s.strip() for s in args.fields.split(",") if s.strip()]
    system_prompt = args.system_prompt or None

    available, note = ensure_model_available(args.model)
    if note:
        print(f"[!] {note}")
    if not available:
        print(f"[i] Could not confirm model '{args.model}'. Attempting anyway.")

    limiter = RateLimiter(args.rate_limit if args.rate_limit > 0 else None)
    inference = _make_inference(options, system_prompt, limiter)

    all_results: List[Dict] = []
    all_structured: List[Dict] = []

    start = time.perf_counter()
    if args.max_concurrency <= 1:
        for fpath in files:
            r, s = _process_file(
                fpath,
                fields=fields,
                model=args.model,
                system_prompt=system_prompt,
                options=options,
                max_image_size=args.max_image_size,
                jpeg_quality=args.jpeg_quality,
                pdf_scale=args.pdf_scale,
                pdf_pages=args.pdf_pages,
                inference=inference,
            )
            all_results.extend(r)
            all_structured.extend(s)
    else:
        with ThreadPoolExecutor(max_workers=args.max_concurrency) as ex:
            futs = [
                ex.submit(
                    _process_file,
                    fpath,
                    fields=fields,
                    model=args.model,
                    system_prompt=system_prompt,
                    options=options,
                    max_image_size=args.max_image_size,
                    jpeg_quality=args.jpeg_quality,
                    pdf_scale=args.pdf_scale,
                    pdf_pages=args.pdf_pages,
                    inference=inference,
                )
                for fpath in files
            ]
            for fut in as_completed(futs):
                r, s = fut.result()
                all_results.extend(r)
                all_structured.extend(s)

    elapsed = time.perf_counter() - start
    print(f"Processed {len(files)} file(s) in {elapsed:.2f}s")

    write_csv(all_results, args.out_results)
    if all_structured:
        write_structured(all_structured, args.out_structured)
        print(f"Wrote: {args.out_results} and {args.out_structured}")
    else:
        print(f"Wrote: {args.out_results}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
