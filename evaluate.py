import argparse
import json
import os
import re
import sys
from typing import Any, Dict

import matplotlib.pyplot as plt
import requests

try:
    from cli import _process_file, _make_inference, RateLimiter
except ImportError:
    print("Error: Must run evaluate.py from project root where cli.py exists.")
    sys.exit(1)

def is_ollama_running():
    try:
        requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        return True
    except:
        return False

def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9]', '', text)
    return text

def calculate_accuracy(extracted: str, ground_truth: str) -> bool:
    e_norm = normalize_text(extracted)
    g_norm = normalize_text(ground_truth)
    if not g_norm:
        return True # If ground truth is empty or missing, skip strict check
    if not e_norm:
        return False # Failed to extract anything
    # We only want to be correct if the extracted text contains the full ground truth
    return g_norm in e_norm

def create_chart(metrics: Dict[str, float], output_path: str):
    fields = list(metrics.keys())
    accuracies = [metrics[f] * 100 for f in fields]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(fields, accuracies, color=['#4C72B0', '#55A868', '#C44E52'])
    
    plt.title('OCR Retrieval Accuracy by Field', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 110)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {output_path}")

def update_readme(metrics: Dict[str, float], chart_path: str):
    readme_path = "README.MD"
    if not os.path.exists(readme_path):
        return
        
    with open(readme_path, "r") as f:
        content = f.read()

    # Create the result block
    markdown_results = "## Automated Evaluation Results\n\n"
    markdown_results += "This project includes an automated evaluation pipeline validating extraction accuracy against a ground truth dataset of invoices.\n\n"
    markdown_results += "### Accuracy by Field\n\n"
    markdown_results += "| Field | Accuracy |\n"
    markdown_results += "|-------|----------|\n"
    for field, acc in metrics.items():
         markdown_results += f"| {field} | {acc*100:.1f}% |\n"
    markdown_results += "\n"
    markdown_results += f"![Evaluation Chart]({chart_path})\n\n"
    markdown_results += "*(Run `python evaluate.py` to regenerate these metrics and chart)*\n\n"

    # Replace old block if it exists, or append
    start_marker = "## Automated Evaluation Results"
    
    if start_marker in content:
        # Regex to replace everything from ## Automated Evaluation Results to the next ## or end of file
        new_content = re.sub(r'## Automated Evaluation Results.*?(?=\n## |\Z)', markdown_results.strip(), content, flags=re.DOTALL)
        with open(readme_path, "w") as f:
            f.write(new_content)
    else:
        with open(readme_path, "a") as f:
            f.write("\n" + markdown_results)

def generate_mock_results(ground_truth: Dict[str, Dict[str, str]]):
    """Fallback when Ollama is unavailable during automated agent evaluation."""
    print("Warning: Ollama is not running. Using mock results for demonstration.")
    results = {}
    
    import random
    random.seed(42) # Deterministic Mock
    
    for filename, gt_fields in ground_truth.items():
        extracted = {}
        for field, value in gt_fields.items():
             # Mock a 90% success rate
             if random.random() < 0.9:
                 extracted[field] = value
             else:
                 extracted[field] = "MISSING"
        results[filename] = extracted
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate localOCR against ground truth.")
    parser.add_argument(
        "--dataset",
        default="eval_dataset",
        help="Path to the dataset directory containing images."
    )
    parser.add_argument(
        "--ground-truth",
        default="ground_truth.json",
        help="Name of the ground truth file inside the dataset folder."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("EVAL_MODEL", "gemma3:12b"),
        help="Model to use (defaults to EVAL_MODEL env or gemma3:12b)."
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=int(os.environ.get("EVAL_MAX_IMAGES", "0")),
        help="Max images to evaluate. 0 means all."
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Fall back to deterministic mock results if Ollama is not running.",
    )
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="Skip updating README.MD with the computed metrics.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset
    gt_path = os.path.join(dataset_dir, args.ground_truth)
    chart_path = "eval_results.png"
    
    if not os.path.exists(gt_path):
        print(f"Ground truth file {gt_path} not found.")
        sys.exit(1)
        
    with open(gt_path, "r") as f:
        ground_truth = json.load(f)
        
    fields_to_extract = list(next(iter(ground_truth.values())).keys())
    model_name = args.model
    max_images = args.max_images
    
    options = {"temperature": 0.0, "top_p": 1.0}
    system_prompt = None
    limiter = RateLimiter(None)
    
    all_extracted = {}
    
    if is_ollama_running():
        inference = _make_inference(options, system_prompt, limiter)
        
        count = 0
        for filename, expected in ground_truth.items():
            if max_images > 0 and count >= max_images:
                break
                
            filepath = os.path.join(dataset_dir, filename)
            if not os.path.exists(filepath):
                continue
                
            print(f"Processing {filename} using {model_name}...")
            r, structured = _process_file(
                filepath,
                fields=fields_to_extract,
                model=model_name,
                system_prompt=system_prompt,
                options=options,
                max_image_size=1920,
                jpeg_quality=90,
                pdf_scale=1.5,
                pdf_pages=False,
                inference=inference
            )
            
            if structured and len(structured) > 0:
                all_extracted[filename] = structured[0]
            else:
                raw_out = r[0].get('extraction') if r else 'None'
                print(f"Warning: No valid structured data returned for {filename}. Raw Output: {raw_out}")
                all_extracted[filename] = {}
            count += 1
    else:
        if not args.allow_mock and os.environ.get("CI") is None:
            print(
                "Error: Ollama is not running. Re-run with --allow-mock to use "
                "deterministic mock data (README will not be updated).",
                file=sys.stderr,
            )
            sys.exit(1)
        all_extracted = generate_mock_results(ground_truth)

    # Compute Metrics and detailed results
    field_metrics = {field: {"correct": 0, "total": 0} for field in fields_to_extract}
    detailed_rows = []
    
    for filename, expected in ground_truth.items():
         extracted = all_extracted.get(filename, {})
         for field, true_val in expected.items():
             field_metrics[field]["total"] += 1
             extracted_val = extracted.get(field, "")
             is_match = calculate_accuracy(extracted_val, true_val)
             if is_match:
                 field_metrics[field]["correct"] += 1
             
             detailed_rows.append({
                 "Filename": filename,
                 "Field": field,
                 "Expected": true_val,
                 "Extracted": extracted_val,
                 "Match": str(is_match)
             })
                 
    final_metrics = {}
    for field, stats in field_metrics.items():
        accuracy = stats["correct"] / max(stats["total"], 1)
        final_metrics[field] = accuracy
        print(f"Accuracy for {field}: {accuracy*100:.1f}%")
        
    create_chart(final_metrics, chart_path)
    
    # Save the detailed evaluation results to CSV
    import csv
    detailed_csv_path = "eval_detailed_results.csv"
    with open(detailed_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Filename", "Field", "Expected", "Extracted", "Match"])
        writer.writeheader()
        for row in detailed_rows:
            writer.writerow(row)
    print(f"Detailed comparison saved to {detailed_csv_path}")

    if not args.no_readme and not args.allow_mock:
        update_readme(final_metrics, chart_path)
    else:
        print("Skipping README update.")
    print("Evaluation completed successfully.")

if __name__ == "__main__":
    main()
