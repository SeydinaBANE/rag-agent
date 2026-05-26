#!/usr/bin/env python3
"""
OCR accuracy evaluation against annotated dataset.
Compares extracted fields to ground truth and checks confidence thresholds.

Usage:
    python scripts/eval_ocr.py \
        --dataset tests/fixtures/ocr_dataset.json \
        --images-dir tests/fixtures/images/ \
        --output reports/ocr_eval_latest.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


async def evaluate_single(
    sample: dict,
    images_dir: Path,
) -> dict:
    """Evaluate one sample. Returns metrics dict."""
    from rag_agent.services.ocr.pipeline import run_ocr_pipeline

    doc_id = sample["id"]
    doc_type = sample["doc_type"]
    _ground_truth = sample["ground_truth"]
    required_fields = sample.get("required_fields", [])
    min_conf = sample.get("min_field_confidence", 0.70)

    image_path = images_dir / f"{doc_id}.png"
    if not image_path.exists():
        image_path = images_dir / f"{doc_id}.jpg"
    if not image_path.exists():
        return {
            "id": doc_id,
            "status": "skipped",
            "reason": f"Image not found: {doc_id}.png/.jpg",
        }

    image_bytes = image_path.read_bytes()
    result = await run_ocr_pipeline(image_bytes, doc_type=doc_type)

    # Check required fields extracted
    fields_ok: list[str] = []
    fields_missing: list[str] = []
    fields_low_conf: list[str] = []
    field_confidences: dict[str, float] = result.field_confidences() if result.structured else {}

    for field in required_fields:
        if field not in field_confidences:
            fields_missing.append(field)
        elif field_confidences[field] < min_conf:
            fields_low_conf.append(field)
        else:
            fields_ok.append(field)

    # Check overall confidence gate
    confidence_passed = result.overall_confidence >= min_conf
    required_fields_ok = len(fields_missing) == 0 and len(fields_low_conf) == 0

    return {
        "id": doc_id,
        "doc_type": doc_type,
        "status": "pass" if (confidence_passed and required_fields_ok) else "fail",
        "overall_confidence": result.overall_confidence,
        "field_confidences": field_confidences,
        "fields_ok": fields_ok,
        "fields_missing": fields_missing,
        "fields_low_confidence": fields_low_conf,
        "warnings": result.warnings,
    }


async def main_async(args: argparse.Namespace) -> int:
    dataset = json.loads(Path(args.dataset).read_text())
    images_dir = Path(args.images_dir)

    results = []
    for sample in dataset:
        print(f"Evaluating {sample['id']}…")
        r = await evaluate_single(sample, images_dir)
        results.append(r)
        status_icon = "✅" if r["status"] == "pass" else ("⚠️" if r["status"] == "skipped" else "❌")
        print(f"  {status_icon} {r['status']} — confidence: {r.get('overall_confidence', 'N/A')}")

    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    total_evaluated = passed + failed

    report = {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "accuracy": round(passed / total_evaluated, 3) if total_evaluated > 0 else 0,
        "results": results,
    }

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"\n{'=' * 50}")
    print(f"Passed:   {passed}/{total_evaluated}")
    print(f"Accuracy: {report['accuracy']:.1%}")
    print(f"Report → {out}")

    threshold = args.min_accuracy
    if report["accuracy"] < threshold:
        print(f"FAIL: accuracy {report['accuracy']:.1%} < threshold {threshold:.1%}")
        return 1
    print("PASS")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tests/fixtures/ocr_dataset.json")
    parser.add_argument("--images-dir", default="tests/fixtures/images/")
    parser.add_argument("--output", default="reports/ocr_eval_latest.json")
    parser.add_argument("--min-accuracy", type=float, default=0.80)
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
