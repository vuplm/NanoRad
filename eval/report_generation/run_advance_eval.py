#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from lightning.fabric.utilities.data import AttributeDict

from capstone_eval.advance_model import R2GenGPT
from capstone_eval.metrics import compute_text_metrics


torch.serialization.add_safe_globals([AttributeDict])


def extract_pred_text(pred_report: str) -> str:
    pred_report = pred_report or ""
    pred_lower = pred_report.lower()

    if "findings :" in pred_lower:
        match = re.search(r"findings :(.*)", pred_report, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else pred_report.strip()

    if "impression :" in pred_lower:
        match = re.search(r"impression :(.*)", pred_report, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else pred_report.strip()

    return pred_report.strip()


def load_test_data(test_json: str) -> List[Dict[str, Any]]:
    with open(test_json, "r", encoding="utf-8") as f:
        return json.load(f)


def run_predictions(
    model: R2GenGPT,
    image_root: str,
    test_json: str,
    out_json: str,
) -> None:
    test_data = load_test_data(test_json)
    results: List[Dict[str, Any]] = []

    for item in tqdm(test_data, desc="Predict"):
        img_paths = [os.path.join(image_root, p) for p in item["image_path"]]
        pred = model.inference(img_paths)[0]
        results.append(
            {
                "id": item.get("id"),
                "gt_report": item.get("report", ""),
                "pred_report": pred,
            }
        )

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def postprocess_predictions(in_json: str, out_json: str) -> None:
    with open(in_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item["pred_after"] = extract_pred_text(item.get("pred_report", ""))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def evaluate(pred_json: str) -> Dict[str, float]:
    with open(pred_json, "r", encoding="utf-8") as f:
        preds = json.load(f)

    gt_list = [p.get("gt_report", "") for p in preds]
    pred_list = [p.get("pred_after", "") for p in preds]

    return compute_text_metrics(gt_list=gt_list, pred_list=pred_list)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint used by R2GenGPT.load_from_checkpoint")
    ap.add_argument("--vision_model", default="microsoft/swin-base-patch4-window7-224")
    ap.add_argument("--llm_model", default="Qwen/Qwen3-0.6B")

    ap.add_argument("--test_json", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--out_prefix", default="predictions_advance")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = ap.parse_args()

    model = R2GenGPT.load_from_checkpoint(
        args.ckpt,
        strict=False,
        vision_model=args.vision_model,
        llm_model=args.llm_model,
    )
    model = model.to(args.device)
    model.eval()

    raw_json = f"{args.out_prefix}.json"
    fixed_json = f"{args.out_prefix}_fixed.json"

    run_predictions(model, args.image_root, args.test_json, raw_json)
    postprocess_predictions(raw_json, fixed_json)

    metrics = evaluate(fixed_json)
    for k, v in metrics.items():
        print(f"{k}: {v}")



if __name__ == "__main__":
    main()
