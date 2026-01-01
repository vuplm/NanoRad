
"""
Shared evaluation utilities for medical report generation.

This module is extracted from the original notebooks and refactored
so both baseline and advanced evaluation scripts can reuse the same logic.
"""
from __future__ import annotations

from typing import Dict, List

def compute_text_metrics(gt_list: List[str], pred_list: List[str]) -> Dict[str, float]:
    """
    Compute BLEU-1..4, METEOR, ROUGE-1/2/L (F-measure) and BERTScore-F1.

    Notes:
    - Requires: nltk, rouge_score, bert_score
    - Assumes English tokenization by whitespace (matching the notebook logic).
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from bert_score import score as bert_score
    from rouge_score import rouge_scorer

    assert len(gt_list) == len(pred_list), "gt_list and pred_list must have same length"

    smooth = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []
    meteor_scores = []
    rouge1_scores, rouge2_scores, rougel_scores = [], [], []

    for g, p in zip(gt_list, pred_list):
        g = (g or "").strip()
        p = (p or "").strip()

        bleu1_scores.append(sentence_bleu([g.split()], p.split(), weights=(1, 0, 0, 0), smoothing_function=smooth))
        bleu2_scores.append(sentence_bleu([g.split()], p.split(), weights=(0.5, 0.5, 0, 0), smoothing_function=smooth))
        bleu3_scores.append(sentence_bleu([g.split()], p.split(), weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth))
        bleu4_scores.append(sentence_bleu([g.split()], p.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))

        meteor_scores.append(meteor_score([g.split()], p.split()))

        r = scorer.score(g, p)
        rouge1_scores.append(r["rouge1"].fmeasure)
        rouge2_scores.append(r["rouge2"].fmeasure)
        rougel_scores.append(r["rougeL"].fmeasure)

    # BERTScore expects list[str]
    P, R, F1 = bert_score(pred_list, gt_list, lang="en", rescale_with_baseline=True)

    def _avg(xs: List[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    return {
        "bleu1": _avg(bleu1_scores),
        "bleu2": _avg(bleu2_scores),
        "bleu3": _avg(bleu3_scores),
        "bleu4": _avg(bleu4_scores),
        "meteor": _avg(meteor_scores),
        "rouge1_f": _avg(rouge1_scores),
        "rouge2_f": _avg(rouge2_scores),
        "rougel_f": _avg(rougel_scores),
        "bertscore_f1": float(F1.mean()),
    }
