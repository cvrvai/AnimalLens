"""
Inter-Rater Reliability & Cohen's Kappa Validator for Ethological Annotations.
Validates multi-annotator consistency before feeding data into ML training pipelines.
"""
from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class KappaReport(BaseModel):
    cohen_kappa: float
    observed_agreement: float
    chance_agreement: float
    categories: List[str]
    sample_count: int
    interpretation: str
    is_valid_for_training: bool
    confusion_matrix: Dict[str, Dict[str, int]]


class CohenKappaValidator:
    """
    Computes Cohen's Kappa statistic between two independent ethologist annotators.
    Formula: Kappa = (Po - Pe) / (1 - Pe)
    """

    @staticmethod
    def compute_kappa(
        annotator_1_labels: List[str],
        annotator_2_labels: List[str],
        threshold: float = 0.75,
    ) -> KappaReport:
        if len(annotator_1_labels) != len(annotator_2_labels):
            raise ValueError(
                f"Label count mismatch: Annotator 1 ({len(annotator_1_labels)}) vs Annotator 2 ({len(annotator_2_labels)})"
            )

        n = len(annotator_1_labels)
        if n == 0:
            return KappaReport(
                cohen_kappa=0.0,
                observed_agreement=0.0,
                chance_agreement=0.0,
                categories=[],
                sample_count=0,
                interpretation="No data provided",
                is_valid_for_training=False,
                confusion_matrix={},
            )

        categories = sorted(list(set(annotator_1_labels) | set(annotator_2_labels)))
        matrix: Dict[str, Dict[str, int]] = {c1: {c2: 0 for c2 in categories} for c1 in categories}

        agreements = 0
        for l1, l2 in zip(annotator_1_labels, annotator_2_labels):
            matrix[l1][l2] += 1
            if l1 == l2:
                agreements += 1

        po = agreements / n

        # Marginal distributions
        r1_counts = collections.Counter(annotator_1_labels)
        r2_counts = collections.Counter(annotator_2_labels)

        pe = sum((r1_counts[c] / n) * (r2_counts[c] / n) for c in categories)

        if pe == 1.0:
            kappa = 1.0
        else:
            kappa = max(-1.0, min(1.0, (po - pe) / (1.0 - pe)))

        # Landis & Koch (1977) interpretation
        if kappa >= 0.81:
            interp = "Almost Perfect Agreement"
        elif kappa >= 0.75:
            interp = "Excellent Reliability (Qualified for Training)"
        elif kappa >= 0.61:
            interp = "Substantial Agreement"
        elif kappa >= 0.41:
            interp = "Moderate Agreement (Needs Refinement)"
        else:
            interp = "Poor / Fair Agreement (Unqualified for Training)"

        return KappaReport(
            cohen_kappa=round(kappa, 4),
            observed_agreement=round(po, 4),
            chance_agreement=round(pe, 4),
            categories=categories,
            sample_count=n,
            interpretation=interp,
            is_valid_for_training=kappa >= threshold,
            confusion_matrix=matrix,
        )
