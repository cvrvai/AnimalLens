"""
Active Learning Uncertainty Triage Engine.
Automatically identifies low-confidence, high-entropy, or kinematic-anomalous events
and routes them into MongoDB active learning queue for human ethologist verification.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from animallens.core.schemas import BehaviorEvent
from animallens.storage.mongodb import MongoDBStorage

logger = logging.getLogger(__name__)


class TriageResult(BaseModel):
    is_uncertain: bool
    reason: Optional[str] = None
    priority: int = 1  # 1 = Low, 2 = Medium, 3 = High (urgent review)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class UncertaintyTriageEngine:
    """
    Evaluates vision perception events against active learning criteria.
    """

    def __init__(
        self,
        min_confidence_threshold: float = 0.50,
        margin_threshold: float = 0.12,
        storage: Optional[MongoDBStorage] = None,
    ) -> None:
        self.min_confidence_threshold = min_confidence_threshold
        self.margin_threshold = margin_threshold
        self.storage = storage

    def evaluate(
        self,
        event: BehaviorEvent,
        kinematics: Optional[Dict[str, Any]] = None,
        save_to_storage: bool = True,
    ) -> TriageResult:
        """
        Evaluate a single BehaviorEvent and auto-persist to active learning queue if uncertain.
        """
        behavior = event.behavior
        conf = behavior.confidence
        kin = kinematics or event.metadata.get("kinematics", {})

        is_uncertain = False
        reasons = []
        priority = 1

        # 1. Check Low Absolute Confidence
        if conf < self.min_confidence_threshold:
            is_uncertain = True
            reasons.append(f"Low confidence ({conf:.2f} < {self.min_confidence_threshold:.2f})")
            priority = max(priority, 2)

        # 2. Check Margin Ambiguity (Top-1 vs Top-2 label entropy)
        if behavior.secondary_labels:
            top_sec = max([list(d.values())[0] for d in behavior.secondary_labels], default=0.0)
            margin = conf - top_sec
            if margin < self.margin_threshold:
                is_uncertain = True
                reasons.append(f"Ambiguous margin between top classes (margin {margin:.2f} < {self.margin_threshold:.2f})")
                priority = max(priority, 2)

        # 3. Check Kinematic Physical Contradictions
        mean_speed = kin.get("mean_speed", 0.0)
        pairwise = kin.get("pairwise", [])

        if behavior.label == "resting" and mean_speed > 0.08:
            is_uncertain = True
            reasons.append(f"Kinematic anomaly: Labeled resting but mean speed is high ({mean_speed:.3f})")
            priority = 3

        if behavior.category == "reproduction" and pairwise:
            closest_dist = min([p.get("distance", 1.0) for p in pairwise], default=1.0)
            if closest_dist > 0.40:
                is_uncertain = True
                reasons.append(f"Spatial anomaly: Reproduction labeled at large distance (IID={closest_dist:.2f})")
                priority = 3

        # Apply flags and persist
        event.behavior.is_uncertain = is_uncertain
        reason_str = "; ".join(reasons) if reasons else None

        if is_uncertain and save_to_storage and self.storage is not None:
            try:
                self.storage.save_uncertainty(
                    event=event,
                    notes=reason_str or "Active learning candidate",
                )
                logger.info(f"Triage: Routed uncertain event {event.event_id} to MongoDB queue.")
            except Exception as e:
                logger.warning(f"Failed to persist uncertain event to storage: {e}")

        return TriageResult(
            is_uncertain=is_uncertain,
            reason=reason_str,
            priority=priority,
            metrics={"confidence": conf, "mean_speed": mean_speed},
        )
