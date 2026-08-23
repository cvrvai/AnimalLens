"""
Standard quantitative ethological sampling protocols.
Scientific Research Methodology based on Altmann (1974) and Martin & Bateson (2007).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from animallens.core.schemas import BehaviorEvent


class EthogramSummary(BaseModel):
    """Ethogram distribution summary for an animal observation session."""
    total_events: int
    duration_seconds: float
    time_budget_percentage: Dict[str, float] = Field(
        default_factory=dict, description="Percentage of total time spent in each behavior category"
    )
    event_frequencies: Dict[str, int] = Field(
        default_factory=dict, description="Count of occurrences per behavior category"
    )
    activity_index: float = Field(
        ..., description="Proportion of active time (locomotion + feeding + social + aggression) vs resting"
    )


class SamplingProtocols:
    """Implementations of formal observational sampling methods."""

    @staticmethod
    def compute_ethogram_time_budget(events: List[BehaviorEvent], total_duration: Optional[float] = None) -> EthogramSummary:
        """
        Calculate quantitative time budget and behavioral frequencies from continuous observation.
        """
        if not events:
            return EthogramSummary(
                total_events=0,
                duration_seconds=total_duration or 0.0,
                time_budget_percentage={},
                event_frequencies={},
                activity_index=0.0,
            )

        duration = total_duration or max(1.0, max(e.temporal.end for e in events))
        category_durations: Dict[str, float] = {}
        category_counts: Dict[str, int] = {}

        for e in events:
            cat = e.behavior.category
            category_counts[cat] = category_counts.get(cat, 0) + 1
            dur = max(0.5, e.temporal.duration)
            category_durations[cat] = category_durations.get(cat, 0.0) + dur

        # Calculate percentages
        time_budget = {
            cat: round((dur / duration) * 100.0, 2)
            for cat, dur in category_durations.items()
        }

        # Activity index: active categories / total
        inactive_time = category_durations.get("resting", 0.0) + category_durations.get("sheltering", 0.0)
        active_time = max(0.0, duration - inactive_time)
        activity_index = round(active_time / max(0.01, duration), 4)

        return EthogramSummary(
            total_events=len(events),
            duration_seconds=round(duration, 2),
            time_budget_percentage=time_budget,
            event_frequencies=category_counts,
            activity_index=min(1.0, activity_index),
        )

    @staticmethod
    def extract_focal_sampling(
        events: List[BehaviorEvent], track_id: int
    ) -> List[BehaviorEvent]:
        """
        Extract focal animal sampling events (filter behavior events involving a specific subject track ID).
        """
        focal_events = []
        for e in events:
            if any(s.track_id == track_id for s in e.subjects):
                focal_events.append(e)
        return focal_events

    @staticmethod
    def extract_scan_sampling(
        events: List[BehaviorEvent], time_interval_seconds: float = 30.0
    ) -> List[BehaviorEvent]:
        """
        Extract instantaneous scan sampling observations at fixed periodic time strides delta t.
        """
        if not events:
            return []

        scans: List[BehaviorEvent] = []
        next_sample_time = 0.0
        sorted_events = sorted(events, key=lambda e: e.temporal.start)

        for e in sorted_events:
            if e.temporal.start >= next_sample_time:
                scans.append(e)
                next_sample_time = e.temporal.start + time_interval_seconds

        return scans
