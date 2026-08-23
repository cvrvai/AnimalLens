"""
Event utilities, dispatching, and active learning serialization.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional
from animallens.core.schemas import BehaviorEvent


class EventCollector:
    """In-memory collector for behavior events with export capabilities."""

    def __init__(self) -> None:
        self.events: List[BehaviorEvent] = []
        self._subscribers: List[Callable[[BehaviorEvent], None]] = []

    def add(self, event: BehaviorEvent) -> None:
        """Add an event and notify subscribers."""
        self.events.append(event)
        for sub in self._subscribers:
            try:
                sub(event)
            except Exception:
                pass

    def subscribe(self, callback: Callable[[BehaviorEvent], None]) -> None:
        """Subscribe a listener callback."""
        self._subscribers.append(callback)

    def get_uncertain_events(self, threshold: float = 0.45) -> List[BehaviorEvent]:
        """Return all events flagged as uncertain or below confidence threshold for active learning."""
        return [
            e for e in self.events
            if e.behavior.is_uncertain or e.behavior.confidence < threshold or e.behavior.label == "unknown"
        ]

    def export_dataset_json(self, file_path: str | Path, uncertain_only: bool = False) -> None:
        """Export events to dataset JSON for training or human-in-the-loop review."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        items = self.get_uncertain_events() if uncertain_only else self.events
        data = [e.model_dump(mode="json") for e in items]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def clear(self) -> None:
        """Clear collected events."""
        self.events.clear()
