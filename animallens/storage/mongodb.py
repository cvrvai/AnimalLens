"""
MongoDB Time-Series Storage, Querying, and Aggregation Engine.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Union
import pymongo
from pymongo import MongoClient
from animallens.core.schemas import BehaviorEvent
from animallens.storage.config import MongoConfig, default_mongo_config

logger = logging.getLogger(__name__)


class MongoDBStorage:
    """
    Primary MongoDB client for persisting BehaviorEvents, video sessions,
    and running real-time ethological aggregation pipelines.
    """

    def __init__(self, config: Optional[MongoConfig] = None, client: Optional[MongoClient] = None):
        self.config = config or default_mongo_config
        self._custom_client = client
        self._client: Optional[MongoClient] = client
        self._connected = False

    @property
    def client(self) -> MongoClient:
        if self._client is None:
            self._client = MongoClient(
                self.config.uri,
                serverSelectionTimeoutMS=self.config.server_selection_timeout_ms,
                connectTimeoutMS=self.config.connect_timeout_ms,
            )
        return self._client

    @property
    def db(self):
        return self.client[self.config.db_name]

    @property
    def events(self):
        return self.db[self.config.events_collection]

    @property
    def sessions(self):
        return self.db[self.config.sessions_collection]

    @property
    def uncertainty_queue(self):
        return self.db[self.config.uncertainty_collection]

    def connect(self) -> bool:
        """Check connection health and ensure indexes."""
        try:
            self.client.admin.command("ping")
            self._connected = True
            self.ensure_indexes()
            return True
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {e}")
            self._connected = False
            return False

    def is_connected(self) -> bool:
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            return False

    def ensure_indexes(self):
        """Create optimized indexes for fast time-series and species filtering."""
        try:
            self.events.create_index([("datetime", pymongo.ASCENDING), ("species.id", pymongo.ASCENDING)])
            self.events.create_index([("event_id", pymongo.ASCENDING)], unique=True)
            self.events.create_index([("behavior.category", pymongo.ASCENDING), ("behavior.label", pymongo.ASCENDING)])
            self.events.create_index([("source.session_id", pymongo.ASCENDING)])
            self.events.create_index([("source.camera_id", pymongo.ASCENDING)])
            self.sessions.create_index([("session_id", pymongo.ASCENDING)], unique=True)
            self.sessions.create_index([("tank_id", pymongo.ASCENDING)])
            self.uncertainty_queue.create_index([("verified_by_human", pymongo.ASCENDING)])
        except Exception as e:
            logger.debug(f"Index creation notice: {e}")

    def save_event(self, event: Union[BehaviorEvent, Dict[str, Any]]) -> str:
        """Insert a single BehaviorEvent into the events collection."""
        data = event.model_dump() if isinstance(event, BehaviorEvent) else dict(event)
        if "datetime" not in data:
            ts = data.get("timestamp", datetime.now(timezone.utc).timestamp())
            data["datetime"] = datetime.fromtimestamp(ts, tz=timezone.utc)

        res = self.events.replace_one({"event_id": data["event_id"]}, data, upsert=True)
        return str(res.upserted_id or data.get("event_id"))

    def save_events(self, events: List[Union[BehaviorEvent, Dict[str, Any]]]) -> int:
        """Bulk insert/upsert a list of BehaviorEvents."""
        if not events:
            return 0

        inserted = 0
        for event in events:
            self.save_event(event)
            inserted += 1
        return inserted

    def save_session(self, session_data: Dict[str, Any]) -> str:
        """Upsert video observation session metadata."""
        sess_id = session_data.get("session_id", f"sess_{int(datetime.now().timestamp())}")
        session_data["session_id"] = sess_id
        if "created_at" not in session_data:
            session_data["created_at"] = datetime.now(timezone.utc)

        self.sessions.replace_one({"session_id": sess_id}, session_data, upsert=True)
        return sess_id

    def save_uncertainty(
        self,
        event: Union[BehaviorEvent, Dict[str, Any]],
        notes: str = "",
        keyframe_uri: Optional[str] = None,
        clip_uri: Optional[str] = None,
    ) -> str:
        """Insert a low-confidence prediction into the active learning queue."""
        data = event.model_dump() if isinstance(event, BehaviorEvent) else dict(event)
        unc_doc = {
            "event_ref_id": data.get("event_id"),
            "species_id": data.get("species", {}).get("id"),
            "behavior_predicted": data.get("behavior"),
            "timestamp": data.get("timestamp"),
            "datetime": datetime.now(timezone.utc),
            "keyframe_uri": keyframe_uri,
            "clip_uri": clip_uri,
            "notes": notes,
            "verified_by_human": False,
            "human_verified_label": None,
            "verified_by": None,
        }
        res = self.uncertainty_queue.insert_one(unc_doc)
        return str(res.inserted_id)

    def get_events(
        self,
        species_id: Optional[str] = None,
        session_id: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query events with optional filtering."""
        query: Dict[str, Any] = {}
        if species_id:
            query["species.id"] = species_id
        if session_id:
            query["source.session_id"] = session_id
        if category:
            query["behavior.category"] = category

        cursor = self.events.find(query, {"_id": 0}).sort("temporal.start", pymongo.ASCENDING).limit(limit)
        return list(cursor)

    def get_transition_matrix(
        self,
        session_id: Optional[str] = None,
        species_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute empirical Markov state transition probabilities using MongoDB Aggregation Pipeline.
        """
        match_stage: Dict[str, Any] = {}
        if session_id:
            match_stage["source.session_id"] = session_id
        if species_id:
            match_stage["species.id"] = species_id

        pipeline = [
            {"$match": match_stage} if match_stage else {"$match": {}},
            {"$sort": {"temporal.start": 1}},
            {
                "$group": {
                    "_id": "$source.session_id",
                    "labels": {"$push": "$behavior.label"},
                    "categories": {"$push": "$behavior.category"},
                }
            },
        ]

        results = list(self.events.aggregate(pipeline))
        if not results:
            return {"states": [], "transitions": {}, "total_transitions": 0}

        counts: Dict[str, Dict[str, int]] = {}
        total_transitions = 0

        for doc in results:
            labels = doc.get("labels", [])
            for t in range(len(labels) - 1):
                s1 = labels[t]
                s2 = labels[t + 1]
                if s1 not in counts:
                    counts[s1] = {}
                counts[s1][s2] = counts[s1].get(s2, 0) + 1
                total_transitions += 1

        # Calculate row-normalized probabilities
        prob_matrix: Dict[str, Dict[str, float]] = {}
        all_states = sorted(list(counts.keys()))
        for s1, row in counts.items():
            row_sum = sum(row.values())
            prob_matrix[s1] = {s2: round(c / row_sum, 4) for s2, c in row.items()}

        return {
            "states": all_states,
            "probabilities": prob_matrix,
            "transition_counts": counts,
            "total_transitions": total_transitions,
        }

    def get_circadian_budget(
        self,
        date_start: Optional[datetime] = None,
        date_end: Optional[datetime] = None,
        species_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compute hourly circadian activity time budget via MongoDB Aggregation.
        """
        match_stage: Dict[str, Any] = {}
        if species_id:
            match_stage["species.id"] = species_id
        if date_start or date_end:
            match_stage["datetime"] = {}
            if date_start:
                match_stage["datetime"]["$gte"] = date_start
            if date_end:
                match_stage["datetime"]["$lte"] = date_end

        pipeline = [
            {"$match": match_stage} if match_stage else {"$match": {}},
            {
                "$group": {
                    "_id": {
                        "category": "$behavior.category",
                    },
                    "total_duration_seconds": {"$sum": "$temporal.duration"},
                    "event_count": {"$sum": 1},
                    "mean_confidence": {"$avg": "$behavior.confidence"},
                }
            },
            {"$sort": {"total_duration_seconds": -1}},
        ]

        results = list(self.events.aggregate(pipeline))
        formatted = []
        for doc in results:
            formatted.append({
                "category": doc["_id"]["category"],
                "total_duration_seconds": round(doc.get("total_duration_seconds", 0.0), 2),
                "event_count": doc.get("event_count", 0),
                "mean_confidence": round(doc.get("mean_confidence", 0.0), 4),
            })
        return formatted

    def get_uncertainty_queue(self, verified: bool = False, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve items from the active learning uncertainty queue."""
        cursor = self.uncertainty_queue.find(
            {"verified_by_human": verified},
            {"_id": {"$toString": "$_id"}, "event_ref_id": 1, "species_id": 1, "behavior_predicted": 1, "notes": 1, "keyframe_uri": 1, "datetime": 1}
        ).limit(limit)
        return list(cursor)

    def verify_uncertainty(self, unc_id: str, verified_label: str, verified_by: str) -> bool:
        """Mark an active learning candidate as verified by human ethologist."""
        from bson import ObjectId
        try:
            oid = ObjectId(unc_id) if ObjectId.is_valid(unc_id) else unc_id
            res = self.uncertainty_queue.update_one(
                {"_id": oid},
                {
                    "$set": {
                        "verified_by_human": True,
                        "human_verified_label": verified_label,
                        "verified_by": verified_by,
                        "verified_at": datetime.now(timezone.utc),
                    }
                }
            )
            return res.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to verify uncertainty item: {e}")
            return False
