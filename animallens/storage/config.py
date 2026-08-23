"""
MongoDB Storage Configuration for AnimalLens.
"""
import os
from pydantic import BaseModel, Field


class MongoConfig(BaseModel):
    """Configuration for MongoDB connection and collections."""
    uri: str = Field(
        default_factory=lambda: os.getenv("ANIMALLENS_MONGO_URI", "mongodb://localhost:27017")
    )
    db_name: str = Field(
        default_factory=lambda: os.getenv("ANIMALLENS_MONGO_DB", "animallens")
    )
    events_collection: str = "events"
    sessions_collection: str = "sessions"
    uncertainty_collection: str = "uncertainty_queue"
    connect_timeout_ms: int = 5000
    server_selection_timeout_ms: int = 5000


default_mongo_config = MongoConfig()
