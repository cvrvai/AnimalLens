"""
Storage and Database module for AnimalLens.
"""
from typing import Optional
from animallens.storage.config import MongoConfig, default_mongo_config
from animallens.storage.mongodb import MongoDBStorage

_global_storage: Optional[MongoDBStorage] = None


def get_storage(config: Optional[MongoConfig] = None) -> MongoDBStorage:
    """Get or create singleton MongoDB storage client."""
    global _global_storage
    if _global_storage is None:
        _global_storage = MongoDBStorage(config=config or default_mongo_config)
    return _global_storage


__all__ = ["MongoConfig", "MongoDBStorage", "get_storage", "default_mongo_config"]
