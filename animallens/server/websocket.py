"""
WebSocket Connection Manager for real-time behavior event streaming.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Set
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket client connections and event broadcasting."""

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast structured event to all active WebSocket listeners."""
        if not self.active_connections:
            return

        message = {
            "type": event_type,
            "data": data,
        }
        text = json.dumps(message)
        dead_sockets = []

        for connection in self.active_connections:
            try:
                await connection.send_text(text)
            except Exception:
                dead_sockets.append(connection)

        for dead in dead_sockets:
            self.disconnect(dead)


# Global WebSocket connection manager
ws_manager = ConnectionManager()
