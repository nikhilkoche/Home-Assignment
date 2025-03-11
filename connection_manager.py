from fastapi import WebSocket
from typing import NoReturn
import asyncio
from dataclasses import dataclass
import hashlib
import time


def create_unique_id(session_id: str) -> str:
   
    return hashlib.sha256(session_id.encode()).hexdigest()[:10]

@dataclass
class Connection:
    session_id: str
    unique_id: str
    websocket: WebSocket
    connected: bool = True  # New flag to track connection state

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, Connection] = {}
        self.client_connections: dict[str, set[str]] = {}

    async def connect(self, client_id: str, websocket: WebSocket) -> tuple[str, str]:
        await websocket.accept()
        session_id: str = f"{client_id}-{time.time()}"
        unique_id: str = create_unique_id(session_id)
        connection = Connection(
            session_id=session_id,
            unique_id=unique_id,
            websocket=websocket,
            connected=True  # Mark connection as active
        )
        self.active_connections[unique_id] = connection
        if client_id not in self.client_connections:
            self.client_connections[client_id] = set()
        self.client_connections[client_id].add(unique_id)
        return session_id, unique_id

    async def disconnect(self, unique_id: str, reason: str = "Connection closed"):
        if unique_id in self.active_connections:
            connection = self.active_connections[unique_id]
            if connection.connected:  
                connection.connected = False  # Mark connection as inactive
                await connection.websocket.close(code=1000, reason=reason)
            client_id = connection.session_id.split("-")[0]
            self.client_connections[client_id].remove(unique_id)
            del self.active_connections[unique_id]

    async def send_json(self, unique_id: str, message: dict, retry_count: int = 3) -> NoReturn:
        if unique_id not in self.active_connections:
            return
        
        connection = self.active_connections[unique_id]
        if not connection.connected:  
            return  # Prevent sending to closed connections

        for attempt in range(retry_count):
            try:
                await connection.websocket.send_json(message)
                break
            except Exception as e:
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                await self.disconnect(unique_id, f"Failed to send message: {str(e)}")
                break
