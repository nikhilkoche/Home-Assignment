from fastapi import WebSocket
from typing import NoReturn
import asyncio
from dataclasses import dataclass
import hashlib
import time


def create_unique_id(session_id: str) -> str:
    """
    Create a unique ID for the client
    
    Args:
        session_id: The session ID for the conversation
    
    Returns:
        str: The unique ID for the client
    """
    return hashlib.sha256(session_id.encode()).hexdigest()[:10]

@dataclass
class Connection:
    """
    Data class to store information about a connection.

    Attributes:
        session_id: The session ID for the conversation
        unique_id: The unique ID for the client
        websocket: The WebSocket connection
    """
    session_id: str
    unique_id: str
    websocket: WebSocket

class ConnectionManager:
    """
    
    A class to manager WebSocket connections. Currently handling message sending, disconnects.

    """

    def __init__(self):
        self.active_connections: dict[str, Connection] = {}
        self.client_connections: dict[str, set[str]] = {}

    async def connect(self, client_id: str, websocket: WebSocket) -> tuple[str, str]:
        """
        Connect a new client to the WebSocket server

        Args:
            client_id: The client ID for the conversation
            websocket: The WebSocket connection

        Returns:
            tuple[str, str]: The session ID and unique ID for the client
        """
        await websocket.accept()
        session_id: str = f"{client_id}-{time.time()}"
        unique_id: str = create_unique_id(session_id)
        connection = Connection(
            session_id=session_id,
            unique_id=unique_id,
            websocket=websocket
        )
        self.active_connections[unique_id] = connection
        if client_id not in self.client_connections:
            self.client_connections[client_id] = set()
        self.client_connections[client_id].add(unique_id)

        return session_id, unique_id

    async def disconnect(self, unique_id: str, reason: str = "Connection closed"):
        """
        Disconnect a client from the WebSocket server

        Args:
            unique_id: The unique ID for the client
            reason: The reason for the disconnection
        """
        if unique_id in self.active_connections:
            await self.active_connections[unique_id].websocket.close(code=1000, reason=reason)
            client_id = self.active_connections[unique_id].session_id.split("-")[0]
            self.client_connections[client_id].remove(unique_id)
            del self.active_connections[unique_id]

    async def send_json(self, unique_id: str, message: dict, retry_count: int = 3) -> NoReturn:
        """
        
        Send a JSON message to a client

        Args:
            unique_id: The unique ID for the client
            message: The message to send
            retry_count: The number of times to retry sending the message
        
        Returns:
            None

        """
        if unique_id not in self.active_connections:
            return
        
        for attempt in range(retry_count):
            try:
                await self.active_connections[unique_id].websocket.send_json(message)
                break
            except Exception as e:
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                await self.disconnect(unique_id, f"Failed to send message: {str(e)}")
                break