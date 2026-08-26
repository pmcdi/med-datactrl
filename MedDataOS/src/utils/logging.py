"""
Logging utilities.
Handles console logging with optional WebSocket broadcasting for web UI.
"""
import json
from datetime import datetime
from typing import Optional, Set
import asyncio

# Global WebSocket manager (will be set by web server)
_websocket_manager: Optional['WebSocketManager'] = None


class WebSocketManager:
    """Manages WebSocket connections and broadcasts log messages."""

    def __init__(self):
        self.active_connections: Set = set()
        self.loop = None

    def set_loop(self, loop):
        """Set the event loop for async operations."""
        self.loop = loop

    def connect(self, websocket):
        """Add a new WebSocket connection."""
        self.active_connections.add(websocket)

    def disconnect(self, websocket):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return

        # Create tasks for all broadcasts
        tasks = []
        for connection in self.active_connections.copy():
            tasks.append(self._send_to_client(connection, message))

        # Wait for all broadcasts to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_to_client(self, connection, message: dict):
        """Send message to a single client."""
        try:
            await connection.send_json(message)
        except Exception:
            # Remove disconnected clients
            self.active_connections.discard(connection)


def set_websocket_manager(manager: WebSocketManager):
    """Set the global WebSocket manager."""
    global _websocket_manager
    _websocket_manager = manager


def _broadcast_log(log_type: str, message: str, metadata: dict = None, header: str = None):
    """Broadcast log to WebSocket clients if manager is available."""
    if _websocket_manager is None or _websocket_manager.loop is None:
        return

    log_data = {
        "type": log_type,
        "header": header or log_type.replace("_", " ").title(),
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {}
    }

    # Schedule the broadcast in the event loop from any thread
    try:
        asyncio.run_coroutine_threadsafe(
            _websocket_manager.broadcast(log_data),
            _websocket_manager.loop
        )
    except Exception as e:
        # Silently fail if broadcast fails
        print(f"Broadcast error: {e}")


def log_user(msg):
    """Log a user message to console and broadcast to web UI."""
    print(f"🧑‍💬 {msg}")
    _broadcast_log("user", msg, header="User Query")


def log_assistant(msg):
    """Log an assistant message to console and broadcast to web UI."""
    print(f"🤖 {msg}")
    _broadcast_log("assistant", msg, header="System Response")


def broadcast_knowledge_bus_update(findings, knowledge_bus: dict, on_complete: callable = None):
    """Broadcast knowledge bus results to update citation badges asynchronously."""
    # Build per-agent knowledge bus mapping
    kb_by_agent = {}
    for f in findings:
        kb_by_agent[f.agent] = knowledge_bus.get(f.agent, {"supported_by": [], "contradicted_by": []})

    _broadcast_log("knowledge_bus_update", "", metadata={"knowledge_bus": kb_by_agent})

    # Callback to patch session history
    if on_complete:
        on_complete(kb_by_agent)


def log_tool_call(name, input_data):
    """Log a tool call to console and broadcast to web UI."""
    message = f"{input_data}"
    print(f"🛠️  {name}: {input_data}")

    # Create readable header from tool name
    header_map = {
        "analyze_clinical_notes": "Clinical Notes Analysis",
        "analyze_chest_xray": "Chest X-Ray Analysis",
        "analyze_ecg": "ECG Analysis",
        "analyze_echo": "Echocardiogram Analysis",
        "analyze_heart_sounds": "Heart Sounds Analysis",
        "analyze_lab_results": "Lab Results Analysis",
        "analyze_medication": "Medication Analysis",
    }
    header = header_map.get(name, name.replace("_", " ").title())

    _broadcast_log("tool_call", message, {"tool_name": name, "input": str(input_data)}, header=header)


def log_tool_result(res):
    """Log a tool result to console and broadcast to web UI."""
    message = json.dumps(res, ensure_ascii=False) if not isinstance(res, str) else res
    print(f"📦 → {message}")
    _broadcast_log("tool_result", message, header="Execution Result")


