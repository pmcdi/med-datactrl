# Web UI for MedDataOS Monitor

Real-time web interface to monitor MedDataOS (Medical Data Operating System) execution with live log streaming.

## Features

- **Patient Sidebar**: Browse 10 patients (P0001–P0010), expand to see conversations, create new ones, switch between them
- **Patient Context**: Active patient demographics (age, sex, blood type, allergies, conditions) injected into AI system prompt
- **Real-time Log Streaming**: See all system logs in real-time via WebSockets
- **Color-coded Log Types**: Different colors for user messages, assistant responses, tool calls, and results
- **Auto-scroll**: Automatically scroll to latest logs
- **Clean UI**: Simple, modern interface with warm/brown design system
- **Non-blocking**: The system runs normally - web UI just observes

## Architecture

```
web/
├── backend/
│   └── server.py           # FastAPI server with WebSocket support
└── frontend/
    ├── index.html          # Main UI page
    ├── style.css           # Styling
    └── app.js              # WebSocket client logic
```

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Web Server

```bash
python web/backend/server.py
```

The server will start on `http://127.0.0.1:8080`

### 3. Open in Browser

Navigate to: `http://127.0.0.1:8080`

### 4. Start the Pipeline

Click the "Start Pipeline" button in the UI to begin the agent execution.

## Log Types

The UI displays different log types with distinct colors:

- 🧑‍💬 **User** (Green): User input messages
- 🤖 **Assistant** (Purple): Assistant responses
- 🛠️ **Tool Call** (Orange): Agent/tool invocations
- 📦 **Tool Result** (Pink): Results from tool executions
- ℹ️ **System** (Blue): System messages

## Technical Details

- **Backend**: FastAPI with WebSocket support
- **Frontend**: Vanilla JavaScript (no framework dependencies)
- **Communication**: WebSocket for real-time bidirectional communication
- **Logging**: Extended `src/utils/logging.py` to broadcast to WebSocket clients
- **Non-invasive**: Original system logic unchanged - logging just broadcasts to UI

## Endpoints

- `GET /` - Serve the web UI
- `WebSocket /ws` - WebSocket endpoint for log streaming
- `POST /api/chat` - Chat with file upload and patient context
- `GET /api/patients` - List all patients
- `GET /api/patients/{id}/conversations` - List conversations for a patient
- `POST /api/patients/{id}/conversations` - Create a new conversation for a patient
- `GET /api/session/{id}` - Get session history
- `POST /api/start` - Start the agent pipeline (legacy)
- `GET /api/health` - Health check endpoint

## Customization

### Change Server Port

Edit `web/backend/server.py`:

```python
start_server(host="127.0.0.1", port=8080)  # Change port here
```

### Modify UI Colors

Edit `web/frontend/style.css` to customize the color scheme.

### Add New Log Types

1. Add icon in `web/frontend/app.js`:
```javascript
const LOG_ICONS = {
    new_type: '🎯'
};
```

2. Add styling in `web/frontend/style.css`:
```css
.log-entry.new_type {
    background: rgba(255, 0, 0, 0.1);
    color: #ff0000;
}
```
