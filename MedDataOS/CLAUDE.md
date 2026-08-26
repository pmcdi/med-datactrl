# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MedDataOS is a multi-agent AI system for automated medical data analysis using Google's Gemini API. It orchestrates specialized agents through a tool-calling pattern where Gemini coordinates data preparation, analysis, and reporting agents. The system supports multimodal citations — orchestrator responses can reference outputs from specialized agents, rendered in the chat as hoverable cards showing the actual data (images, audio, video, tables, text). A left sidebar enables multi-patient, multi-conversation workflows — users select from 10 patients (P0001–P0010), create per-patient conversations, and switch between them.

## Running the Application

### Web Server (Primary Interface)
```bash
python web/backend/server.py
```
- Starts FastAPI server on `http://127.0.0.1:8080`
- WebSocket streaming at `ws://127.0.0.1:8080/ws`

### CLI Mode (Legacy)
```bash
python -m src.main
```
Uses `INITIAL_QUERY` from `src/agents/prompts.py`

### Environment Setup
Create `.env` file in project root:
```
GEMINI_API_KEY=your_api_key_here
```

## Core Architecture

### Tool-Calling Orchestration Pattern

The system uses Gemini's function calling to coordinate specialized agents. **Critical flow:**

1. **User Query → Orchestrator** (`src/core/orchestrator.py`)
   - Receives query + optional file upload + optional patient context
   - When a patient is active, prepends patient demographics (age, sex, blood type, allergies, conditions) to the system instruction
   - Creates `GenerateContentConfig` with tool definitions
   - Sends to Gemini 2.5 Flash with conversation history

2. **Tool Calling Loop**
   - Gemini returns `function_call` with tool name + arguments
   - Orchestrator maps tool name to agent config
   - Executes agent via `execute_agent(plan, agent_config)`
   - Attaches result as `function_response`
   - Loops until Gemini stops calling tools

3. **Agent Execution** (`src/agents/executors.py`)
   - Takes "plan" (instructions) from Gemini
   - Sends plan to Gemini with agent-specific system prompt
   - Gemini generates Python code
   - Code saved to `workspace/{agent_name}/generated_script.py`
   - Executed via subprocess
   - Summary generated and saved to `shared_knowledge.xml`

### Agent Types

| Agent | Tool Name | Input | Output Location |
|-------|-----------|-------|-----------------|
| **Preparation** | `prepare_data_for_analysis` | Raw CSV/Excel | `workspace/preparation/output_dataset.csv` |
| **Analysis** | `prepare_analysis` | Cleaned CSV | `workspace/analysis/model.joblib` |
| **Report** | `generate_analysis_report` | Model + data | `workspace/report/analysis_report.txt` |
| **Chest X-ray** | `classify_chest_xray` | Image URL | External API response |

**Key Pattern:** Each agent:
1. Receives a structured "plan" (not raw data)
2. Generates Python code dynamically based on plan
3. Executes code in isolated workspace
4. Outputs to predictable file location for next agent

### WebSocket Streaming Implementation

**Backend:** `web/backend/server.py`
- `WebSocketManager` class manages active connections
- Logging functions call `_broadcast_log()` which uses `asyncio.run_coroutine_threadsafe()` to cross thread boundary from orchestrator thread to async event loop

**Frontend:** `web/frontend/app.js`
- Message filtering logic:
  - `assistant` → Chat bubble only (stops processing flag)
  - `tool_call`/`tool_result` → Execution log entries
  - `user` → Skipped (already shown locally)

**Critical Detail:** Logs broadcast from background thread require `asyncio.run_coroutine_threadsafe()` to schedule in event loop safely.

### Patient Sidebar & Multi-Conversation

The UI has a collapsible left sidebar listing 10 patients (P0001–P0010) loaded from `multimodal-data/patient-info/P*.json`. Each patient expands to show its conversations and a "+ New" button.

**Frontend state:** `activePatientId`, `activeConversationId`, `patients[]`, `sidebarOpen`

**Flow:**
1. On load, `loadPatients()` fetches `GET /api/patients` and renders the list
2. Clicking a patient header expands it and fetches `GET /api/patients/{id}/conversations`
3. "+ New" calls `POST /api/patients/{id}/conversations` to create a session, then switches to it
4. Clicking a conversation calls `GET /api/session/{id}` to load history and replays messages in the chat
5. `sendMessage()` includes `patient_id` in the FormData; sidebar refreshes after send and after assistant response

**Sidebar toggle:** Hamburger button in the header toggles `.collapsed` class on the sidebar (CSS transition on `width`).

### Session Management

**In-memory storage:** `sessions = {}`
- Structure: `{"history": [...], "processing": bool, "uploaded_file": str, "patient_id": str}`
- Session ID stored in frontend localStorage
- Survives page refresh but NOT server restart
- `processing` flag prevents concurrent requests per session
- `patient_id` links session to a patient (set on creation or first message)

### File Upload Pipeline

1. User uploads CSV/Excel via `/api/chat` (multipart/form-data)
2. Backend validates extension: `.csv` or `.xlsx`
3. Saves to `data/input/{session_id}_{filename}`
4. Orchestrator modifies system instruction when file present:
   ```
   IMPORTANT: The user has uploaded a file at: {uploaded_file}
   You MUST call the prepare_data_for_analysis tool first...
   ```
5. Gemini sees file path and plans appropriate tool sequence

## Key Files & Their Roles

### Orchestration
- `src/core/orchestrator.py` - Main control loop, tool calling, conversation history management
- `src/core/gemini_client.py` - Gemini API client setup
- `src/tools/definitions.py` - Function declarations for Gemini tool calling

### Agent Implementation
- `src/agents/executors.py` - Code generation, execution, summarization
- `src/agents/prompts.py` - System prompts for each agent (defines library constraints)

### Web Interface
- `web/backend/server.py` - FastAPI server, session management, file uploads, patient/conversation APIs, WebSocket broadcasting. Serves multimodal data at `/multimodal-data/*`
- `web/frontend/app.js` - WebSocket client, message filtering, markdown rendering, citation parsing and card rendering, patient sidebar logic (load, expand, switch, create conversations)
- `web/frontend/index.html` - UI structure (header with sidebar toggle, sidebar + main flex layout)
- `web/frontend/style.css` - Styling including sidebar, markdown element styles, and citation cards

### Multimodal Data
- `multimodal-data/` - Static patient data files served at `/multimodal-data/*`

### Utilities
- `src/utils/logging.py` - WebSocket manager, XML persistence, log broadcasting functions

## Citation System

### Overview
The orchestrator can cite specific agent outputs inline in its response. Citations are rendered in the chat as hoverable cards showing the actual multimodal data.

### Model Output Format
The orchestrator returns JSON (not plain text) when citations are present:
```json
{
  "response": "Patient has chest pain [cite:1]. X-ray is normal [cite:2].",
  "citations": [
    { "id": "1", "agent": "clinical_notes", "file": "clinical-notes.txt", "web_path": "/multimodal-data/clinical-notes.txt", "summary": "..." },
    { "id": "2", "agent": "chest_xray",     "file": "chest-x-ray.png",    "web_path": "/multimodal-data/chest-x-ray.png",    "summary": "..." }
  ]
}
```

### Reference Format
- Inline token: `[cite:N]` where N is the citation ID
- Frontend regex: `/\[cite:(\w+)\]/g`
- Rendered as: `<sup class="citation" data-citation="...">N</sup>`

### Multimodal Agent Names
| Agent ID | Data Type | File Format |
|----------|-----------|-------------|
| `clinical_notes` | Clinical text notes | `.txt` |
| `chest_xray` | Chest X-ray image | `.png` |
| `ecg` | Electrocardiogram | `.svg` (HL7 source) |
| `heart_sounds` | Heart auscultation | `.wav` |
| `echo` | Echocardiogram | `.mp4` |
| `lab_results` | Lab results chart | `.png` (HL7v2 source) |
| `medication` | Medication history | `.csv` |

### Citation Card Rendering (Frontend)
Cards are fixed size (340px wide, 210px content area). Content adapts by file type:
- `.png` / `.jpg` → `<img>` with dark letterbox (`object-fit: contain`)
- `.svg` → `<img>` with **white** background (ECG waveform is dark on white)
- `.mp4` → `<video>` with controls
- `.wav` → `<audio>` with controls, centered in card with ♫ icon
- `.csv` → scrollable HTML table (all rows, first 5 columns)
- `.txt` → scrollable text excerpt

### Hover Interaction
- Hover citation badge → card appears below (flips above if near screen bottom)
- Mouse moves to card → card stays open (150ms delay prevents flicker)
- Mouse leaves card → card hides
- Audio/video are fully playable within the card (`pointer-events: auto`)

### Demo
A hardcoded demo dialog is available via the **Load Demo** button in the chat header. It simulates a full cardiac assessment for patient P0001 using all 7 data modalities. Load Demo also sets P0001 as the active patient and expands it in the sidebar.

## Non-Obvious Implementation Details

### Thread Safety
- Orchestrator runs in background thread (`threading.Thread`)
- WebSocket broadcasts cross thread boundary via `asyncio.run_coroutine_threadsafe()`
- Session `processing` flag prevents race conditions

### Agent Library Constraints
Each agent has strict library limitations defined in system prompts:
- **Preparation:** pandas, numpy only
- **Analysis:** scikit-learn, joblib only
- **Report:** pandas, joblib, json only

**Why:** Ensures generated code is predictable and reduces security surface area

### Data Continuity Pattern
Agents don't pass data directly - they write/read from predictable file locations:
```
Preparation → workspace/preparation/output_dataset.csv
Analysis (reads above) → workspace/analysis/model.joblib
Report (reads both above) → workspace/report/analysis_report.txt
```

### Markdown Rendering
- Frontend loads `marked.js` and `DOMPurify` from CDN
- Only assistant messages rendered as markdown (security)
- User messages kept as escaped plain text
- Styling in `.message-content` scopes all markdown elements

### Code Cleanup Pattern
Generated Python code may include markdown code blocks:
```python
if python_code.startswith("```python"):
    python_code = python_code[len("```python"):].strip()
if python_code.endswith("```"):
    python_code = python_code[:-len("```")].strip()
```

## Common Modification Patterns

### Adding a New Agent
1. Create function declaration in `src/tools/definitions.py`
2. Add system prompt to `src/agents/prompts.py`
3. Add agent config mapping in `orchestrator.py`:
   ```python
   agent_configs = {
       "your_tool_name": {
           'system_prompt': YOUR_AGENT_SYSTEM_PROMPT,
           'agent_name': 'your_agent',
           'plan_key': 'plan'
       }
   }
   ```
4. Ensure workspace directory exists: `workspace/your_agent/`

### Modifying Agent Behavior
- Edit system prompts in `src/agents/prompts.py`
- Change allowed libraries
- Modify input/output location instructions
- Adjust code generation expectations

### Changing WebSocket Log Display
- Frontend filtering: `web/frontend/app.js` in `handleWebSocketMessage()`
- Backend broadcast control: `src/utils/logging.py` in log functions
- Message types: `user`, `assistant`, `tool_call`, `tool_result`, `system`

### Adjusting Session Behavior
- Timeout logic: Add expiry checking in `web/backend/server.py`
- History limits: Slice `session["history"]` before passing to orchestrator
- Concurrent handling: Modify `processing` flag logic

## Important Constraints

### Security
- Generated code executes with server process permissions
- No sandboxing of agent code execution
- File uploads saved without encryption
- DOMPurify sanitizes markdown to prevent XSS

### Limitations
- Single-user system (no authentication)
- Sessions not persistent across server restarts
- No parallel agent execution (sequential only)
- No automatic retry on agent failures

### API Dependencies
- Requires Google Gemini API key
- All agents use `gemini-2.5-pro` model
- Chest X-ray agent expects external service at `localhost:8000`

## Troubleshooting

### Port Already in Use
```bash
lsof -ti :8080 | xargs kill -9
```

### Agent Execution Failures
Check generated script: `workspace/{agent_name}/generated_script.py`
- Verify library imports
- Check file paths
- Review subprocess output in logs

### WebSocket Disconnects
- Check browser console for errors
- Verify server is running
- Clear browser cache (files have version query params)

### File Upload Issues
- Verify `python-multipart` installed
- Check file extension validation
- Ensure `data/input/` directory exists and is writable
- For Excel files, ensure `openpyxl` is installed
