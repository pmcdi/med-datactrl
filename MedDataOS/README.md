# MedDataOS - Medical Data Operating System

A multi-agent AI system for multimodal medical data analysis using Google's Gemini API. Specialized agents analyze seven clinical data modalities — clinical notes, chest X-rays, ECGs, echocardiograms, heart sounds, lab results, and medications — and a knowledge bus cross-references findings to surface clinical correlations, contradictions, and actionable insights.

## Features

- **Multimodal Analysis** - Seven specialized agents each handle a distinct clinical data type
- **Knowledge Bus** - Automated cross-referencing of findings across modalities (supports/contradicts with clinical reasoning)
- **Inline Citations** - Orchestrator responses reference agent findings via `[cite:N]` tokens, rendered as hoverable cards showing the actual data (images, audio, video, tables, text)
- **Patient Sidebar** - Browse 10 patients (P0001-P0010), create per-patient conversations, and switch between them
- **Real-time Streaming** - WebSocket streaming of agent execution logs and tool calls
- **Patient-Aware Context** - Active patient demographics and available modalities automatically injected into the AI system prompt

## Architecture

The system uses a tool-calling orchestration pattern where Gemini coordinates specialized agents:

```
User Query
    |
Orchestrator (Gemini 3.1 Pro)
    |
    |--- dynamically selects relevant agents based on query + available data
    |
    v
+-----------------+  +-----------+  +-------+  +------+
| Clinical Notes  |  | Chest     |  |  ECG  |  | Echo |
| Agent (.txt)    |  | X-ray     |  | (.svg)|  |(.mp4)|
|                 |  | Agent     |  |       |  |      |
|                 |  | (.png)    |  |       |  |      |
+-----------------+  +-----------+  +-------+  +------+

+-----------------+  +-----------+  +------------+
| Heart Sounds    |  | Lab       |  | Medication |
| Agent (.wav)    |  | Results   |  | Agent      |
|                 |  | Agent     |  | (.csv)     |
|                 |  | (.png)    |  |            |
+-----------------+  +-----------+  +------------+
    |
    v
Knowledge Bus (cross-references all findings)
    |
    v
Response Assembly (narrative + citations)
```

Each agent returns a structured finding with a clinical summary. The knowledge bus then identifies which findings support or contradict each other across modalities. The orchestrator weaves these into a clinical narrative with inline citations.

## Agents

| Agent | Modality | Input Format |
|-------|----------|-------------|
| ClinicalNotesAgent | clinical_notes | `.txt` |
| ChestXrayAgent | chest_xray | `.png` |
| EcgAgent | ecg | `.svg` |
| EchoAgent | echo | `.mp4` |
| HeartSoundsAgent | heart_sounds | `.wav` |
| LabResultsAgent | lab_results | `.png` |
| MedicationAgent | medication | `.csv` |

## Prerequisites

- Python 3.10 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

## Installation

```bash
cd MedDataOS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_api_key_here
```

## Running the Application

```bash
python web/backend/server.py
```

The server starts on `http://127.0.0.1:8080`. Open this URL in your browser.

## Usage

### Select a Patient

Use the sidebar on the left to browse patients (P0001-P0010). Click a patient to expand their conversation list, then click **+ New** to start a conversation. The patient's demographics and available data modalities are automatically included as context.

### Ask Clinical Questions

Type a question and press Send:

```
"What are the key cardiac findings for this patient?"
"Summarize the ECG and echo results and flag any concerns."
"Are there any medication interactions given the lab results?"
```

The system will:
1. Select the relevant agents based on your question and available data
2. Run each agent to extract findings from the patient's data
3. Cross-reference findings via the knowledge bus
4. Return a clinical narrative with inline citations

### View Results

- **Chat bubbles** - Clinical narrative with hoverable citation badges
- **Citation cards** - Hover a `[cite:N]` badge to see the source data (X-ray image, ECG waveform, audio player, lab chart, etc.)
- **Execution logs** - Real-time agent activity shown below the chat

## Project Structure

```
MedDataOS/
├── src/
│   ├── agents/               # Agent implementations
│   │   ├── base_agent.py     # Base class + Finding dataclass
│   │   ├── clinical_notes_agent.py
│   │   ├── chest_xray_agent.py
│   │   ├── ecg_agent.py
│   │   ├── echo_agent.py
│   │   ├── heart_sounds_agent.py
│   │   ├── lab_results_agent.py
│   │   ├── medication_agent.py
│   │   ├── knowledge_bus.py  # Cross-referencing engine
│   │   └── prompts.py        # System prompts (orchestrator + knowledge bus)
│   ├── core/
│   │   ├── orchestrator.py   # Main agent coordinator + tool-calling loop
│   │   ├── gemini_client.py  # Gemini API client
│   │   └── response_assembly.py  # Citation + knowledge bus assembly
│   ├── tools/
│   │   └── definitions.py    # Gemini function calling schemas
│   └── utils/
│       └── logging.py        # WebSocket manager + log broadcasting
├── multimodal-data/          # Patient data files
│   ├── patient-info/         # Patient demographics (P0001-P0010.json)
│   ├── clinical-notes/       # .txt files
│   ├── chest-xray/           # .png files
│   ├── ecg/                  # .svg files
│   ├── echo/                 # .mp4 files
│   ├── heart-sounds/         # .wav files
│   ├── lab-results/          # .png files
│   └── medications/          # .csv files
├── stubs/                    # Pre-computed agent findings (P0001-P0010.json)
├── web/
│   ├── backend/
│   │   └── server.py         # FastAPI server + REST/WebSocket endpoints
│   └── frontend/
│       ├── index.html        # UI layout (sidebar + chat)
│       ├── app.js            # Chat, WebSocket, sidebar, citation rendering
│       └── style.css         # Styling
├── tests/                    # Test suite
├── requirements.txt
└── .env                      # API keys (create this)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the web UI |
| `GET` | `/api/patients` | List all patients with demographics |
| `GET` | `/api/patients/{id}/conversations` | List conversations for a patient |
| `POST` | `/api/patients/{id}/conversations` | Create a new conversation |
| `POST` | `/api/chat` | Send a message (multipart/form-data) |
| `GET` | `/api/session/{id}` | Get session history |
| `WebSocket` | `/ws` | Real-time log streaming |
| `GET` | `/api/health` | Health check |

## Key Technologies

- **Google Gemini 3.1 Pro** - LLM orchestration via function calling
- **FastAPI** - Async web framework
- **WebSockets** - Real-time streaming
- **Vanilla JavaScript** - No frontend framework dependencies

## Security Notes

- Sessions are stored in-memory (not persistent across server restarts)
- No authentication or authorization (single-user system)
- No sandboxing of code execution

## License

This project is for educational and research purposes.
