// WebSocket connection
let ws = null;

// Session management
let sessionId = localStorage.getItem('meddata_session_id') || null;
let isProcessing = false;
let attachedFile = null;
// Sidebar state
let activePatientId = null;
let activeConversationId = null;
let patients = [];
let sidebarOpen = true;

// Insights panel state
let insightsOpen = false;
let patientPins = {};    // patient_id -> [{pin_id, citation}]
let patientReviews = {};  // patient_id -> {"agent|web_path": {"0": true, ...}}
let modalCitation = null; // citation currently shown in modal
let currentModalPin = null; // full pin object for modal (or null if not pinned)
let lastUserQuery = '';     // last query text for pin provenance

// DOM elements
const chatContainer = document.getElementById('chat-container');
const connectionStatus = document.getElementById('connection-status');
const connectionText = document.getElementById('connection-text');
const sendBtn = document.getElementById('send-btn');
const clearBtn = document.getElementById('clear-btn');
const messageInput = document.getElementById('message-input');
const attachBtn = document.getElementById('attach-btn');
const fileInput = document.getElementById('file-input');
const fileAttachment = document.getElementById('file-attachment');
const fileName = document.getElementById('file-name');
const fileRemove = document.getElementById('file-remove');
const citationPopup = document.getElementById('citation-popup');
const sidebarEl = document.getElementById('sidebar');
const sidebarContent = document.getElementById('sidebar-content');
const sidebarToggle = document.getElementById('sidebar-toggle');
const activePatientLabel = document.getElementById('active-patient-label');
const insightsPanel = document.getElementById('insights-panel');
const insightsContent = document.getElementById('insights-content');
const insightsToggle = document.getElementById('insights-toggle');
const modalPin = document.getElementById('modal-pin');
const modalTabToggle = document.getElementById('modal-tab-toggle');
const modalAnnotationText = document.getElementById('modal-annotation-text');
const modalAnnotationTags = document.getElementById('modal-annotation-tags');
const modalAgentTab = document.getElementById('modal-agent-tab');
const modalAgentPanel = document.getElementById('modal-agent-panel');
const modalProvenance = document.getElementById('modal-provenance');
const modalProvenancePanel = document.getElementById('modal-provenance-panel');

// ─── Sidebar: Load Patients ─────────────────────────────────
async function loadPatients() {
    try {
        const res = await fetch('/api/patients');
        patients = await res.json();
        renderPatientList();

        // Auto-expand patient from URL param (?patient=P0003), or default to P0001
        const urlPatientId = new URLSearchParams(window.location.search).get('patient') || 'P0001';
        const targetPatient = document.querySelector(`.sidebar-patient[data-patient-id="${urlPatientId}"]`)
            || document.querySelector('.sidebar-patient[data-patient-id="P0001"]');
        if (targetPatient) {
            await togglePatient(targetPatient);
            const firstConvo = targetPatient.querySelector('.conversation-item');
            if (firstConvo) {
                await switchConversation(firstConvo.dataset.patientId, firstConvo.dataset.sessionId);
            }
        }
    } catch (e) {
        console.error('Failed to load patients:', e);
    }
}

// ─── Patient Tooltip ────────────────────────────────────────
const patientTooltip = document.createElement('div');
patientTooltip.className = 'patient-tooltip';
document.body.appendChild(patientTooltip);
const patientDataMap = {};  // id -> full patient object
let patientTooltipShowTimer = null;
let patientTooltipHideTimer = null;

function formatShortDate(iso) {
    const d = new Date(iso + 'T00:00:00');
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    const now = new Date();
    const month = months[d.getMonth()];
    const day = d.getDate();
    return d.getFullYear() === now.getFullYear() ? `${month} ${day}` : `${month} ${day}, ${d.getFullYear()}`;
}

function formatMessageTime(iso) {
    const d = new Date(iso);
    let h = d.getHours();
    const m = d.getMinutes().toString().padStart(2, '0');
    const ampm = h >= 12 ? 'PM' : 'AM';
    h = h % 12 || 12;
    return `${h}:${m} ${ampm}`;
}

function showPatientTooltip(headerEl, patient) {
    clearTimeout(patientTooltipHideTimer);
    patientTooltipShowTimer = setTimeout(() => {
        // Build content
        const conditionsHTML = patient.conditions.length
            ? patient.conditions.map(c => `<span class="patient-tooltip-chip">${escapeHtml(c)}</span>`).join('')
            : '<span class="patient-tooltip-none">None recorded</span>';
        const allergiesHTML = patient.allergies.length
            ? patient.allergies.map(a => `<span class="patient-tooltip-chip allergy">${escapeHtml(a)}</span>`).join('')
            : '<span class="patient-tooltip-none">No known allergies</span>';
        const modsHTML = patient.modalities.map(m => {
            const dateStr = m.date ? formatShortDate(m.date) : '';
            return `<span class="patient-tooltip-mod">${escapeHtml(m.label)}${dateStr ? `<span class="patient-tooltip-mod-date">${dateStr}</span>` : ''}</span>`;
        }).join('');

        patientTooltip.innerHTML = `
            <div class="patient-tooltip-header">
                <span class="patient-tooltip-name">${escapeHtml(patient.name)}</span>
                <span class="patient-tooltip-demo">${patient.age} · ${patient.sex}${patient.blood_type ? ' · ' + patient.blood_type : ''}</span>
            </div>
            <div class="patient-tooltip-section">
                <span class="patient-tooltip-label">Conditions</span>
                <div class="patient-tooltip-chips">${conditionsHTML}</div>
            </div>
            <div class="patient-tooltip-section">
                <span class="patient-tooltip-label">Allergies</span>
                <div class="patient-tooltip-chips">${allergiesHTML}</div>
            </div>
            ${modsHTML ? `<div class="patient-tooltip-section">
                <span class="patient-tooltip-label">Data</span>
                <div class="patient-tooltip-modalities">${modsHTML}</div>
            </div>` : ''}
        `;

        // Position to right of sidebar
        const rect = headerEl.getBoundingClientRect();
        const sidebarRect = sidebarEl.getBoundingClientRect();
        let top = rect.top;
        let left = sidebarRect.right + 8;

        // Ensure it doesn't overflow the bottom of the viewport
        patientTooltip.classList.add('visible');
        const tooltipHeight = patientTooltip.offsetHeight;
        if (top + tooltipHeight > window.innerHeight - 12) {
            top = window.innerHeight - tooltipHeight - 12;
        }
        if (top < 12) top = 12;

        patientTooltip.style.top = top + 'px';
        patientTooltip.style.left = left + 'px';
    }, 200);
}

function hidePatientTooltip() {
    clearTimeout(patientTooltipShowTimer);
    patientTooltipHideTimer = setTimeout(() => {
        patientTooltip.classList.remove('visible');
    }, 150);
}

function renderPatientList() {
    sidebarContent.innerHTML = '';
    patients.forEach(p => {
        patientDataMap[p.id] = p;
        const div = document.createElement('div');
        div.className = 'sidebar-patient';
        div.dataset.patientId = p.id;
        div.dataset.patientName = p.name;

        div.innerHTML = `
            <div class="sidebar-patient-header">
                <span class="sidebar-patient-arrow">▶</span>
                <span class="sidebar-patient-name">${escapeHtml(p.name)}</span>
                <span class="sidebar-patient-meta">${p.age}, ${p.sex}</span>
            </div>
            <div class="sidebar-patient-body">
                <div class="conversation-list"></div>
                <button class="btn-new-conversation" data-patient-id="${p.id}">+ New</button>
            </div>
        `;

        // Tooltip hover handlers
        const header = div.querySelector('.sidebar-patient-header');
        header.addEventListener('mouseenter', () => showPatientTooltip(header, p));
        header.addEventListener('mouseleave', hidePatientTooltip);

        sidebarContent.appendChild(div);
    });
}

// Keep tooltip open when hovering over it
patientTooltip.addEventListener('mouseenter', () => clearTimeout(patientTooltipHideTimer));
patientTooltip.addEventListener('mouseleave', hidePatientTooltip);

// ─── Sidebar: Toggle ────────────────────────────────────────
function toggleSidebar() {
    sidebarOpen = !sidebarOpen;
    sidebarEl.classList.toggle('collapsed', !sidebarOpen);
}

// ─── Insights Panel ─────────────────────────────────────────
function toggleInsights() {
    insightsOpen = !insightsOpen;
    insightsPanel.classList.toggle('collapsed', !insightsOpen);
}

async function fetchPins(patientId) {
    try {
        const res = await fetch(`/api/patients/${patientId}/pins`);
        patientPins[patientId] = await res.json();
    } catch (e) {
        console.error('Failed to fetch pins:', e);
        patientPins[patientId] = [];
    }
}

async function fetchReviews(patientId) {
    try {
        const res = await fetch(`/api/patients/${patientId}/reviews`);
        if (res.ok) patientReviews[patientId] = await res.json();
        else patientReviews[patientId] = {};
    } catch (e) {
        console.error('Failed to fetch reviews:', e);
        patientReviews[patientId] = {};
    }
}

async function addPin(patientId, pinData) {
    try {
        const body = { ...pinData };
        if (body.type === 'citation') body.query = lastUserQuery;
        await fetch(`/api/patients/${patientId}/pins`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        await fetchPins(patientId);
        renderInsightsPanel();
        highlightPinnedText();
    } catch (e) {
        console.error('Failed to add pin:', e);
    }
}

async function removePin(patientId, pinId) {
    try {
        await fetch(`/api/patients/${patientId}/pins/${pinId}`, { method: 'DELETE' });
        await fetchPins(patientId);
        renderInsightsPanel();
        highlightPinnedText();
    } catch (e) {
        console.error('Failed to remove pin:', e);
    }
}

function isPinned(patientId, citation) {
    return !!findPinId(patientId, citation);
}

function findPinId(patientId, citation) {
    const pins = patientPins[patientId] || [];
    const match = pins.find(p =>
        p.type !== 'text' && p.citation &&
        p.citation.agent === citation.agent && p.citation.web_path === citation.web_path
    );
    return match ? match.pin_id : null;
}

function isTextPinned(patientId, text) {
    return !!findTextPinId(patientId, text);
}

function findTextPinId(patientId, text) {
    const pins = patientPins[patientId] || [];
    const match = pins.find(p => p.type === 'text' && p.text === text);
    return match ? match.pin_id : null;
}

// Update citation badges in chat to reflect review status
function updateCitationBadgeStates() {
    if (!activePatientId) return;
    const reviews = patientReviews[activePatientId] || {};
    chatContainer.querySelectorAll('.citation').forEach(badge => {
        badge.classList.remove('citation-review-none', 'citation-review-partial', 'citation-review-complete', 'citation-has-conflict');
        try {
            const citation = JSON.parse(decodeURIComponent(badge.dataset.citation));
            const key = citation.agent + '|' + citation.web_path;
            const items = parseSummaryToChecklist(citation.summary);
            const total = items ? items.length : 0;
            const state = reviews[key] || {};
            const checked = items ? items.filter((_, i) => state[String(i)]).length : 0;

            if (total > 0) {
                if (checked === 0) {
                    badge.classList.add('citation-review-none');
                } else if (checked >= total) {
                    badge.classList.add('citation-review-complete');
                } else {
                    badge.classList.add('citation-review-partial');
                }
            }
            const conflict = getCitationConflict(citation);
            if (conflict) {
                badge.classList.add('citation-has-conflict');
            }
        } catch (e) { /* skip malformed */ }
    });
}

function buildPinPreview(citation) {
    const type = getCitationType(citation.file);
    if (type === 'image') {
        const isSvg = citation.file.endsWith('.svg');
        return `<div class="pinned-card-preview"><img src="${citation.web_path}" alt="" style="${isSvg ? 'background:#f8f5f0;object-fit:contain;' : ''}"></div>`;
    }
    const typeLabels = { video: 'Video', audio: 'Audio', csv: 'Table', text: 'Text' };
    const label = typeLabels[type] || 'Data';
    return `<div class="pinned-card-preview"><span class="pinned-card-preview-label">${label}</span></div>`;
}

function buildPinSummaryHTML(summary) {
    const items = parseSummaryToChecklist(summary);
    if (!items) return escapeHtml(summary);
    const maxShow = 2;
    const shown = items.slice(0, maxShow);
    const remaining = items.length - maxShow;
    return '<ul class="pin-summary-list">'
        + shown.map(s => `<li>${escapeHtml(s)}</li>`).join('')
        + (remaining > 0 ? `<li class="pin-summary-more">${remaining} more</li>` : '')
        + '</ul>';
}

function renderInsightsPanel() {
    const pins = (activePatientId && patientPins[activePatientId]) || [];
    if (pins.length === 0) {
        insightsContent.innerHTML = '<div class="empty-state"><div class="empty-state-icon">&#x1F4CC;</div><div class="empty-state-text">No pinned citations</div></div>';
        return;
    }
    insightsContent.innerHTML = '';
    pins.forEach(p => {
        const card = document.createElement('div');
        card.className = 'pinned-card';
        card.dataset.pinId = p.pin_id;

        if (p.type === 'text') {
            card.dataset.pinType = 'text';
            card.dataset.text = p.text;
            card.dataset.source = p.source || '';
            const maxLen = 120;
            const truncated = p.text.length > maxLen
                ? escapeHtml(p.text.slice(0, maxLen)) + '<span class="pinned-text-more">\u2026more</span>'
                : escapeHtml(p.text);
            card.innerHTML = `
                <div class="pinned-card-agent">Pinned Text</div>
                <div class="pinned-card-summary">${truncated}</div>
                ${p.source ? `<span class="pinned-card-source" data-session-id="${escapeHtml(p.source)}" title="Go to source">View in chat &#x2192;</span>` : ''}
                <button class="btn-unpin" title="Unpin">&times;</button>
            `;
        } else {
            card.dataset.pinType = 'citation';
            card.dataset.citation = JSON.stringify(p.citation);
            card.dataset.source = p.source || '';
            const agentLabel = agentCardLabels[p.citation.agent] || p.citation.agent.replace(/_/g, ' ');
            const previewHTML = buildPinPreview(p.citation);
            const annotationText = p.annotations && p.annotations.text ? p.annotations.text : '';
            const annotationTags = p.annotations && p.annotations.tags && p.annotations.tags.length > 0 ? p.annotations.tags : [];
            const reviewKey = p.citation.agent + '|' + p.citation.web_path;
            const reviews = patientReviews[activePatientId] || {};
            const items = parseSummaryToChecklist(p.citation.summary);
            let progressHTML = '';
            if (items) {
                const state = reviews[reviewKey] || {};
                const checked = items.filter((_, i) => state[String(i)]).length;
                const total = items.length;
                const pct = Math.round((checked / total) * 100);
                const cls = checked === 0 ? 'review-none' : checked >= total ? 'review-complete' : 'review-partial';
                progressHTML = `<div class="pinned-card-progress ${cls}">
                    <div class="progress-track"><div class="progress-fill" style="width:${pct}%"></div></div>
                    <span class="progress-label">${checked}/${total}</span>
                </div>`;
            }
            const conflict = getCitationConflict(p.citation);
            const conflictHTML = conflict
                ? `<div class="pinned-card-conflict">\u26A0 Contradicted by ${escapeHtml(conflict.firstAgent)}</div>`
                : '';
            card.innerHTML = `
                <div class="pinned-card-agent">${escapeHtml(agentLabel)}</div>
                ${previewHTML}
                <div class="pinned-card-summary">${buildPinSummaryHTML(p.citation.summary)}</div>
                ${annotationText ? `<div class="pinned-card-annotation">${escapeHtml(annotationText.slice(0, 60))}${annotationText.length > 60 ? '…' : ''}</div>` : ''}
                ${annotationTags.length > 0 ? `<div class="pinned-card-tags">${annotationTags.map(t => `<span class="pinned-card-tag">${escapeHtml(t)}</span>`).join('')}</div>` : ''}
                ${progressHTML}
                ${conflictHTML}
                ${p.source ? `<span class="pinned-card-source" data-session-id="${escapeHtml(p.source)}" title="Go to source">View in chat &#x2192;</span>` : ''}
                <button class="btn-unpin" title="Unpin">&times;</button>
            `;
        }
        insightsContent.appendChild(card);
    });
}

// Insights panel event delegation
insightsContent.addEventListener('click', (e) => {
    // Unpin button
    const unpinBtn = e.target.closest('.btn-unpin');
    if (unpinBtn) {
        e.stopPropagation();
        const card = unpinBtn.closest('.pinned-card');
        if (card && activePatientId) {
            removePin(activePatientId, card.dataset.pinId);
        }
        return;
    }

    // Source link click → navigate to conversation
    const sourceLink = e.target.closest('.pinned-card-source');
    if (sourceLink) {
        e.stopPropagation();
        const sid = sourceLink.dataset.sessionId;
        const card = sourceLink.closest('.pinned-card');
        const text = card ? card.dataset.text : '';
        if (sid && activePatientId) {
            switchConversation(activePatientId, sid).then(() => {
                if (text) scrollToTextInChat(text);
                else if (card && card.dataset.pinType === 'citation') {
                    try { scrollToCitationInChat(JSON.parse(card.dataset.citation)); } catch (e) {}
                }
            });
        }
        return;
    }

    // Preview click → open citation modal
    const preview = e.target.closest('.pinned-card-preview');
    if (preview) {
        e.stopPropagation();
        const card = preview.closest('.pinned-card');
        if (card && card.dataset.pinType === 'citation') {
            const citation = JSON.parse(card.dataset.citation);
            const pinId = card.dataset.pinId;
            const pin = (patientPins[activePatientId] || []).find(p => p.pin_id === pinId) || null;
            openCitationModal(citation, pin);
        }
        return;
    }

    // Card click → navigate to source conversation
    const card = e.target.closest('.pinned-card');
    if (card) {
        const sid = card.dataset.source;
        if (card.dataset.pinType === 'text') {
            const text = card.dataset.text;
            if (sid && activePatientId) {
                switchConversation(activePatientId, sid).then(() => {
                    scrollToTextInChat(text);
                });
            }
        } else if (card.dataset.pinType === 'citation') {
            if (sid && activePatientId) {
                switchConversation(activePatientId, sid).then(() => {
                    try { scrollToCitationInChat(JSON.parse(card.dataset.citation)); } catch (e) {}
                });
            }
        }
    }
});

insightsToggle.addEventListener('click', toggleInsights);

// ─── Sidebar: Expand / Collapse Patient ─────────────────────
async function togglePatient(patientEl) {
    const wasExpanded = patientEl.classList.contains('expanded');

    if (wasExpanded) {
        patientEl.classList.remove('expanded');
    } else {
        patientEl.classList.add('expanded');
        // Load conversations for this patient
        const pid = patientEl.dataset.patientId;
        await loadConversations(pid, patientEl);
    }
}

async function loadConversations(patientId, patientEl) {
    try {
        const res = await fetch(`/api/patients/${patientId}/conversations`);
        const convos = await res.json();
        const listEl = patientEl.querySelector('.conversation-list');
        listEl.innerHTML = '';
        convos.forEach(c => {
            const item = document.createElement('div');
            item.className = 'conversation-item';
            if (c.session_id === activeConversationId) {
                item.classList.add('active');
            }
            item.dataset.sessionId = c.session_id;
            item.dataset.patientId = patientId;
            item.textContent = c.preview || 'New conversation';
            listEl.appendChild(item);
        });
    } catch (e) {
        console.error('Failed to load conversations:', e);
    }
}

// ─── Sidebar: Switch Conversation ───────────────────────────
async function switchConversation(patientId, sid) {
    const changingPatient = activePatientId !== patientId;
    activePatientId = patientId;
    activeConversationId = sid;
    sessionId = sid;
    localStorage.setItem('meddata_session_id', sid);

    // Update active patient label with name
    const patientEl = document.querySelector(`.sidebar-patient[data-patient-id="${patientId}"]`);
    const patientName = patientEl ? patientEl.dataset.patientName : patientId;
    activePatientLabel.textContent = `— ${patientName}`;

    // Highlight active conversation in sidebar
    document.querySelectorAll('.conversation-item').forEach(el => {
        el.classList.toggle('active', el.dataset.sessionId === sid);
    });

    // Only re-fetch pins when switching to a different patient;
    // within the same patient the local cache is authoritative and
    // a fresh fetch would race with any in-flight PATCH requests.
    if (changingPatient) {
        await Promise.all([fetchPins(patientId), fetchReviews(patientId)]);
    }
    renderInsightsPanel();

    // Clear chat and load history
    chatContainer.innerHTML = '';
    try {
        const res = await fetch(`/api/session/${sid}`);
        const data = await res.json();
        if (data.status === 'success' && data.history) {
            data.history.forEach(msg => {
                addChatMessage(msg.role, msg.content, msg.timestamp);
            });
        }
    } catch (e) {
        console.error('Failed to load session:', e);
    }
    highlightPinnedText();
    updateCitationBadgeStates();
}

// ─── Sidebar: Create New Conversation ───────────────────────
async function createNewConversation(patientId) {
    try {
        const res = await fetch(`/api/patients/${patientId}/conversations`, { method: 'POST' });
        const data = await res.json();
        // Refresh conversation list
        const patientEl = document.querySelector(`.sidebar-patient[data-patient-id="${patientId}"]`);
        if (patientEl) {
            await loadConversations(patientId, patientEl);
        }
        // Switch to new conversation
        await switchConversation(patientId, data.session_id);
    } catch (e) {
        console.error('Failed to create conversation:', e);
    }
}

// ─── Sidebar: Refresh active patient's conversation list ────
async function refreshSidebarConversations() {
    if (!activePatientId) return;
    const patientEl = document.querySelector(`.sidebar-patient[data-patient-id="${activePatientId}"]`);
    if (patientEl) {
        await loadConversations(activePatientId, patientEl);
    }
}

// ─── Sidebar: Event Delegation ──────────────────────────────
sidebarContent.addEventListener('click', async (e) => {
    // New conversation button
    const newBtn = e.target.closest('.btn-new-conversation');
    if (newBtn) {
        await createNewConversation(newBtn.dataset.patientId);
        return;
    }

    // Conversation item click
    const convoItem = e.target.closest('.conversation-item');
    if (convoItem) {
        await switchConversation(convoItem.dataset.patientId, convoItem.dataset.sessionId);
        return;
    }

    // Patient header click (expand/collapse)
    const header = e.target.closest('.sidebar-patient-header');
    if (header) {
        const patientEl = header.closest('.sidebar-patient');
        await togglePatient(patientEl);
    }
});

sidebarToggle.addEventListener('click', toggleSidebar);

// Connect to WebSocket
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
        updateConnectionStatus(true);
        updateSendButton();
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        updateConnectionStatus(false);
        updateSendButton();
        // Attempt to reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

// Handle WebSocket messages
function handleWebSocketMessage(data) {
    // Skip initial connection messages
    if (data.type === 'system' && data.message === 'Connected to MedDataOS Monitor') {
        return;
    }

    // Handle assistant messages - show as chat bubble only
    if (data.type === 'assistant') {
        removeTypingIndicator();
        isProcessing = false;
        updateSendButton();
        addChatMessage('assistant', data.message, data.timestamp || new Date().toISOString());
        updateCitationBadgeStates();
        // Refresh sidebar to update conversation preview
        refreshSidebarConversations();
        return;
    }

    // Skip user messages - already shown as chat bubble when sent
    if (data.type === 'user') {
        return;
    }

    // Handle knowledge bus update — patch citation badges with cross-reference data
    if (data.type === 'knowledge_bus_update') {
        const kbByAgent = data.metadata?.knowledge_bus || {};
        chatContainer.querySelectorAll('.citation').forEach(badge => {
            try {
                const citation = JSON.parse(decodeURIComponent(badge.dataset.citation));
                if (citation.agent in kbByAgent) {
                    citation.knowledge_bus = kbByAgent[citation.agent];
                    citation._kb_loaded = true;
                    badge.dataset.citation = encodeURIComponent(JSON.stringify(citation));
                }
            } catch (e) { /* skip malformed */ }
        });
        updateCitationBadgeStates();
        // Re-render provenance panel if modal is open and flipped to provenance
        if (modalCitation && citationModal.classList.contains('open') && citationModal.classList.contains('flipped')) {
            // Refresh modalCitation with updated knowledge bus
            if (modalCitation.agent in kbByAgent) {
                modalCitation.knowledge_bus = kbByAgent[modalCitation.agent];
                modalCitation._kb_loaded = true;
                modalProvenancePanel.innerHTML = buildProvenanceHTML(modalCitation);
                // Update conflict indicator on provenance tab
                const provTab = modalTabToggle.querySelector('.modal-tab[data-tab="provenance"]');
                if (provTab) {
                    const conflict = getCitationConflict(modalCitation);
                    provTab.classList.toggle('tab-has-conflict', !!conflict);
                }
            }
        }
        return;
    }

    // tool_call / tool_result are silently consumed — the loading indicator
    // already tells the user something is happening.
}

// Update connection status UI
function updateConnectionStatus(connected) {
    if (connected) {
        connectionStatus.className = 'status connected';
        connectionText.textContent = 'Connected';
    } else {
        connectionStatus.className = 'status disconnected';
        connectionText.textContent = 'Disconnected';
    }
}

// Update send button state
function updateSendButton() {
    const isConnected = ws && ws.readyState === WebSocket.OPEN;
    const hasMessage = messageInput.value.trim().length > 0;
    sendBtn.disabled = !isConnected || isProcessing || !hasMessage;
    attachBtn.disabled = !isConnected || isProcessing;
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type
    const fileNameLower = file.name.toLowerCase();
    if (!fileNameLower.endsWith('.csv') && !fileNameLower.endsWith('.xlsx')) {
        alert('Only CSV and Excel (.csv, .xlsx) files are allowed');
        fileInput.value = '';
        return;
    }

    attachedFile = file;
    fileName.textContent = file.name;
    fileAttachment.style.display = 'flex';
}

// Remove attached file
function removeFile() {
    attachedFile = null;
    fileInput.value = '';
    fileAttachment.style.display = 'none';
    fileName.textContent = '';
}

// Agent ID → short display label for citation badges
const agentBadgeLabels = {
    'clinical_notes': 'Notes',
    'chest_xray':     'X-Ray',
    'ecg':            'ECG',
    'heart_sounds':   'Heart Sounds',
    'echo':           'Echo',
    'lab_results':    'Labs',
    'medication':     'Meds',
};

// Agent ID → full display name for citation card header
const agentCardLabels = {
    'clinical_notes': 'Findings from Clinical Notes',
    'chest_xray':     'Findings from Chest X-Ray',
    'ecg':            'Findings from ECG',
    'heart_sounds':   'Findings from Heart Sounds',
    'echo':           'Findings from Echocardiogram',
    'lab_results':    'Findings from Lab Results',
    'medication':     'Findings from Medications',
};

// Agent-specific checklist section headers
const agentFindingsLabel = {
    'clinical_notes': 'Clinical Summary',
    'chest_xray':     'Radiographic Findings',
    'ecg':            'ECG Findings',
    'heart_sounds':   'Auscultation Findings',
    'echo':           'Echocardiographic Findings',
    'lab_results':    'Lab Findings',
    'medication':     'Medication Review',
};

// Provenance pipeline steps per agent (static processing stages)
const provenanceSteps = {
    'clinical_notes': [
        { label: 'Data Origin', desc: 'Physician documentation, EHR export' },
        { label: 'NLP Extraction', desc: 'Named entity recognition: conditions, medications, vitals' },
        { label: 'Summarization', desc: 'Key findings extraction, temporal ordering' },
    ],
    'chest_xray': [
        { label: 'Data Origin', desc: 'Posteroanterior chest radiograph, digital acquisition' },
        { label: 'DICOM Processing', desc: 'DICOM \u2192 PNG, window/level normalization' },
        { label: 'AI Analysis', desc: 'CheXNet DenseNet-121, 18 pathology classification' },
    ],
    'ecg': [
        { label: 'Data Origin', desc: '12-lead resting ECG, bedside recording' },
        { label: 'Format Conversion', desc: 'HL7v2 aECG \u2192 SVG waveform rendering' },
        { label: 'Signal Analysis', desc: 'Baseline correction, QRS detection, interval measurement' },
        { label: 'AI Interpretation', desc: 'ECG-FM transformer, trained on 1.3M ECGs' },
    ],
    'heart_sounds': [
        { label: 'Data Origin', desc: 'Digital auscultation, 4 standard positions' },
        { label: 'Audio Processing', desc: 'WAV normalization, noise reduction, segmentation' },
        { label: 'AI Classification', desc: 'HeartNet CNN, murmur detection & grading' },
    ],
    'echo': [
        { label: 'Data Origin', desc: 'Transthoracic echocardiogram, standard views' },
        { label: 'Video Processing', desc: 'DICOM cine \u2192 MP4, frame extraction' },
        { label: 'Chamber Quantification', desc: 'LV/RV volumes, wall motion, EF estimation' },
        { label: 'AI Analysis', desc: 'EchoNet-Dynamic, video-based EF prediction' },
    ],
    'lab_results': [
        { label: 'Data Origin', desc: 'Laboratory panel, hospital LIS' },
        { label: 'HL7 Parsing', desc: 'HL7v2 OBX segments \u2192 structured results with reference ranges' },
        { label: 'Trend Analysis', desc: 'Delta checks, critical value flagging, longitudinal trends' },
    ],
    'medication': [
        { label: 'Data Origin', desc: 'Medication reconciliation, pharmacy system' },
        { label: 'Regimen Parsing', desc: 'Active/discontinued classification, dose timeline construction' },
        { label: 'Interaction Screening', desc: 'Drug-drug interaction check, contraindication review' },
    ],
};

// Agent info for tooltip (per agent type) — model-card style
const agentInfo = {
    'clinical_notes': {
        model: 'GatorTron 8.9B',
        modelUrl: 'https://huggingface.co/UFNLP/gatortron-large',
        purpose: 'Extracts diagnoses, medications, and clinical relationships from unstructured physician notes',
        input: 'Free-text clinical notes (.txt)',
        output: 'Named entities (conditions, medications, vitals), clinical relationships, temporal ordering',
        method: 'Transformer-based clinical NLP trained on 90B words of EHR text',
        accuracy: 'F1 +7.5% vs ClinicalBERT',
        paper: 'https://www.nature.com/articles/s41746-022-00742-2',
        license: 'Apache 2.0',
        limitations: 'May miss implicit clinical reasoning or abbreviations not seen in training data. Not validated for non-English notes.',
    },
    'chest_xray': {
        model: 'TorchXRayVision',
        modelUrl: 'https://huggingface.co/torchxrayvision/densenet121-res224-all',
        purpose: 'Detects and classifies thoracic pathologies on frontal chest radiographs',
        input: 'Frontal chest X-ray (PNG)',
        output: '18 pathology classifications: atelectasis, cardiomegaly, effusion, infiltration, pneumothorax, and others',
        method: 'DenseNet-121 multi-label classifier trained on 700k+ images from 6 merged public datasets',
        accuracy: 'AUC 0.84 (18 pathologies)',
        paper: 'https://arxiv.org/abs/2111.00595',
        license: 'Apache 2.0',
        limitations: 'Reduced accuracy on lateral views and portable/bedside radiographs. May miss subtle nodules <1 cm.',
    },
    'ecg': {
        model: 'ECG-FM',
        modelUrl: 'https://huggingface.co/wanglab/ecg-fm',
        purpose: 'Interprets 12-lead ECG waveforms for rhythm, conduction, and structural abnormalities',
        input: '12-lead ECG (HL7 aECG \u2192 SVG)',
        output: 'Rhythm classification, multi-label diagnostic statements, LVEF prediction',
        method: 'Transformer foundation model pretrained on 1.3M ECGs via self-supervised learning',
        accuracy: 'AUC 0.93\u20130.99',
        paper: 'https://arxiv.org/abs/2408.05178',
        license: 'MIT',
        limitations: 'May miss subtle arrhythmias or pacemaker-related artifacts. Not validated for pediatric ECGs.',
    },
    'heart_sounds': {
        model: 'Eko EFAST',
        modelUrl: 'https://www.ekohealth.com/blogs/newsroom/fda-clearance-efast-sensora',
        purpose: 'Screens for structural heart murmurs and atrial fibrillation from auscultation recordings',
        input: 'Heart sound recording (WAV)',
        output: 'Murmur detection (present/absent, timing, character), atrial fibrillation detection',
        method: 'Masked autoencoder foundation model trained on 4M+ de-identified recordings',
        accuracy: '86% sens / 84% spec',
        paper: null,
        license: 'Proprietary',
        limitations: 'Performance degrades with ambient noise or poor stethoscope contact. May not detect low-grade (1/6) murmurs.',
    },
    'echo': {
        model: 'EchoNet-Dynamic',
        modelUrl: 'https://echonet.github.io/dynamic/',
        purpose: 'Estimates left ventricular ejection fraction and detects systolic dysfunction from echo video',
        input: 'Apical 4-chamber echocardiogram (MP4)',
        output: 'LV ejection fraction estimation, LV segmentation, beat-to-beat cardiac function tracking',
        method: 'Video-based 3D CNN (R2+1D ResNet) with semantic segmentation, trained on 10k+ studies',
        accuracy: 'AUC 0.97 (EF MAE 4.1%)',
        paper: 'https://www.nature.com/articles/s41586-020-2145-8',
        license: 'Stanford Non-Commercial',
        limitations: 'Optimized for apical 4-chamber view only. Accuracy decreases with poor acoustic windows or foreshortened views.',
    },
    'lab_results': {
        model: 'Rule-based + Gemini 3.1 Pro',
        modelUrl: null,
        purpose: 'Flags abnormal values, identifies trends, and interprets clinical significance of lab panels',
        input: 'Structured lab results (HL7v2 \u2192 PNG chart)',
        output: 'Abnormal value flags, delta checks, critical value alerts, longitudinal trend analysis',
        method: 'Reference range engine with delta checks + LLM-based clinical interpretation',
        accuracy: 'Rule-based',
        paper: null,
        license: 'Proprietary',
        limitations: 'Reference ranges may not account for age/sex/ethnicity-specific norms. LLM interpretation requires clinical verification.',
    },
    'medication': {
        model: 'DrugBank + Gemini 3.1 Pro',
        modelUrl: 'https://go.drugbank.com/',
        purpose: 'Reviews medication lists for drug interactions, contraindications, and dosing concerns',
        input: 'Medication history (CSV)',
        output: 'Drug-drug interactions, contraindication flags, dosing timeline, regimen classification',
        method: 'Curated pharmaceutical knowledge base (500k+ products) with LLM reasoning layer',
        accuracy: 'Knowledge-base',
        paper: null,
        license: 'CC BY-NC 4.0',
        limitations: 'May not reflect most recent FDA labeling changes. Does not account for patient-specific pharmacogenomics.',
    },
};

// Extract data date for a citation from active patient's data_dates
function getCitationDate(citation) {
    if (!activePatientId || !citation.web_path) return '';
    const patient = patientDataMap[activePatientId];
    if (!patient || !patient.data_dates) return '';
    // web_path like "/multimodal-data/ecg/P0001.svg" → extract "ecg"
    const parts = citation.web_path.split('/');
    const modalityDir = parts.length >= 3 ? parts[2] : '';
    const date = patient.data_dates[modalityDir];
    return date ? formatShortDate(date) : '';
}

// Check if a citation has knowledge_bus conflicts
function getCitationConflict(citation) {
    const bus = citation.knowledge_bus;
    if (!bus || !bus.contradicted_by || !bus.contradicted_by.length) return null;
    const first = bus.contradicted_by[0];
    const label = agentBadgeLabels[first.agent] || first.agent.replace(/_/g, ' ');
    return { count: bus.contradicted_by.length, firstAgent: label, finding: first.finding };
}

// Replace [cite:X] tokens with hoverable <sup> badges
function renderWithCitations(html, citations) {
    const citationMap = {};
    citations.forEach(c => { citationMap[c.id] = c; });

    return html.replace(/\[cite:(\w+)\]/g, (match, id) => {
        const citation = citationMap[id];
        if (!citation) return match;
        const data = encodeURIComponent(JSON.stringify(citation));
        const label = agentBadgeLabels[citation.agent] || citation.agent.replace(/_/g, ' ');
        const dateStr = getCitationDate(citation);
        return `<span class="citation-group"><sup class="citation" data-citation="${data}">${escapeHtml(label)}</sup>${dateStr ? `<span class="citation-date">${dateStr}</span>` : ''}</span>`;
    });
}

// Build a minimal citation object for a given agent + active patient (for KB hover cards)
const agentDirMap = {
    'clinical_notes': { dir: 'clinical-notes', ext: 'txt' },
    'chest_xray':     { dir: 'chest-xray',     ext: 'png' },
    'ecg':            { dir: 'ecg',             ext: 'svg' },
    'heart_sounds':   { dir: 'heart-sounds',    ext: 'wav' },
    'echo':           { dir: 'echo',            ext: 'mp4' },
    'lab_results':    { dir: 'lab-results',     ext: 'png' },
    'medication':     { dir: 'medications',     ext: 'csv' },
};

function buildCitationForAgent(agent) {
    if (!activePatientId) return null;
    const mapping = agentDirMap[agent];
    if (!mapping) return null;
    const file = `${activePatientId}.${mapping.ext}`;
    const web_path = `/multimodal-data/${mapping.dir}/${file}`;
    return { id: '0', agent, file, web_path, summary: '' };
}

// Derive modality type from file extension
function getCitationType(file) {
    const ext = file.split('.').pop().toLowerCase();
    return { png: 'image', jpg: 'image', jpeg: 'image', svg: 'image',
             csv: 'csv', txt: 'text', mp4: 'video', wav: 'audio' }[ext] || 'text';
}

// Convert plain text to reflowed HTML paragraphs (collapse hard line-wraps)
function plainTextToHTML(text) {
    return text
        .split(/\n\s*\n/)                          // split on blank lines → paragraphs
        .map(p => p.replace(/\s*\n\s*/g, ' ').trim()) // collapse single newlines
        .filter(Boolean)
        .map(p => `<p>${escapeHtml(p)}</p>`)
        .join('');
}

// Parse CSV text into an HTML table (first maxRows rows, first maxCols columns)
function csvToHTML(csv, maxRows = 100, maxCols = 5) {
    const lines = csv.trim().split('\n').filter(l => l.trim());
    const allHeaders = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
    const headers = allHeaders.slice(0, maxCols);
    const truncated = allHeaders.length > maxCols;

    let html = '<div class="card-table-wrap"><table class="card-table"><thead><tr>';
    headers.forEach(h => html += `<th>${escapeHtml(h)}</th>`);
    if (truncated) html += `<th class="dim">+${allHeaders.length - maxCols}</th>`;
    html += '</tr></thead><tbody>';

    lines.slice(1, maxRows + 1).forEach(line => {
        const cells = line.split(',').map(c => c.trim().replace(/^"|"$/g, ''));
        html += '<tr>';
        cells.slice(0, maxCols).forEach(cell => html += `<td>${escapeHtml(cell)}</td>`);
        if (truncated) html += '<td class="dim">…</td>';
        html += '</tr>';
    });
    html += '</tbody></table></div>';
    return html;
}

// Position popup near anchor, flip up if near bottom of screen
function positionPopup(anchor) {
    const rect = anchor.getBoundingClientRect();
    const h = citationPopup.offsetHeight;
    const top = (window.innerHeight - rect.bottom > h + 12)
        ? rect.bottom + 8
        : rect.top - h - 8;
    citationPopup.style.left = `${Math.min(rect.left, window.innerWidth - 360)}px`;
    citationPopup.style.top = `${Math.max(8, top)}px`;
}

let hideTimeout = null;

async function showCitationPopup(anchor, citation) {
    clearTimeout(hideTimeout);

    const type = getCitationType(citation.file);
    const agentLabel = agentCardLabels[citation.agent] || citation.agent.replace(/_/g, ' ');

    // Build initial content (immediate for media, loading state for text/csv)
    let contentHTML = '';
    if (type === 'image') {
        const isSvg = citation.file.endsWith('.svg');
        contentHTML = `<img class="card-image${isSvg ? ' card-image-svg' : ''}" src="${citation.web_path}" alt="${escapeHtml(agentLabel)}">`;
    } else if (type === 'video') {
        contentHTML = `<video class="card-video" src="${citation.web_path}" controls></video>`;
    } else if (type === 'audio') {
        contentHTML = `<div class="card-audio-wrap"><audio class="card-audio" src="${citation.web_path}" controls></audio></div>`;
    } else {
        contentHTML = `<div class="card-loading">Loading…</div>`;
    }

    const pinBtnHTML = activePatientId
        ? `<button class="btn-pin-card${isPinned(activePatientId, citation) ? ' pinned' : ''}" data-citation="${encodeURIComponent(JSON.stringify(citation))}" title="Pin citation">&#x1F4CC;</button>`
        : '';

    // Build summary as bullet list or fallback to plain text
    let summaryHTML = '';
    const summaryItems = parseSummaryToChecklist(citation.summary);
    if (summaryItems) {
        const maxShow = 3;
        const shown = summaryItems.slice(0, maxShow);
        const remaining = summaryItems.length - maxShow;
        summaryHTML = '<ul class="card-summary-list">'
            + shown.map(s => `<li>${escapeHtml(s)}</li>`).join('')
            + (remaining > 0 ? `<li class="card-summary-more">${remaining} more — click to review</li>` : '')
            + '</ul>';
    } else {
        summaryHTML = escapeHtml(citation.summary);
    }

    const conflict = getCitationConflict(citation);
    const conflictHTML = conflict
        ? `<div class="card-conflict">\u26A0 Contradicted by ${escapeHtml(conflict.firstAgent)}${conflict.count > 1 ? ` +${conflict.count - 1}` : ''}</div>`
        : '';

    citationPopup.innerHTML = `
        <div class="card-header">
            <span class="card-agent">${escapeHtml(agentLabel)}</span>
            ${pinBtnHTML}
        </div>
        <div class="card-content" id="card-content-inner">${contentHTML}</div>
        <div class="card-summary">${summaryHTML}</div>
        ${conflictHTML}
    `;

    citationPopup.style.display = 'block';
    positionPopup(anchor);

    // Fetch and render text-based content
    if (type === 'csv' || type === 'text') {
        try {
            const text = await fetch(citation.web_path + '?t=' + Date.now()).then(r => r.text());
            const inner = document.getElementById('card-content-inner');
            if (inner) {
                inner.innerHTML = type === 'csv'
                    ? csvToHTML(text)
                    : `<div class="card-text">${plainTextToHTML(text.slice(0, 600))}${text.length > 600 ? '<p>…</p>' : ''}</div>`;
                positionPopup(anchor);
            }
        } catch {
            const inner = document.getElementById('card-content-inner');
            if (inner) inner.innerHTML = `<div class="card-text card-error">Could not load content.</div>`;
        }
    }
}

// Add chat message to UI
function showTypingIndicator() {
    removeTypingIndicator();
    const indicator = document.createElement('div');
    indicator.className = 'chat-message assistant typing-indicator';
    indicator.id = 'typing-indicator';
    indicator.innerHTML = `
        <div class="message-header">MedDataOS</div>
        <div class="message-bubble">
            <div class="message-content typing-dots">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    chatContainer.appendChild(indicator);
    scrollToBottom();
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) indicator.remove();
}

function renderAssistantContent(rawContent) {
    let responseText = rawContent;
    let citations = [];
    try {
        const parsed = JSON.parse(rawContent);
        if (parsed.response && parsed.citations) {
            responseText = parsed.response;
            citations = parsed.citations;
        }
    } catch (e) { /* plain text, use as-is */ }

    const rawHtml = marked.parse(responseText);
    const safeHtml = DOMPurify.sanitize(rawHtml);
    const withWarnings = safeHtml.replace(/⚠️\s*(<strong>.*?<\/strong>)/g, '<span class="warning-group">⚠️\u00A0$1</span>');
    return citations.length > 0 ? renderWithCitations(withWarnings, citations) : withWarnings;
}

function addChatMessage(role, content, timestamp) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}`;
    const label = role === 'user' ? 'You' : 'MedDataOS';

    // Render markdown for assistant messages, plain text for user messages
    let messageContent;
    if (role === 'assistant') {
        messageContent = renderAssistantContent(content);
    } else {
        // User messages stay as plain text
        messageContent = escapeHtml(content);
    }

    const timeStr = timestamp ? formatMessageTime(timestamp) : '';
    messageDiv.innerHTML = `
        <div class="message-header">${escapeHtml(label)}</div>
        <div class="message-bubble">
            <div class="message-content">${messageContent}</div>
        </div>
        ${timeStr ? `<span class="message-time">${timeStr}</span>` : ''}
    `;

    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

// Map tool names to friendly agent names
const agentNameMap = {
    'prepare_data_for_analysis': 'Preparation Agent',
    'prepare_analysis': 'Analysis Agent',
    'generate_analysis_report': 'Report Agent',
    'classify_chest_xray': 'Chest X-Ray Agent'
};

// Track the last agent that was called
let lastAgentName = '';

// Add detailed log entry (for WebSocket streaming logs)
function addLogEntry(data) {
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${data.type}`;

    const timestamp = formatTimestamp(data.timestamp);

    // Determine title based on type
    let title = '';
    let icon = '';

    if (data.type === 'tool_call') {
        // Extract tool name from header or message
        const toolName = data.header || data.message.split(':')[0] || '';
        const agentName = agentNameMap[toolName.trim()] || toolName;
        lastAgentName = agentName; // Track for matching with result
        icon = '🤖';
        title = `${icon} ${agentName}: Executing`;
    } else if (data.type === 'tool_result') {
        // Use the last agent name that was called
        icon = '✅';
        title = `${icon} ${lastAgentName}: Completed`;
    }

    logEntry.innerHTML = `
        <div class="log-header">
            <span class="log-title">${escapeHtml(title)}</span>
            <span class="log-timestamp">${timestamp}</span>
        </div>
        <div class="log-message">${escapeHtml(data.message)}</div>
    `;

    chatContainer.appendChild(logEntry);
    scrollToBottom();
}

// Scroll to bottom of chat
function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Format timestamp
function formatTimestamp(isoString) {
    if (!isoString) return '--:--:--';
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', { hour12: false });
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Format ISO date string for display
function formatDate(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    return d.toLocaleString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
    });
}

// Debounce utility
function debounce(fn, ms) {
    let t;
    return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

async function patchReview(patientId, key, checklist_state) {
    try {
        await fetch(`/api/patients/${patientId}/reviews`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ key, checklist_state }),
        });
        patientReviews[patientId] = patientReviews[patientId] || {};
        patientReviews[patientId][key] = checklist_state;
    } catch (e) {
        console.error('Failed to patch review:', e);
    }
}

const debouncedPatchReview = debounce((patientId, key, state) => {
    patchReview(patientId, key, state);
}, 500);

// Split summary text into checklist items; returns null if < 2 items
function parseSummaryToChecklist(text) {
    const parts = text.split(/\.\s+|\n+/)
        .map(s => s.replace(/\.$/, '').trim())
        .filter(s => s.length >= 20);
    if (parts.length < 2) return null;
    return parts.slice(0, 8);
}

// Build checklist UL element from items + saved state
function renderModalChecklist(items, checklist_state) {
    const ul = document.createElement('ul');
    ul.className = 'findings-checklist';
    items.forEach((item, i) => {
        const li = document.createElement('li');
        const checked = checklist_state[String(i)] || false;
        if (checked) li.classList.add('checked');
        li.innerHTML = `
            <input type="checkbox" id="chk-${i}"${checked ? ' checked' : ''}>
            <label for="chk-${i}">${escapeHtml(item)}</label>
        `;
        ul.appendChild(li);
    });
    return ul;
}

// PATCH a pin's annotations or checklist_state; updates local cache
async function patchPin(patientId, pinId, data) {
    try {
        await fetch(`/api/patients/${patientId}/pins/${pinId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        // Update local cache directly
        const pins = patientPins[patientId] || [];
        const pin = pins.find(p => p.pin_id === pinId);
        if (pin) {
            if (data.annotations !== undefined) pin.annotations = data.annotations;
        }
    } catch (e) {
        console.error('Failed to patch pin:', e);
    }
}

const debouncedPatchAnnotations = debounce((patientId, pinId, annotations) => {
    patchPin(patientId, pinId, { annotations });
}, 500);

// Clear chat
function clearChat() {
    chatContainer.innerHTML = '';
    // Clear session and start fresh
    sessionId = null;
    activeConversationId = null;
    activePatientId = null;
    activePatientLabel.textContent = '';
    localStorage.removeItem('meddata_session_id');
    // Clear attached file
    removeFile();
    // Remove active highlight from sidebar
    document.querySelectorAll('.conversation-item.active').forEach(el => {
        el.classList.remove('active');
    });
    // Clear insights panel
    renderInsightsPanel();
}

// Send message
async function sendMessage() {
    const message = messageInput.value.trim();

    if (!message) {
        return;
    }

    // Capture query for provenance before clearing input
    lastUserQuery = message;

    // Disable input while processing
    isProcessing = true;
    updateSendButton();
    messageInput.disabled = true;
    attachBtn.disabled = true;

    // Build display message
    let displayMessage = message;
    if (attachedFile) {
        displayMessage += `\n\n📎 Attached: ${attachedFile.name}`;
    }

    // Add user message to chat immediately
    addChatMessage('user', displayMessage, new Date().toISOString());

    // Show loading indicator
    showTypingIndicator();

    // Clear input
    messageInput.value = '';

    try {
        // Prepare request body
        const formData = new FormData();
        formData.append('message', message);
        if (sessionId) {
            formData.append('session_id', sessionId);
        }
        if (activePatientId) {
            formData.append('patient_id', activePatientId);
        }
        if (attachedFile) {
            formData.append('file', attachedFile);
        }

        const response = await fetch('/api/chat', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.status === 'processing') {
            // Store session ID
            if (result.session_id) {
                sessionId = result.session_id;
                activeConversationId = result.session_id;
                localStorage.setItem('meddata_session_id', sessionId);
            }
            // Clear attached file after successful send
            removeFile();
            // Refresh sidebar to show conversation preview
            refreshSidebarConversations();
            // Response will come via WebSocket
        } else if (result.status === 'error') {
            removeTypingIndicator();
            alert(result.message || 'Failed to send message');
            isProcessing = false;
            updateSendButton();
        }
    } catch (error) {
        removeTypingIndicator();
        console.error('Error sending message:', error);
        alert('Failed to communicate with server');
        isProcessing = false;
        updateSendButton();
    } finally {
        messageInput.disabled = false;
        messageInput.focus();
    }
}

// Citation hover — event delegation on chat container
chatContainer.addEventListener('mouseover', (e) => {
    const target = e.target.closest('.citation');
    if (!target) return;
    const citation = JSON.parse(decodeURIComponent(target.dataset.citation));
    showCitationPopup(target, citation);
});

chatContainer.addEventListener('mouseout', (e) => {
    if (!e.target.closest('.citation')) return;
    hideTimeout = setTimeout(() => { citationPopup.style.display = 'none'; }, 150);
});

// Citation hover on provenance panel (knowledge bus badges)
modalProvenancePanel.addEventListener('mouseover', (e) => {
    const target = e.target.closest('.citation');
    if (!target) return;
    const citation = JSON.parse(decodeURIComponent(target.dataset.citation));
    showCitationPopup(target, citation);
});
modalProvenancePanel.addEventListener('mouseout', (e) => {
    if (!e.target.closest('.citation')) return;
    hideTimeout = setTimeout(() => { citationPopup.style.display = 'none'; }, 150);
});

// Keep card open when mouse moves onto it (needed for audio/video interaction)
citationPopup.addEventListener('mouseenter', () => clearTimeout(hideTimeout));
citationPopup.addEventListener('mouseleave', () => { citationPopup.style.display = 'none'; });

// Pin button on hover card
citationPopup.addEventListener('click', async (e) => {
    const pinBtn = e.target.closest('.btn-pin-card');
    if (!pinBtn || !activePatientId) return;
    e.stopPropagation();
    const citation = JSON.parse(decodeURIComponent(pinBtn.dataset.citation));
    const existingPinId = findPinId(activePatientId, citation);
    if (existingPinId) {
        await removePin(activePatientId, existingPinId);
        pinBtn.classList.remove('pinned');
    } else {
        await addPin(activePatientId, { type: 'citation', citation, source: getActiveSessionId() });
        pinBtn.classList.add('pinned');
    }
});

// ─── Citation Detail Modal ──────────────────────────────────
const citationBackdrop = document.getElementById('citation-backdrop');
const citationModal = document.getElementById('citation-modal');
const modalAgent = document.getElementById('modal-agent');
const modalBody = document.getElementById('modal-body');
const modalSummary = document.getElementById('modal-summary');
const modalClose = document.getElementById('modal-close');

// Build modal table from CSV (larger version — all columns)
function csvToModalHTML(csv, maxRows = 200) {
    const lines = csv.trim().split('\n').filter(l => l.trim());
    const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));

    let html = '<div class="modal-table-wrap"><table><thead><tr>';
    headers.forEach(h => html += `<th>${escapeHtml(h)}</th>`);
    html += '</tr></thead><tbody>';

    lines.slice(1, maxRows + 1).forEach(line => {
        const cells = line.split(',').map(c => c.trim().replace(/^"|"$/g, ''));
        html += '<tr>';
        cells.forEach(cell => html += `<td>${escapeHtml(cell)}</td>`);
        html += '</tr>';
    });
    html += '</tbody></table></div>';
    return html;
}

async function openCitationModal(citation, pin = null) {
    modalCitation = citation;

    // Look up pin if not provided
    if (!pin && activePatientId) {
        const pins = patientPins[activePatientId] || [];
        pin = pins.find(p =>
            p.type !== 'text' && p.citation &&
            p.citation.agent === citation.agent && p.citation.web_path === citation.web_path
        ) || null;
    }
    currentModalPin = pin;

    const type = getCitationType(citation.file);
    const agentLabel = agentCardLabels[citation.agent] || citation.agent.replace(/_/g, ' ');

    modalAgent.textContent = agentLabel;

    if (activePatientId) {
        modalPin.style.display = '';
        modalPin.classList.toggle('pinned', isPinned(activePatientId, citation));
    } else {
        modalPin.style.display = 'none';
    }

    // ── Agent info panel ──────────────────────────────────
    const info = agentInfo[citation.agent];
    if (info) {
        const paperHTML = info.paper
            ? `<a href="${escapeHtml(info.paper)}" target="_blank" rel="noopener" class="agent-model-link">${info.paper.includes('arxiv') ? 'arXiv' : 'Paper'} &#x2197;</a>`
            : 'N/A';
        modalAgentPanel.innerHTML = `
            <div class="agent-info-tooltip-body">
                <div class="agent-info-tooltip-row">
                    <span class="agent-info-tooltip-label">Model</span>
                    <span class="agent-info-tooltip-value">${info.modelUrl ? `<a href="${escapeHtml(info.modelUrl)}" target="_blank" rel="noopener" class="agent-model-link">${escapeHtml(info.model)} &#x2197;</a>` : escapeHtml(info.model)}</span>
                </div>
                <div class="agent-info-tooltip-row">
                    <span class="agent-info-tooltip-label">Purpose</span>
                    <span class="agent-info-tooltip-value">${escapeHtml(info.purpose)}</span>
                </div>
                <div class="agent-info-tooltip-row">
                    <span class="agent-info-tooltip-label">Input</span>
                    <span class="agent-info-tooltip-value">${escapeHtml(info.input)}</span>
                </div>
                <div class="agent-info-tooltip-row">
                    <span class="agent-info-tooltip-label">Output</span>
                    <span class="agent-info-tooltip-value">${escapeHtml(info.output)}</span>
                </div>
                <div class="agent-info-tooltip-row">
                    <span class="agent-info-tooltip-label">Method</span>
                    <span class="agent-info-tooltip-value">${escapeHtml(info.method)}</span>
                </div>
                <div class="agent-info-tooltip-row">
                    <span class="agent-info-tooltip-label">Accuracy</span>
                    <span class="agent-info-tooltip-value">${escapeHtml(info.accuracy)}</span>
                </div>
                <div class="agent-info-tooltip-row">
                    <span class="agent-info-tooltip-label">Paper</span>
                    <span class="agent-info-tooltip-value">${paperHTML}</span>
                </div>
                <div class="agent-info-tooltip-row">
                    <span class="agent-info-tooltip-label">License</span>
                    <span class="agent-info-tooltip-value">${escapeHtml(info.license)}</span>
                </div>
                <div class="agent-info-tooltip-row">
                    <span class="agent-info-tooltip-label">Limits</span>
                    <span class="agent-info-tooltip-value">${escapeHtml(info.limitations)}</span>
                </div>
            </div>
            <div class="agent-info-tooltip-disclaimer">Advisory only — verify clinically</div>
        `;
        modalAgentTab.style.display = '';
    } else {
        modalAgentTab.style.display = 'none';
    }

    document.querySelector('.modal-tab[data-tab="provenance"]').style.display = '';
    document.querySelector('.modal-tab[data-tab="agent"]').style.display = '';
    modalAnnotationTags.parentElement.style.display = '';

    // ── Footer provenance ──────────────────────────────────
    let provenanceHTML = '';
    const citDate = getCitationDate(citation);
    if (citDate) {
        provenanceHTML += `<div>Source data: ${citDate}</div>`;
    }
    modalProvenance.innerHTML = provenanceHTML;

    // ── Annotations ─────────────────────────────────────────
    const annotations = pin ? (pin.annotations || { text: '', tags: [] }) : { text: '', tags: [] };
    modalAnnotationText.value = annotations.text || '';
    modalAnnotationText.style.height = 'auto';
    setTimeout(() => {
        modalAnnotationText.style.height = modalAnnotationText.scrollHeight + 'px';
    }, 0);
    renderAnnotationTags(annotations.tags || []);

    // ── Build body content ───────────────────────────────────
    let contentHTML = '';
    if (type === 'image') {
        const isSvg = citation.file.endsWith('.svg');
        contentHTML = `<img src="${citation.web_path}" alt="${escapeHtml(agentLabel)}"${isSvg ? ' class="modal-image-svg"' : ''}>`;
    } else if (type === 'video') {
        contentHTML = `<video src="${citation.web_path}" controls autoplay></video>`;
    } else if (type === 'audio') {
        contentHTML = `<div class="modal-audio-wrap"><div class="audio-label">Heart sound recording</div><audio src="${citation.web_path}" controls></audio></div>`;
    } else {
        contentHTML = '<div class="modal-loading">Loading\u2026</div>';
    }
    modalBody.innerHTML = contentHTML;

    // ── Summary or Checklist ────────────────────────
    const checklistItems = parseSummaryToChecklist(citation.summary);
    if (checklistItems) {
        const reviewKey = citation.agent + '|' + citation.web_path;
        const checklist_state = (patientReviews[activePatientId] || {})[reviewKey] || {};
        modalSummary.innerHTML = '';
        modalSummary.style.padding = '0';
        // Section header
        const headerLabel = agentFindingsLabel[citation.agent] || 'Findings';
        const header = document.createElement('div');
        header.className = 'findings-header';
        header.innerHTML = `<span class="findings-header-label">${escapeHtml(headerLabel)}</span><span class="findings-header-hint">verify each</span>`;
        modalSummary.appendChild(header);
        const ul = renderModalChecklist(checklistItems, checklist_state);
        modalSummary.appendChild(ul);
        // Checkbox change handlers — persist to pin if pinned, visual-only otherwise
        ul.querySelectorAll('input[type="checkbox"]').forEach((cb, i) => {
            cb.addEventListener('change', async () => {
                const li = cb.closest('li');
                li.classList.toggle('checked', cb.checked);
                if (!activePatientId) return;
                const rKey = citation.agent + '|' + citation.web_path;
                const current = (patientReviews[activePatientId] || {})[rKey] || {};
                current[String(i)] = cb.checked;
                patientReviews[activePatientId] = patientReviews[activePatientId] || {};
                patientReviews[activePatientId][rKey] = current;
                debouncedPatchReview(activePatientId, rKey, { ...current });
                updateCitationBadgeStates();
                renderInsightsPanel();
            });
        });
    } else {
        modalSummary.style.padding = '';
        modalSummary.textContent = citation.summary;
    }

    // Mark Provenance tab if citation has conflicts
    const provTab = modalTabToggle.querySelector('.modal-tab[data-tab="provenance"]');
    if (provTab) {
        const conflict = getCitationConflict(citation);
        provTab.classList.toggle('tab-has-conflict', !!conflict);
    }

    // Show modal
    citationBackdrop.classList.add('open');
    citationModal.classList.add('open');

    // Fetch text-based content
    if (type === 'csv' || type === 'text') {
        try {
            const text = await fetch(citation.web_path + '?t=' + Date.now()).then(r => r.text());
            modalBody.innerHTML = type === 'csv'
                ? csvToModalHTML(text)
                : `<div class="modal-text">${plainTextToHTML(text)}</div>`;
        } catch {
            modalBody.innerHTML = '<div class="modal-text" style="color: var(--critical);">Could not load content.</div>';
        }
    }
}

// Pin the current modal citation if not already pinned; returns true on success
async function ensurePinned() {
    if (currentModalPin) return true;
    if (!activePatientId || !modalCitation) return false;
    await addPin(activePatientId, { type: 'citation', citation: modalCitation, source: getActiveSessionId() });
    const pins = patientPins[activePatientId] || [];
    currentModalPin = pins.find(p =>
        p.type !== 'text' && p.citation &&
        p.citation.agent === modalCitation.agent && p.citation.web_path === modalCitation.web_path
    ) || null;
    if (currentModalPin) modalPin.classList.add('pinned');
    return !!currentModalPin;
}

// Render predefined tag chips with active state
function renderAnnotationTags(activeTags) {
    const predefinedTags = ['#follow-up', '#critical', '#reviewed', '#discuss', '#normal'];
    modalAnnotationTags.innerHTML = '';
    predefinedTags.forEach(tag => {
        const btn = document.createElement('button');
        btn.className = 'annotation-tag' + (activeTags.includes(tag) ? ' active' : '');
        btn.textContent = tag;
        btn.addEventListener('click', async () => {
            if (!activePatientId) return;
            if (!currentModalPin && !(await ensurePinned())) return;
            const ann = currentModalPin.annotations = currentModalPin.annotations || { text: '', tags: [] };
            const tags = ann.tags || [];
            const idx = tags.indexOf(tag);
            if (idx === -1) tags.push(tag); else tags.splice(idx, 1);
            ann.tags = tags;
            renderAnnotationTags(tags);
            renderInsightsPanel();
            patchPin(activePatientId, currentModalPin.pin_id, { annotations: { ...ann } });
        });
        modalAnnotationTags.appendChild(btn);
    });
}

// Build provenance pipeline HTML for a citation
function buildProvenanceHTML(citation) {
    const steps = provenanceSteps[citation.agent] || [];
    const bus = citation.knowledge_bus || { supported_by: [], contradicted_by: [] };
    const agentLabel = agentBadgeLabels[citation.agent] || citation.agent.replace(/_/g, ' ');

    // Map agent name to data_dates directory key
    const agentToDirKey = (agent) => {
        if (agent === 'medication') return 'medications';
        return agent.replace(/_/g, '-');
    };

    // Extract data date for this citation's modality
    const patient = activePatientId ? patientDataMap[activePatientId] : null;
    let dataDate = '';
    if (patient?.data_dates && citation.web_path) {
        const parts = citation.web_path.split('/');
        const modalityDir = parts.length >= 3 ? parts[2] : '';
        const rawDate = patient.data_dates[modalityDir] || '';
        if (rawDate) dataDate = formatShortDate(rawDate);
    }

    // Look up date for a knowledge bus entry by agent name
    const getAgentDate = (agent) => {
        if (!patient?.data_dates) return '';
        const raw = patient.data_dates[agentToDirKey(agent)] || '';
        return raw ? formatShortDate(raw) : '';
    };

    let html = '<div class="prov-description">How this citation was produced \u2014 from raw patient data through agent analysis to cross-modal verification.</div>';
    html += '<div class="prov-pipeline">';

    // Processing steps
    const todayDate = formatShortDate(new Date().toISOString().slice(0, 10));
    for (const step of steps) {
        const dateStr = step.label === 'Data Origin' ? dataDate : todayDate;
        const timeHTML = dateStr ? `<span class="prov-step-time">${dateStr}</span>` : '';
        html += `<div class="prov-step">
            <div class="prov-step-label">${escapeHtml(step.label)}${timeHTML}</div>
            <div class="prov-step-desc">${escapeHtml(step.desc)}</div>
        </div>`;
    }

    // Separator before knowledge bus
    html += '<div class="prov-separator"></div>';

    // Knowledge bus node
    const hasBus = (bus.supported_by && bus.supported_by.length) || (bus.contradicted_by && bus.contradicted_by.length);
    html += '<div class="prov-bus">';
    html += '<div class="prov-bus-label">Cross-Modal Verification</div>';

    // Helper: build a mini citation badge for a KB entry agent
    const buildKBBadge = (entry) => {
        const entryAgent = agentBadgeLabels[entry.agent] || entry.agent.replace(/_/g, ' ');
        const entryDate = getAgentDate(entry.agent);
        // Try to build a citation object for hover
        const citObj = buildCitationForAgent(entry.agent);
        if (citObj) {
            const data = encodeURIComponent(JSON.stringify(citObj));
            return `<span class="citation-group"><sup class="citation prov-bus-badge" data-citation="${data}">${escapeHtml(entryAgent)}</sup>${entryDate ? `<span class="citation-date">${entryDate}</span>` : ''}</span>`;
        }
        return `<span class="prov-bus-agent">${escapeHtml(entryAgent)}</span>${entryDate ? `<span class="prov-step-time">${entryDate}</span>` : ''}`;
    };

    if (hasBus) {
        if (bus.supported_by && bus.supported_by.length) {
            html += '<div class="prov-bus-group">';
            html += '<div class="prov-bus-group-label supported">\u2713 Supported by</div>';
            for (const entry of bus.supported_by) {
                html += `<div class="prov-bus-entry">
                    <span class="prov-bus-icon supported">\u2713</span>
                    ${buildKBBadge(entry)}
                    <span class="prov-bus-finding">${escapeHtml(entry.finding)}</span>
                    <span class="prov-bus-reason">\u2014 ${escapeHtml(entry.reason)}</span>
                </div>`;
            }
            html += '</div>';
        }
        if (bus.contradicted_by && bus.contradicted_by.length) {
            html += '<div class="prov-bus-group">';
            html += '<div class="prov-bus-group-label contradicted">\u2717 Contradicted by</div>';
            for (const entry of bus.contradicted_by) {
                html += `<div class="prov-bus-entry">
                    <span class="prov-bus-icon contradicted">\u2717</span>
                    ${buildKBBadge(entry)}
                    <span class="prov-bus-finding">${escapeHtml(entry.finding)}</span>
                    <span class="prov-bus-reason">\u2014 ${escapeHtml(entry.reason)}</span>
                </div>`;
            }
            html += '</div>';
        }
    } else if (citation._kb_loaded) {
        html += '<div class="prov-bus-empty">No cross-modal correlations found</div>';
    } else {
        html += '<div class="prov-bus-loading"><span class="prov-bus-spinner"></span>Analyzing cross-modal correlations\u2026</div>';
    }
    html += '</div>';

    // Citation generated node
    html += `<div class="prov-citation">
        <div class="prov-citation-label">Citation Generated<span class="prov-step-time">${todayDate}</span></div>
        <div class="prov-citation-text">Gemini 3.1 Pro \u2014 orchestrator synthesis and citation formatting</div>
    </div>`;

    html += '</div>';
    return html;
}

function closeCitationModal() {
    citationBackdrop.classList.remove('open');
    citationModal.classList.remove('open');
    citationModal.classList.remove('flipped');
    citationModal.classList.remove('agent-view');
    citationModal.style.height = '';
    // Reset tabs to data
    modalTabToggle.querySelectorAll('.modal-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === 'data'));
    // Stop any playing media
    modalBody.querySelectorAll('video, audio').forEach(el => { el.pause(); el.src = ''; });
    modalBody.innerHTML = '';
    modalProvenancePanel.innerHTML = '';
    modalAgentPanel.innerHTML = '';
    currentModalPin = null;
}

// Click citation → open modal
chatContainer.addEventListener('click', (e) => {
    const target = e.target.closest('.citation');
    if (!target) return;
    e.preventDefault();
    // Hide hover card
    citationPopup.style.display = 'none';
    clearTimeout(hideTimeout);
    const citation = JSON.parse(decodeURIComponent(target.dataset.citation));
    openCitationModal(citation);
});

modalClose.addEventListener('click', closeCitationModal);
citationBackdrop.addEventListener('click', closeCitationModal);

// Modal pin button (citation pins only — text pins navigate directly)
modalPin.addEventListener('click', async () => {
    if (!activePatientId || !modalCitation) return;
    const existingPinId = findPinId(activePatientId, modalCitation);
    if (existingPinId) {
        await removePin(activePatientId, existingPinId);
        modalPin.classList.remove('pinned');
    } else {
        await addPin(activePatientId, { type: 'citation', citation: modalCitation, source: getActiveSessionId() });
        modalPin.classList.add('pinned');
    }
});

// Tab toggle — switch between data, provenance, and agent views
modalTabToggle.addEventListener('click', (e) => {
    const tab = e.target.closest('.modal-tab');
    if (!tab || !modalCitation) return;
    const target = tab.dataset.tab;
    // Lock height before toggling to prevent jump
    if (target !== 'data' && !citationModal.style.height) {
        citationModal.style.height = citationModal.offsetHeight + 'px';
    }
    citationModal.classList.toggle('flipped', target === 'provenance');
    citationModal.classList.toggle('agent-view', target === 'agent');
    // Update active tab
    modalTabToggle.querySelectorAll('.modal-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === target));
    // Build provenance content on first flip
    if (target === 'provenance' && !modalProvenancePanel.innerHTML) {
        modalProvenancePanel.innerHTML = buildProvenanceHTML(modalCitation);
    }
    if (target === 'data') {
        citationModal.style.height = '';
    }
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && lightboxEl) {
        closeLightbox();
        return;
    }
    if (e.key === 'Escape' && citationModal.classList.contains('open')) {
        if (citationModal.classList.contains('flipped') || citationModal.classList.contains('agent-view')) {
            citationModal.classList.remove('flipped');
            citationModal.classList.remove('agent-view');
            citationModal.style.height = '';
            modalTabToggle.querySelectorAll('.modal-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === 'data'));
        } else {
            closeCitationModal();
        }
    }
});

// Image lightbox
let lightboxEl = null;
function openLightbox(src) {
    if (lightboxEl) lightboxEl.remove();
    lightboxEl = document.createElement('div');
    lightboxEl.className = 'image-lightbox';
    lightboxEl.innerHTML = `<img src="${src}" alt="">`;
    lightboxEl.addEventListener('click', closeLightbox);
    document.body.appendChild(lightboxEl);
    requestAnimationFrame(() => lightboxEl.classList.add('open'));
}
function closeLightbox() {
    if (!lightboxEl) return;
    lightboxEl.classList.remove('open');
    lightboxEl.addEventListener('transitionend', () => { lightboxEl.remove(); lightboxEl = null; }, { once: true });
}
modalBody.addEventListener('click', (e) => {
    if (e.target.tagName === 'IMG') {
        openLightbox(e.target.src);
    }
});


// Annotation textarea: auto-grow + debounced save (auto-pins if needed)
modalAnnotationText.addEventListener('input', async () => {
    modalAnnotationText.style.height = 'auto';
    modalAnnotationText.style.height = modalAnnotationText.scrollHeight + 'px';
    if (!activePatientId) return;
    if (!currentModalPin && !(await ensurePinned())) return;
    currentModalPin.annotations = currentModalPin.annotations || { text: '', tags: [] };
    currentModalPin.annotations.text = modalAnnotationText.value;
    debouncedPatchAnnotations(activePatientId, currentModalPin.pin_id,
        { ...currentModalPin.annotations });
});

// ─── Pinned Text Highlighting ────────────────────────────────
function clearPinnedHighlights() {
    chatContainer.querySelectorAll('mark.pinned-highlight').forEach(mark => {
        const parent = mark.parentNode;
        parent.replaceChild(document.createTextNode(mark.textContent), mark);
        parent.normalize();
    });
}

function highlightPinnedText() {
    clearPinnedHighlights();
    if (!activePatientId) return;
    const pins = patientPins[activePatientId] || [];
    const textPins = pins.filter(p => p.type === 'text').map(p => p.text);
    if (textPins.length === 0) return;

    const bubbles = chatContainer.querySelectorAll('.chat-message.assistant .message-content');
    bubbles.forEach(bubble => {
        textPins.forEach(pinText => {
            highlightTextInNode(bubble, pinText);
        });
    });
}

function highlightTextInNode(root, searchText) {
    // Collect text nodes, skipping those inside citation badges and date spans
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
        acceptNode(node) {
            const el = node.parentElement;
            if (el && (el.closest('.citation') || el.closest('.citation-date'))) return NodeFilter.FILTER_REJECT;
            return NodeFilter.FILTER_ACCEPT;
        }
    });
    const textNodes = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode);
    if (textNodes.length === 0) return;

    // Build combined string and map each char to its text node + offset
    let combined = '';
    const charMap = []; // charMap[i] = { node, offset }
    for (const node of textNodes) {
        for (let j = 0; j < node.textContent.length; j++) {
            charMap.push({ node, offset: j });
            combined += node.textContent[j];
        }
    }

    // Find all occurrences in the combined string
    const matches = [];
    let searchFrom = 0;
    while (true) {
        const idx = combined.indexOf(searchText, searchFrom);
        if (idx === -1) break;
        matches.push(idx);
        searchFrom = idx + 1;
    }

    // Create highlights in reverse order to preserve offsets
    for (let i = matches.length - 1; i >= 0; i--) {
        try {
            const startIdx = matches[i];
            const endIdx = startIdx + searchText.length - 1;
            const startInfo = charMap[startIdx];
            const endInfo = charMap[endIdx];

            // If the match spans a single text node, wrap simply
            if (startInfo.node === endInfo.node) {
                const range = document.createRange();
                range.setStart(startInfo.node, startInfo.offset);
                range.setEnd(endInfo.node, endInfo.offset + 1);
                const mark = document.createElement('mark');
                mark.className = 'pinned-highlight';
                range.surroundContents(mark);
            } else {
                // Multi-node span: wrap each text node segment individually
                const segments = [];
                let curNode = null;
                let segStart = 0;
                for (let c = startIdx; c <= endIdx; c++) {
                    if (charMap[c].node !== curNode) {
                        if (curNode) segments.push({ node: curNode, start: segStart, end: charMap[c - 1].offset + 1 });
                        curNode = charMap[c].node;
                        segStart = charMap[c].offset;
                    }
                }
                if (curNode) segments.push({ node: curNode, start: segStart, end: charMap[endIdx].offset + 1 });

                // Wrap segments in reverse to preserve earlier offsets
                for (let s = segments.length - 1; s >= 0; s--) {
                    const seg = segments[s];
                    const range = document.createRange();
                    range.setStart(seg.node, seg.start);
                    range.setEnd(seg.node, seg.end);
                    const mark = document.createElement('mark');
                    mark.className = 'pinned-highlight';
                    range.surroundContents(mark);
                }
            }
        } catch (e) { /* skip if wrapping fails at element boundary */ }
    }
}

function scrollToTextInChat(text) {
    if (!text) return;
    // Find the first mark whose text starts the pinned text, then collect
    // all sibling marks in the same parent that together form the full highlight
    // (marks may be separated by citation <sup> or date <span> elements).
    const marks = chatContainer.querySelectorAll('mark.pinned-highlight');
    for (const mark of marks) {
        if (mark.textContent === text || text.startsWith(mark.textContent)) {
            mark.scrollIntoView({ behavior: 'smooth', block: 'center' });
            // Collect all marks in the same parent that belong to this highlight
            const parent = mark.parentNode;
            const allMarks = [];
            let found = false;
            for (const child of parent.childNodes) {
                if (child === mark) found = true;
                if (found && child.nodeType === Node.ELEMENT_NODE &&
                    child.tagName === 'MARK' && child.classList.contains('pinned-highlight')) {
                    allMarks.push(child);
                }
            }
            if (allMarks.length === 0) allMarks.push(mark);
            allMarks.forEach(m => {
                m.style.transition = 'background 0.3s';
                m.style.background = 'rgba(166, 120, 48, 0.35)';
                setTimeout(() => { m.style.background = ''; }, 1200);
            });
            return;
        }
    }
    // Fallback: search text nodes directly
    const bubbles = chatContainer.querySelectorAll('.chat-message.assistant .message-content');
    for (const bubble of bubbles) {
        if (bubble.textContent.includes(text)) {
            bubble.closest('.chat-message').scrollIntoView({ behavior: 'smooth', block: 'center' });
            return;
        }
    }
}

function scrollToCitationInChat(citation) {
    if (!citation) return;
    const badges = chatContainer.querySelectorAll('sup.citation');
    for (const badge of badges) {
        try {
            const c = JSON.parse(decodeURIComponent(badge.dataset.citation));
            if (c.agent === citation.agent && c.web_path === citation.web_path) {
                badge.scrollIntoView({ behavior: 'smooth', block: 'center' });
                badge.style.transition = 'background 0.3s';
                badge.style.background = 'rgba(166, 120, 48, 0.35)';
                setTimeout(() => { badge.style.background = ''; }, 1200);
                return;
            }
        } catch (e) { /* skip malformed */ }
    }
}

// ─── Text Selection Pin Button ──────────────────────────────
const textSelectPin = document.createElement('button');
textSelectPin.className = 'text-select-pin';
textSelectPin.innerHTML = '&#x1F4CC;<span class="text-select-pin-label">Pin</span>';
document.body.appendChild(textSelectPin);

function getActiveSessionId() {
    return activeConversationId || sessionId || '';
}

function hideTextSelectPin() {
    textSelectPin.style.display = 'none';
}

// Extract selected text, skipping citation badge and date elements
function getCleanSelectionText(sel) {
    if (!sel.rangeCount) return '';
    const range = sel.getRangeAt(0);
    const frag = range.cloneContents();
    // Remove citation badges and date spans from the clone
    frag.querySelectorAll('.citation, .citation-date').forEach(el => el.remove());
    return (frag.textContent || '').trim();
}

chatContainer.addEventListener('mouseup', (e) => {
    // Ignore if clicking on citation badges or the pin button itself
    if (e.target.closest('.citation') || e.target.closest('.text-select-pin')) return;
    if (!activePatientId) return;

    const sel = window.getSelection();
    const text = getCleanSelectionText(sel);
    if (!text) { hideTextSelectPin(); return; }

    // Check if selection is inside an assistant message bubble
    if (!sel.anchorNode) { hideTextSelectPin(); return; }
    const bubble = sel.anchorNode.nodeType === Node.TEXT_NODE
        ? sel.anchorNode.parentElement.closest('.chat-message.assistant .message-bubble')
        : sel.anchorNode.closest('.chat-message.assistant .message-bubble');
    if (!bubble) { hideTextSelectPin(); return; }

    // Position above selection
    const range = sel.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    textSelectPin.style.display = 'block';
    const btnW = textSelectPin.offsetWidth;
    let left = rect.left + (rect.width / 2) - (btnW / 2);
    left = Math.max(8, Math.min(left, window.innerWidth - btnW - 8));
    let top = rect.top - 36;
    if (top < 8) top = rect.bottom + 8;
    textSelectPin.style.left = `${left}px`;
    textSelectPin.style.top = `${top}px`;
});

textSelectPin.addEventListener('mousedown', async (e) => {
    e.preventDefault();
    e.stopPropagation();
    const sel = window.getSelection();
    const text = getCleanSelectionText(sel);
    if (!text || !activePatientId) return;

    const sourceSessionId = getActiveSessionId();
    hideTextSelectPin();
    await addPin(activePatientId, { type: 'text', text, source: sourceSessionId });
    sel.removeAllRanges();
});

document.addEventListener('mousedown', (e) => {
    if (e.target.closest('.text-select-pin')) return;
    hideTextSelectPin();
});

// Event listeners
sendBtn.addEventListener('click', sendMessage);
clearBtn.addEventListener('click', clearChat);
attachBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
fileRemove.addEventListener('click', removeFile);

// Update send button when typing
messageInput.addEventListener('input', updateSendButton);

// Allow Enter key to send (with Shift+Enter for new line)
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!sendBtn.disabled) {
            sendMessage();
        }
    }
});

// Initialize
connectWebSocket();
loadPatients();

