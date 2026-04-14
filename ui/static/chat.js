const messages = document.getElementById('messages');
const queryInput = document.getElementById('query');
const sendBtn = document.getElementById('sendBtn');
const newConversationBtn = document.getElementById('newConversationBtn');
let sessionId = localStorage.getItem('procurement_session_id') || null;
let activeController = null;
const STREAM_IDLE_TIMEOUT_MS = 45000;

function addMessage(text, role) {
    const el = document.createElement('div');
    el.className = `msg ${role}`;
    el.textContent = text;
    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
}

addMessage('Hello! I can help with procurement requests. What would you like to buy?', 'assistant');

async function resetConversation() {
    if (activeController) {
        activeController.abort();
        activeController = null;
    }

    const currentSessionId = sessionId;
    sessionId = null;
    localStorage.removeItem('procurement_session_id');

    if (currentSessionId) {
        try {
            await fetch(`/chat/session/${encodeURIComponent(currentSessionId)}`, { method: 'DELETE' });
        } catch (err) {
            // Ignore cleanup failures and still reset local state.
        }
    }

    messages.innerHTML = '';
    addMessage('Hello! I can help with procurement requests. What would you like to buy?', 'assistant');
    queryInput.focus();
}

async function sendMessage() {
    const message = queryInput.value.trim();
    if (!message) return;
    if (activeController) return;

    addMessage(message, 'user');
    queryInput.value = '';
    sendBtn.disabled = true;
    if (newConversationBtn) newConversationBtn.disabled = true;

    const loading = document.createElement('div');
    loading.className = 'msg assistant';
    loading.textContent = 'Thinking...';
    messages.appendChild(loading);
    messages.scrollTop = messages.scrollHeight;

    try {
        let streamSucceeded = false;

        for (let attempt = 0; attempt < 2; attempt += 1) {
            const controller = new AbortController();
            activeController = controller;

            let idleTimer = null;
            const resetIdleTimer = () => {
                if (idleTimer) clearTimeout(idleTimer);
                idleTimer = setTimeout(() => controller.abort('stream_timeout'), STREAM_IDLE_TIMEOUT_MS);
            };

            let receivedAnyChunk = false;

            try {
                resetIdleTimer();
                const res = await fetch('/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, session_id: sessionId }),
                    signal: controller.signal
                });

                if (!res.ok) {
                    const data = await res.json();
                    loading.remove();
                    addMessage(`Error: ${data.detail || 'Unable to process request.'}`, 'assistant');
                    return;
                }

                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let streamText = '';

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    receivedAnyChunk = true;
                    resetIdleTimer();
                    buffer += decoder.decode(value, { stream: true });

                    const events = buffer.split('\n\n');
                    buffer = events.pop() || '';

                    for (const rawEvent of events) {
                        const lines = rawEvent.split('\n');
                        const eventType = lines.find((l) => l.startsWith('event:'))?.replace('event:', '').trim();
                        const dataLines = lines
                            .filter((l) => l.startsWith('data:'))
                            .map((l) => l.replace('data:', '').trim());
                        if (!dataLines.length) continue;

                        let payload;
                        try {
                            payload = JSON.parse(dataLines.join('\n'));
                        } catch (_) {
                            continue;
                        }

                        if (eventType === 'meta' && payload.session_id) {
                            sessionId = payload.session_id;
                            localStorage.setItem('procurement_session_id', sessionId);
                        }

                        if (eventType === 'token') {
                            streamText += payload.text || '';
                            loading.textContent = streamText.trim() || 'Thinking...';
                            messages.scrollTop = messages.scrollHeight;
                        }

                        if (eventType === 'done') {
                            loading.textContent = payload.answer || streamText.trim() || 'No response generated.';
                        }

                        if (eventType === 'error') {
                            loading.textContent = `Error: ${payload.detail || 'Unable to process request.'}`;
                        }
                    }
                }

                streamSucceeded = true;
                break;
            } catch (err) {
                const isAbort = err.name === 'AbortError';
                const isTimeoutAbort = isAbort && controller.signal.reason === 'stream_timeout';

                if (isTimeoutAbort) {
                    loading.textContent = 'Request timed out while streaming response. Please try again.';
                    break;
                }

                if (isAbort) {
                    loading.remove();
                    break;
                }

                const shouldRetry = attempt === 0 && !receivedAnyChunk;
                if (shouldRetry) {
                    loading.textContent = 'Connection interrupted. Retrying...';
                    continue;
                }

                loading.remove();
                addMessage('Network error while calling chat service.', 'assistant');
                break;
            } finally {
                if (idleTimer) clearTimeout(idleTimer);
                if (activeController === controller) {
                    activeController = null;
                }
            }
        }

        if (!streamSucceeded && loading.isConnected && loading.textContent === 'Thinking...') {
            loading.textContent = 'No response generated.';
        }
    } catch (err) {
        loading.remove();
        addMessage('Network error while calling chat service.', 'assistant');
    } finally {
        sendBtn.disabled = false;
        if (newConversationBtn) newConversationBtn.disabled = false;
        queryInput.focus();
    }
}

sendBtn.addEventListener('click', sendMessage);
if (newConversationBtn) {
    newConversationBtn.addEventListener('click', resetConversation);
}
queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
