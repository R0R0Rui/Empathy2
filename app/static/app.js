const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chatForm");
const inputEl = document.querySelector("#messageInput");
const sendButtonEl = document.querySelector("#sendButton");
const restartButtonEl = document.querySelector("#restartButton");

let sessionId = null;
let busy = false;

function setBusy(nextBusy) {
  busy = nextBusy;
  sendButtonEl.disabled = nextBusy;
  restartButtonEl.disabled = nextBusy;
}

function scrollToBottom() {
  window.requestAnimationFrame(() => {
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  });
}

function autosizeInput() {
  inputEl.style.height = "0px";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 170)}px`;
}

function clearEmptyState() {
  const emptyStateEl = document.querySelector("#emptyState");
  if (emptyStateEl) {
    emptyStateEl.remove();
  }
}

function createBubble(role, text, options = {}) {
  clearEmptyState();
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}${options.error ? " error" : ""}${options.typing ? " typing" : ""}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrapper.appendChild(bubble);

  if (options.meta?.length) {
    const meta = document.createElement("div");
    meta.className = "meta";
    for (const item of options.meta) {
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.textContent = item;
      meta.appendChild(pill);
    }
    wrapper.appendChild(meta);
  }

  messagesEl.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

function formatEmotion(emotion) {
  return `${emotion.label} ${Math.round(emotion.confidence * 100)}%`;
}

function responseMeta(response) {
  const emotions = response.emotions;
  const performance = response.performance;
  return [
    formatEmotion(emotions.primary),
    formatEmotion(emotions.secondary),
    formatEmotion(emotions.tertiary),
    `${Math.round(performance.total_ms)} ms`,
  ];
}

async function postChat(message) {
  const payload = { message };
  if (sessionId) {
    payload.session_id = sessionId;
  }

  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Request failed with ${response.status}`);
  }
  return data;
}

async function resetConversation() {
  if (busy) {
    return;
  }

  const previousSessionId = sessionId;
  sessionId = null;
  messagesEl.innerHTML = `
    <div class="empty-state" id="emptyState">
      <h1>What’s on your mind?</h1>
    </div>
  `;

  if (previousSessionId) {
    try {
      await fetch(`/sessions/${encodeURIComponent(previousSessionId)}/reset`, { method: "POST" });
    } catch {
      // A page-local reset is still valid if the server-side cleanup request fails.
    }
  }

  inputEl.value = "";
  autosizeInput();
  inputEl.focus();
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) {
    return;
  }

  const message = inputEl.value.trim();
  if (!message) {
    return;
  }

  inputEl.value = "";
  autosizeInput();
  createBubble("user", message);
  const typingEl = createBubble("assistant", "Thinking…", { typing: true });

  setBusy(true);
  try {
    const response = await postChat(message);
    sessionId = response.session_id;
    typingEl.remove();
    createBubble("assistant", response.reply, { meta: responseMeta(response) });
  } catch (error) {
    typingEl.remove();
    createBubble("assistant", error.message, { error: true });
  } finally {
    setBusy(false);
    inputEl.focus();
  }
});

inputEl.addEventListener("input", autosizeInput);
inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});
restartButtonEl.addEventListener("click", resetConversation);
autosizeInput();
