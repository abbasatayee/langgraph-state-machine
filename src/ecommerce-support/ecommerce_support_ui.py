import json
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from ecommerce_support_agent import (
    MODEL_NAME,
    app,
    _extract_interrupt_payload,
    _latest_ai_text,
    _redacted_memory,
)


HOST = "127.0.0.1"
PORT = 8080


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>E-commerce Support Agent</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1d2433;
      --muted: #667085;
      --line: #d8dde6;
      --accent: #1463ff;
      --accent-dark: #0f4fd1;
      --danger: #b42318;
      --ok: #067647;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      display: grid;
      place-items: center;
    }

    main {
      width: min(1120px, 100vw);
      height: min(820px, 100vh);
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      background: var(--panel);
      border: 1px solid var(--line);
    }

    .chat {
      min-width: 0;
      display: grid;
      grid-template-rows: auto 1fr auto;
      border-right: 1px solid var(--line);
    }

    header, aside {
      padding: 18px 20px;
    }

    header {
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
    }

    h1 {
      font-size: 18px;
      line-height: 1.2;
      margin: 0 0 4px;
    }

    .sub {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }

    .thread {
      color: var(--muted);
      font-size: 12px;
      text-align: right;
      overflow-wrap: anywhere;
    }

    #messages {
      overflow-y: auto;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      background: #fbfcfe;
    }

    .msg {
      max-width: min(680px, 92%);
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      line-height: 1.45;
      font-size: 14px;
      white-space: pre-wrap;
    }

    .user {
      align-self: flex-end;
      background: #eaf1ff;
      border-color: #b8cdfd;
    }

    .agent {
      align-self: flex-start;
      background: #ffffff;
    }

    .approval {
      align-self: stretch;
      max-width: none;
      border-color: #f6c177;
      background: #fff8eb;
    }

    .approval-actions {
      display: flex;
      gap: 10px;
      margin-top: 12px;
    }

    form {
      padding: 14px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      border-top: 1px solid var(--line);
      background: var(--panel);
    }

    textarea {
      min-height: 48px;
      max-height: 140px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      font: inherit;
      line-height: 1.35;
    }

    button {
      border: 0;
      border-radius: 8px;
      padding: 0 16px;
      min-height: 42px;
      font: inherit;
      font-weight: 650;
      color: #ffffff;
      background: var(--accent);
      cursor: pointer;
    }

    button:hover { background: var(--accent-dark); }
    button:disabled { opacity: .55; cursor: wait; }
    button.reject { background: var(--danger); }
    button.approve { background: var(--ok); }
    button.secondary {
      color: var(--text);
      background: #eef1f6;
    }

    aside {
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 18px;
      background: #ffffff;
      min-width: 0;
    }

    aside h2 {
      font-size: 14px;
      margin: 0 0 8px;
    }

    .examples {
      display: grid;
      gap: 8px;
    }

    .example {
      width: 100%;
      text-align: left;
      color: var(--text);
      background: #f3f5f8;
      font-weight: 500;
      padding: 10px;
      min-height: auto;
    }

    pre {
      margin: 0;
      padding: 12px;
      overflow: auto;
      background: #f7f8fb;
      border: 1px solid var(--line);
      border-radius: 8px;
      font-size: 12px;
      color: var(--muted);
    }

    @media (max-width: 860px) {
      body { display: block; }
      main {
        height: 100vh;
        border: 0;
        grid-template-columns: 1fr;
      }
      aside { display: none; }
      .chat { border-right: 0; }
      header { align-items: flex-start; flex-direction: column; }
      .thread { text-align: left; }
    }
  </style>
</head>
<body>
  <main>
    <section class="chat">
      <header>
        <div>
          <h1>E-commerce Support Agent</h1>
          <p class="sub">LangGraph multi-agent support with verified memory and refund approval.</p>
        </div>
        <div class="thread">
          <div>Model: <span id="model"></span></div>
          <div>Thread: <span id="thread"></span></div>
        </div>
      </header>

      <section id="messages" aria-live="polite"></section>

      <form id="chat-form">
        <textarea id="message" placeholder="Ask about an order, refund, shipping, or reorder..." autocomplete="off"></textarea>
        <button id="send" type="submit">Send</button>
      </form>
    </section>

    <aside>
      <section>
        <h2>Try These</h2>
        <div class="examples">
          <button class="example" type="button">Where is my order ORD-456? My email is abbas@example.com and DOB is 1995-04-20.</button>
          <button class="example" type="button">Can I reorder the same item?</button>
          <button class="example" type="button">I forgot my order ID. Can you find my recent orders? My email is abbas@example.com and DOB is 1995-04-20.</button>
          <button class="example" type="button">Please refund order ORD-123. My email is abbas@example.com and DOB is 1995-04-20.</button>
        </div>
      </section>

      <section>
        <h2>Verified Memory</h2>
        <button class="secondary" id="refresh-memory" type="button">Refresh</button>
      </section>

      <pre id="memory">{}</pre>
    </aside>
  </main>

  <script>
    const threadId = localStorage.getItem("support_thread_id") || `ui-${crypto.randomUUID()}`;
    localStorage.setItem("support_thread_id", threadId);

    const modelEl = document.querySelector("#model");
    const threadEl = document.querySelector("#thread");
    const messagesEl = document.querySelector("#messages");
    const formEl = document.querySelector("#chat-form");
    const messageEl = document.querySelector("#message");
    const sendEl = document.querySelector("#send");
    const memoryEl = document.querySelector("#memory");

    modelEl.textContent = "__MODEL_NAME__";
    threadEl.textContent = threadId;

    function addMessage(role, text, extraClass = "") {
      const div = document.createElement("div");
      div.className = `msg ${role} ${extraClass}`.trim();
      div.textContent = text;
      messagesEl.appendChild(div);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return div;
    }

    function setBusy(busy) {
      sendEl.disabled = busy;
      messageEl.disabled = busy;
    }

    async function postJson(path, payload) {
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Request failed");
      return data;
    }

    async function refreshMemory() {
      const response = await fetch(`/memory?thread_id=${encodeURIComponent(threadId)}`);
      const data = await response.json();
      memoryEl.textContent = JSON.stringify(data.memory || {}, null, 2);
    }

    function showApproval(interruptPayload) {
      const box = addMessage("agent", JSON.stringify(interruptPayload, null, 2), "approval");
      const actions = document.createElement("div");
      actions.className = "approval-actions";

      const approve = document.createElement("button");
      approve.className = "approve";
      approve.textContent = "Approve Refund";

      const reject = document.createElement("button");
      reject.className = "reject";
      reject.textContent = "Reject";

      actions.append(approve, reject);
      box.appendChild(actions);

      async function decide(approved) {
        approve.disabled = true;
        reject.disabled = true;
        const data = await postJson("/approve", { thread_id: threadId, approved });
        if (data.answer) addMessage("agent", data.answer);
        if (data.interrupt) showApproval(data.interrupt);
        refreshMemory();
      }

      approve.addEventListener("click", () => decide(true));
      reject.addEventListener("click", () => decide(false));
    }

    async function sendMessage(text) {
      addMessage("user", text);
      setBusy(true);
      try {
        const data = await postJson("/chat", { thread_id: threadId, message: text });
        if (data.answer) addMessage("agent", data.answer);
        if (data.interrupt) showApproval(data.interrupt);
        refreshMemory();
      } catch (error) {
        addMessage("agent", error.message);
      } finally {
        setBusy(false);
        messageEl.focus();
      }
    }

    formEl.addEventListener("submit", (event) => {
      event.preventDefault();
      const text = messageEl.value.trim();
      if (!text) return;
      messageEl.value = "";
      sendMessage(text);
    });

    messageEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        formEl.requestSubmit();
      }
    });

    document.querySelectorAll(".example").forEach((button) => {
      button.addEventListener("click", () => {
        messageEl.value = button.textContent;
        messageEl.focus();
      });
    });

    document.querySelector("#refresh-memory").addEventListener("click", refreshMemory);

    addMessage("agent", "Hi. I can help with orders, shipping, refunds, and reorders. For order-specific requests, include your order ID, email, and date of birth first.");
    refreshMemory();
  </script>
</body>
</html>
""".replace("__MODEL_NAME__", MODEL_NAME)


def graph_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json(handler: BaseHTTPRequestHandler) -> dict:
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length == 0:
        return {}
    raw_body = handler.rfile.read(content_length).decode("utf-8")
    return json.loads(raw_body)


def run_chat(message: str, thread_id: str) -> dict:
    result = app.invoke(
        {
            "messages": [HumanMessage(content=message)],
            "tool_call_count": 0,
        },
        config=graph_config(thread_id),
    )

    interrupt_payload = _extract_interrupt_payload(result)
    if interrupt_payload is not None:
        return {"thread_id": thread_id, "interrupt": interrupt_payload, "answer": None}

    return {"thread_id": thread_id, "interrupt": None, "answer": _latest_ai_text(result)}


def resume_approval(approved: bool, thread_id: str) -> dict:
    result = app.invoke(Command(resume=approved), config=graph_config(thread_id))

    interrupt_payload = _extract_interrupt_payload(result)
    if interrupt_payload is not None:
        return {"thread_id": thread_id, "interrupt": interrupt_payload, "answer": None}

    return {"thread_id": thread_id, "interrupt": None, "answer": _latest_ai_text(result)}


def get_memory(thread_id: str) -> dict:
    snapshot = app.get_state(graph_config(thread_id))
    values = snapshot.values if snapshot else {}
    return {"thread_id": thread_id, "memory": _redacted_memory(values.get("memory"))}


class SupportUIHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/memory":
            query = parse_qs(parsed.query)
            thread_id = query.get("thread_id", [f"ui-{uuid.uuid4().hex[:8]}"])[0]
            json_response(self, 200, get_memory(thread_id))
            return

        if parsed.path == "/health":
            json_response(self, 200, {"ok": True, "model": MODEL_NAME})
            return

        json_response(self, 404, {"error": "Not found"})

    def do_POST(self) -> None:
        try:
            payload = read_json(self)
            thread_id = payload.get("thread_id") or f"ui-{uuid.uuid4().hex[:8]}"

            if self.path == "/chat":
                message = str(payload.get("message", "")).strip()
                if not message:
                    json_response(self, 400, {"error": "Missing message."})
                    return
                json_response(self, 200, run_chat(message, thread_id))
                return

            if self.path == "/approve":
                approved = bool(payload.get("approved"))
                json_response(self, 200, resume_approval(approved, thread_id))
                return

            json_response(self, 404, {"error": "Not found"})
        except Exception as exc:
            json_response(self, 500, {"error": str(exc)})

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")


def run_server() -> None:
    server = ThreadingHTTPServer((HOST, PORT), SupportUIHandler)
    print(f"E-commerce Support UI running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
