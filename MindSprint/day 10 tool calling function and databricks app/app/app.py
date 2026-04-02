"""
Databricks App — Simple Claude Agent
Uses mlflow.deployments for model calls — no manual host/token URL construction.
Credentials are auto-discovered from the Databricks Apps runtime.
"""

import ast
import asyncio
import json
import operator
import os
import uuid

from databricks.sdk import WorkspaceClient
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from mlflow.deployments import get_deploy_client
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL = os.environ.get("CLAUDE_MODEL", "databricks-claude-sonnet-4")

SYSTEM_PROMPT = """You are a helpful data assistant running inside Databricks.
You have access to two tools:
- calculator: for any arithmetic or mathematical expressions
- run_sql: to query Unity Catalog tables with read-only Spark SQL

Always use a tool when the question involves numbers or data.
Be concise and direct in your answers."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression. Use for any arithmetic or numerical calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '(120 * 0.15) + 30'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Run a read-only SQL query against Unity Catalog. Use SHOW CATALOGS or SHOW TABLES to explore.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A SELECT or SHOW SQL statement."}
                },
                "required": ["query"],
            },
        },
    },
]

SESSIONS: dict[str, list[dict]] = {}
app = FastAPI(title="Claude Agent")

# Auto-discover credentials — works in Databricks Apps without any env var setup
deploy_client = get_deploy_client("databricks")
sdk_client    = WorkspaceClient()


# ── Tools ───────────────────────────────────────────────────────────────────────

def calculator(expression: str) -> str:
    SAFE_OPS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.USub: operator.neg,
    }
    def _eval(node):
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.BinOp):
            return SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return SAFE_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported: {ast.dump(node)}")
    try:
        return str(_eval(ast.parse(expression.strip(), mode="eval").body))
    except Exception as e:
        return f"Error: {e}"


def _get_warehouse_id() -> str:
    wh_id = os.environ.get("SQL_WAREHOUSE_ID")
    if wh_id:
        return wh_id
    warehouses = list(sdk_client.warehouses.list())
    if not warehouses:
        raise RuntimeError("No SQL warehouses found.")
    return warehouses[0].id


def run_sql(query: str) -> str:
    blocked = ["insert", "update", "delete", "drop", "create", "alter", "truncate", "merge"]
    if any(k in query.lower() for k in blocked):
        return json.dumps({"error": "Only SELECT/SHOW queries allowed."})
    try:
        from databricks.sdk.service.sql import StatementState
        result = sdk_client.statement_execution.execute_statement(
            statement=query,
            warehouse_id=_get_warehouse_id(),
        )
        if result.status.state != StatementState.SUCCEEDED:
            return json.dumps({"error": str(result.status.error)})
        cols = [c.name for c in (result.manifest.schema.columns or [])]
        rows = []
        if result.result and result.result.data_array:
            for row in result.result.data_array[:50]:
                rows.append(dict(zip(cols, row)))
        return json.dumps(rows, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


TOOL_HANDLERS = {
    "calculator": lambda a: calculator(a["expression"]),
    "run_sql":    lambda a: run_sql(a["query"]),
}


# ── Agent (sync, offloaded to thread pool) ─────────────────────────────────────

def _run_agent_sync(messages: list[dict]) -> str:
    for _ in range(10):
        response = deploy_client.predict(
            endpoint=MODEL,
            inputs={
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                "tools": TOOLS,
                "max_tokens": 2048,
            },
        )
        msg        = response["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        messages.append(msg)

        if not tool_calls:
            return msg.get("content", "")

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = json.loads(tc["function"]["arguments"])
            result  = TOOL_HANDLERS.get(fn_name, lambda _: "Unknown tool")(fn_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })
    return "Reached max reasoning steps."


async def run_agent(messages: list[dict]) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_agent_sync, messages)


async def word_stream(text: str):
    words, buf = text.split(), []
    for i, w in enumerate(words):
        buf.append(w)
        if len(buf) >= 4 or i == len(words) - 1:
            yield f"data: {json.dumps({'delta': ' '.join(buf) + ' '})}\n\n"
            buf = []
            await asyncio.sleep(0.02)
    yield f"data: {json.dumps({'done': True})}\n\n"


# ── Routes ──────────────────────────────────────────────────────────────────────

class ChatReq(BaseModel):
    message: str
    session_id: str | None = None

class ClearReq(BaseModel):
    session_id: str


@app.post("/chat")
async def chat(req: ChatReq):
    sid      = req.session_id or str(uuid.uuid4())
    messages = SESSIONS.setdefault(sid, [])
    messages.append({"role": "user", "content": req.message})
    try:
        answer = await run_agent(messages)
    except Exception as e:
        answer = f"Error: {e}"
    messages.append({"role": "assistant", "content": answer})
    SESSIONS[sid] = messages[-40:]

    async def stream():
        yield f"data: {json.dumps({'session_id': sid})}\n\n"
        async for chunk in word_stream(answer):
            yield chunk

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/clear")
async def clear(req: ClearReq):
    SESSIONS.pop(req.session_id, None)
    return {"ok": True}


@app.get("/health")
async def health():
    return {"model": MODEL, "status": "ok"}


# ── Frontend ────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Databricks Agent</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0d0f16;--surf:#161923;--surf2:#1e2233;--border:#272d42;--text:#dde1ef;
      --muted:#737a96;--accent:#7c6dfa;--accent2:#6558e8;--user:#1a2445;
      --radius:14px;--font:"Inter",system-ui,sans-serif}
body{font-family:var(--font);background:var(--bg);color:var(--text);
     height:100vh;display:flex;flex-direction:column;overflow:hidden}
header{padding:14px 22px;background:var(--surf);border-bottom:1px solid var(--border);
       display:flex;align-items:center;gap:12px;flex-shrink:0}
.logo{width:34px;height:34px;background:var(--accent);border-radius:9px;
      display:flex;align-items:center;justify-content:center;font-size:16px}
header h1{font-size:15px;font-weight:600}
header small{font-size:11px;color:var(--muted);display:block;margin-top:1px}
.badge{margin-left:auto;padding:3px 10px;background:#162a1a;border:1px solid #1f4d25;
       border-radius:20px;font-size:11px;color:#4caf75}
#msgs{flex:1;overflow-y:auto;padding:22px;display:flex;flex-direction:column;
      gap:18px;scroll-behavior:smooth}
#msgs::-webkit-scrollbar{width:5px}
#msgs::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
.row{display:flex;gap:10px;max-width:780px;width:100%}
.row.user{align-self:flex-end;flex-direction:row-reverse}
.row.bot{align-self:flex-start}
.av{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;
    justify-content:center;font-size:13px;flex-shrink:0;margin-top:2px}
.row.user .av{background:var(--accent)}
.row.bot  .av{background:var(--surf2);border:1px solid var(--border)}
.bbl{padding:11px 15px;border-radius:var(--radius);font-size:14px;line-height:1.65;max-width:660px}
.row.user .bbl{background:var(--user);border:1px solid #283870;border-bottom-right-radius:4px}
.row.bot  .bbl{background:var(--surf2);border:1px solid var(--border);border-bottom-left-radius:4px}
.cur{display:inline-block;width:2px;height:13px;background:var(--accent);
     margin-left:2px;vertical-align:middle;animation:blink .7s steps(1) infinite}
@keyframes blink{50%{opacity:0}}
#empty{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
       gap:8px;color:var(--muted);text-align:center;padding-bottom:60px}
#empty .icon{font-size:42px;margin-bottom:4px}
#empty h2{color:var(--text);font-size:17px;font-weight:600}
.chips{display:flex;flex-wrap:wrap;gap:7px;max-width:780px;margin:0 auto 10px}
.chip{padding:6px 14px;background:var(--surf2);border:1px solid var(--border);
      border-radius:20px;font-size:12px;color:var(--muted);cursor:pointer;transition:all .15s}
.chip:hover{border-color:var(--accent);color:var(--text)}
footer{padding:14px 22px;background:var(--surf);border-top:1px solid var(--border);flex-shrink:0}
.irow{display:flex;gap:8px;max-width:780px;margin:0 auto}
textarea{flex:1;background:var(--surf2);border:1px solid var(--border);border-radius:var(--radius);
         color:var(--text);font-family:var(--font);font-size:14px;padding:11px 15px;
         resize:none;outline:none;line-height:1.5;max-height:130px;transition:border-color .15s}
textarea:focus{border-color:var(--accent)}
textarea::placeholder{color:var(--muted)}
.send{width:42px;height:42px;background:var(--accent);border:none;border-radius:10px;color:#fff;
      cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;
      align-self:flex-end;transition:background .15s,transform .1s}
.send:hover{background:var(--accent2)}.send:active{transform:scale(.94)}
.send:disabled{opacity:.35;cursor:not-allowed}
.clr{width:auto;padding:0 13px;background:transparent;border:1px solid var(--border);
     border-radius:10px;color:var(--muted);font-size:12px;cursor:pointer;
     align-self:flex-end;height:42px;transition:all .15s}
.clr:hover{border-color:var(--text);color:var(--text)}
</style>
</head>
<body>
<header>
  <div class="logo">⚡</div>
  <div><h1>Databricks Agent</h1><small>Claude · calculator · Spark SQL</small></div>
  <div class="badge">● Online</div>
</header>
<div id="msgs">
  <div id="empty">
    <div class="icon">🤖</div>
    <h2>What can I help you with?</h2>
    <p>Ask me anything — I can do math and query your Unity Catalog tables.</p>
  </div>
</div>
<footer>
  <div class="chips" id="chips">
    <div class="chip" onclick="ask(this)">What catalogs exist in this workspace?</div>
    <div class="chip" onclick="ask(this)">Compound interest: $10k at 8% for 20 years</div>
    <div class="chip" onclick="ask(this)">Show tables in the default catalog</div>
    <div class="chip" onclick="ask(this)">What's 15% tip on a $87.50 bill?</div>
  </div>
  <div class="irow">
    <textarea id="inp" rows="1" placeholder="Ask anything…"
      onkeydown="onKey(event)" oninput="resize(this)"></textarea>
    <button class="clr" onclick="clear_()">Clear</button>
    <button class="send" id="sbtn" onclick="send()">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <line x1="22" y1="2" x2="11" y2="13"/>
        <polygon points="22 2 15 22 11 13 2 9 22 2"/>
      </svg>
    </button>
  </div>
</footer>
<script>
let sid=null,busy=false;
const msgsEl=document.getElementById("msgs"),inp=document.getElementById("inp"),
      sbtn=document.getElementById("sbtn"),empty=document.getElementById("empty"),
      chips=document.getElementById("chips");
function resize(el){el.style.height="auto";el.style.height=Math.min(el.scrollHeight,130)+"px"}
function onKey(e){if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send()}}
function ask(el){inp.value=el.textContent;resize(inp);send()}
function scroll(){msgsEl.scrollTop=msgsEl.scrollHeight}
function addBubble(role,html){
  empty.style.display="none";chips.style.display="none";
  const row=document.createElement("div");row.className="row "+role;
  const av=document.createElement("div");av.className="av";av.textContent=role==="user"?"👤":"🤖";
  const bbl=document.createElement("div");bbl.className="bbl";bbl.innerHTML=html;
  row.appendChild(av);row.appendChild(bbl);msgsEl.appendChild(row);scroll();return bbl;
}
function fmt(t){
  return t.replace(/\*\*(.*?)\*\*/g,"<strong>$1</strong>")
    .replace(/`([^`]+)`/g,"<code style='background:#0f0d20;padding:1px 5px;border-radius:4px;font-size:12px'>$1</code>")
    .split("\n\n").map(p=>"<p style='margin-bottom:8px'>"+p.trim()+"</p>").join("");
}
async function send(){
  const msg=inp.value.trim();if(!msg||busy)return;
  addBubble("user",msg);inp.value="";resize(inp);sbtn.disabled=true;busy=true;
  const bbl=addBubble("bot","");
  const cur=document.createElement("span");cur.className="cur";bbl.appendChild(cur);
  let full="";
  try{
    const r=await fetch("/chat",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({message:msg,session_id:sid})});
    const reader=r.body.getReader(),dec=new TextDecoder();
    while(true){
      const{done,value}=await reader.read();if(done)break;
      for(const line of dec.decode(value).split("\n")){
        if(!line.startsWith("data: "))continue;
        const d=JSON.parse(line.slice(6));
        if(d.session_id)sid=d.session_id;
        if(d.delta){full+=d.delta;bbl.textContent=full;bbl.appendChild(cur)}
        if(d.done){cur.remove();bbl.innerHTML=fmt(full)}
      }
      scroll();
    }
  }catch(e){cur.remove();bbl.innerHTML="<span style='color:#f87171'>Error: "+e.message+"</span>"}
  sbtn.disabled=false;busy=false;inp.focus();
}
async function clear_(){
  if(sid)await fetch("/clear",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({session_id:sid})});
  sid=null;msgsEl.innerHTML="";msgsEl.appendChild(empty);
  empty.style.display="flex";chips.style.display="flex";
}
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML)
