import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.requests import Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.messages import AIMessage, HumanMessage

from core.config import load_environment
from models.schemas import ChatIn
from services.agent import get_executor

load_environment()

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Procurement Agent MVP")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "ui" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "ui" / "templates"))
executor = get_executor()
session_history = defaultdict(list)


def resolve_session_id(session_id: Optional[str]) -> str:
    return session_id or str(uuid4())


def invoke_agent(message: str, sid: str) -> str:
    history = session_history[sid]
    result = executor.invoke({"input": message, "chat_history": history})
    answer = result.get("output", "No response generated.")
    history.append(HumanMessage(content=message))
    history.append(AIMessage(content=answer))
    return answer


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/chat")
def chat_get_help():
    return {
        "detail": "Method Not Allowed for GET /chat. Use POST /chat with JSON body: {\"message\": \"...\"}"
    }

@app.post("/chat")
def chat(payload: ChatIn):
    sid = resolve_session_id(payload.session_id)
    try:
        answer = invoke_agent(payload.message, sid)
        return {"answer": answer, "session_id": sid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {e}")


@app.post("/chat/stream")
async def chat_stream(payload: ChatIn):
    sid = resolve_session_id(payload.session_id)

    async def event_generator():
        yield sse_event("meta", {"session_id": sid})
        yield sse_event("status", {"message": "Thinking..."})
        try:
            answer = await asyncio.to_thread(invoke_agent, payload.message, sid)
            built = ""
            for token in answer.split(" "):
                built = f"{built} {token}".strip()
                yield sse_event("token", {"text": token + " ", "partial": built})
            yield sse_event("done", {"answer": answer, "session_id": sid})
        except Exception as e:
            yield sse_event("error", {"detail": f"Chat processing failed: {e}"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.delete("/chat/session/{session_id}")
def clear_chat_session(session_id: str):
    existed = session_id in session_history
    session_history.pop(session_id, None)
    return {"cleared": existed, "session_id": session_id}