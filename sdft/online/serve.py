"""FastAPI server: OpenAI-compatible chat + feedback + demo endpoints.

OpenClaw-RL harness interop
---------------------------
OpenClaw-RL (github.com/Gen-Verse/OpenClaw-RL) turns live agent traffic into RL
signals via an OpenAI-compatible proxy that reads three custom fields per turn:

    X-Session-Id   / body "session_id"    trajectory grouping key
    X-Turn-Type    / body "turn_type"     "main" (train on it) | "side" (skip)
    X-Session-Done / body "session_done"  conversation boundary

We honor the same contract so this server is a drop-in for an OpenClaw client:
`turn_type == "side"` turns are served but excluded from learning; `session_done`
closes the conversation (harvesting accepted-reply self-demonstrations). Where
OpenClaw-RL learns via cluster RL against an automated reward, we learn on-device
via SDFT from the *user's own* interaction — see docs/OPENCLAW.md.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .controller import OnlineController


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    conversation_id: str | None = None
    session_id: str | None = None      # OpenClaw alias for conversation_id
    turn_type: str | None = None       # "main" | "side"
    session_done: bool | None = None
    max_tokens: int | None = None
    temperature: float | None = None


class FeedbackRequest(BaseModel):
    conversation_id: str
    corrected_text: str
    message_id: str | None = None


class CloseRequest(BaseModel):
    conversation_id: str


def create_app(controller: OnlineController) -> FastAPI:
    app = FastAPI(title="local online SDFT")
    app.state.controller = controller
    static = Path(__file__).parent / "static"

    def _header(req: Request, name: str) -> str | None:
        return req.headers.get(name)

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.post("/v1/chat/completions")
    def chat_completions(body: ChatCompletionRequest, request: Request):
        # OpenClaw harness fields via header or body.
        conv_id = (
            body.conversation_id or body.session_id
            or _header(request, "X-Session-Id") or uuid.uuid4().hex[:8]
        )
        turn_type = body.turn_type or _header(request, "X-Turn-Type") or "main"
        done_hdr = _header(request, "X-Session-Done")
        session_done = body.session_done or (done_hdr not in (None, "", "0", "false", "False"))

        last_user = next((m for m in reversed(body.messages) if m.role == "user"), None)
        if last_user is None:
            raise HTTPException(400, "no user message in request")

        overrides = {}
        if body.max_tokens is not None:
            overrides["max_new_tokens"] = body.max_tokens
        if body.temperature is not None:
            overrides["temperature"] = body.temperature

        if turn_type == "side":
            # Serve housekeeping turns without recording them as training signal.
            reply = controller.backend.generate(
                [m.model_dump() for m in body.messages], **overrides
            )
            msg_id, update = None, None
        else:
            msg_id, reply = controller.chat(conv_id, last_user.content, **overrides)
            update = controller.maybe_update()

        if session_done:
            controller.close_conversation(conv_id)
            controller.maybe_update()

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "conversation_id": conv_id,
            "message_id": msg_id,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": reply}}],
            "sdft": {"turn_type": turn_type,
                     "update_ran": update is not None,
                     "adapter_version": update.adapter_version if update else None},
        }

    @app.post("/v1/feedback")
    def feedback(body: FeedbackRequest):
        mid = body.message_id or _last_assistant_id(controller, body.conversation_id)
        demo = controller.correct(body.conversation_id, mid, body.corrected_text)
        if demo is None:
            raise HTTPException(404, "message not found or not an assistant message")
        run = controller.maybe_update()
        return {"status": "recorded", "demonstration_id": demo.id, "update_ran": run is not None}

    @app.post("/v1/conversation/close")
    def close(body: CloseRequest):
        demos = controller.close_conversation(body.conversation_id)
        run = controller.maybe_update()
        return {"harvested": len(demos), "update_ran": run is not None}

    @app.post("/v1/train")
    def train_now():
        run = controller.maybe_update(force=True)
        if run is None:
            return {"status": "no demonstrations"}
        return {"status": "trained", "steps": run.steps, "metrics": run.metrics,
                "adapter_version": run.adapter_version}

    @app.post("/v1/rollback")
    def rollback(version: int | None = None):
        av = controller.rollback(version)
        if av is None:
            raise HTTPException(404, "no adapter to roll back to")
        return {"status": "rolled back", "version": av.version}

    @app.get("/v1/stats")
    def stats():
        return controller.stats()

    # ---- demo: "Airplane-Mode Coach" ------------------------------------

    @app.get("/v1/demo/eval")
    def demo_eval():
        """success@held-out for the configured reward task (the demo curve)."""
        if controller._reward_fn is None:
            raise HTTPException(400, "no online.reward_fn configured for the demo")
        from .demo import HELDOUT_PROMPTS, success_on

        with controller._lock:
            res = success_on(controller.backend, controller._reward_fn, HELDOUT_PROMPTS)
        res["active_adapter"] = controller.stats()["active_adapter"]
        return res

    @app.post("/v1/demo/coach")
    def demo_coach(n: int = 2):
        """Coach on n fresh prompts (reward-selected self-distillation), then update."""
        if controller._reward_fn is None:
            raise HTTPException(400, "no online.reward_fn configured for the demo")
        from .demo import COACH_PROMPTS

        import uuid as _uuid
        start = controller.stats()["demonstrations"]
        conv = "coach-" + _uuid.uuid4().hex[:6]
        for i in range(n):
            controller.chat(conv, COACH_PROMPTS[(start + i) % len(COACH_PROMPTS)])
        run = controller.maybe_update(force=True)
        return {"coached": n, "harvested": controller.stats()["demonstrations"] - start,
                "update_ran": run is not None,
                "adapter_version": run.adapter_version if run else None,
                "loss": run.metrics.get("loss") if run else None}

    return app


def _last_assistant_id(controller: OnlineController, conversation_id: str) -> str | None:
    for m in reversed(controller.store.conversation_messages(conversation_id)):
        if m.role == "assistant":
            return m.id
    return None
