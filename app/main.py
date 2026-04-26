from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, settings
from app.model_downloader import ensure_emotion_classifier
from app.models.emotion import EmotionClassifier
from app.models.generator import UnslothGenerator
from app.prompting import build_classifier_input, build_generation_messages
from app.safety import crisis_reply, detect_crisis
from app.schemas import ChatRequest, ChatResponse, ResetResponse
from app.session import InMemorySessionStore
# --- LLM #1 Understanding additions ---------------------------------
from app.prompting import build_generation_messages_with_understanding
from app.schemas import UnderstandingPayload
from app.understanding import UnderstandingAnalyzer


STATIC_DIR = settings.project_root / "app" / "static"
session_store = InMemorySessionStore(max_turns_per_session=settings.max_turns_per_session)
# UnderstandingAnalyzer is stateless and lightweight: it consumes the
# GoEmotions classifier output (already produced upstream) plus a few
# rule-based / lexicon-based modules. No new ML model is loaded here.
understanding_analyzer = UnderstandingAnalyzer()
emotion_classifier: Any
generator: Any


def _build_emotion_classifier(config: Settings) -> Any:
    return EmotionClassifier(
        model_dir=config.emotion_model_dir,
        max_length=config.emotion_max_length,
    )


def _build_generator(config: Settings) -> Any:
    return UnslothGenerator(
        base_model=config.llm_base_model,
        adapter_model=config.llm_adapter_model,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global emotion_classifier, generator
    if settings.auto_download_models and settings.missing_local_model_files():
        ensure_emotion_classifier(
            settings.emotion_model_dir,
            repo_id=settings.emotion_model_repo,
            revision=settings.emotion_model_revision,
        )
    settings.validate_local_model_files()
    emotion_classifier = _build_emotion_classifier(settings)
    generator = _build_generator(settings)
    yield


app = FastAPI(
    title="Empathy AI Backend",
    version="0.1.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "sessions": session_store.size(),
        "emotion_model_dir_exists": settings.emotion_model_dir.exists(),
        "emotion_model_source": getattr(emotion_classifier, "source", "not-loaded"),
        "llm_base_model": settings.llm_base_model,
        "llm_adapter_source": getattr(generator, "adapter_source", "not-loaded"),
        "generation_backend": "unsloth",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    total_started = perf_counter()
    session_id = session_store.ensure_session(request.session_id)
    history_limit = request.history_limit or settings.history_limit
    history = session_store.get_history(session_id, limit=history_limit)
    classifier_input = build_classifier_input(request.message, history)

    classifier_started = perf_counter()
    try:
        emotions = await asyncio.to_thread(emotion_classifier.predict, classifier_input)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Emotion classifier failed: {exc}") from exc
    classifier_ms = (perf_counter() - classifier_started) * 1000.0

    # ---- LLM #1 Understanding layer (additive; failure is non-fatal) ----
    # We treat the analyzer as best-effort: if anything in the lexicon /
    # rule-based stack raises, we still return a working /chat reply with
    # `understanding=None`. Existing clients that ignore `understanding`
    # are unaffected.
    understanding_started = perf_counter()
    understanding_payload: UnderstandingPayload | None = None
    try:
        understanding_state = understanding_analyzer.analyze(
            user_message=request.message,
            history=history,
            emotions=emotions,
        )
        understanding_payload = UnderstandingPayload(**understanding_state.model_dump())
    except Exception:
        understanding_payload = None
    understanding_ms = (perf_counter() - understanding_started) * 1000.0

    generation_ms = 0.0
    crisis_detected = detect_crisis(request.message)
    # --- LLM #1 Understanding additions: allow understanding layer
    # to ESCALATE crisis_detected, never to lower it (one-way escalation).
    if understanding_payload is not None and understanding_payload.safety_flag in {"medium", "high"}:
        crisis_detected = True
    if crisis_detected:
        reply = crisis_reply()
    else:
        messages = build_generation_messages(
            message=request.message,
            history=history,
            emotions=emotions,
        )
        # --- LLM #1 Understanding additions: if the analyzer ran,
        # rebuild `messages` using the wrapper that injects the
        # understanding payload. Otherwise, the line above stays in effect.
        if understanding_payload is not None:
            messages = build_generation_messages_with_understanding(
                message=request.message,
                history=history,
                emotions=emotions,
                understanding=understanding_payload.model_dump(),
            )
        generation_started = perf_counter()
        try:
            reply = await asyncio.to_thread(generator.generate, messages)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"LLM generation failed: {exc}") from exc
        generation_ms = (perf_counter() - generation_started) * 1000.0

    session_store.append_exchange(session_id, request.message, reply)
    total_ms = (perf_counter() - total_started) * 1000.0
    return ChatResponse(
        session_id=session_id,
        reply=reply,
        emotions=emotions,
        performance={
            "classifier_ms": round(classifier_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "total_ms": round(total_ms, 2),
            "understanding_ms": round(understanding_ms, 2),
        },
        safety={"crisis_detected": crisis_detected},
        understanding=understanding_payload,
    )


@app.post("/sessions/{session_id}/reset", response_model=ResetResponse)
async def reset_session(session_id: str) -> ResetResponse:
    session_store.reset(session_id)
    return ResetResponse(session_id=session_id)


@app.post("/sessions/reset-all", response_model=ResetResponse)
async def reset_all_sessions() -> ResetResponse:
    cleared = session_store.clear_all()
    return ResetResponse(cleared_sessions=cleared)


@app.post("/admin/restart-session-buffer", response_model=ResetResponse)
async def restart_session_buffer() -> ResetResponse:
    cleared = session_store.clear_all()
    return ResetResponse(cleared_sessions=cleared)
