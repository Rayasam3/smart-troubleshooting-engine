"""REST API (PDF section 5).

  POST /v1/troubleshoot   {"query": "...", "siis_response": "<optional raw text>"}  -> plan + meta
  GET  /health            200 {"status": "ok"} once model, index and cache are loaded; 503 while starting
  GET  /v1/info           engine / cache details (handy for demos and debugging)
  GET  /                  optional one-page demo UI

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError

from app.schema import ContextDeeplinkResponse

log = logging.getLogger("api")
STATIC = Path(__file__).parent / "static"


class TroubleshootRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    siis_response: Optional[Union[str, dict]] = None


def _empty(query: str, fallback: str, started: float, **meta) -> dict:
    return {"query": query, "query_variations": [], "response": {"contexts": []},
            "meta": {"latency_ms": round((time.perf_counter() - started) * 1000, 1), "cache_hit": False,
                     "model": "none", "cost_usd": 0.0, "fallback": fallback, **meta}}


def create_app(engine_factory: Optional[Callable] = None, background: bool = True) -> FastAPI:
    state = {"engine": None, "error": None, "ready_s": None}

    def load():
        t0 = time.perf_counter()
        try:
            if engine_factory is None:
                from app.engine import TroubleshootEngine
                state["engine"] = TroubleshootEngine()
            else:
                state["engine"] = engine_factory()
            state["ready_s"] = round(time.perf_counter() - t0, 1)
            log.info("engine ready in %ss", state["ready_s"])
        except Exception as exc:          # keep the server up and report the problem on /health
            state["error"] = str(exc)
            log.exception("engine failed to load")

    @asynccontextmanager
    async def lifespan(_app):
        # load in the background so /health can answer 503 "starting" instead of the port being dead
        if background:
            threading.Thread(target=load, daemon=True).start()
        else:
            load()
        yield

    app = FastAPI(title="Smart Guided Troubleshooting Engine", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        if state["engine"] is not None:
            return {"status": "ok"}
        if state["error"]:
            return JSONResponse({"status": "error", "detail": state["error"]}, status_code=503)
        return JSONResponse({"status": "starting"}, status_code=503)

    @app.get("/v1/info")
    def info():
        eng = state["engine"]
        if eng is None:
            return JSONResponse({"status": "starting"}, status_code=503)
        return {"embedder": eng.embedder.name, "cached_plans": len(eng.cache),
                "cached_phrasings": len(eng.cache.door_plan), "catalog_deeplinks": len(eng.mapper.index.entries),
                "extractor": getattr(eng.pipeline.extractor, "name", "?"), "startup_s": state["ready_s"]}

    @app.post("/v1/troubleshoot")
    def troubleshoot(req: TroubleshootRequest):
        started = time.perf_counter()
        eng = state["engine"]
        if eng is None:
            return JSONResponse(_empty(req.query, "engine_starting", started), status_code=503)
        try:
            result = eng.handle(req.query, req.siis_response)
            ContextDeeplinkResponse.model_validate(result["response"])       # last line of defence
            return result
        except ValidationError as exc:
            log.error("schema violation blocked: %s", exc)
            return _empty(req.query, "validation_error", started)
        except Exception as exc:                                              # never leak a stack trace
            log.exception("troubleshoot failed")
            return JSONResponse(_empty(req.query, "internal_error", started, error=type(exc).__name__),
                                status_code=500)

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        return JSONResponse({"error": "internal_error", "type": type(exc).__name__}, status_code=500)

    @app.get("/", include_in_schema=False)
    def demo():
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()