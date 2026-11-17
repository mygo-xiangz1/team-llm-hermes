"""RunPod template HTTP contract on :8000; supervised llama-server on loopback :8080."""

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.request

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


LLAMA_URL = "http://127.0.0.1:8080"
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "1200"))
DEFAULT_MODEL = "/data/models/Qwen3.8-Flash-Next-Uncensored-AD-4.25bpw-Q4_K_M-00001-of-00003.gguf"
DEFAULT_MMPROJ = "/data/models/mmproj-Qwen3.8-Flash-Next-Uncensored-BF16.gguf"
DEFAULT_MTP = "/data/models/mtp-shared-Q8_0.gguf"
HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "content-length",
}


def log(message):
    print(message, flush=True)


def model_path(value):
    """Honor /data; also support Serverless's documented /runpod-volume mount."""
    path = Path(value)
    if not path.is_file() and value.startswith("/data/"):
        root = os.getenv("VOLUME_FALLBACK_ROOT", "/runpod-volume")
        alternative = Path(root) / value.removeprefix("/data/") if root else path
        if alternative.is_file():
            log(f"[volume] {path} -> {alternative}")
            path = alternative
    if not path.is_file():
        raise FileNotFoundError(f"Model file missing: {path}; check the volume mount")
    return path


def find_binary():
    explicit = os.getenv("LLAMA_SERVER_BIN", "")
    root = Path(os.getenv("LLAMA_BIN_DIR", "/opt/llama.cpp"))
    candidates = [Path(explicit)] if explicit else sorted(root.rglob("llama-server"))
    candidates = sorted({p.resolve() for p in candidates if p.is_file() and os.access(p, os.X_OK)})
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one executable llama-server under {root}; set LLAMA_SERVER_BIN")
    return candidates[0], root


def command(use_mtp):
    binary, root = find_binary()
    model = model_path(os.getenv("MODEL_FILE", DEFAULT_MODEL))
    split = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", model.name)
    if split:
        prefix, _, count = split.groups()
        for index in range(1, int(count) + 1):
            shard = model.with_name(f"{prefix}-{index:05d}-of-{count}.gguf")
            if not shard.is_file():
                raise FileNotFoundError(f"GGUF shard missing: {shard}")
    args = [
        str(binary), "--host", "127.0.0.1", "--port", "8080",
        "--model", str(model), "--alias", os.getenv("MODEL_ALIAS", "team-hermes"),
        "-ngl", os.getenv("N_GPU_LAYERS", "99"),
        "--flash-attn", os.getenv("FLASH_ATTN", "on"),
        "-c", os.getenv("CTX_SIZE", "262144"),
        "--parallel", os.getenv("PARALLEL", "4"),
        "--cache-type-k", os.getenv("CACHE_TYPE_K", "q8_0"),
        "--cache-type-v", os.getenv("CACHE_TYPE_V", "q8_0"),
        "--timeout", str(int(REQUEST_TIMEOUT)), "--jinja",
    ]
    mmproj = os.getenv("MMPROJ_FILE", DEFAULT_MMPROJ)
    if mmproj:
        args += ["--mmproj", str(model_path(mmproj))]
    # Deliberately no default --chat-template-kwargs: Hermes owns per-request thinking.
    kwargs = os.getenv("CHAT_TEMPLATE_KWARGS", "").strip()
    if kwargs:
        if not isinstance(json.loads(kwargs), dict):
            raise ValueError("CHAT_TEMPLATE_KWARGS must be a JSON object")
        args += ["--chat-template-kwargs", kwargs]
    args += shlex.split(os.getenv("LLAMA_EXTRA_ARGS", ""))
    if use_mtp:
        mtp = model_path(os.getenv("MTP_FILE", DEFAULT_MTP))
        args += ["--spec-draft-model", str(mtp), "--spec-type", "draft-mtp",
                 "--spec-draft-n-max", os.getenv("SPEC_DRAFT_MAX", "4")]
    child_env = os.environ.copy()
    lib_dirs = sorted({str(p.parent) for p in root.rglob("*.so*") if p.is_file()})
    child_env["LD_LIBRARY_PATH"] = ":".join(
        [str(binary.parent), *lib_dirs, child_env.get("LD_LIBRARY_PATH", "")]
    )
    return args, child_env, binary.parent


class LlamaSupervisor:
    def __init__(self):
        self.ready = threading.Event()
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.process = None
        self.thread = None
        # Never send local readiness probes through an HTTP_PROXY from the environment.
        self.probe = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def start(self):
        self.thread = threading.Thread(target=self.run, name="llama-supervisor", daemon=True)
        self.thread.start()

    def backend_ready(self):
        try:
            with self.probe.open(f"{LLAMA_URL}/health", timeout=2) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    @staticmethod
    def read_logs(process):
        with process.stdout:
            for line in process.stdout:
                log(f"[llama] {line.rstrip()}")

    @staticmethod
    def terminate(process):
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            except ProcessLookupError:
                pass

    def run(self):
        use_mtp = os.getenv("SPEC_MTP", "0") == "1"
        fallback_used = False
        initial = max(1.0, float(os.getenv("RESTART_INITIAL_SECONDS", "2")))
        maximum = max(initial, float(os.getenv("RESTART_MAX_SECONDS", "60")))
        delay = initial
        startup_timeout = float(os.getenv("LLAMA_STARTUP_TIMEOUT", "1200"))
        while not self.stopping.is_set():
            process = None
            was_ready = False
            first_ready = None
            self.ready.clear()
            try:
                args, child_env, cwd = command(use_mtp)
                log(f"[supervisor] starting {args[0]} (MTP={use_mtp}, ctx={os.getenv('CTX_SIZE', '262144')})")
                # Start while holding the lock so shutdown cannot miss a newly created child.
                with self.lock:
                    if self.stopping.is_set():
                        break
                    process = subprocess.Popen(
                        args, env=child_env, cwd=cwd, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                        errors="replace", bufsize=1,
                    )
                    self.process = process
                threading.Thread(target=self.read_logs, args=(process,), daemon=True).start()
                started = time.monotonic()
                while not self.stopping.is_set() and process.poll() is None:
                    if self.backend_ready():
                        self.ready.set()
                        if not was_ready:
                            first_ready = time.monotonic()
                            was_ready = True
                            log("[supervisor] llama-server ready")
                    else:
                        self.ready.clear()
                    if not was_ready and time.monotonic() - started > startup_timeout:
                        log(f"[supervisor] startup exceeded {startup_timeout:g}s; terminating")
                        self.terminate(process)
                        break
                    self.stopping.wait(1)
                if not self.stopping.is_set():
                    log(f"[supervisor] llama-server exited: {process.poll()}")
            except Exception as exc:
                log(f"[supervisor] startup/monitor failure: {type(exc).__name__}: {exc}")
            finally:
                self.ready.clear()
                self.terminate(process)
                with self.lock:
                    self.process = None
            if self.stopping.is_set():
                break
            if use_mtp and not was_ready and not fallback_used:
                # Exactly one immediate startup retry without BOTH speculative flags.
                use_mtp = False
                fallback_used = True
                log("[supervisor] retrying once without --spec-type / --spec-draft-max")
                continue
            if first_ready is not None and time.monotonic() - first_ready >= 60:
                delay = initial
            log(f"[supervisor] restart in {delay:g}s")
            self.stopping.wait(delay)
            delay = min(delay * 2, maximum)

    def close(self):
        self.stopping.set()
        self.ready.clear()
        # run() owns termination, avoiding two threads signalling the same process.
        if self.thread is not None:
            self.thread.join(timeout=30)


@asynccontextmanager
async def lifespan(app):
    supervisor = LlamaSupervisor()
    app.state.supervisor = supervisor
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(REQUEST_TIMEOUT), trust_env=False,
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
    )
    supervisor.start()
    log("[proxy] listening on 0.0.0.0:8000; loading llama-server in background")
    try:
        yield
    finally:
        await asyncio.to_thread(supervisor.close)
        await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


def api_error(message, status, kind="upstream_error"):
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)


@app.get("/")
async def root():
    return {"message": "ok"}


@app.get("/ping")
async def ping():
    return {"message": "ok"}


@app.get("/health")
async def health(request: Request):
    # The template contract deliberately requires HTTP 200 even during cold loading.
    if request.app.state.supervisor.ready.is_set():
        try:
            response = await request.app.state.client.get(f"{LLAMA_URL}/health", timeout=2)
            if response.status_code == 200:
                return {"message": "ready"}
        except httpx.HTTPError:
            pass
    return {"message": "booting"}


@app.get("/ws")
async def ws_stub():
    return {"message": "stub"}


@app.get("/get_status")
async def status_stub(request: Request):
    return {"status": "ready" if request.app.state.supervisor.ready.is_set() else "booting"}


def filtered_headers(headers):
    connection_tokens = {token.strip().lower() for token in headers.get("connection", "").split(",")}
    excluded = HOP_HEADERS | connection_tokens
    return {key: value for key, value in headers.items() if key.lower() not in excluded}


async def forward(request: Request, path):
    # Four-person team gate: TEAM_API_KEY_1..4 must be set, distinct and non-empty.
    # Key accepted via X-API-Key or Authorization: Bearer (Hermes sends the latter).
    keys = {os.getenv(f"TEAM_API_KEY_{i}", "") for i in range(1, 5)}
    supplied = request.headers.get("x-api-key", "")
    if not supplied:
        authz = request.headers.get("authorization", "")
        for pfx in ("Bearer ", "bearer "):
            if authz.startswith(pfx):
                supplied = authz[len(pfx):]
    if len(keys) != 4 or "" in keys or supplied not in keys:
        return JSONResponse(
            {"error": {"message": "Invalid API key", "type": "authentication_error"}},
            status_code=401,
        )
    if not request.app.state.supervisor.ready.is_set():
        return JSONResponse({"message": "booting"}, status_code=503, headers={"Retry-After": "5"})
    body = await request.body()
    streaming = False
    if request.method == "POST":
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return api_error("Request body must be a JSON object", 400, "invalid_request_error")
        if not isinstance(payload, dict):
            return api_error("Request body must be a JSON object", 400, "invalid_request_error")
        streaming = payload.get("stream") is True
    headers = filtered_headers(request.headers)
    # RunPod authenticates the user token; do not send that credential to llama-server.
    for key in ("host", "authorization", "cookie", "accept-encoding"):
        headers.pop(key, None)
    headers["accept-encoding"] = "identity"
    client = request.app.state.client
    upstream = None
    try:
        outgoing = client.build_request(
            request.method, f"{LLAMA_URL}{path}", content=body,
            params=request.query_params.multi_items(), headers=headers,
        )
        # Open outside the generator so upstream status/errors arrive before HTTP headers.
        upstream = await client.send(outgoing, stream=True)
        response_headers = filtered_headers(upstream.headers)
        is_sse = "text/event-stream" in upstream.headers.get("content-type", "").lower()
        if upstream.is_success and (streaming or is_sse):
            response_headers["cache-control"] = "no-cache"
            response_headers["x-accel-buffering"] = "no"

            async def chunks():
                try:
                    # Raw bytes preserve SSE lines, blank separators, tool/reasoning deltas,
                    # and the upstream data: [DONE] terminator, without JSON re-encoding.
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                except httpx.HTTPError as exc:
                    log(f"[proxy] SSE interrupted: {type(exc).__name__}")
                    raise  # Never fabricate [DONE] after a broken stream.
                finally:
                    await asyncio.shield(upstream.aclose())

            return StreamingResponse(
                chunks(), status_code=upstream.status_code,
                headers=response_headers, media_type="text/event-stream",
            )
        content = await upstream.aread()
        # aread() decodes content encodings; do not retain a stale encoding header.
        response_headers.pop("content-encoding", None)
        return Response(content=content, status_code=upstream.status_code, headers=response_headers)
    except httpx.TimeoutException:
        log("[proxy] llama-server request timed out")
        return api_error("llama-server request timed out", 504, "upstream_timeout")
    except httpx.HTTPError as exc:
        log(f"[proxy] llama-server connection failed: {type(exc).__name__}")
        return api_error("llama-server is unavailable", 502)
    finally:
        # Successful SSE owns its response until the iterator closes it.
        if upstream is not None and (not upstream.is_success or not (streaming or is_sse)):
            await upstream.aclose()


@app.post("/handle_request")
async def handle_request(request: Request):
    # The template passes the entire OpenAI body here; do not unwrap an "input" field.
    return await forward(request, "/v1/chat/completions")


async def openai_proxy(request: Request):
    return await forward(request, request.url.path)


for route, method in (
    ("/v1/models", "GET"),
    ("/v1/chat/completions", "POST"),
    ("/v1/completions", "POST"),
    ("/v1/embeddings", "POST"),
):
    app.add_api_route(route, openai_proxy, methods=[method])


if __name__ == "__main__":
    import uvicorn

    # Exactly one ASGI worker: multiple workers would start competing llama processes.
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1, access_log=False)
