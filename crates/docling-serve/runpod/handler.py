"""RunPod queue worker for docling-serve: sizes itself to the GPU it lands on.

RunPod queue endpoints hand jobs to a worker through the `runpod` SDK, not as
HTTP requests, and the worker decides how many it takes at once. One
docling-serve process converts one PDF at a time (its warm pipeline is shared
behind a lock), so this worker starts one process per document the GPU can
hold — two on a 16 GB card, more on a bigger one, rather than leaving a 48 GB
card running a single book — and tells RunPod to send it that many jobs at
once. Each job goes to an idle process's `/v1/convert/r2` on localhost.

A job's `input` is exactly that endpoint's request body (pre-signed source,
chunk, Markdown and manifest URLs; see crates/docling-serve/README.md). The
result is the endpoint's answer; a 4xx/5xx comes back as the job's `error`
(with `status`), so RunPod marks the job failed and its retry policy and the
caller's webhook see why.

Sizing, all overridable in the endpoint's environment:
  DOCLING_POOL_SIZE         processes, overriding the computation below
  DOCLING_POOL_GB_PER_DOC   GPU memory one conversion needs (default 5)
  DOCLING_POOL_RESERVE_GB   GPU memory left free (default 1.5)
  DOCLING_POOL_MAX          upper bound (default 8)
The count is also capped by CPU cores (one conversion keeps about two busy)
and host RAM (about 4 GB per process); with no GPU visible it is 1.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

import runpod

SERVE = os.environ.get("DOCLING_SERVE_BIN", "/usr/local/bin/docling-serve")
BASE_PORT = 9001


def env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def gpu_memory_gb():
    """Total memory of the first visible GPU in GB, or None without one."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.split()
        return float(out[0]) / 1024.0 if out else None
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def host_memory_gb():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024.0 / 1024.0
    except OSError:
        pass
    return None


def pool_size():
    """How many conversions this machine runs at once (see the module doc)."""
    fixed = os.environ.get("DOCLING_POOL_SIZE")
    if fixed:
        return max(1, int(fixed))
    gpu = gpu_memory_gb()
    if gpu is None:
        return 1
    per_doc = env_float("DOCLING_POOL_GB_PER_DOC", 5)
    reserve = env_float("DOCLING_POOL_RESERVE_GB", 1.5)
    n = int((gpu - reserve) // per_doc)
    n = min(n, int(env_float("DOCLING_POOL_MAX", 8)))
    n = min(n, max(1, (os.cpu_count() or 2) // 2))
    ram = host_memory_gb()
    if ram:
        n = min(n, max(1, int(ram // 4)))
    return max(1, n)


def ready(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/ready", timeout=2) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def start_servers(n):
    """Start n docling-serve processes and wait until every one has its models
    loaded (`--warmup`): a job must never land on a cold process."""
    procs = []
    for i in range(n):
        port = BASE_PORT + i
        procs.append(subprocess.Popen(
            [SERVE, "--addr", f"127.0.0.1:{port}", "--allow-url-fetch", "--warmup",
             "--concurrency", "1", "--no-ui"],
            stdout=sys.stdout, stderr=sys.stderr,
        ))
    deadline = time.monotonic() + float(os.environ.get("DOCLING_POOL_START_TIMEOUT", 600))
    pending = set(range(n))
    while pending:
        for i in list(pending):
            if procs[i].poll() is not None:
                raise RuntimeError(f"docling-serve on port {BASE_PORT + i} exited "
                                   f"with {procs[i].returncode} during warm-up")
            if ready(BASE_PORT + i):
                pending.discard(i)
        if pending and time.monotonic() > deadline:
            raise RuntimeError(f"docling-serve not ready after the start timeout: "
                               f"ports {sorted(BASE_PORT + i for i in pending)}")
        time.sleep(0.5)
    return procs


def post(port, body):
    """POST a job to one process's /v1/convert/r2: (status, parsed answer)."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/convert/r2",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=None) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw or b"{}")
        except ValueError:
            return e.code, {"error": raw.decode(errors="replace")[:2000]}


POOL = pool_size()
print(f"docling worker: GPU {gpu_memory_gb()} GB, host {host_memory_gb()} GB, "
      f"{os.cpu_count()} CPUs -> {POOL} conversion(s) at once", flush=True)
PROCS = start_servers(POOL)
FREE = None  # asyncio.Queue of idle ports, made on the SDK's event loop


async def handler(job):
    global FREE
    if FREE is None:
        FREE = asyncio.Queue()
        for i in range(POOL):
            FREE.put_nowait(BASE_PORT + i)
    body = job.get("input") or {}
    port = await FREE.get()
    try:
        if PROCS[port - BASE_PORT].poll() is not None:
            return {"error": f"docling-serve on port {port} is not running", "status": 503}
        status, answer = await asyncio.to_thread(post, port, body)
    finally:
        FREE.put_nowait(port)
    if status >= 400:
        message = answer.get("error") if isinstance(answer, dict) else None
        return {"error": message or f"HTTP {status}", "status": status}
    return answer


runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": lambda current: POOL,
})
