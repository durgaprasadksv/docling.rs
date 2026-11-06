# Deploying `docling-serve` in Containers

This guide covers running the [`docling-serve`](../crates/docling-serve) HTTP conversion service using Docker, Docker Compose, and Kubernetes/OpenShift.

---

## Container Registry & Prebuilt Images

The following container images are published on **GitHub Container Registry (GHCR)** for every release the maintainer publishes (`release.yml`; version tags plus `latest`):

#### 📦 Distributed Images

| Image | Description | Architectures |
|---|---|---|
| [`ghcr.io/docling-project/docling-rs-serve`](https://github.com/docling-project/docling.rs/pkgs/container/docling-rs-serve) | High-performance document conversion HTTP API with PDF, DOCX, PPTX, XLSX, HTML, images, and audio/video models pre-installed (zero Python runtime dependencies, CPU). | `linux/amd64`, `linux/arm64` |
| [`ghcr.io/docling-project/docling-rs`](https://github.com/docling-project/docling.rs/pkgs/container/docling-rs) | The `docling-rs` CLI with the same models baked in: batch conversion of local files without installing Rust (CPU). Entrypoint `docling-rs`, working directory `/data`. | `linux/amd64`, `linux/arm64` |
| [`ghcr.io/docling-project/docling-rs-serve-cuda`](https://github.com/docling-project/docling.rs/pkgs/container/docling-rs-serve-cuda) | NVIDIA CUDA 13 GPU-accelerated HTTP conversion API (CUDA 13 + cuDNN 9, Linux x86_64). Explicit version tags (`v1.28.0`, `master`), no `:latest`. | `linux/amd64` |
| [`ghcr.io/docling-project/docling-rs-cuda`](https://github.com/docling-project/docling.rs/pkgs/container/docling-rs-cuda) | NVIDIA CUDA 13 GPU-accelerated `docling-rs` CLI. Explicit version tags (`v1.28.0`, `master`), no `:latest`. | `linux/amd64` |
> [!NOTE]
> **Image Naming**: The `-rs` in `docling-rs-serve` / `docling-rs` differentiates this native Rust implementation from the Python `docling-project/docling-serve` image repository on GHCR. Both images are targets of the same [`Dockerfile`](../crates/docling-serve/Dockerfile) and share every layer but the final binary.

```bash
# Pull the docling-serve HTTP API image (Rust engine):
docker pull ghcr.io/docling-project/docling-rs-serve:latest

# Pull the CLI image and convert files from the current directory:
docker pull ghcr.io/docling-project/docling-rs:latest
docker run --rm -v "$PWD:/data" ghcr.io/docling-project/docling-rs:latest report.pdf --to md
docker run --rm -v "$PWD:/data" ghcr.io/docling-project/docling-rs:latest --input docs --output converted --to json
```

### Supported Architectures & Tags

| Architecture | Platform | Notes |
|---|---|---|
| `linux/amd64` | x86-64 | Standard 64-bit Linux |
| `linux/arm64` | ARM64 / aarch64 | Apple Silicon, AWS Graviton, Ampere |

#### Tagging Scheme

- **CPU images (`docling-rs-serve`, `docling-rs`)**:
  - `latest`: Latest release or master build
  - `v1.28.0`, `1.28.0`: Exact release version
  - `1.28`, `1`: Floating major/minor tags
  - `master`: Bleeding-edge master branch build
- **CUDA GPU images (`docling-rs-serve-cuda`, `docling-rs-cuda`)**:
  - `v1.28.0`, `1.28.0`, `1.28`, `1`: Explicit release version tags
  - `master`: Master branch build
  - *(No floating `:latest` tag, following upstream docling-serve CUDA tagging policy)*

---

## Quick Start with Docker

Run the container exposing port `5001`:

```bash
docker run -d \
  --name docling-serve \
  -p 5001:5001 \
  --restart unless-stopped \
  ghcr.io/docling-project/docling-rs-serve:latest
```

Check health:

```bash
curl http://localhost:5001/health
# => {"status":"ok"}
```

Convert a document (PDF, DOCX, PPTX, XLSX, HTML, Images, Audio/Video) to Markdown:

```bash
curl -F file=@document.pdf http://localhost:5001/v1/convert
```

---

### A self-built CLI image (`examples/Dockerfile`)

For a self-contained CLI image with models exported from PyTorch, [`examples/Dockerfile`](../examples/Dockerfile)
is a 3-stage build that bakes the binary, native libs, and models into a slim runtime stage:

```bash
docker build -f examples/Dockerfile -t docling-rs .
docker run --rm -v "$PWD:/data" docling-rs /data/input.pdf          # Markdown to stdout
docker run --rm -v "$PWD:/data" docling-rs /data/input.pdf --to json
```

Both `linux/amd64` and `linux/arm64` build (#281) — everything in the image
is arch-neutral or built from source.

## Deployment with Docker Compose

Preconfigured compose files are available in [`examples/docker-compose/`](../examples/docker-compose/):

### Standalone Service (`docker-compose.yml`)

```yaml
services:
  docling-serve:
    image: ghcr.io/docling-project/docling-rs-serve:latest
    container_name: docling-serve
    restart: unless-stopped
    ports:
      # Bind to loopback interface only (127.0.0.1) for unauthenticated access
      - "127.0.0.1:5001:5001"
    environment:
      - DOCLING_RS_NO_ARENA=1
      - DOCLING_RS_MAX_MEMORY_MB=4096
      - DOCLING_RS_MEMORY_WATERMARK_PCT=85
      - RUST_LOG=info
      # 0 = the API only, no web UI page at / (#892)
      - DOCLING_RS_ENABLE_UI=1
    command:
      - "docling-serve"
      - "--addr"
      - "0.0.0.0:5001"
      - "--concurrency"
      - "2"
      - "--warmup"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5001/ready"]
      interval: 15s
      timeout: 5s
      retries: 5
      start_period: 45s
    deploy:
      resources:
        limits:
          cpus: "4.0"
          memory: 4G
        reservations:
          cpus: "1.0"
          memory: 1G
```

Launch with:

```bash
cd examples/docker-compose
docker compose up -d
```

### Production Stack with Caddy Reverse Proxy (`docker-compose.caddy.yml`)

For production setups needing automatic TLS certificates, Basic Auth, or header security:

```bash
cd examples/docker-compose
docker compose -f docker-compose.caddy.yml up -d
```
### NVIDIA CUDA GPU Stack (`docker-compose.cuda.yml`)

For hosts equipped with NVIDIA GPUs and the NVIDIA Container Toolkit:

```bash
cd examples/docker-compose
docker compose -f docker-compose.cuda.yml up -d
```

---
## Installing without containers — `install.sh`

`scripts/install/install.sh` installs a self-contained tree — for a dev box or
a pipeline step:

```bash
curl -fsSL https://raw.githubusercontent.com/docling-project/docling.rs/master/scripts/install/install.sh | bash
docling-rs your.pdf > out.md
```

It grabs the **prebuilt CLI binary** from the newest
[GitHub Release](https://github.com/docling-project/docling.rs/releases) that carries one
(Linux x64/arm64/s390x, macOS arm64; `DOCLING_RS_FROM_SOURCE=1` opts out) and only falls back to
building from source when no matching asset exists — in that case it checks
for a Rust toolchain (installs one via rustup if `cargo` is missing) and runs
`cargo build --release -p docling-cli`. Either way it installs the
binary + all models under `/usr/local/docling.rs`, symlinks
`/usr/local/bin/docling-rs`, and writes `/etc/profile.d/docling-rs.sh` with
the `DOCLING_*` exports. The env file is a convenience for other
consumers of the model tree — the CLI itself resolves `.models/`
**relative to its own (symlink-resolved) location**, so the
command works from any directory with no environment at all. ONNX Runtime is
statically linked; nothing else lands outside the prefix.

Knobs (env vars before the call): `DOCLING_RS_PREFIX` (default
`/usr/local/docling.rs`), `DOCLING_RS_BIN_DIR`, `DOCLING_RS_REF` (git ref
to build), `DOCLING_RS_NO_ASR=1` (skip the ~150 MB Whisper models),
`DOCLING_RS_SUDO=0` (never escalate). Re-running is idempotent — it only
fetches missing model files. Uninstall:
`rm -rf /usr/local/docling.rs /usr/local/bin/docling-rs /etc/profile.d/docling-rs.sh`.

## Configuration & Environment Variables

`docling-serve` is configured via CLI arguments and environment variables:

### CLI Arguments

Every server flag, with its default, is in the serve README's
[Server flags](../crates/docling-serve/README.md#server-flags) table. The
published images run `docling-serve --addr 0.0.0.0:5001 --warmup`: bound to
all interfaces inside the container, and `/ready` answers 503 until the
PDF/image models have loaded (and stays 503, with the error, if loading fails).

### Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `DOCLING_RS_SYSTEM_FONTS` | `0` in the images | PDF fallback fonts come only from `.models/fonts` + `DOCLING_RS_FONT_DIRS`, not the host's directories — the same layout input on every host (#633) |
| `DOCLING_RS_FONT_DIRS` | the image's Liberation + DejaVu dirs | Font directories for fonts a PDF does not embed; extend it when you install a font for another script |
| `DOCLING_RS_ENABLE_UI` | `1` | `0` serves the API only: no web UI page at `/` (404 there), the logo gone; `/v1`, `/openapi.yaml`, the probes and `/metrics` unchanged (#892). Upstream docling-serve's `DOCLING_SERVE_ENABLE_UI` is read when it is unset; the `--no-ui` / `--enable-ui` flags beat both |
| `DOCLING_RS_NO_ARENA` | `1` | Disables ONNX Runtime CPU arena to prevent RSS heap ratcheting (#263) |
| `DOCLING_RS_MAX_MEMORY_MB` | `0` (or cgroup) | Memory ceiling for admission control in MiB |
| `DOCLING_RS_MEMORY_WATERMARK_PCT` | `85` | Watermark % above which new requests get HTTP 503 Retry-After |
| `DOCLING_RS_PDF_WORKERS` | CPU count | Worker pool size for concurrent PDF page processing |
| `DOCLING_RS_PAGE_BITMAP` | `adaptive` | When the 2× page bitmap (TableFormer, picture crops, OCR) of a page with a text layer is rendered (#897): `adaptive` follows the document's recent pages; `eager` renders every page's during layout; `on-demand` only the ones read, after layout (least CPU) |
| `DOCLING_RS_TF_INTRA` | auto (#262) | Derived from cgroup CPU quota (#262); explicitly narrows ONNX intra-op threads for TableFormer decoder |
| `DOCLING_RS_GRAPH_CACHE_DIR` | `$XDG_CACHE_HOME/docling-rs/graphs` (else `~/.cache/…`) | Where ONNX Runtime's optimized graphs are cached between processes (CPU provider only; session creation for the layout model ~0.8 s → ~0.15 s) |
| `DOCLING_RS_NO_GRAPH_CACHE` | `0` | `1` disables the optimized-graph cache (models load and optimize from scratch every process) |
| `DOCLING_RS_OCR_SESSIONS` | worker thread budget (1–8) | Parallel single-thread OCR recognition lanes per worker; output is byte-identical at any count |
| `DOCLING_RS_ORT_PROFILE` | unset | A directory: every ONNX session writes ORT's per-operator profile there (`<model>_<timestamp>.json`, Chrome-trace format); `scripts/dev/ort_profile_summary.py` summarizes one |
| `DOCLING_RS_MAX_RASTER_PAGES` | `100` | Max page count for `to=images` PDF page rasterization |
| `DOCLING_RS_MAX_FETCH_BYTES` | `268435456` (256MB) | Max response size for remote URL fetching |
| `DOCLING_RS_OCR_LANG` | `en` | Default OCR language: under PP-OCR `en` or `ch` (multilingual); under `DOCLING_RS_OCR_ENGINE=tesseract` Tesseract's language list — tessdata stems joined by `+` (`rus+eng`) or BCP-47 tags — whose `.traineddata` must be in the tessdata dir (`DOCLING_RS_TESSDATA_DIR`). A request's `ocr_lang` overrides it |
| `DOCLING_RS_OCR_MODE` | `default` | OCR region selection (`default`, `full_page`, `layout_regions`) |
| `DOCLING_RS_FP32` | `0` | Force FP32 model precision instead of INT8 |
| `DOCLING_RS_MODELS_DIR` | `.models` | Directory override for ONNX model weights |
| `DOCLING_FFMPEG` | `ffmpeg` | Path to `ffmpeg` binary for video frame extraction |
| `RUST_LOG` | `info` | Logging verbosity (`error`, `warn`, `info`, `debug`, `trace`) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | OTLP/gRPC endpoint for distributed trace export (#297) |
| `OTEL_SERVICE_NAME` | `docling-serve` | OpenTelemetry service name |

---

## Resource Sizing & Performance Tuning

1. **Memory Arena & Admission Control (#263)**:
   By default, `DOCLING_RS_NO_ARENA=1` is enabled. Without this, ONNX Runtime's allocator retains mapped heap memory indefinitely. Combined with automatic `malloc_trim` calls, process memory drops ~3× after heavy PDF conversions. The server also pins glibc's mmap threshold at 4 MiB on startup (#669): glibc otherwise raises it each time a large buffer is freed, after which page bitmaps and tensors are carved from per-thread arenas and stay resident — measured, RSS crept 528 → 754 MB over six mixed PDF requests and now holds at ~590 MB, at the same latency. Since #871 every process that loads an ONNX model pins it the same way — the CLI, the Python and Node bindings and the C library too, which also trim after each conversion: on linux/arm64 the TableFormer decoder's ever-longer KV caches otherwise fragmented the heap until a long table ran out of memory under a 6 GiB limit. With `decoder_kv_layers.onnx` in the models (fetched by `download_dependencies.sh` and the Python `download_models()`; the Node package picks it up from `.models/`) the decoder no longer allocates those caches at all: they live in two buffers per table that ONNX Runtime writes in place. `GLIBC_TUNABLES=glibc.malloc.mmap_threshold=<bytes>` (or `MALLOC_MMAP_THRESHOLD_`) overrides the pinned value.
   Setting `DOCLING_RS_MAX_MEMORY_MB=4096` ensures the server sheds load with `503 + Retry-After` when under memory pressure, rather than being OOM-killed.

2. **Concurrency (`--concurrency`)**:
   One warm ML pipeline is shared across concurrent requests. For typical workloads, `--concurrency 2` or `--concurrency 4` provides high throughput without CPU saturation.

3. **CPU Sizing**:
   Allocate at least 2 vCPUs (recommended 4+ vCPUs) for production workloads with heavy scanned PDFs and OCR.

---

## GPU Throughput Benchmark (serverless sizing)

To size a GPU deployment — a RunPod serverless endpoint, say — measure the
server's own per-stage numbers rather than client wall time. The
`/v1/convert/file` response carries `processing_time` and, when the server
runs with `DOCLING_RS_TIMING=1`, a `timings` object: `{stage: {ms, calls}}`
for that request's layout, OCR, TableFormer, render and formula-decode
stages (the timer is process-wide, so run `--concurrency 1` while measuring,
or overlapping requests' stages mix).

A CUDA image with the formula recognizer baked in — the default image fetches
no enrichment models, so `do_formula_enrichment` would silently degrade to
undecoded formulas:

```bash
docker build -f crates/docling-serve/Dockerfile --target runtime-cuda \
  --build-arg EXTRA_ASSETS="--enrich --no-int8" \
  --build-arg CUDA_FEATURES=cuda,speech --build-arg MATHCAT_RULES=1 \
  -t docling-rs-serve-cuda:bench .
docker run --gpus all -p 5001:5001 \
  -e DOCLING_RS_TIMING=1 -e DOCLING_RS_WORD_PROV=1 -e DOCLING_RS_SPEECH=1 \
  docling-rs-serve-cuda:bench --concurrency 1
```

(`--no-int8` matters on a GPU: the CUDA provider does not run the INT8
decoder's dynamically quantized matmuls, so those nodes fall back to the CPU
with a copy every token — measured at 26 ms per generated token on an RTX
PRO 4000 against the fp32 graph's figure below. `CUDA_FEATURES` /
`MATHCAT_RULES` only matter for the `speech` map on formula items;
`DOCLING_RS_WORD_PROV=1` adds the per-word boxes to the JSON. Both are
additive and off by default.)

Then drive the corpus through it:

```bash
scripts/bench/gpu_serve_bench.sh http://host:5001 \
  scanned-textbook.pdf:220-221 university-physics.pdf:280-281 \
  chemistry.pdf:620-620 tests/data/pdf/sources/code_and_formula.pdf
```

One line per request: client wall time, the server's `processing_time`,
pages, formula items, and the four heaviest stages in milliseconds. The
first run of each file is cold (model and execution-provider load); set
`DOCLING_BENCH_RUNS=3` to see the warm figures.

### Measured (RunPod, RTX PRO 4000 Blackwell 24 GB, 2026-11-06)

Warm second run, `--concurrency 1`, fp32 models, the formula decoder's KV
cache device-resident (`DOCLING_RS_ENRICH_IOBINDING`, the default under a
GPU provider). The same box's 48-core CPU provider for comparison:

| input | pages | formula items | GPU s | per page | CPU (48 cores) s |
|---|---:|---:|---:|---:|---:|
| arXiv paper 2206.01062 (digital, 5 tables) | 9 | 0 | 1.8 | 0.2 | 6.5 |
| OpenStax Chemistry 2e p760 (digital) | 1 | 2 | 1.9 | 1.9 | 8.9 |
| OpenStax Chemistry 2e p620 (digital) | 1 | 3 | 4.0 | 4.0 | 28.8 |
| OpenStax University Physics p280–281 (digital) | 2 | 4 | 9.2 | 4.6 | 40.4 |
| 1900s scanned arithmetic textbook, 12 pages | 12 | 4 | 19.1 | 1.6 | 50.3 |
| the same book's algebra pages 220–221 (equation-dense) | 2 | 16 | 34.6 | 17.3 | 262.6 |

Where the time goes, from the per-stage timings:

- **Formula decode** is the cost on any page with equations: ~10 ms per
  generated token (`code_formula.step`), ~2 s per formula item on these
  pages. Two things had to be true for that: the fp32 decoder (the INT8
  one's quantized matmuls run on the CPU under CUDA — 26 ms/token) and the
  device-resident cache (the plain loop copied the whole KV cache to and
  from the host every token — a formula took 3–9 s either way).
- **Scanned pages** cost ~1.6 s each before any formula: the embedded scan is
  decoded and resampled for every render of the page — layout, the 2x
  bitmap and the enrichment render each decode it again
  (`render.image_decode`, 12 calls for 2 pages). Caching the decoded
  samples across the renders of a page is the next lever for scanned books.
- Crop preprocessing (`code_formula.preprocess`, three Lanczos resizes of
  each crop) was 290 ms per crop with the scalar resize; it now goes
  through the SIMD resizer like the page bitmaps.
- The GPU sits at ~35 % utilization: the per-token loop is launch-bound
  (30 layers of small kernels per step). Decoding several formulas of a
  page as one batch would raise throughput several-fold at the same
  latency per step; fp16 or fused-attention graphs would cut the step time.

### Concurrency on one GPU

The pipeline runs one page worker under a GPU provider unless
`DOCLING_RS_PDF_WORKERS` raises it, and the formula recognizer is a pool of
lanes (`DOCLING_RS_ENRICH_LANES`, one per worker on a GPU). On a 12-page
equation-dense scan (61 formula items), warm:

| workers / lanes | wall s | per page | GPU memory |
|---|---:|---:|---:|
| 1 / 1 | 116.7 | 9.7 | 3.0 GB |
| 2 / 2 | 79.2 | 6.6 | 5.8 GB |
| 4 / 4 | 91.2 | 7.6 | 11.0 GB |
| 2 / 2, decodes stopped at the first repeated cycle | 57.5 | 4.8 | 5.8 GB |

The last row is the same configuration after two fixes the run exposed: the
formula budget counted the prompt, so a tall block's 17 image tiles left it
a 0-token budget and the block was dropped (61 → 68 formula items once
counted in generated tokens), and a decode that has started looping ran to
the whole 1,024-token budget — six per run, most of the decode time — where
stopping at the first repeated cycle gives the same empty result at a
fraction of the cost.

Two is the sweet spot on a 24 GB card: beyond it the per-token steps of
concurrent decodes slow each other and the single render helper thread
serializes the page bitmaps. The same change took the 12-page scanned book
without formulas from 19 s to 14.7 s. **Recommended serverless
configuration:** `DOCLING_RS_PDF_WORKERS=2 DOCLING_RS_ENRICH_LANES=2`, one
request at a time per worker (`--concurrency 1`), on the 16 GB tier — 5.8
GB of GPU memory fits it, and it is 23 % cheaper than the 24 GB tier.

A second document in flight on the same GPU does not add throughput: the
server holds one warm pipeline per process, so with `--concurrency 2` two
of the 12-page scans submitted together finished at 72 s and 146 s against
74 s for one alone — queued, not overlapped — and a second server process
per GPU would reproduce the four-lane row above. Scale by GPUs (serverless
workers), not by stacking instances on one.

### Cost per book (serverless)

RunPod's 16 GB serverless tier (A4000 / A4500 / RTX 4000 class) lists at
$0.58 per hour, $0.000161 per second, the 24 GB tier at $0.75, both billed
per second with no per-request charge. With the recommended configuration
(two workers, two lanes, SIMD crop resizes) the per-page figures are:

| page kind | seconds | cost (16 GB tier) |
|---|---:|---:|
| digital text page, tables, no formulas | 0.2 | $0.00003 |
| digital page with 2–3 formulas | 2–3.5 | $0.0003–0.0006 |
| scanned page, no formulas | 1.2 | $0.0002 |
| scanned page, equation-dense (5–6 formulas) | 4.8 | $0.0008 |

So a 400-page scanned novel is about 8 minutes and $0.08; a 400-page digital
physics textbook with formulas on a third of its pages about 8 minutes and
$0.08; a 400-page scanned mathematics textbook with a quarter of its pages
equation-dense about 14 minutes and $0.14 (down from $0.45 before the fp32
decoder, the device-resident cache, the SIMD resizes, the second worker and
the loop detector). Add one cold start per worker (~8 s, the model and
provider load)
and whatever idle timeout the endpoint keeps a worker warm for.

The next reductions, in order of gain: decode a page's formulas as one
batch (the per-token loop is launch-bound, so a batch of eight costs about
what one does), and decode the embedded scan once per page instead of once
per render (`render.image_decode`, most of a scanned page's remaining
time). Together they put the scanned mathematics textbook under $0.10.

## Offline & Air-Gapped Deployment

The container image is fully self-contained by default. If you need to build a custom slim image or mount pre-cached assets from host storage:

### Building with `--build-arg FETCH_ASSETS=0`

```bash
docker build -f crates/docling-serve/Dockerfile --build-arg FETCH_ASSETS=0 -t docling-serve-slim .
```

### Baking in more speech-recognition models (`ASR_MODELS`)

The image always carries Whisper tiny, the default audio/video model. Named
presets (`whisper_tiny_en`, `whisper_base_en`, `whisper_small_en`,
`whisper_distil_small_en`, `parakeet_tdt_0.6b_v3`) are added with the
`ASR_MODELS` build arg — space- or comma-separated `--asr-model` names of
`download_dependencies.sh` — and picked per request with the `asr_model`
option:

```bash
docker build -f crates/docling-serve/Dockerfile \
  --build-arg ASR_MODELS="parakeet_tdt_0.6b_v3 whisper_small_en" -t docling-serve-asr .
curl -F files=@interview_de.mp3 -F asr_model=parakeet_tdt_0.6b_v3 \
  http://localhost:5001/v1/convert
```

Parakeet TDT 0.6B v3 adds ~670 MB (int8 encoder + decoder-joint + the Silero
VAD).

### Tesseract and its languages (`TESSERACT_VERSION`, `TESSDATA_LANGS`)

The images carry Tesseract 5.5.3, built from source by
`scripts/install/build_tesseract.sh` (the CLI alone, against the distro's
Leptonica; cross-compiled for s390x) — the distro packages are older
(Debian trixie 5.5.0, Ubuntu 24.04 5.3.4). Its language models come from
[tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast) 4.1.0, the
same LSTM models Debian's `tesseract-ocr-*` packages ship. By default
(`TESSDATA_LANGS=all`) the image carries every language model — 125 of them,
`osd` (orientation detection) included, ~340 MB — so any `ocr_lang` combination
works (`rus+eng`, `deu+fra`, `chi_sim`, …). The build arg, space- or
comma-separated, also takes tessdata stems for a slimmer image, and `scripts`
for the script models (`script/Cyrillic`, `script/Latin`, …; another ~330 MB);
a stem tessdata_fast does not have fails the build:

```bash
# slim: three languages (~18 MB)
docker build -f crates/docling-serve/Dockerfile \
  --build-arg TESSDATA_LANGS="eng osd rus" -t docling-serve-ocr .
# everything: languages and scripts (~670 MB)
docker build -f crates/docling-serve/Dockerfile \
  --build-arg TESSDATA_LANGS="all scripts" -t docling-serve-ocr .
curl -F files=@scan_ru.pdf -F ocr_engine=tesseract -F ocr_lang=rus+eng \
  http://localhost:5001/v1/convert
```

`GET /v1/config` lists what the server's Tesseract has
(`tesseract_languages`), and the web UI's OCR-language hints come from it.

A running container takes more models as files in
`/usr/local/share/tessdata` (`-v ./ukr.traineddata:/usr/local/share/tessdata/ukr.traineddata:ro`,
or a derived image) — not a mount over the directory, which also holds the
`configs/` the CLI reads its `tsv` output format from — and
`--build-arg TESSERACT_VERSION=<tag>` builds another upstream release.
List the first language whose text dominates: Tesseract reads mixed text in
that script first (`rus+eng` for Russian with some English, `eng+rus` for
the reverse).

### Running with Mounted Assets

```bash
docker run -d \
  -p 5001:5001 \
  -v /host/path/.models:/app/.models:ro \
  docling-serve-slim
```

---

## Endpoints & Health Checks

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Liveness probe: returns `{"status":"ok"}` immediately |
| `/ready` | `GET` | Readiness probe. With `--warmup`: `503 {"status":"warming_up"}` while the models load, then `200 {"status":"ready","models":"warm"}` — or `503 {"status":"warmup_failed","error":…}` if they cannot load. Without it: `200 {"status":"ready","models":"lazy"}` at once (models load on the first PDF/image request) |
| `/v1/config` | `GET` | Server capabilities, active memory, URL fetch status, resolved models, and Tesseract's installed languages (`tesseract_languages`) |
| `/metrics` | `GET` | Prometheus metrics (request counters, latency histograms, active conversions) |
| `/v1/convert` | `POST` | Synchronous conversion (multipart upload or JSON URL) |
| `/v1/convert/async` | `POST` | Asynchronous conversion: returns `202 {"task_id":"..."}` |
| `/v1/status/{id}` | `GET` | Poll status of an async conversion task |
| `/v1/result/{id}` | `GET` | Retrieve the completed async conversion result |
| `/openapi.yaml` | `GET` | OpenAPI 3.1 schema specification |
| `/` | `GET` | The web UI (playground + docs); with `DOCLING_RS_ENABLE_UI=0` / `--no-ui` a 404 naming the API (#892) |

---

## Observability

### Prometheus Metrics

`docling-serve` exports Prometheus metrics out of the box at `GET /metrics`:

- `docling_http_requests_total`: Total HTTP requests partitioned by status code and method
- `docling_http_in_flight_requests`: Gauge of current in-flight requests
- `docling_http_request_duration_seconds`: Request latency histogram
- `docling_conversions_total`: Conversions partitioned by format and status
- `docling_conversion_duration_seconds`: Conversion execution duration histogram

### OpenTelemetry Tracing

Set `OTEL_EXPORTER_OTLP_ENDPOINT` to stream distributed tracing spans over gRPC:

```bash
-e OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317 \
-e OTEL_SERVICE_NAME=docling-serve
```

---

## Security Considerations

1. **URL Fetching & SSRF Protection**:
   URL inputs (`{"url": "..."}`) are **disabled by default**. Passing `--allow-url-fetch` enables them. Even when enabled, requests to any address that is not globally routable — private, loopback, link-local, cloud metadata (`169.254.169.254`, `127.0.0.1`, `10.0.0.0/8`, multicast, reserved, NAT64, etc.) — are blocked at connect time, so DNS rebinding cannot get around the check (#752), and redirects are forbidden.

2. **Authentication**:
   `docling-serve` does not provide built-in authentication. Always bind to `127.0.0.1` or front the container with a reverse proxy (such as Caddy, NGINX, Traefik, or an API gateway) with authentication before exposing it publicly.

---

## API Usage Examples

### 1. Synchronous File Conversion (Markdown Output)

```bash
curl -X POST http://localhost:5001/v1/convert \
  -F "file=@annual_report.pdf"
```

### 2. Convert to Docling JSON with Embedded Images

```bash
curl -X POST "http://localhost:5001/v1/convert?to=json&images=embedded" \
  -F "file=@document.docx"
```

### 3. Convert to Chunk Records for RAG

```bash
curl -X POST "http://localhost:5001/v1/convert?to=chunks&chunker=hybrid" \
  -F "file=@paper.pdf"
```

### 4. Asynchronous Conversion (for Large Documents)

```bash
# Submit conversion
TASK=$(curl -s -X POST http://localhost:5001/v1/convert/async \
  -F "file=@large_book.pdf")
TASK_ID=$(echo $TASK | jq -r .task_id)

# Poll status
curl http://localhost:5001/v1/status/$TASK_ID
# => {"status":"success"}

# Retrieve result
curl http://localhost:5001/v1/result/$TASK_ID
```
