# Docker on the GPU box

Three services in `compose.yaml` (decisions D3, D6, D27; background in `docs/reviews/architecture/doc.md` 3.1, a closed record: its `--api-key` example is superseded by this page):

| Service | Profile | Image | Purpose |
|---|---|---|---|
| `judge` | `judge` | `vllm/vllm-openai` (pinned) | Local judge: Qwen2.5-7B-Instruct-AWQ with structured outputs and prefix caching. No published port, reachable only on the isolated internal network `judge_net` |
| `judge-proxy` | `judge` | `nginxinc/nginx-unprivileged` (pinned) | The only way in: publishes `127.0.0.1:8000`, forwards `/v1/*` and `/health`, rejects everything else, any unexpected `Host` header and any ambiguous URI. Starts only once the judge is healthy. Config: `docker/judge-proxy.conf` |
| `work` | `train` | built from `Dockerfile` | Project environment: CUDA 13.2 runtime, uv, the locked venv in `/opt/venv`, the repository bind-mounted at `/workspace` |

## Setup

1. Create `.env` from `.env.example` (git-ignored; never commit it or paste its values into logs):

   ```
   VLLM_API_KEY=<openssl rand -hex 32>
   HF_HOME=<host directory for the Hugging Face cache; default ~/.cache/huggingface>
   HOST_UID=<id -u>
   HOST_GID=<id -g>
   ```

   `HOST_UID` and `HOST_GID` default to 1000. They are needed when your user is not 1000:1000, so that the containers write nothing as root into the repository or the cache. `work` bakes them in at build time: rebuild it after changing them.
2. `docker compose --profile judge config --quiet` and `docker compose --profile train config --quiet` must both succeed. The key is not checked at `config` time (so the `train` profile works without it); the judge refuses to start without a key of at least 32 characters, and `up` reports that failure when run with `--wait`.
3. First time only, download the model into the cache. The judge runs offline (`HF_HUB_OFFLINE=1`) and mounts the cache read-only, so it never downloads or writes:

   ```
   docker compose --profile train run --rm work uv run hf download Qwen/Qwen2.5-7B-Instruct-AWQ --revision <revision from params.yaml>
   ```

## The profiles take turns (D27)

With `--gpu-memory-utilization 0.85` the judge holds about 13 GB of the 16 GB card. Never run `judge` and `work` at the same time: training next to the judge runs out of memory. Stop one before starting the other.

### judge (labeling)

```
docker compose --profile judge up -d --wait --wait-timeout 1500   # blocks until the judge is healthy
docker compose --profile judge logs -f judge
docker compose --profile judge down                                  # frees the GPU
```

- Startup takes about 4 minutes from a warm cache (CUDA graph capture and FlashInfer autotune dominate). The healthcheck allows 20 minutes before it counts failures. The proxy waits for a healthy judge (`depends_on: service_healthy`), so `up` blocks that long even without `--wait`, and if the judge exits (a bad key, a missing model) `up` stops with `dependency failed to start: container ... judge ... exited (1)`. Read the reason with `docker compose --profile judge logs judge`.
- Every request to `/v1/*` needs `Authorization: Bearer $VLLM_API_KEY`. `/health` is open. Every other path (`/metrics`, `/tokenize`, `/invocations`, `/docs`, `/openapi.json`, ...) returns 404 with or without the key, so vLLM's `/metrics` is not reachable from outside the container; other methods than GET and POST return 403.
- The proxy forwards the URI that nginx normalized, never the raw one: each `proxy_pass` carries a URI part. On top of that it answers 400 to any raw path that is not made of `A-Z a-z 0-9 _ . / -`, or that contains `..` or `//`, so encoded or dot-segment tricks (`/metrics/../v1/`, `/metrics%2F..%2Fv1%2F`, `/v1/%2e%2e/metrics`) never reach vLLM. Query strings pass through unchanged. `tests/test_judge_proxy_conf.py` checks these rules against the config.
- The proxy accepts only these `Host` headers: `localhost:8000`, `127.0.0.1:8000` and `judge-proxy:8080` (the name `work` uses). Anything else gets 403, which blocks DNS rebinding from a browser. Point clients at `http://127.0.0.1:8000/v1`, not at another hostname.
- The port is published as `127.0.0.1:8000:8080`. vLLM has no access control beyond the key, so it never gets a published port. The network it sits on is `internal` (no route to the internet) and uses `gateway_mode_ipv4: isolated` (Docker 28 or later), so the host has no address on it: the judge container's IP is unreachable from the host, and the proxy is the only way in. Never change the proxy binding to `0.0.0.0`.
- Residual: the proxy container is also reachable from the host at its address on the `default` network (for example `172.23.0.2:8080`). That is the same proxy with the same rules and key, only on a local bridge, not a bypass.
- From the Raspberry Pi: `ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8000:127.0.0.1:8000 <gpu-box>`, then use `http://127.0.0.1:8000/v1` with the same key.
- The model is pinned by `--revision` and `--tokenizer-revision` (a commit hash); `trust_remote_code` is off. `--generation-config=vllm` stops vLLM from applying Qwen's `generation_config.json` (repetition penalty, top-k), so the judge samples only with the parameters the request sets. Usage statistics and Hugging Face telemetry are off.
- The key reaches vLLM through the `VLLM_API_KEY` environment variable, not the command line, so it does not appear in `ps` or in the container's command. It is still in the container environment (`docker inspect`), which only your user and root can read.
- For structured output send either `response_format` (`json_schema`) or `structured_outputs: {"json": <schema>}` in the request body.
- `cap_drop: [ALL]` and `no-new-privileges` apply to every service. The proxy runs as uid 101 on an unprivileged port with a read-only root filesystem.

### work (training, DVC, tests)

```
docker compose --profile train build work
docker compose --profile train run --rm work                         # interactive bash
docker compose --profile train run --rm work uv run pytest -m "not integration"
docker compose --profile train run --rm work uv run mypy src         # authoritative type check (H8)
docker compose --profile train run --rm work uv run dvc repro
```

- The venv is in `/opt/venv`, outside the bind mount, so `/workspace` does not hide it. `UV_LOCKED=1` makes every `uv run` fail instead of re-resolving when `uv.lock` is stale. `uv run` installs `banhammer` itself into the venv on each start (about a second).
- `.git` is mounted read-only and `.env` is replaced by an empty file, so a process in the container (or a dependency it runs) cannot rewrite history or read the keys. Create `.env` before the first `run`; Docker would otherwise create the mount point as root. `dvc repro` works inside the container (checked with the empty pipeline). **`dvc exp run` and anything else that writes Git refs or commits runs on the host**, not in the container.
- `work` is on the default network; the judge is reachable from it only as `http://judge-proxy:8080/v1` (`JUDGE_URL`), and only if the judge profile is deliberately up, which D27 forbids while training. Because `.env` is hidden, the key would have to be passed explicitly.
- `shm_size: 8gb` is for `DataLoader` workers; Docker's default of 64 MB crashes them.
- `judge` and `work` each reserve one GPU (`count: 1`), not all of them; `tests/test_compose_hardening.py` checks it.
- Rebuild after any change to `pyproject.toml` or `uv.lock`. Only those two files are in the build context (`.dockerignore`).

## Tunables (H7)

The model id, revisions, `--max-model-len` and `--gpu-memory-utilization` are tunables, so `params.yaml` (`judge:` section) is their source. Compose cannot read `params.yaml`, so the same values are literals in `compose.yaml`, and `tests/test_compose_params.py` fails when they differ. To change one, edit both and run the tests.

## Pins

Every image is pinned by tag and by digest. The digests are multi-arch index digests, verified on 2026-11-25.

| Image | Tag | Digest | Why |
|---|---|---|---|
| `vllm/vllm-openai` | `v0.34.0` | `sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1` | Latest stable release (2026-11-16). CUDA 13.0.2 image with `TORCH_CUDA_ARCH_LIST` including 12.0 (sm_120). The host driver must not be affected by NVIDIA bulletin 5861: see "Host driver" |
| `nvidia/cuda` | `13.2.1-cudnn-runtime-ubuntu24.04` | `sha256:1c0c68dbf3258d32b446a02cb4be05c8478b65d320d7856ad33c4bbdf898ca86` | Matches the `cu132` torch wheels in `uv.lock` (torch 2.15 pins `cuda-toolkit==13.2.2`; 13.2.1 is the newest 13.2 `cudnn-runtime` tag). The wheels bring their own CUDA libraries, so the image supplies the base OS and the driver-compatibility check (`NVIDIA_REQUIRE_CUDA` accepts driver 580). `apt-get upgrade -y` in the `Dockerfile` brings the OS layer (built 2026-04) up to date at each build |
| `ghcr.io/astral-sh/uv` | `0.12.0` | `sha256:606e70c71c852d03f611b1e56a195d08648507018a7057fab82c4974c4eae105` | Same uv as the one that wrote `uv.lock` and as the `uv_build` range in `pyproject.toml` (`>=0.12.0,<0.13`) |
| `nginxinc/nginx-unprivileged` | `1.30.6-alpine` | `sha256:f5116ed4606f3eb015dd4a8e28cb9b351476dc0ad4bc6e9f93b19cec130cdf03` | Stable branch (2026-10-19), 26 MB, runs as uid 101 on port 8080 so it works with all capabilities dropped and a read-only root. nginx expresses the path allow-list and the exact `Host` check declaratively; Caddy would do the same but needs more configuration for strict `Host` rejection |
| `Qwen/Qwen2.5-7B-Instruct-AWQ` | revision `b25037543e9394b818fdfca67ab2a00ecc7dd641` | n/a | `main` on 2026-11-25 (last modified 2024-10-09), Apache-2.0, safetensors |

### Host driver

Recorded host driver: **595.104.02**, open kernel module (`nvidia-smi` reports CUDA 13.2), installed 2026-11-26. Update this line whenever the driver changes.

- **Minimum safe driver.** NVIDIA security bulletin 5861 (2026-09-30) covers CVE-2026-47501, an out-of-bounds write in the kernel driver that lets a container escalate to the host through `/dev/nvidia*`. Fixed releases per branch: R580 580.178.04, 595.91.07, 610.57.04, 615.71.09 (the list `tests/test_nvidia_driver.py` checks). The box was on 580.159.03, which is affected; on Pop!_OS, `sudo apt install --only-upgrade nvidia-driver-580-open` moved it to 595.104.02, because Pop's 580.178.04 `nvidia-driver-580-open` is a transition package that depends on `nvidia-driver-595-open` (upgrading only `system76-driver-nvidia` does not change the driver). Do not run GPU containers (`work`, `judge`) on an affected driver; the `gpu`-marked driver test fails on one; `cap_drop` and `no-new-privileges` do not protect against a kernel bug reached through the device nodes.
- The `cu132` torch wheels and the CUDA 13.2 base image run natively on the 595 driver (CUDA 13.2); the vLLM image (CUDA 13.0) runs on it as an older 13.x runtime. On an R580 driver (native CUDA 13.0) they ran through CUDA minor-version compatibility within the 13.x family, checked on 580.159.03. After any driver change, rerun the GPU check (`torch.cuda.is_available()`, the arch list with sm_120, a CUDA matmul) in `work`.
- At each milestone, check the driver version against the NVIDIA security bulletins (NVIDIA PSIRT, https://www.nvidia.com/en-us/security/) and note the date in `docs/dependencies.md`, "Advisory checks".

### Bumping a pin

1. **vLLM.** Read the release notes for changes to defaults and to the structured-output and AWQ paths, check the Docker table for the CUDA version, and confirm the driver in `nvidia-smi` supports it. Pull the new tag, copy its digest (`docker image inspect --format '{{.Id}}'` gives the index digest of a freshly pulled multi-arch tag), edit `compose.yaml`, and rerun the smoke test: the smoke list below.
2. **CUDA base.** Keep the minor version equal to the torch wheel index (`cu132` in `pyproject.toml`). Edit the `FROM` line, rebuild, run the GPU check and the tests above. When the torch index moves to a new CUDA minor, check first that the host driver supports it (see "Host driver").
3. **uv.** Change the `COPY --from` line together with the `uv_build` range in `pyproject.toml`; rebuild so `uv sync --locked` verifies the lockfile.
4. **nginx.** Edit the tag and digest in `compose.yaml`; rerun the path, URI and `Host` checks below.
5. **Qwen revision.** Take the commit hash from `https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct-AWQ`, set it in `params.yaml` and in both `--revision` flags of `compose.yaml`, prefetch it with the `hf download` command, and note that labels produced under the old revision are not reproducible with the new one.
6. Update the table above and the `tool` rows in `docs/dependencies.md`.

## Smoke list (after any change to the images, `compose.yaml` or `docker/judge-proxy.conf`)

With the judge profile up and `KEY` taken from `.env` (do not print it):

- `/v1/chat/completions`: 200 with the key, 401 without; a structured-output request parses; a `"stream": true` request delivers chunks incrementally; `/v1/models?x=1` returns 200.
- `/health`: 200 without a key.
- 404 with and without the key, GET and POST: `/metrics`, `/tokenize`, `/detokenize`, `/invocations`, `/docs`, `/openapi.json`, `/redoc`, `/ping`, `/version`.
- With `curl --path-as-is`, 400 and never an unauthenticated vLLM answer: `/metrics/../v1/`, `/metrics/../health`, `/metrics%2F..%2Fv1%2F`, `/v1/%2e%2e/metrics`, `/invocations/../v1/models`, `/v1//models`, `//metrics`.
- `Host` header: 403 for `evil.example.com`, `localhost`, `127.0.0.1` (no port) and `[::1]:8000`; 200 for `localhost:8000`, `127.0.0.1:8000` and `judge-proxy:8080`.
- `ss -ltnp` shows only `127.0.0.1:8000`; `docker port` of the judge is empty; `curl` from the host to the judge container IP on port 8000 gets no answer.
- `docker compose --profile judge logs judge | grep -c -F "$KEY"` prints 0.
- Prefix caching and the metrics: `/metrics` is blocked by the proxy, so read `vllm:prefix_cache_hits_total` from inside: `docker compose --profile judge exec judge python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/metrics').read().decode())"`.

## Rebuilding the `work` image

`apt-get upgrade -y` in the `Dockerfile` pulls the newest OS packages at build time, so two builds of the same `Dockerfile` can differ: the image is not reproducible from the repository alone (the base image digest, the uv digest and `uv.lock` still are). The trade-off is accepted so that OS security fixes arrive without waiting for a new base image. The apt layer is cached like any other, so a plain `build` reuses the old packages. To pick up OS fixes run:

```
docker compose --profile train build --pull --no-cache work
```

Rebuild this way at each milestone (it re-downloads the uv cache, about seven minutes) and after any bulletin that names a package in the image.

## Disk

The vLLM image is about 36 GB unpacked, `work` about 18 GB, the Qwen weights 5.6 GB. The Docker build cache keeps the uv download cache (several GB); `docker builder prune` reclaims it, at the cost of a seven-minute rebuild of `work`.

## Known behaviour

- vLLM 0.34 sizes the KV cache from the memory free after warmup (extensible KV cache) and defaults to a block size of 64. With `--gpu-memory-utilization 0.85` it used about 12.5 GiB on this box and left the desktop session's 1.2 GB alone.
- Prefix caching only reuses whole 64-token blocks, so a shared ruleset prefix shorter than 64 tokens never hits. D17 prefixes are far longer.
- Files that `hf download` writes into the cache belong to `HOST_UID`; do not run the judge as root against the same cache, or it will leave root-owned files in it.
