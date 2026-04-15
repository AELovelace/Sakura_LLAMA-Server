# DoLLAMACPP Frontend

A small Python + Qt frontend for:

- searching Hugging Face model repos
- listing GGUF files inside a selected repo
- downloading a selected GGUF model locally
- launching up to 4 `llama-server` instances
- exposing an Ollama-compatible API proxy for drop-in app integration
- sending a quick test chat prompt to the running server

## What this version does

This starter app now covers the full first workflow:

1. point the app at your local `llama-server` executable
2. search Hugging Face for a model repo
3. inspect repo metadata and available `.gguf` files
4. download a selected file into the local `models/` folder
5. assign the model file to one of 4 server slots
6. choose backend per slot (`CUDA`, `HIP`, `Vulkan`, `CPU`)
7. optionally assign GPU IDs per slot and choose multi-GPU mode (`parallel` or `pooled`)
8. launch each server independently and send a chat request from inside the app
9. run an Ollama-compatible proxy on `http://127.0.0.1:11434` and route requests to your server slots

It is meant as a clean foundation we can extend with presets, model management, health checks, and better server controls.

## Requirements

- Python 3.10+
- a working `llama.cpp` build or binary release that includes `llama-server`

You can get `llama.cpp` from the official repo:

- https://github.com/ggml-org/llama.cpp

The upstream README currently shows `llama-server -hf ...` support as well, but this frontend downloads a selected GGUF file locally first and then launches `llama-server` with `-m <file>`.

The `llama.cpp` README also documents its OpenAI-compatible server interface at `http://localhost:8080/v1/chat/completions`. The in-app chat panel uses that endpoint.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

## Notes

- The app stores server and chat settings in `frontend_config.json`.
- Downloaded models go into `models/`.
- For gated Hugging Face repos, paste a token into the `HF Token` field.
- In **App Settings**, configure global `llama-server` executables for `CUDA`, `HIP`, `Vulkan`, and `CPU` once (used by all server slots).
- Runtime URL fields default to `https://github.com/ggml-org/llama.cpp/releases`; missing binaries are auto-fetched from latest release assets.
- In **Server Settings**, each server slot has independent model path, host/port, context size, extra args, and checkbox-based device assignment.
- In **Server Settings**, each server slot also has an **Ollama Model** alias used by the compatibility proxy for model routing.
- For HIP and Vulkan builds, point a server slot to your `hip-llama` or `vulkan-llama` `llama-server` executable (or leave path empty to use simple autodetect paths).
- Use **Start Ollama Proxy** to expose Ollama-style endpoints on a configurable host/port (default `127.0.0.1:11434`).
- Model downloads show live transferred bytes, total size when available, and estimated throughput.
- The repository panel shows basic metadata such as author, tags, downloads, and license when available.
- The chat panel is a lightweight tester for any selected server slot, not a full chat client yet.

## Ollama API compatibility

The in-app proxy implements these endpoints:

- `GET /api/tags`
- `POST /api/show`
- `POST /api/generate`
- `POST /api/chat`

Routing behavior:

- `model` (or `name`) is matched to a server slot by that slot's **Ollama Model** alias.
- If no model is provided or no exact alias matches, requests route to the configured **Ollama default** server.

Device assignment behavior:

- Select devices per server with checkboxes (CPU and/or detected GPUs).
- Valid combinations are CPU-only, or one/more GPUs from the same backend.
- Backend executable is auto-selected from App Settings based on selected devices.

Runtime fetch behavior:

- If a selected backend runtime is missing locally, the app runs `scripts/fetch_runtime_binaries.py` automatically.
- Downloaded runtime archives are extracted into versioned subfolders under each backend folder (for example `hip-llama/llama-b8795-bin-win-hip-radeon-x64/`).
- Auto-detection prefers the newest `llama-server` binary found for that backend.
- For Windows, release asset selection prefers these targets from latest ggml release:
	- CUDA: `win-cuda-12.x`
	- Vulkan: `win-vulkan-x64`
	- HIP: `win-hip-radeon-x64`
	- CPU: `win-cpu-x64`

Typical client setup:

- Set `OLLAMA_HOST=http://127.0.0.1:11434`
- Keep using your existing Ollama client library or HTTP calls

## Good first test

Search for small GGUF repos such as:

- `gemma 3 1b gguf`
- `qwen2.5 0.5b gguf`
- `tinyllama gguf`

Then choose a smaller quantized file like `Q4_K_M` or `Q4_0` to validate the workflow quickly.
