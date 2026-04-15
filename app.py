from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests
from huggingface_hub import HfApi, hf_hub_url
from PySide6.QtCore import QProcess, QProcessEnvironment, QRunnable, QThreadPool, Qt, QObject, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


APP_NAME = "DoLLAMACPP Frontend"
CONFIG_PATH = Path("frontend_config.json")
MODELS_DIR = Path("models")
LLAMA_CPP_RELEASES_URL = "https://github.com/ggml-org/llama.cpp/releases"


@dataclass
class ModelSearchResult:
    repo_id: str
    likes: int
    downloads: int
    last_modified: str


@dataclass
class RepoFile:
    name: str
    size_text: str
    size_bytes: int | None = None


@dataclass
class RepoDetails:
    repo_id: str
    author: str
    downloads: int
    likes: int
    last_modified: str
    library_name: str
    pipeline_tag: str
    license_name: str
    gated: bool
    private: bool
    tags: list[str]
    summary: str


@dataclass
class GPUDevice:
    key: str
    label: str
    backend: str
    env_id: str


@dataclass
class ServerSlot:
    index: int
    process: QProcess
    backend_label: QLabel
    ollama_model_input: QLineEdit
    model_path_input: QLineEdit
    host_input: QLineEdit
    port_input: QSpinBox
    ctx_size_input: QSpinBox
    extra_args_input: QLineEdit
    device_box: QGroupBox
    device_checkboxes: dict[str, QCheckBox]
    split_mode_input: QComboBox
    start_button: QPushButton
    stop_button: QPushButton
    status_label: QLabel
    proxy_status_label: QLabel


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(object)


class Worker(QRunnable):
    def __init__(
        self,
        fn: Callable[..., Any],
        *args: Any,
        use_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.use_progress = use_progress
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            call_kwargs = dict(self.kwargs)
            if self.use_progress:
                call_kwargs["progress_callback"] = self.signals.progress.emit
            result = self.fn(*self.args, **call_kwargs)
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))
            return
        self.signals.finished.emit(result)


class HuggingFaceClient:
    def __init__(self) -> None:
        self.api = HfApi()

    def search_models(self, query: str, token: str | None = None) -> list[ModelSearchResult]:
        query = query.strip()
        if not query:
            return []

        base_kwargs = {
            "search": query,
            "sort": "downloads",
            "limit": 25,
            "token": token or None,
        }

        try:
            models = self.api.list_models(direction=-1, **base_kwargs)
        except TypeError as exc:
            if "direction" not in str(exc):
                raise
            models = self.api.list_models(**base_kwargs)

        results: list[ModelSearchResult] = []
        for model in models:
            model_id = getattr(model, "id", "")
            if not model_id:
                continue
            results.append(
                ModelSearchResult(
                    repo_id=model_id,
                    likes=int(getattr(model, "likes", 0) or 0),
                    downloads=int(getattr(model, "downloads", 0) or 0),
                    last_modified=self._format_date(getattr(model, "last_modified", None)),
                )
            )
        return results

    def get_repo_details(self, repo_id: str, token: str | None = None) -> tuple[RepoDetails, list[RepoFile]]:
        info = self.api.model_info(repo_id=repo_id, token=token or None, files_metadata=True)
        card_data = getattr(info, "cardData", None) or getattr(info, "card_data", None) or {}
        summary = self._coalesce_summary(card_data)
        license_name = str(card_data.get("license") or "") if isinstance(card_data, dict) else ""

        details = RepoDetails(
            repo_id=repo_id,
            author=str(getattr(info, "author", "") or ""),
            downloads=int(getattr(info, "downloads", 0) or 0),
            likes=int(getattr(info, "likes", 0) or 0),
            last_modified=self._format_date(getattr(info, "last_modified", None)),
            library_name=str(getattr(info, "library_name", "") or ""),
            pipeline_tag=str(getattr(info, "pipeline_tag", "") or ""),
            license_name=license_name,
            gated=bool(getattr(info, "gated", False)),
            private=bool(getattr(info, "private", False)),
            tags=list(getattr(info, "tags", []) or []),
            summary=summary,
        )

        files: list[RepoFile] = []
        for sibling in info.siblings or []:
            name = getattr(sibling, "rfilename", "")
            if not name.lower().endswith(".gguf"):
                continue
            size_value = getattr(sibling, "size", None)
            files.append(
                RepoFile(
                    name=name,
                    size_text=self._format_size(size_value),
                    size_bytes=size_value,
                )
            )

        return details, sorted(files, key=lambda item: item.name.lower())

    def download_file(
        self,
        repo_id: str,
        filename: str,
        destination_dir: Path,
        token: str | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> Path:
        destination_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name
        destination = destination_dir / safe_name
        url = hf_hub_url(repo_id=repo_id, filename=filename)
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        downloaded = 0
        started = time.monotonic()

        with requests.get(url, headers=headers, stream=True, timeout=30) as response:
            response.raise_for_status()
            total_header = response.headers.get("Content-Length")
            total_bytes = int(total_header) if total_header and total_header.isdigit() else None

            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        elapsed = max(time.monotonic() - started, 0.001)
                        progress_callback(
                            {
                                "downloaded": downloaded,
                                "total": total_bytes,
                                "speed": downloaded / elapsed,
                                "filename": safe_name,
                            }
                        )

        return destination

    @staticmethod
    def _coalesce_summary(card_data: Any) -> str:
        if not isinstance(card_data, dict):
            return ""
        for key in ("summary", "description", "model_summary"):
            value = card_data.get(key)
            if value:
                return str(value).strip()
        return ""

    @staticmethod
    def _format_date(value: Any) -> str:
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d")
        return str(value or "")

    @staticmethod
    def _format_size(size_value: int | None) -> str:
        if not size_value:
            return ""
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(size_value)
        for unit in units:
            if size < 1024 or unit == units[-1]:
                if unit == "B":
                    return f"{int(size)} {unit}"
                return f"{size:.1f} {unit}"
            size /= 1024
        return ""


class LlamaServerClient:
    def chat(
        self,
        base_url: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        url = f"{base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": "local-model",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("No choices returned from llama-server.")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict)).strip()
        return str(content).strip()


class OllamaCompatProxy:
    def __init__(
        self,
        get_snapshot: Callable[[], dict[str, Any]],
    ) -> None:
        self._get_snapshot = get_snapshot
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def is_running(self) -> bool:
        return self._server is not None and self._thread is not None and self._thread.is_alive()

    def start(self, host: str, port: int) -> None:
        if self.is_running():
            raise RuntimeError("Ollama compatibility proxy is already running.")

        handler_cls = self._build_handler_class()
        self._server = ThreadingHTTPServer((host, port), handler_cls)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._server:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None

    def _build_handler_class(self) -> type[BaseHTTPRequestHandler]:
        parent = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "DoLLAMACPP-OllamaProxy/1.0"

            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/api/tags":
                    payload = parent._build_tags_payload()
                    self._send_json(200, payload)
                    return
                self._send_json(404, {"error": f"Unsupported endpoint: {parsed.path}"})

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/api/show":
                    parent._handle_show(self)
                    return
                if parsed.path == "/api/generate":
                    parent._handle_generate(self)
                    return
                if parsed.path == "/api/chat":
                    parent._handle_chat(self)
                    return
                self._send_json(404, {"error": f"Unsupported endpoint: {parsed.path}"})

            def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_ndjson(self, status_code: int, payloads: list[dict[str, Any]]) -> None:
                body = b"".join((json.dumps(item) + "\n").encode("utf-8") for item in payloads)
                self.send_response(status_code)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler

    def _read_json(self, handler: BaseHTTPRequestHandler) -> dict[str, Any]:
        length_header = handler.headers.get("Content-Length")
        if not length_header:
            return {}
        try:
            length = int(length_header)
        except ValueError as exc:
            raise RuntimeError("Invalid Content-Length header.") from exc
        raw = handler.rfile.read(length) if length > 0 else b""
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Request body must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Request body JSON must be an object.")
        return payload

    def _build_tags_payload(self) -> dict[str, Any]:
        snapshot = self._get_snapshot()
        models = []
        for slot in snapshot.get("slots", []):
            model_name = str(slot.get("ollama_model", "")).strip()
            if not model_name:
                continue
            model_path = str(slot.get("model_path", "")).strip()
            size_bytes = 0
            if model_path and Path(model_path).exists():
                try:
                    size_bytes = int(Path(model_path).stat().st_size)
                except OSError:
                    size_bytes = 0
            models.append(
                {
                    "name": model_name,
                    "model": model_name,
                    "modified_at": datetime.utcnow().isoformat() + "Z",
                    "size": size_bytes,
                    "digest": "",
                    "details": {
                        "format": "gguf",
                        "family": "llama.cpp",
                        "parameter_size": "",
                        "quantization_level": "",
                    },
                }
            )
        return {"models": models}

    def _send_error(self, handler: BaseHTTPRequestHandler, message: str, status_code: int = 400) -> None:
        handler._send_json(status_code, {"error": message})  # type: ignore[attr-defined]

    def _resolve_slot(self, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        snapshot = self._get_snapshot()
        slots = list(snapshot.get("slots", []))
        requested_model = str(payload.get("model") or payload.get("name") or "").strip()

        if requested_model:
            for slot in slots:
                slot_name = str(slot.get("ollama_model", "")).strip()
                if requested_model == slot_name:
                    return slot, requested_model

        default_index = int(snapshot.get("default_server", 0) or 0)
        if 0 <= default_index < len(slots):
            fallback = slots[default_index]
            if not requested_model:
                requested_model = str(fallback.get("ollama_model", "")).strip() or f"server-{default_index + 1}"
            return fallback, requested_model

        if slots:
            fallback = slots[0]
            if not requested_model:
                requested_model = str(fallback.get("ollama_model", "")).strip() or "server-1"
            return fallback, requested_model

        return None, requested_model or None

    @staticmethod
    def _extract_text_from_chat_choice(data: dict[str, Any]) -> str:
        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text", "")))
            return "\n".join(parts).strip()
        return str(content).strip()

    @staticmethod
    def _extract_text_from_completion(data: dict[str, Any]) -> str:
        choices = data.get("choices", [])
        if not choices:
            return ""
        text = choices[0].get("text", "")
        return str(text).strip()

    def _handle_show(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            payload = self._read_json(handler)
        except RuntimeError as exc:
            self._send_error(handler, str(exc), 400)
            return

        slot, model_name = self._resolve_slot(payload)
        if slot is None:
            self._send_error(handler, "No server slots are configured.", 400)
            return

        model_path = str(slot.get("model_path", "")).strip()
        details = {
            "parent_model": "",
            "format": "gguf",
            "family": "llama.cpp",
            "families": ["llama.cpp"],
            "parameter_size": "",
            "quantization_level": "",
        }
        payload_out = {
            "license": "",
            "modelfile": f"FROM {model_path}" if model_path else "",
            "parameters": "",
            "template": "",
            "system": "",
            "details": details,
            "model_info": {
                "model_name": model_name or "",
                "server_slot": int(slot.get("index", 0)) + 1,
            },
        }
        handler._send_json(200, payload_out)  # type: ignore[attr-defined]

    def _forward_chat(
        self,
        slot: dict[str, Any],
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        base_url = f"http://{slot.get('host', '127.0.0.1')}:{int(slot.get('port', 8080))}"
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": "local-model",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=300,
        )
        response.raise_for_status()
        return self._extract_text_from_chat_choice(response.json())

    def _forward_completion(
        self,
        slot: dict[str, Any],
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        base_url = f"http://{slot.get('host', '127.0.0.1')}:{int(slot.get('port', 8080))}"
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/completions",
            json={
                "model": "local-model",
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=300,
        )
        response.raise_for_status()
        text = self._extract_text_from_completion(response.json())
        if text:
            return text

        chat_response = requests.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": "local-model",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=300,
        )
        chat_response.raise_for_status()
        return self._extract_text_from_chat_choice(chat_response.json())

    def _handle_generate(self, handler: BaseHTTPRequestHandler) -> None:
        started = time.time()
        try:
            payload = self._read_json(handler)
        except RuntimeError as exc:
            self._send_error(handler, str(exc), 400)
            return

        slot, model_name = self._resolve_slot(payload)
        if slot is None:
            self._send_error(handler, "No server slots are configured.", 400)
            return

        prompt = str(payload.get("prompt", "") or "")
        if not prompt:
            self._send_error(handler, "Missing 'prompt'.", 400)
            return

        options = payload.get("options", {})
        if not isinstance(options, dict):
            options = {}
        temperature = float(options.get("temperature", payload.get("temperature", 0.7)) or 0.7)
        max_tokens = int(options.get("num_predict", payload.get("num_predict", 256)) or 256)

        try:
            text = self._forward_completion(slot, prompt, temperature, max_tokens)
        except requests.RequestException as exc:
            self._send_error(handler, f"Upstream llama-server request failed: {exc}", 502)
            return

        stream = bool(payload.get("stream", False))
        created_at = datetime.utcnow().isoformat() + "Z"
        total_duration = int((time.time() - started) * 1_000_000_000)
        response_chunk = {
            "model": model_name or "",
            "created_at": created_at,
            "response": text,
            "done": True,
            "done_reason": "stop",
            "total_duration": total_duration,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 0,
            "eval_count": 0,
            "eval_duration": 0,
        }

        if stream:
            handler._send_ndjson(200, [response_chunk])  # type: ignore[attr-defined]
            return
        handler._send_json(200, response_chunk)  # type: ignore[attr-defined]

    def _handle_chat(self, handler: BaseHTTPRequestHandler) -> None:
        started = time.time()
        try:
            payload = self._read_json(handler)
        except RuntimeError as exc:
            self._send_error(handler, str(exc), 400)
            return

        slot, model_name = self._resolve_slot(payload)
        if slot is None:
            self._send_error(handler, "No server slots are configured.", 400)
            return

        raw_messages = payload.get("messages", [])
        if not isinstance(raw_messages, list):
            self._send_error(handler, "'messages' must be a list.", 400)
            return

        messages: list[dict[str, str]] = []
        for item in raw_messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "user") or "user")
            content = item.get("content", "")
            if isinstance(content, list):
                content_text = "\n".join(
                    str(part.get("text", "")) for part in content if isinstance(part, dict)
                ).strip()
            else:
                content_text = str(content or "")
            messages.append({"role": role, "content": content_text})

        if not messages:
            self._send_error(handler, "No valid messages provided.", 400)
            return

        options = payload.get("options", {})
        if not isinstance(options, dict):
            options = {}
        temperature = float(options.get("temperature", payload.get("temperature", 0.7)) or 0.7)
        max_tokens = int(options.get("num_predict", payload.get("num_predict", 256)) or 256)

        try:
            text = self._forward_chat(slot, messages, temperature, max_tokens)
        except requests.RequestException as exc:
            self._send_error(handler, f"Upstream llama-server request failed: {exc}", 502)
            return

        stream = bool(payload.get("stream", False))
        created_at = datetime.utcnow().isoformat() + "Z"
        total_duration = int((time.time() - started) * 1_000_000_000)
        response_chunk = {
            "model": model_name or "",
            "created_at": created_at,
            "message": {"role": "assistant", "content": text},
            "done": True,
            "done_reason": "stop",
            "total_duration": total_duration,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 0,
            "eval_count": 0,
            "eval_duration": 0,
        }

        if stream:
            handler._send_ndjson(200, [response_chunk])  # type: ignore[attr-defined]
            return
        handler._send_json(200, response_chunk)  # type: ignore[attr-defined]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1380, 900)

        self.thread_pool = QThreadPool()
        self.hf_client = HuggingFaceClient()
        self.server_client = LlamaServerClient()
        self.search_results: list[ModelSearchResult] = []
        self.repo_files: list[RepoFile] = []
        self.repo_details: RepoDetails | None = None
        self.chat_history: list[dict[str, str]] = []
        self.pending_user_message = ""
        self.available_devices: list[GPUDevice] = self._detect_available_devices()
        self.server_slots: list[ServerSlot] = []
        self.server_running_states = [False, False, False, False]
        self._snapshot_lock = threading.Lock()
        self._ollama_snapshot: dict[str, Any] = {
            "default_server": 0,
            "slots": [],
        }
        self.ollama_proxies: list[OllamaCompatProxy] = [
            OllamaCompatProxy(lambda i=i: self._get_slot_snapshot(i))
            for i in range(4)
        ]

        self._build_ui()
        self._apply_theme()
        self._load_config()
        self._set_ollama_proxy_state(False, "Ollama API proxy is stopped.")
        for slot in self.server_slots:
            self._set_server_state(slot.index, False)
        self._refresh_ollama_snapshot()
        self._update_proxy_port_labels()

    def _apply_theme(self) -> None:
        bg_path = (Path(__file__).resolve().parent / "assets" / "bg.jpg")
        bg_uri = bg_path.as_uri().replace("\"", "\\\"") if bg_path.exists() else ""

        bg_style = f'border-image: url("{bg_uri}") 0 0 0 0 stretch stretch;' if bg_uri else "background-color: #261b23;"

        self.setStyleSheet(
            f"""
            QMainWindow {{
                background: #000000;
                color: #4f2f40;
            }}

            QWidget#appRoot {{
                {bg_style}
            }}

            QWidget {{
                color: #4f2f40;
                selection-background-color: rgba(236, 127, 182, 220);
                selection-color: #ffffff;
            }}

            QTabWidget::pane {{
                border: 1px solid rgba(255, 222, 238, 95);
                background: rgba(255, 237, 246, 175);
                border-radius: 8px;
            }}

            QTabBar::tab {{
                background: rgba(255, 205, 227, 215);
                color: #5a3447;
                border: 1px solid rgba(255, 222, 238, 90);
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 8px 12px;
                margin-right: 4px;
            }}

            QTabBar::tab:selected {{
                background: rgba(250, 170, 206, 230);
                color: #4e2c3e;
            }}

            QGroupBox {{
                background: rgba(255, 242, 249, 176);
                border: 1px solid rgba(255, 222, 238, 80);
                border-radius: 8px;
                margin-top: 12px;
                padding: 8px;
                font-weight: 600;
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #6a3f55;
            }}

            QLineEdit,
            QTextEdit,
            QPlainTextEdit,
            QListWidget,
            QTableWidget,
            QComboBox,
            QSpinBox {{
                background: rgba(255, 250, 253, 226);
                color: #4f2f40;
                border: 1px solid rgba(255, 216, 236, 95);
                border-radius: 6px;
                padding: 6px;
            }}

            QTableWidget,
            QListWidget,
            QPlainTextEdit,
            QTextEdit {{
                alternate-background-color: rgba(255, 236, 246, 200);
            }}

            QTableWidget::item,
            QListWidget::item {{
                background: transparent;
            }}

            QTableWidget::item:selected,
            QListWidget::item:selected {{
                background: rgba(237, 121, 177, 210);
                color: #ffffff;
            }}

            QPushButton {{
                background: rgba(247, 168, 206, 238);
                color: #4b2a3b;
                border: 1px solid rgba(255, 236, 245, 130);
                border-radius: 6px;
                padding: 7px 12px;
                font-weight: 600;
            }}

            QPushButton:hover {{
                background: rgba(252, 185, 218, 242);
            }}

            QPushButton:pressed {{
                background: rgba(232, 141, 188, 242);
            }}

            QPushButton:disabled {{
                background: rgba(236, 210, 222, 205);
                color: rgba(120, 89, 105, 185);
            }}

            QProgressBar {{
                border: 1px solid rgba(255, 216, 236, 95);
                border-radius: 6px;
                background: rgba(255, 243, 249, 210);
                text-align: center;
                color: #593749;
            }}

            QProgressBar::chunk {{
                border-radius: 4px;
                background: rgba(238, 138, 188, 232);
            }}

            QHeaderView::section {{
                background: rgba(249, 198, 221, 226);
                color: #5a3447;
                border: 1px solid rgba(255, 220, 238, 95);
                padding: 6px;
            }}

            QScrollArea,
            QSplitter,
            QLabel {{
                background: transparent;
            }}
            """
        )

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("appRoot")
        main_layout = QVBoxLayout(root)

        tabs = QTabWidget()
        tabs.addTab(self._build_app_settings_tab(), "App Settings")
        tabs.addTab(self._build_hf_tab(), "Hugging Face Search")
        tabs.addTab(self._build_run_panel(), "Server Settings")
        tabs.addTab(self._build_log_panel(), "Server Log")
        tabs.addTab(self._build_chat_panel(), "Chat Tester")

        main_layout.addWidget(tabs)
        self.setCentralWidget(root)

    def _build_app_settings_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        llama_box = QGroupBox("llama.cpp Runtime Paths")
        llama_layout = QFormLayout(llama_box)

        self.cuda_llama_path_input = QLineEdit()
        self.hip_llama_path_input = QLineEdit()
        self.vulkan_llama_path_input = QLineEdit()
        self.cpu_llama_path_input = QLineEdit()
        self.cuda_runtime_url_input = QLineEdit()
        self.hip_runtime_url_input = QLineEdit()
        self.vulkan_runtime_url_input = QLineEdit()
        self.cpu_runtime_url_input = QLineEdit()
        self.cuda_runtime_url_input.setPlaceholderText(LLAMA_CPP_RELEASES_URL)
        self.hip_runtime_url_input.setPlaceholderText(LLAMA_CPP_RELEASES_URL)
        self.vulkan_runtime_url_input.setPlaceholderText(LLAMA_CPP_RELEASES_URL)
        self.cpu_runtime_url_input.setPlaceholderText(LLAMA_CPP_RELEASES_URL)

        llama_layout.addRow("CUDA", self._build_path_row(self.cuda_llama_path_input, "cuda"))
        llama_layout.addRow("HIP", self._build_path_row(self.hip_llama_path_input, "hip"))
        llama_layout.addRow("Vulkan", self._build_path_row(self.vulkan_llama_path_input, "vulkan"))
        llama_layout.addRow("CPU", self._build_path_row(self.cpu_llama_path_input, "cpu"))
        llama_layout.addRow("CUDA URL", self.cuda_runtime_url_input)
        llama_layout.addRow("HIP URL", self.hip_runtime_url_input)
        llama_layout.addRow("Vulkan URL", self.vulkan_runtime_url_input)
        llama_layout.addRow("CPU URL", self.cpu_runtime_url_input)

        self.detect_paths_button = QPushButton("Auto Detect Paths")
        self.detect_paths_button.clicked.connect(self._auto_fill_llama_paths)
        llama_layout.addRow("", self.detect_paths_button)

        devices_box = QGroupBox("Detected Devices")
        devices_layout = QVBoxLayout(devices_box)
        self.detected_devices_label = QLabel(self._describe_devices(self.available_devices))
        self.detected_devices_label.setWordWrap(True)
        self.refresh_devices_button = QPushButton("Refresh Device List")
        self.refresh_devices_button.clicked.connect(self.refresh_available_devices)
        devices_layout.addWidget(self.detected_devices_label)
        devices_layout.addWidget(self.refresh_devices_button)

        layout.addWidget(llama_box)
        layout.addWidget(devices_box)
        layout.addStretch(1)
        return container

    def _build_path_row(self, target_input: QLineEdit, backend: str) -> QWidget:
        row = QHBoxLayout()
        browse = QPushButton("Browse")
        browse.clicked.connect(lambda _checked=False, b=backend: self.pick_global_llama_server(b))
        row.addWidget(target_input)
        row.addWidget(browse)
        return self._wrap_layout(row)

    def _llama_path_input_for_backend(self, backend: str) -> QLineEdit:
        key = backend.strip().lower()
        if key == "cuda":
            return self.cuda_llama_path_input
        if key == "hip":
            return self.hip_llama_path_input
        if key == "vulkan":
            return self.vulkan_llama_path_input
        return self.cpu_llama_path_input

    def _describe_devices(self, devices: list[GPUDevice]) -> str:
        if not devices:
            return "No compute devices detected."
        return "\n".join(f"- {device.label}" for device in devices)

    def pick_global_llama_server(self, backend: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, f"Select {backend.upper()} llama-server executable")
        if not path:
            return
        self._llama_path_input_for_backend(backend).setText(path)
        self._save_config()

    def _auto_fill_llama_paths(self) -> None:
        for backend in ("cuda", "hip", "vulkan", "cpu"):
            target = self._llama_path_input_for_backend(backend)
            if target.text().strip():
                continue
            detected = self._auto_detect_llama_server(backend)
            if detected:
                target.setText(detected)
        self._save_config()

    def refresh_available_devices(self) -> None:
        self.available_devices = self._detect_available_devices()
        self.detected_devices_label.setText(self._describe_devices(self.available_devices))
        for slot in self.server_slots:
            previous = [key for key, checkbox in slot.device_checkboxes.items() if checkbox.isChecked()]
            self._render_slot_device_checkboxes(slot, previous)
        self._save_config()

    def _build_hf_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_search_panel())
        splitter.addWidget(self._build_repo_details_panel())
        splitter.setSizes([760, 520])

        layout.addWidget(splitter)
        return container

    def _build_search_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        search_box = QGroupBox("Hugging Face Search")
        search_layout = QVBoxLayout(search_box)

        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search GGUF repos, e.g. qwen gguf, gemma gguf, tinyllama gguf")
        self.search_input.returnPressed.connect(self.search_models)
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_models)
        search_row.addWidget(self.search_input)
        search_row.addWidget(self.search_button)

        self.results_table = QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(["Repository", "Downloads", "Likes", "Updated"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.itemSelectionChanged.connect(self.load_selected_repo)

        self.search_status = QLabel("Enter a query to search Hugging Face.")

        search_layout.addLayout(search_row)
        search_layout.addWidget(self.results_table)
        search_layout.addWidget(self.search_status)

        files_box = QGroupBox("GGUF Files")
        files_layout = QVBoxLayout(files_box)
        self.files_list = QListWidget()
        self.files_list.itemSelectionChanged.connect(self._sync_selected_file_path)

        download_row = QHBoxLayout()
        self.download_button = QPushButton("Download Selected File")
        self.download_button.clicked.connect(self.download_selected_file)
        self.download_button.setEnabled(False)
        self.browse_models_button = QPushButton("Browse Models Folder")
        self.browse_models_button.clicked.connect(lambda _checked=False: self.pick_model_file())
        download_row.addWidget(self.download_button)
        download_row.addWidget(self.browse_models_button)

        self.download_progress = QProgressBar()
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_status = QLabel("No download started.")

        files_layout.addWidget(self.files_list)
        files_layout.addLayout(download_row)
        files_layout.addWidget(self.download_progress)
        files_layout.addWidget(self.download_status)

        layout.addWidget(search_box)
        layout.addWidget(files_box)
        return container

    def _build_right_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._build_run_panel())
        layout.addWidget(self._build_repo_details_panel())
        return container

    def _build_run_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        global_box = QGroupBox("Global Server Settings")
        global_layout = QFormLayout(global_box)

        self.hf_token_input = QLineEdit()
        self.hf_token_input.setPlaceholderText("Optional Hugging Face token for gated/private repos")
        self.hf_token_input.setEchoMode(QLineEdit.Password)
        global_layout.addRow("HF Token", self.hf_token_input)

        self.model_target_server_selector = QComboBox()
        self.model_target_server_selector.addItems([f"Server {index + 1}" for index in range(4)])
        global_layout.addRow("HF model target", self.model_target_server_selector)

        self.ollama_default_server_selector = QComboBox()
        self.ollama_default_server_selector.addItems([f"Server {index + 1}" for index in range(4)])
        global_layout.addRow("Ollama default", self.ollama_default_server_selector)

        ollama_proxy_layout = QGridLayout()
        self.ollama_host_input = QLineEdit("127.0.0.1")
        self.ollama_port_input = QSpinBox()
        self.ollama_port_input.setRange(1, 65535)
        self.ollama_port_input.setValue(11434)
        ollama_proxy_layout.addWidget(QLabel("Host"), 0, 0)
        ollama_proxy_layout.addWidget(self.ollama_host_input, 0, 1)
        ollama_proxy_layout.addWidget(QLabel("Base Port"), 0, 2)
        ollama_proxy_layout.addWidget(self.ollama_port_input, 0, 3)
        global_layout.addRow("Ollama API", self._wrap_layout(ollama_proxy_layout))
        self.ollama_host_input.textChanged.connect(self._update_proxy_port_labels)
        self.ollama_port_input.valueChanged.connect(self._update_proxy_port_labels)

        ollama_controls = QHBoxLayout()
        self.start_ollama_proxy_button = QPushButton("Start Ollama Proxy")
        self.start_ollama_proxy_button.clicked.connect(self.start_ollama_proxy)
        self.stop_ollama_proxy_button = QPushButton("Stop Ollama Proxy")
        self.stop_ollama_proxy_button.clicked.connect(self.stop_ollama_proxy)
        self.test_ollama_proxy_button = QPushButton("Test Ollama Proxy")
        self.test_ollama_proxy_button.clicked.connect(self.test_ollama_proxy)
        self.stop_ollama_proxy_button.setEnabled(False)
        ollama_controls.addWidget(self.start_ollama_proxy_button)
        ollama_controls.addWidget(self.stop_ollama_proxy_button)
        ollama_controls.addWidget(self.test_ollama_proxy_button)
        global_layout.addRow("Proxy Control", self._wrap_layout(ollama_controls))

        self.ollama_proxy_status = QLabel("Ollama API proxy is stopped.")
        global_layout.addRow("Proxy Status", self.ollama_proxy_status)
        self.ollama_proxy_test_status = QLabel("Proxy test idle.")
        global_layout.addRow("Proxy Test", self.ollama_proxy_test_status)

        generation_layout = QGridLayout()
        self.temperature_input = QSpinBox()
        self.temperature_input.setRange(0, 200)
        self.temperature_input.setValue(70)
        self.chat_max_tokens_input = QSpinBox()
        self.chat_max_tokens_input.setRange(1, 32768)
        self.chat_max_tokens_input.setValue(512)
        generation_layout.addWidget(QLabel("Temp x100"), 0, 0)
        generation_layout.addWidget(self.temperature_input, 0, 1)
        generation_layout.addWidget(QLabel("Reply Tokens"), 0, 2)
        generation_layout.addWidget(self.chat_max_tokens_input, 0, 3)
        global_layout.addRow("Chat Defaults", self._wrap_layout(generation_layout))

        self.server_slots = []
        slots_wrapper = QWidget()
        slots_layout = QVBoxLayout(slots_wrapper)
        for index in range(4):
            slots_layout.addWidget(self._build_server_slot(index))
        slots_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(slots_wrapper)

        layout.addWidget(global_box)
        layout.addWidget(scroll, stretch=1)
        return container

    def _build_server_slot(self, index: int) -> QWidget:
        box = QGroupBox(f"Server {index + 1}")
        form = QFormLayout(box)

        ollama_model_input = QLineEdit(f"server-{index + 1}")
        ollama_model_input.setPlaceholderText("Ollama model alias for this server slot")
        form.addRow("Ollama Model", ollama_model_input)

        backend_label = QLabel("Auto")
        form.addRow("Backend", backend_label)

        model_path_input = QLineEdit()
        model_path_input.setPlaceholderText("Path to downloaded .gguf file")
        browse_model = QPushButton("Browse")
        browse_model.clicked.connect(lambda _checked=False, i=index: self.pick_model_file(i))
        model_row = QHBoxLayout()
        model_row.addWidget(model_path_input)
        model_row.addWidget(browse_model)
        form.addRow("Model File", self._wrap_layout(model_row))

        device_box = QGroupBox("Assigned Devices")
        device_layout = QVBoxLayout(device_box)
        device_layout.setContentsMargins(8, 8, 8, 8)
        form.addRow("Devices", device_box)

        split_mode_input = QComboBox()
        split_mode_input.addItems(["parallel", "pooled"])
        form.addRow("Multi-GPU Mode", split_mode_input)

        network_layout = QGridLayout()
        host_input = QLineEdit("127.0.0.1")
        port_input = QSpinBox()
        port_input.setRange(1, 65535)
        port_input.setValue(8080 + index)
        network_layout.addWidget(QLabel("Host"), 0, 0)
        network_layout.addWidget(host_input, 0, 1)
        network_layout.addWidget(QLabel("Port"), 0, 2)
        network_layout.addWidget(port_input, 0, 3)
        form.addRow("Network", self._wrap_layout(network_layout))

        ctx_size_input = QSpinBox()
        ctx_size_input.setRange(256, 1048576)
        ctx_size_input.setSingleStep(256)
        ctx_size_input.setValue(4096)
        form.addRow("Context", ctx_size_input)

        extra_args_input = QLineEdit()
        extra_args_input.setPlaceholderText("Optional extra args, e.g. --n-gpu-layers 999")
        form.addRow("Extra Args", extra_args_input)

        controls_row = QHBoxLayout()
        start_button = QPushButton("Start")
        stop_button = QPushButton("Stop")
        stop_button.setEnabled(False)
        start_button.clicked.connect(lambda _checked=False, i=index: self.start_server(i))
        stop_button.clicked.connect(lambda _checked=False, i=index: self.stop_server(i))
        controls_row.addWidget(start_button)
        controls_row.addWidget(stop_button)
        form.addRow("Controls", self._wrap_layout(controls_row))

        status_label = QLabel("Server is stopped.")
        form.addRow("Status", status_label)

        proxy_status_label = QLabel("Proxy stopped.")
        form.addRow("Proxy URL", proxy_status_label)

        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(lambda i=index: self._append_server_output(i))
        process.errorOccurred.connect(lambda error, i=index: self._handle_process_error(i, error))
        process.started.connect(lambda i=index: self._set_server_state(i, True))
        process.finished.connect(lambda *_args, i=index: self._set_server_state(i, False))

        self.server_slots.append(
            ServerSlot(
                index=index,
                process=process,
                backend_label=backend_label,
                ollama_model_input=ollama_model_input,
                model_path_input=model_path_input,
                host_input=host_input,
                port_input=port_input,
                ctx_size_input=ctx_size_input,
                extra_args_input=extra_args_input,
                device_box=device_box,
                device_checkboxes={},
                split_mode_input=split_mode_input,
                start_button=start_button,
                stop_button=stop_button,
                status_label=status_label,
                proxy_status_label=proxy_status_label,
            )
        )

        self._render_slot_device_checkboxes(self.server_slots[-1], ["cpu"])

        return box
    def _build_repo_details_panel(self) -> QWidget:
        box = QGroupBox("Repository Details")
        layout = QVBoxLayout(box)

        self.repo_title_label = QLabel("Select a Hugging Face repo to inspect it.")
        self.repo_title_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.repo_meta_label = QLabel("")
        self.repo_meta_label.setWordWrap(True)
        self.repo_meta_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.repo_summary_text = QPlainTextEdit()
        self.repo_summary_text.setReadOnly(True)
        self.repo_summary_text.setPlaceholderText("Repo summary, tags, and metadata will appear here.")

        layout.addWidget(self.repo_title_label)
        layout.addWidget(self.repo_meta_label)
        layout.addWidget(self.repo_summary_text)
        return box

    def _build_log_panel(self) -> QWidget:
        box = QGroupBox("Server Log")
        layout = QVBoxLayout(box)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.clicked.connect(self.log_output.clear)
        layout.addWidget(self.clear_log_button)
        layout.addWidget(self.log_output)
        return box

    def _build_chat_panel(self) -> QWidget:
        box = QGroupBox("Chat Tester")
        layout = QVBoxLayout(box)

        self.chat_history_output = QPlainTextEdit()
        self.chat_history_output.setReadOnly(True)
        self.chat_history_output.setPlaceholderText("Start the server, then send a prompt here.")

        self.system_prompt_input = QLineEdit()
        self.system_prompt_input.setPlaceholderText("Optional system prompt")

        self.chat_server_selector = QComboBox()
        self.chat_server_selector.addItems([f"Server {index + 1}" for index in range(4)])

        self.chat_input = QTextEdit()
        self.chat_input.setPlaceholderText("Write a test prompt for the running model...")
        self.chat_input.setFixedHeight(120)

        buttons_row = QHBoxLayout()
        self.send_chat_button = QPushButton("Send Prompt")
        self.send_chat_button.clicked.connect(self.send_chat_message)
        self.clear_chat_button = QPushButton("Clear Chat")
        self.clear_chat_button.clicked.connect(self.clear_chat)
        buttons_row.addWidget(self.send_chat_button)
        buttons_row.addWidget(self.clear_chat_button)

        self.chat_status = QLabel("Chat is idle.")

        layout.addWidget(self.chat_history_output)
        layout.addWidget(QLabel("Target Server"))
        layout.addWidget(self.chat_server_selector)
        layout.addWidget(self.system_prompt_input)
        layout.addWidget(self.chat_input)
        layout.addLayout(buttons_row)
        layout.addWidget(self.chat_status)
        return box

    def _wrap_layout(self, inner_layout: QGridLayout | QHBoxLayout) -> QWidget:
        widget = QWidget()
        widget.setLayout(inner_layout)
        return widget

    @staticmethod
    def _run_probe_command(args: list[str]) -> str:
        try:
            completed = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=4,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        if completed.returncode != 0:
            return ""
        return completed.stdout or ""

    def _detect_available_devices(self) -> list[GPUDevice]:
        devices: list[GPUDevice] = [GPUDevice(key="cpu", label="CPU", backend="cpu", env_id="")]
        seen = {"cpu"}

        nvidia_output = self._run_probe_command([
            "nvidia-smi",
            "--query-gpu=index,name",
            "--format=csv,noheader",
        ])
        for line in nvidia_output.splitlines():
            row = line.strip()
            if not row:
                continue
            parts = [part.strip() for part in row.split(",", 1)]
            if not parts:
                continue
            gpu_id = parts[0]
            gpu_name = parts[1] if len(parts) > 1 else f"NVIDIA GPU {gpu_id}"
            key = f"cuda:{gpu_id}"
            if key in seen:
                continue
            devices.append(GPUDevice(key=key, label=f"CUDA GPU {gpu_id}: {gpu_name}", backend="cuda", env_id=gpu_id))
            seen.add(key)

        rocm_output = self._run_probe_command(["rocm-smi", "--showproductname"])
        for line in rocm_output.splitlines():
            match = re.search(r"GPU\[(\d+)\].*?:\s*(.+)", line)
            if not match:
                continue
            gpu_id = match.group(1)
            gpu_name = match.group(2).strip()
            key = f"hip:{gpu_id}"
            if key in seen:
                continue
            devices.append(GPUDevice(key=key, label=f"HIP GPU {gpu_id}: {gpu_name}", backend="hip", env_id=gpu_id))
            seen.add(key)

        vulkan_output = self._run_probe_command(["vulkaninfo", "--summary"])
        for line in vulkan_output.splitlines():
            match = re.search(r"GPU(\d+)\s*:\s*(.+)", line)
            if not match:
                continue
            gpu_id = match.group(1)
            gpu_name = match.group(2).strip()
            key = f"vulkan:{gpu_id}"
            if key in seen:
                continue
            devices.append(GPUDevice(key=key, label=f"Vulkan GPU {gpu_id}: {gpu_name}", backend="vulkan", env_id=gpu_id))
            seen.add(key)

        return devices

    def _render_slot_device_checkboxes(self, slot: ServerSlot, selected_keys: list[str] | None = None) -> None:
        selected = set(selected_keys or [])
        layout = slot.device_box.layout()
        if not isinstance(layout, QVBoxLayout):
            layout = QVBoxLayout(slot.device_box)
            slot.device_box.setLayout(layout)
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        slot.device_checkboxes = {}
        for device in self.available_devices:
            checkbox = QCheckBox(device.label)
            checkbox.setChecked(device.key in selected)
            checkbox.toggled.connect(self._save_config)
            layout.addWidget(checkbox)
            slot.device_checkboxes[device.key] = checkbox

        if not any(checkbox.isChecked() for checkbox in slot.device_checkboxes.values()) and "cpu" in slot.device_checkboxes:
            slot.device_checkboxes["cpu"].setChecked(True)

    def _load_config(self) -> None:
        if not CONFIG_PATH.exists():
            return
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return

        self.hf_token_input.setText(data.get("hf_token", ""))
        self.system_prompt_input.setText(data.get("system_prompt", ""))
        self.temperature_input.setValue(int(data.get("temperature_x100", 70)))
        self.chat_max_tokens_input.setValue(int(data.get("chat_max_tokens", 512)))

        model_target = int(data.get("model_target_server", 0) or 0)
        self.model_target_server_selector.setCurrentIndex(max(0, min(3, model_target)))

        ollama_default = int(data.get("ollama_default_server", 0) or 0)
        self.ollama_default_server_selector.setCurrentIndex(max(0, min(3, ollama_default)))

        self.ollama_host_input.setText(str(data.get("ollama_host", "127.0.0.1") or "127.0.0.1"))
        self.ollama_port_input.setValue(int(data.get("ollama_port", 11434) or 11434))

        llama_paths = data.get("llama_paths", {})
        if not isinstance(llama_paths, dict):
            llama_paths = {}
        self.cuda_llama_path_input.setText(
            str(llama_paths.get("cuda", data.get("llama_server_path", "")) or "")
        )
        self.hip_llama_path_input.setText(str(llama_paths.get("hip", "") or ""))
        self.vulkan_llama_path_input.setText(str(llama_paths.get("vulkan", "") or ""))
        self.cpu_llama_path_input.setText(str(llama_paths.get("cpu", "") or ""))

        runtime_binary_urls = data.get("runtime_binary_urls", {})
        if not isinstance(runtime_binary_urls, dict):
            runtime_binary_urls = {}
        self.cuda_runtime_url_input.setText(str(runtime_binary_urls.get("cuda", "") or ""))
        self.hip_runtime_url_input.setText(str(runtime_binary_urls.get("hip", "") or ""))
        self.vulkan_runtime_url_input.setText(str(runtime_binary_urls.get("vulkan", "") or ""))
        self.cpu_runtime_url_input.setText(str(runtime_binary_urls.get("cpu", "") or ""))
        if not self.cuda_runtime_url_input.text().strip():
            self.cuda_runtime_url_input.setText(LLAMA_CPP_RELEASES_URL)
        if not self.hip_runtime_url_input.text().strip():
            self.hip_runtime_url_input.setText(LLAMA_CPP_RELEASES_URL)
        if not self.vulkan_runtime_url_input.text().strip():
            self.vulkan_runtime_url_input.setText(LLAMA_CPP_RELEASES_URL)
        if not self.cpu_runtime_url_input.text().strip():
            self.cpu_runtime_url_input.setText(LLAMA_CPP_RELEASES_URL)

        chat_target = int(data.get("chat_server", 0) or 0)
        self.chat_server_selector.setCurrentIndex(max(0, min(3, chat_target)))

        servers = data.get("servers")
        if not isinstance(servers, list):
            servers = [
                {
                    "backend": "cpu",
                    "model_path": data.get("model_path", ""),
                    "host": data.get("host", "127.0.0.1"),
                    "port": int(data.get("port", 8080) or 8080),
                    "ctx_size": int(data.get("ctx_size", 4096) or 4096),
                    "extra_args": data.get("extra_args", ""),
                    "device_keys": ["cpu"],
                    "split_mode": "parallel",
                }
            ]

        for index, slot in enumerate(self.server_slots):
            server_data = servers[index] if index < len(servers) and isinstance(servers[index], dict) else {}
            slot.ollama_model_input.setText(
                str(server_data.get("ollama_model", f"server-{index + 1}") or f"server-{index + 1}")
            )
            slot.model_path_input.setText(str(server_data.get("model_path", "") or ""))
            slot.host_input.setText(str(server_data.get("host", f"127.0.0.1") or "127.0.0.1"))
            slot.port_input.setValue(int(server_data.get("port", 8080 + index) or (8080 + index)))
            slot.ctx_size_input.setValue(int(server_data.get("ctx_size", 4096) or 4096))
            slot.extra_args_input.setText(str(server_data.get("extra_args", "") or ""))

            selected_device_keys = server_data.get("device_keys")
            if isinstance(selected_device_keys, list):
                normalized_keys = [str(item) for item in selected_device_keys if isinstance(item, str)]
            else:
                legacy_backend = str(server_data.get("backend", "cpu") or "cpu").strip().lower()
                legacy_gpu_text = str(server_data.get("gpu_assignment", "") or "")
                if legacy_gpu_text:
                    ids = [segment.strip() for segment in legacy_gpu_text.split(",") if segment.strip()]
                    normalized_keys = [f"{legacy_backend}:{gpu_id}" for gpu_id in ids]
                elif legacy_backend == "cpu":
                    normalized_keys = ["cpu"]
                else:
                    normalized_keys = []

            self._render_slot_device_checkboxes(slot, normalized_keys)
            split_mode = str(server_data.get("split_mode", "parallel") or "parallel")
            split_idx = slot.split_mode_input.findText(split_mode, Qt.MatchFixedString)
            slot.split_mode_input.setCurrentIndex(split_idx if split_idx >= 0 else 0)

    def _save_config(self) -> None:
        servers = []
        for slot in self.server_slots:
            selected_keys = [key for key, checkbox in slot.device_checkboxes.items() if checkbox.isChecked()]
            servers.append(
                {
                    "ollama_model": slot.ollama_model_input.text().strip(),
                    "model_path": slot.model_path_input.text().strip(),
                    "host": slot.host_input.text().strip() or "127.0.0.1",
                    "port": slot.port_input.value(),
                    "ctx_size": slot.ctx_size_input.value(),
                    "extra_args": slot.extra_args_input.text().strip(),
                    "device_keys": selected_keys,
                    "split_mode": slot.split_mode_input.currentText(),
                }
            )

        data = {
            "hf_token": self.hf_token_input.text().strip(),
            "system_prompt": self.system_prompt_input.text().strip(),
            "temperature_x100": self.temperature_input.value(),
            "chat_max_tokens": self.chat_max_tokens_input.value(),
            "model_target_server": self.model_target_server_selector.currentIndex(),
            "ollama_default_server": self.ollama_default_server_selector.currentIndex(),
            "ollama_host": self.ollama_host_input.text().strip() or "127.0.0.1",
            "ollama_port": self.ollama_port_input.value(),
            "llama_paths": {
                "cuda": self.cuda_llama_path_input.text().strip(),
                "hip": self.hip_llama_path_input.text().strip(),
                "vulkan": self.vulkan_llama_path_input.text().strip(),
                "cpu": self.cpu_llama_path_input.text().strip(),
            },
            "runtime_binary_urls": {
                "cuda": self.cuda_runtime_url_input.text().strip(),
                "hip": self.hip_runtime_url_input.text().strip(),
                "vulkan": self.vulkan_runtime_url_input.text().strip(),
                "cpu": self.cpu_runtime_url_input.text().strip(),
            },
            "chat_server": self.chat_server_selector.currentIndex(),
            "servers": servers,
        }
        CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._refresh_ollama_snapshot()

    def pick_model_file(self, slot_index: int | None = None) -> None:
        target_index = self._resolve_server_index(slot_index)
        MODELS_DIR.mkdir(exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select GGUF model",
            str(MODELS_DIR.resolve()),
            "GGUF files (*.gguf);;All files (*)",
        )
        if path:
            self.server_slots[target_index].model_path_input.setText(path)
            self._save_config()

    def search_models(self) -> None:
        query = self.search_input.text().strip()
        token = self.hf_token_input.text().strip()

        self.search_button.setEnabled(False)
        self.search_status.setText("Searching Hugging Face...")
        self.results_table.setRowCount(0)
        self.files_list.clear()
        self.download_button.setEnabled(False)
        self._clear_repo_details("Searching...")

        worker = Worker(self.hf_client.search_models, query, token)
        worker.signals.finished.connect(self._populate_search_results)
        worker.signals.error.connect(self._show_search_error)
        self.thread_pool.start(worker)

    def _populate_search_results(self, results: list[ModelSearchResult]) -> None:
        self.search_button.setEnabled(True)
        self.search_results = results
        self.results_table.setRowCount(len(results))

        for row, result in enumerate(results):
            self.results_table.setItem(row, 0, QTableWidgetItem(result.repo_id))
            self.results_table.setItem(row, 1, QTableWidgetItem(f"{result.downloads:,}"))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{result.likes:,}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(result.last_modified))

        if results:
            self.search_status.setText(f"Found {len(results)} repositories. Select one to inspect it.")
            self.results_table.selectRow(0)
        else:
            self.search_status.setText("No matching repositories found.")
            self._clear_repo_details("No repo selected.")

    def _show_search_error(self, message: str) -> None:
        self.search_button.setEnabled(True)
        self.search_status.setText("Search failed.")
        self._clear_repo_details("Search failed.")
        QMessageBox.critical(self, "Search Failed", message)

    def load_selected_repo(self) -> None:
        selected_row = self.results_table.currentRow()
        if selected_row < 0 or selected_row >= len(self.search_results):
            return

        repo_id = self.search_results[selected_row].repo_id
        token = self.hf_token_input.text().strip()
        self.files_list.clear()
        self.files_list.addItem("Loading GGUF files...")
        self.download_button.setEnabled(False)
        self.repo_title_label.setText(repo_id)
        self.repo_meta_label.setText("Loading metadata...")
        self.repo_summary_text.setPlainText("")

        worker = Worker(self.hf_client.get_repo_details, repo_id, token)
        worker.signals.finished.connect(self._populate_repo_details)
        worker.signals.error.connect(self._show_repo_files_error)
        self.thread_pool.start(worker)

    def _populate_repo_details(self, payload: tuple[RepoDetails, list[RepoFile]]) -> None:
        details, files = payload
        self.repo_details = details
        self.repo_files = files

        self.repo_title_label.setText(details.repo_id)
        meta_parts = [
            f"Author: {details.author or 'unknown'}",
            f"Downloads: {details.downloads:,}",
            f"Likes: {details.likes:,}",
            f"Updated: {details.last_modified or 'unknown'}",
        ]
        if details.license_name:
            meta_parts.append(f"License: {details.license_name}")
        if details.pipeline_tag:
            meta_parts.append(f"Pipeline: {details.pipeline_tag}")
        if details.library_name:
            meta_parts.append(f"Library: {details.library_name}")
        if details.gated:
            meta_parts.append("Gated repo")
        if details.private:
            meta_parts.append("Private repo")
        summary_lines = []
        if details.summary:
            summary_lines.append(details.summary)
        if details.tags:
            summary_lines.append("")
            summary_lines.append("Tags: " + ", ".join(details.tags[:20]))

        self.repo_meta_label.setText(" | ".join(meta_parts))
        self.repo_summary_text.setPlainText("\n".join(summary_lines).strip())

        self.files_list.clear()
        if not files:
            self.files_list.addItem("No GGUF files were found in this repo.")
            self.download_button.setEnabled(False)
            return

        for file_info in files:
            label = file_info.name if not file_info.size_text else f"{file_info.name} ({file_info.size_text})"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, file_info.name)
            self.files_list.addItem(item)

        self.files_list.setCurrentRow(0)
        self.download_button.setEnabled(True)

    def _show_repo_files_error(self, message: str) -> None:
        self.files_list.clear()
        self.files_list.addItem("Failed to load files.")
        self._clear_repo_details("Failed to load repository details.")
        QMessageBox.critical(self, "Repository Load Failed", message)

    def _clear_repo_details(self, title: str) -> None:
        self.repo_details = None
        self.repo_title_label.setText(title)
        self.repo_meta_label.setText("")
        self.repo_summary_text.setPlainText("")

    def _selected_repo_id(self) -> str:
        row = self.results_table.currentRow()
        if row < 0 or row >= len(self.search_results):
            return ""
        return self.search_results[row].repo_id

    def _selected_filename(self) -> str:
        item = self.files_list.currentItem()
        if item is None:
            return ""
        return item.data(Qt.UserRole) or ""

    def _sync_selected_file_path(self) -> None:
        filename = self._selected_filename()
        if not filename:
            return
        target_index = self._resolve_server_index(None)
        self.server_slots[target_index].model_path_input.setText(str((MODELS_DIR / Path(filename).name).resolve()))
        self._save_config()

    def download_selected_file(self) -> None:
        repo_id = self._selected_repo_id()
        filename = self._selected_filename()
        token = self.hf_token_input.text().strip()

        if not repo_id or not filename:
            QMessageBox.warning(self, "No File Selected", "Select a GGUF file first.")
            return

        self.download_button.setEnabled(False)
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_status.setText(f"Starting download for {Path(filename).name}...")

        worker = Worker(
            self.hf_client.download_file,
            repo_id,
            filename,
            MODELS_DIR,
            token,
            use_progress=True,
        )
        worker.signals.progress.connect(self._update_download_progress)
        worker.signals.finished.connect(self._download_complete)
        worker.signals.error.connect(self._download_failed)
        self.thread_pool.start(worker)

    def _update_download_progress(self, payload: dict[str, Any]) -> None:
        downloaded = int(payload.get("downloaded", 0) or 0)
        total = payload.get("total")
        speed = float(payload.get("speed", 0.0) or 0.0)
        filename = str(payload.get("filename", "model.gguf"))

        self.download_progress.setRange(0, 100)
        if total:
            percent = min(100, int(downloaded * 100 / total))
            self.download_progress.setValue(percent)
            self.download_status.setText(
                f"Downloading {filename}: {self._format_bytes(downloaded)} / {self._format_bytes(total)}"
                f" at {self._format_bytes(speed)}/s"
            )
        else:
            self.download_progress.setValue(0)
            self.download_status.setText(
                f"Downloading {filename}: {self._format_bytes(downloaded)} at {self._format_bytes(speed)}/s"
            )

    def _download_complete(self, path: Path) -> None:
        self.download_button.setEnabled(True)
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(100)
        resolved = str(path.resolve())
        target_index = self._resolve_server_index(None)
        self.server_slots[target_index].model_path_input.setText(resolved)
        self.download_status.setText(f"Downloaded to {resolved}")
        self._save_config()

    def _download_failed(self, message: str) -> None:
        self.download_button.setEnabled(True)
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_status.setText("Download failed.")
        QMessageBox.critical(self, "Download Failed", message)

    def _resolve_server_index(self, slot_index: int | None) -> int:
        if slot_index is not None and 0 <= slot_index < len(self.server_slots):
            return slot_index
        selected = self.model_target_server_selector.currentIndex()
        if 0 <= selected < len(self.server_slots):
            return selected
        return 0

    def _chat_server_index(self) -> int:
        selected = self.chat_server_selector.currentIndex()
        if 0 <= selected < len(self.server_slots):
            return selected
        return 0

    def _server_base_url(self, index: int) -> str:
        slot = self.server_slots[index]
        host = slot.host_input.text().strip() or "127.0.0.1"
        return f"http://{host}:{slot.port_input.value()}"

    def _refresh_ollama_snapshot(self) -> None:
        slots_payload: list[dict[str, Any]] = []
        for idx, slot in enumerate(self.server_slots):
            self._update_backend_label(slot)
            selected_devices = self._selected_devices(slot)
            backend = self._infer_backend(selected_devices)
            slots_payload.append(
                {
                    "index": idx,
                    "ollama_model": slot.ollama_model_input.text().strip(),
                    "host": slot.host_input.text().strip() or "127.0.0.1",
                    "port": slot.port_input.value(),
                    "model_path": slot.model_path_input.text().strip(),
                    "backend": backend or "",
                    "running": bool(self.server_running_states[idx]) if idx < len(self.server_running_states) else False,
                }
            )

        snapshot = {
            "default_server": self.ollama_default_server_selector.currentIndex(),
            "slots": slots_payload,
        }
        with self._snapshot_lock:
            self._ollama_snapshot = snapshot

    def _get_ollama_snapshot(self) -> dict[str, Any]:
        with self._snapshot_lock:
            return deepcopy(self._ollama_snapshot)

    def _get_slot_snapshot(self, slot_index: int) -> dict[str, Any]:
        with self._snapshot_lock:
            full = deepcopy(self._ollama_snapshot)
        slots = full.get("slots", [])
        slot_data = slots[slot_index] if slot_index < len(slots) else {}
        return {
            "default_server": 0,
            "slots": [slot_data] if slot_data else [],
        }

    def _set_ollama_proxy_state(self, running: bool, status_message: str) -> None:
        self.start_ollama_proxy_button.setEnabled(not running)
        self.stop_ollama_proxy_button.setEnabled(running)
        self.ollama_proxy_status.setText(status_message)

    def start_ollama_proxy(self) -> None:
        host = self.ollama_host_input.text().strip() or "127.0.0.1"
        base_port = self.ollama_port_input.value()
        self._refresh_ollama_snapshot()
        self._save_config()

        started = 0
        errors: list[str] = []
        for i, proxy in enumerate(self.ollama_proxies):
            port = base_port + i
            try:
                proxy.start(host, port)
                started += 1
                if i < len(self.server_slots):
                    self.server_slots[i].proxy_status_label.setText(f"http://{host}:{port}")
                self.log_output.appendPlainText(f"[OLLAMA] S{i + 1} proxy at http://{host}:{port}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"S{i + 1}: {exc}")

        if started == 0:
            QMessageBox.critical(self, "Ollama Proxy Failed", "\n".join(errors) or "All proxies failed to start.")
            self._set_ollama_proxy_state(False, "Ollama API proxy failed to start.")
            return

        status = f"Running — S1={base_port}, S2={base_port + 1}, S3={base_port + 2}, S4={base_port + 3}"
        if errors:
            status += f"  ({len(errors)} failed: {'; '.join(errors)})"
        self._set_ollama_proxy_state(True, status)

    def stop_ollama_proxy(self) -> None:
        for i, proxy in enumerate(self.ollama_proxies):
            proxy.stop()
        self._set_ollama_proxy_state(False, "Ollama API proxy is stopped.")
        self.log_output.appendPlainText("[OLLAMA] all proxies stopped")
        self._update_proxy_port_labels()

    def test_ollama_proxy(self) -> None:
        host = self.ollama_host_input.text().strip() or "127.0.0.1"
        base_port = self.ollama_port_input.value()

        running_index = next(
            (i for i, proxy in enumerate(self.ollama_proxies) if proxy.is_running()), None
        )
        if running_index is None:
            self.ollama_proxy_test_status.setText("No proxy is running.")
            return

        port = base_port + running_index
        base_url = f"http://{host}:{port}"

        self.test_ollama_proxy_button.setEnabled(False)
        self.ollama_proxy_test_status.setText(f"Testing {base_url}/api/tags ...")

        worker = Worker(self._probe_ollama_proxy, base_url)
        worker.signals.finished.connect(self._ollama_proxy_test_complete)
        worker.signals.error.connect(self._ollama_proxy_test_failed)
        self.thread_pool.start(worker)

    def _probe_ollama_proxy(self, base_url: str) -> dict[str, Any]:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        if not isinstance(models, list):
            models = []
        return {
            "base_url": base_url,
            "count": len(models),
        }

    def _ollama_proxy_test_complete(self, payload: dict[str, Any]) -> None:
        self.test_ollama_proxy_button.setEnabled(True)
        base_url = str(payload.get("base_url", "")).strip()
        count = int(payload.get("count", 0) or 0)
        self.ollama_proxy_test_status.setText(f"OK: {base_url}/api/tags returned {count} model(s).")

    def _ollama_proxy_test_failed(self, message: str) -> None:
        self.test_ollama_proxy_button.setEnabled(True)
        self.ollama_proxy_test_status.setText(f"Failed: {message}")

    def _update_proxy_port_labels(self) -> None:
        if not hasattr(self, "server_slots") or not hasattr(self, "ollama_proxies"):
            return
        host = self.ollama_host_input.text().strip() or "127.0.0.1"
        base_port = self.ollama_port_input.value()
        for i, slot in enumerate(self.server_slots):
            proxy = self.ollama_proxies[i] if i < len(self.ollama_proxies) else None
            if proxy and proxy.is_running():
                slot.proxy_status_label.setText(f"http://{host}:{base_port + i}")
            else:
                slot.proxy_status_label.setText(f"Proxy stopped (port {base_port + i})")

    def _auto_detect_llama_server(self, backend: str) -> str:
        backend_key = backend.strip().lower()
        backend_candidates: dict[str, list[Path]] = {
            "cuda": [
                Path("llama.cpp/build/bin/Release/llama-server.exe"),
                Path("llama.cpp/build/bin/llama-server.exe"),
                Path("llama.cpp/bin/llama-server.exe"),
                Path("llama.cpp/build/bin/Release/llama-server"),
                Path("llama.cpp/build/bin/llama-server"),
                Path("llama.cpp/bin/llama-server"),
            ],
            "hip": [
                Path("hip-llama/llama-server.exe"),
                Path("hip-llama/bin/llama-server.exe"),
                Path("hip-llama/build/bin/Release/llama-server.exe"),
                Path("hip-llama/llama-server"),
                Path("hip-llama/bin/llama-server"),
                Path("hip-llama/build/bin/Release/llama-server"),
            ],
            "vulkan": [
                Path("vulkan-llama/llama-server.exe"),
                Path("vulkan-llama/bin/llama-server.exe"),
                Path("vulkan-llama/build/bin/Release/llama-server.exe"),
                Path("vulkan-llama/llama-server"),
                Path("vulkan-llama/bin/llama-server"),
                Path("vulkan-llama/build/bin/Release/llama-server"),
            ],
            "cpu": [
                Path("llama.cpp/build/bin/Release/llama-server.exe"),
                Path("llama.cpp/build/bin/llama-server.exe"),
                Path("llama.cpp/bin/llama-server.exe"),
                Path("llama.cpp/build/bin/Release/llama-server"),
                Path("llama.cpp/build/bin/llama-server"),
                Path("llama.cpp/bin/llama-server"),
            ],
        }

        for candidate in backend_candidates.get(backend_key, []):
            if candidate.exists():
                return str(candidate.resolve())

        backend_roots: dict[str, Path] = {
            "cuda": Path("llama.cpp"),
            "hip": Path("hip-llama"),
            "vulkan": Path("vulkan-llama"),
            "cpu": Path("cpu-llama"),
        }
        backend_root = backend_roots.get(backend_key)
        if backend_root and backend_root.exists():
            target_name = "llama-server.exe" if os.name == "nt" else "llama-server"
            matches = [path for path in backend_root.rglob(target_name) if path.is_file()]
            if matches:
                newest = max(matches, key=lambda path: path.stat().st_mtime)
                return str(newest.resolve())

        return ""

    def _selected_devices(self, slot: ServerSlot) -> list[GPUDevice]:
        selected: list[GPUDevice] = []
        by_key = {device.key: device for device in self.available_devices}
        for key, checkbox in slot.device_checkboxes.items():
            if checkbox.isChecked() and key in by_key:
                selected.append(by_key[key])
        return selected

    @staticmethod
    def _infer_backend(selected_devices: list[GPUDevice]) -> str | None:
        if not selected_devices:
            return None

        backends = {device.backend for device in selected_devices}
        if "cpu" in backends and len(backends) > 1:
            return None
        if len(backends) != 1:
            return None
        return next(iter(backends))

    def _resolve_llama_server_for_backend(self, backend: str) -> str:
        configured = self._llama_path_input_for_backend(backend).text().strip()
        if configured and Path(configured).exists():
            return configured

        detected = self._auto_detect_llama_server(backend)
        if detected:
            self._llama_path_input_for_backend(backend).setText(detected)
            return detected

        fetched = self._fetch_runtime_binary(backend)
        if fetched:
            self._llama_path_input_for_backend(backend).setText(fetched)
            self._save_config()
            return fetched
        return ""

    def _fetch_runtime_binary(self, backend: str) -> str:
        workspace = Path(__file__).resolve().parent
        script_path = workspace / "scripts" / "fetch_runtime_binaries.py"
        if not script_path.exists():
            self.log_output.appendPlainText(f"[RUNTIME] fetch script not found: {script_path}")
            return ""

        self.log_output.appendPlainText(f"[RUNTIME] missing {backend.upper()} runtime, attempting fetch...")

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--backend",
                    backend,
                    "--workspace",
                    str(workspace),
                ],
                capture_output=True,
                text=True,
                timeout=900,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self.log_output.appendPlainText(f"[RUNTIME] fetch failed: {exc}")
            return ""

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        if completed.returncode != 0:
            if stderr:
                self.log_output.appendPlainText(f"[RUNTIME] {stderr}")
            else:
                self.log_output.appendPlainText("[RUNTIME] fetch failed with unknown error.")
            return ""

        if not stdout:
            self.log_output.appendPlainText("[RUNTIME] fetch completed but no runtime path was returned.")
            return ""

        runtime_path = stdout.splitlines()[-1].strip()
        if not runtime_path or not Path(runtime_path).exists():
            self.log_output.appendPlainText(f"[RUNTIME] fetch returned invalid path: {runtime_path}")
            return ""

        self.log_output.appendPlainText(f"[RUNTIME] fetched {backend.upper()} runtime: {runtime_path}")
        return runtime_path

    def _update_backend_label(self, slot: ServerSlot) -> None:
        backend = self._infer_backend(self._selected_devices(slot))
        slot.backend_label.setText(backend.upper() if backend else "Invalid")

    def start_server(self, index: int) -> None:
        slot = self.server_slots[index]
        if slot.process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Server Running", f"Server {index + 1} is already running.")
            return

        selected_devices = self._selected_devices(slot)
        backend = self._infer_backend(selected_devices)
        if backend is None:
            QMessageBox.warning(
                self,
                "Invalid device selection",
                f"Server {index + 1}: choose either CPU only, or one/more GPUs from the same backend.",
            )
            return

        llama_path = self._resolve_llama_server_for_backend(backend)
        model_path = slot.model_path_input.text().strip()
        host = slot.host_input.text().strip() or "127.0.0.1"

        if not llama_path or not Path(llama_path).exists():
            QMessageBox.warning(
                self,
                "Missing llama-server",
                f"Server {index + 1}: set a valid {backend.upper()} llama-server path in App Settings.",
            )
            return

        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(self, "Missing model", f"Server {index + 1}: set a valid GGUF model path.")
            return

        arguments = [
            "-m",
            model_path,
            "--host",
            host,
            "--port",
            str(slot.port_input.value()),
            "-c",
            str(slot.ctx_size_input.value()),
        ]

        gpu_ids = [device.env_id for device in selected_devices if device.backend == backend and device.env_id]
        if len(gpu_ids) > 1 and slot.split_mode_input.currentText() == "pooled":
            arguments.extend(["--main-gpu", "0", "--tensor-split", ",".join(["1"] * len(gpu_ids))])

        extra_args = slot.extra_args_input.text().strip()
        if extra_args:
            arguments.extend(shlex.split(extra_args, posix=False))

        environment = QProcessEnvironment.systemEnvironment()
        visible_ids = ",".join(gpu_ids)
        backend_key = backend.lower()
        if visible_ids:
            if backend_key == "cuda":
                environment.insert("CUDA_VISIBLE_DEVICES", visible_ids)
            elif backend_key == "hip":
                environment.insert("HIP_VISIBLE_DEVICES", visible_ids)
                environment.insert("ROCR_VISIBLE_DEVICES", visible_ids)
            elif backend_key == "vulkan":
                environment.insert("GGML_VK_VISIBLE_DEVICES", visible_ids)
                environment.insert("VK_VISIBLE_DEVICES", visible_ids)

        slot.process.setProcessEnvironment(environment)
        self._save_config()
        self.log_output.appendPlainText(f"[S{index + 1}] > {llama_path} {' '.join(arguments)}")
        slot.process.start(llama_path, arguments)

    def stop_server(self, index: int) -> None:
        slot = self.server_slots[index]
        if slot.process.state() == QProcess.NotRunning:
            return
        slot.process.kill()
        slot.process.waitForFinished(3000)
    def send_chat_message(self) -> None:
        prompt = self.chat_input.toPlainText().strip()
        if not prompt:
            QMessageBox.information(self, "Empty Prompt", "Enter a prompt first.")
            return

        chat_index = self._chat_server_index()
        base_url = self._server_base_url(chat_index)
        messages: list[dict[str, str]] = []
        system_prompt = self.system_prompt_input.text().strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.chat_history)
        messages.append({"role": "user", "content": prompt})

        self.pending_user_message = prompt
        self.send_chat_button.setEnabled(False)
        self.chat_status.setText(f"Sending request to Server {chat_index + 1} at {base_url}/v1/chat/completions ...")
        self._append_chat_message("user", prompt)
        self.chat_input.clear()

        worker = Worker(
            self.server_client.chat,
            base_url,
            messages,
            self.chat_max_tokens_input.value(),
            self.temperature_input.value() / 100.0,
        )
        worker.signals.finished.connect(self._chat_complete)
        worker.signals.error.connect(self._chat_failed)
        self.thread_pool.start(worker)

    def _chat_complete(self, reply: str) -> None:
        self.send_chat_button.setEnabled(True)
        self.chat_status.setText("Chat reply received.")
        if self.pending_user_message:
            self.chat_history.append({"role": "user", "content": self.pending_user_message})
        self.chat_history.append({"role": "assistant", "content": reply})
        self.pending_user_message = ""
        self._append_chat_message("assistant", reply)
        self._save_config()

    def _chat_failed(self, message: str) -> None:
        self.send_chat_button.setEnabled(True)
        self.chat_status.setText("Chat request failed.")
        self.pending_user_message = ""
        self.chat_history_output.appendPlainText(f"[error]\n{message}\n")
        QMessageBox.critical(self, "Chat Failed", message)

    def clear_chat(self) -> None:
        self.chat_history.clear()
        self.pending_user_message = ""
        self.chat_history_output.clear()
        self.chat_status.setText("Chat cleared.")

    def _append_chat_message(self, role: str, content: str) -> None:
        self.chat_history_output.appendPlainText(f"[{role}]\n{content}\n")

    def _append_server_output(self, index: int) -> None:
        slot = self.server_slots[index]
        text = bytes(slot.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not text:
            return

        lines = text.rstrip().splitlines()
        if not lines:
            return
        for line in lines:
            self.log_output.appendPlainText(f"[S{index + 1}] {line}")

    def _handle_process_error(self, index: int, _error: QProcess.ProcessError) -> None:
        self.server_slots[index].status_label.setText("Server failed to start or crashed.")

    def _set_server_state(self, index: int, running: bool) -> None:
        slot = self.server_slots[index]
        self.server_running_states[index] = running
        slot.start_button.setEnabled(not running)
        slot.stop_button.setEnabled(running)
        slot.status_label.setText(
            f"Running at {self._server_base_url(index)}" if running else "Server is stopped."
        )
        self._refresh_ollama_snapshot()

    @staticmethod
    def _format_bytes(size_value: float | int) -> str:
        size = float(size_value)
        units = ["B", "KB", "MB", "GB", "TB"]
        for unit in units:
            if size < 1024 or unit == units[-1]:
                if unit == "B":
                    return f"{int(size)} {unit}"
                return f"{size:.1f} {unit}"
            size /= 1024
        return "0 B"

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.stop_ollama_proxy()
        self._save_config()
        for slot in self.server_slots:
            if slot.process.state() != QProcess.NotRunning:
                slot.process.kill()
                slot.process.waitForFinished(2000)
        super().closeEvent(event)


def main() -> int:
    os.makedirs(MODELS_DIR, exist_ok=True)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
