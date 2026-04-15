from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


@dataclass
class ChatConfig:
    base_url: str
    model: str
    system: str
    timeout: float


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)


class Worker(QRunnable):
    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))
            return
        self.signals.finished.emit(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick PySide6 chat tester for Ollama-compatible /api/chat endpoints.")
    parser.add_argument("--base-url", default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "server-1"))
    parser.add_argument("--system", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def _extract_content(payload: dict[str, Any]) -> str:
    message = payload.get("message", {})
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def send_chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: float,
) -> str:
    url = f"{base_url.rstrip('/')}/api/chat"
    response = requests.post(
        url,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
        },
        timeout=timeout,
    )

    if not response.ok:
        details = response.text.strip()
        try:
            payload = response.json()
            if isinstance(payload, dict):
                error_text = payload.get("error")
                if isinstance(error_text, str) and error_text.strip():
                    details = error_text.strip()
        except Exception:
            pass

        summary = f"HTTP {response.status_code} from {url}"
        if details:
            summary = f"{summary}: {details}"
        raise RuntimeError(summary)

    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected response payload (expected JSON object).")

    text = _extract_content(data)
    if text:
        return text

    raw = data.get("response", "")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()

    return json.dumps(data, ensure_ascii=False)


class QuickChatWindow(QMainWindow):
    def __init__(self, config: ChatConfig) -> None:
        super().__init__()
        self.setWindowTitle("Quick Ollama Proxy Chat")
        self.resize(900, 680)

        self.thread_pool = QThreadPool.globalInstance()
        self.messages: list[dict[str, str]] = []
        self.pending_user_message = ""

        root = QWidget()
        root.setObjectName("appRoot")
        layout = QVBoxLayout(root)

        settings_box = QGroupBox("Connection")
        settings_form = QFormLayout(settings_box)

        self.base_url_input = QLineEdit(config.base_url)
        self.model_input = QLineEdit(config.model)
        self.system_input = QLineEdit(config.system)
        self.timeout_input = QLineEdit(str(config.timeout))

        settings_form.addRow("Base URL", self.base_url_input)
        settings_form.addRow("Model", self.model_input)
        settings_form.addRow("System Prompt", self.system_input)
        settings_form.addRow("Timeout (s)", self.timeout_input)

        control_row = QHBoxLayout()
        self.ping_button = QPushButton("Ping /api/tags")
        self.ping_button.clicked.connect(self.ping_tags)
        self.reset_button = QPushButton("Reset Chat")
        self.reset_button.clicked.connect(self.reset_chat)
        control_row.addWidget(self.ping_button)
        control_row.addWidget(self.reset_button)
        settings_form.addRow("Actions", self._wrap_layout(control_row))

        self.status_label = QLabel("Ready.")
        settings_form.addRow("Status", self.status_label)

        chat_box = QGroupBox("Chat")
        chat_layout = QVBoxLayout(chat_box)

        self.chat_log = QPlainTextEdit()
        self.chat_log.setReadOnly(True)
        self.chat_log.setPlaceholderText("Messages will appear here.")

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Type message and press Enter...")
        self.prompt_input.returnPressed.connect(self.send_message)

        send_row = QHBoxLayout()
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        send_row.addWidget(self.prompt_input)
        send_row.addWidget(self.send_button)

        chat_layout.addWidget(self.chat_log)
        chat_layout.addLayout(send_row)

        layout.addWidget(settings_box)
        layout.addWidget(chat_box, stretch=1)
        self.setCentralWidget(root)

        self._apply_theme()
        self._init_system_message()

    @staticmethod
    def _wrap_layout(layout: QHBoxLayout) -> QWidget:
        container = QWidget()
        container.setLayout(layout)
        return container

    def _apply_theme(self) -> None:
        bg_path = (Path(__file__).resolve().parent.parent / "assets" / "bg.jpg")
        bg_uri = bg_path.as_uri().replace('"', '\\"') if bg_path.exists() else ""
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
            QPlainTextEdit {{
                background: rgba(255, 250, 253, 226);
                color: #4f2f40;
                border: 1px solid rgba(255, 216, 236, 95);
                border-radius: 6px;
                padding: 6px;
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

            QLabel {{
                background: transparent;
            }}
            """
        )

    def _append_chat(self, speaker: str, text: str) -> None:
        self.chat_log.appendPlainText(f"{speaker}: {text}")

    def _init_system_message(self) -> None:
        self.messages = []
        system_text = self.system_input.text().strip()
        if system_text:
            self.messages.append({"role": "system", "content": system_text})

    def _set_busy(self, busy: bool, status: str) -> None:
        self.send_button.setEnabled(not busy)
        self.ping_button.setEnabled(not busy)
        self.prompt_input.setEnabled(not busy)
        self.status_label.setText(status)

    def reset_chat(self) -> None:
        self.chat_log.clear()
        self._init_system_message()
        self.status_label.setText("Chat reset.")

    def _current_timeout(self) -> float:
        raw = self.timeout_input.text().strip() or "120"
        try:
            timeout = float(raw)
        except ValueError:
            raise RuntimeError("Timeout must be a number.")
        if timeout <= 0:
            raise RuntimeError("Timeout must be greater than 0.")
        return timeout

    def ping_tags(self) -> None:
        base_url = self.base_url_input.text().strip()
        if not base_url:
            QMessageBox.warning(self, "Missing URL", "Set a Base URL first.")
            return
        try:
            timeout = self._current_timeout()
        except RuntimeError as exc:
            QMessageBox.warning(self, "Invalid Timeout", str(exc))
            return

        self._set_busy(True, f"Pinging {base_url.rstrip('/')}/api/tags ...")
        worker = Worker(self._probe_tags, base_url, timeout)
        worker.signals.finished.connect(self._ping_complete)
        worker.signals.error.connect(self._ping_failed)
        self.thread_pool.start(worker)

    @staticmethod
    def _probe_tags(base_url: str, timeout: float) -> dict[str, Any]:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", []) if isinstance(payload, dict) else []
        if not isinstance(models, list):
            models = []
        return {"count": len(models)}

    def _ping_complete(self, payload: dict[str, Any]) -> None:
        count = int(payload.get("count", 0) or 0)
        self._set_busy(False, f"Connected. /api/tags returned {count} model(s).")

    def _ping_failed(self, message: str) -> None:
        self._set_busy(False, f"Ping failed: {message}")

    def send_message(self) -> None:
        text = self.prompt_input.text().strip()
        if not text:
            return

        base_url = self.base_url_input.text().strip()
        model = self.model_input.text().strip()
        if not base_url:
            QMessageBox.warning(self, "Missing URL", "Set a Base URL first.")
            return
        if not model:
            QMessageBox.warning(self, "Missing Model", "Set a model alias first.")
            return
        try:
            timeout = self._current_timeout()
        except RuntimeError as exc:
            QMessageBox.warning(self, "Invalid Timeout", str(exc))
            return

        self.pending_user_message = text
        self.messages.append({"role": "user", "content": text})
        self._append_chat("you", text)
        self.prompt_input.clear()
        self._set_busy(True, "Waiting for response...")

        worker = Worker(
            send_chat,
            base_url=base_url,
            model=model,
            messages=list(self.messages),
            timeout=timeout,
        )
        worker.signals.finished.connect(self._chat_complete)
        worker.signals.error.connect(self._chat_failed)
        self.thread_pool.start(worker)

    def _chat_complete(self, reply: str) -> None:
        self._append_chat("bot", str(reply))
        self.messages.append({"role": "assistant", "content": str(reply)})
        self._set_busy(False, "Ready.")
        self.prompt_input.setFocus(Qt.TabFocusReason)

    def _chat_failed(self, message: str) -> None:
        if self.messages and self.messages[-1].get("role") == "user":
            self.messages.pop()
        self._append_chat("error", message)
        self._set_busy(False, "Request failed.")
        self.prompt_input.setFocus(Qt.TabFocusReason)


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    window = QuickChatWindow(
        ChatConfig(
            base_url=args.base_url,
            model=args.model,
            system=args.system,
            timeout=args.timeout,
        )
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
