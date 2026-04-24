from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlretrieve

import requests
from huggingface_hub import HfApi, hf_hub_url, snapshot_download
from PySide6.QtCore import QProcess, QProcessEnvironment, QRunnable, QThreadPool, Qt, QObject, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
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
    QMenu,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QSystemTrayIcon,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    import psutil
except Exception:  # noqa: BLE001
    psutil = None

try:
    import pynvml
except Exception:  # noqa: BLE001
    pynvml = None

try:
    import clr  # type: ignore
except Exception:  # noqa: BLE001
    clr = None


APP_NAME = "DoLLAMACPP Frontend"
CONFIG_PATH = Path("frontend_config.json")
MODELS_DIR = Path("models")
LLAMA_CPP_RELEASES_URL = "https://github.com/ggml-org/llama.cpp/releases"
OLLAMA_BLOBS_DIR = Path(os.environ.get("OLLAMA_MODELS", "")) / "blobs" if os.environ.get("OLLAMA_MODELS") else Path.home() / ".ollama" / "models" / "blobs"
PROJECT_ROOT = Path(__file__).resolve().parent
SAKURA_LIB_DIR = PROJECT_ROOT / "lib"
SAKURA_LHM_VERSION = "0.9.6"
SAKURA_LHM_ARCHIVE_NAME = "lhm_netfx.zip"
SAKURA_LHM_ARCHIVE_URLS = [
    (
        "https://github.com/LibreHardwareMonitor/LibreHardwareMonitor/releases/download/"
        f"v{SAKURA_LHM_VERSION}/LibreHardwareMonitor-net472.zip"
    ),
    (
        "https://sourceforge.net/projects/librehardwaremonitor.mirror/files/"
        f"v{SAKURA_LHM_VERSION}/LibreHardwareMonitor-net472.zip/download"
    ),
]


def normalize_connect_host(host: str) -> str:
    value = host.strip()
    if not value:
        return "127.0.0.1"
    lowered = value.lower()
    if lowered in {"0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return value


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
class SakuraGPUStats:
    name: str
    vram_total_mib: int
    vram_used_mib: int
    vram_percent: float
    core_clock_mhz: int | None
    util_percent: float | None
    power_watts: float | None
    temperature_c: float | None = None
    shared_total_mib: int = 0
    shared_used_mib: int = 0
    shared_percent: float = 0.0


@dataclass
class SakuraSystemStats:
    cpu_percent: float
    cpu_power_watts: float | None
    cpu_temp_c: float | None
    ram_total_gib: float
    ram_used_gib: float
    ram_percent: float
    ram_power_watts: float | None
    ram_temp_c: float | None


def ensure_sakura_lhm_lib_available() -> tuple[Path | None, str | None]:
    target_dll = SAKURA_LIB_DIR / "LibreHardwareMonitorLib.dll"
    if target_dll.exists():
        return target_dll.resolve(), None

    archive_path = PROJECT_ROOT / SAKURA_LHM_ARCHIVE_NAME

    try:
        SAKURA_LIB_DIR.mkdir(parents=True, exist_ok=True)

        downloaded = False
        if not archive_path.exists():
            last_error: Exception | None = None
            for archive_url in SAKURA_LHM_ARCHIVE_URLS:
                try:
                    urlretrieve(archive_url, archive_path)
                    downloaded = True
                    break
                except (URLError, OSError) as exc:
                    last_error = exc
            if not downloaded:
                raise last_error or OSError("download failed")

        extracted_any = False
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                member_path = Path(member.filename)
                if member.is_dir():
                    continue
                if len(member_path.parts) != 1:
                    continue
                if member_path.suffix.lower() != ".dll":
                    continue

                destination = SAKURA_LIB_DIR / member_path.name
                with archive.open(member) as src, destination.open("wb") as dst:
                    dst.write(src.read())
                extracted_any = True

        if target_dll.exists():
            source_label = "download" if downloaded else "cache"
            return target_dll.resolve(), f"LibreHardwareMonitor: restored {SAKURA_LHM_VERSION} from {source_label}"

        if extracted_any:
            return None, "LibreHardwareMonitor: archive extracted but LibreHardwareMonitorLib.dll was not found"
        return None, "LibreHardwareMonitor: archive did not contain any root-level DLLs"
    except (URLError, OSError, zipfile.BadZipFile) as exc:
        return None, f"LibreHardwareMonitor: auto-fetch failed ({exc})"


def _find_sakura_lhm_dll() -> Path | None:
    candidates: list[Path] = []

    env_path = os.environ.get("LHM_DLL_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            PROJECT_ROOT / "LibreHardwareMonitorLib.dll",
            SAKURA_LIB_DIR / "LibreHardwareMonitorLib.dll",
            Path(os.environ.get("ProgramFiles", "")) / "LibreHardwareMonitor" / "LibreHardwareMonitorLib.dll",
            Path(os.environ.get("ProgramFiles(x86)", "")) / "LibreHardwareMonitor" / "LibreHardwareMonitorLib.dll",
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Programs"
            / "LibreHardwareMonitor"
            / "LibreHardwareMonitorLib.dll",
        ]
    )

    for candidate in candidates:
        if str(candidate).strip() and candidate.exists():
            return candidate.resolve()
    return None


class SakuraNvidiaMonitor:
    def __init__(self) -> None:
        self.ready = False
        self.handles: list[Any] = []
        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            for index in range(count):
                self.handles.append(pynvml.nvmlDeviceGetHandleByIndex(index))
            self.ready = bool(self.handles)
        except Exception:  # noqa: BLE001
            self.ready = False

    def collect(self) -> list[SakuraGPUStats]:
        if not self.ready:
            return []

        stats: list[SakuraGPUStats] = []
        for handle in self.handles:
            try:
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "ignore")

                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_total_mib = int(mem.total / (1024 * 1024))
                vram_used_mib = int(mem.used / (1024 * 1024))
                vram_percent = (vram_used_mib / vram_total_mib * 100.0) if vram_total_mib else 0.0

                util_percent: float | None = None
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    util_percent = float(util.gpu)
                except Exception:  # noqa: BLE001
                    util_percent = None

                core_clock_mhz: int | None = None
                try:
                    core_clock_mhz = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS))
                except Exception:  # noqa: BLE001
                    core_clock_mhz = None

                power_watts: float | None = None
                try:
                    power_watts = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
                except Exception:  # noqa: BLE001
                    power_watts = None

                temperature_c: float | None = None
                try:
                    temperature_c = float(
                        pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    )
                except Exception:  # noqa: BLE001
                    temperature_c = None

                stats.append(
                    SakuraGPUStats(
                        name=str(name),
                        vram_total_mib=vram_total_mib,
                        vram_used_mib=vram_used_mib,
                        vram_percent=vram_percent,
                        core_clock_mhz=core_clock_mhz,
                        util_percent=util_percent,
                        power_watts=power_watts,
                        temperature_c=temperature_c,
                    )
                )
            except Exception:  # noqa: BLE001
                continue

        return stats

    def shutdown(self) -> None:
        if not self.ready:
            return
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


class SakuraLibreHardwareMonitorBridge:
    def __init__(self) -> None:
        self.available = False
        self.status = "LibreHardwareMonitor: unavailable"
        self._computer: Any = None
        self._sensor_type_power = None
        self._sensor_type_load = None
        self._sensor_type_clock = None
        self._sensor_type_data = None
        self._sensor_type_small_data = None
        self._sensor_type_temperature = None

        if os.name != "nt":
            self.status = "LibreHardwareMonitor: Windows only"
            return
        if clr is None:
            if sys.version_info >= (3, 14):
                self.status = "LibreHardwareMonitor: pythonnet unavailable on Python 3.14+"
            else:
                self.status = "LibreHardwareMonitor: pythonnet not installed"
            return

        dll_path = _find_sakura_lhm_dll()
        auto_dll: Path | None = None
        auto_status: str | None = None
        if dll_path is None:
            auto_dll, auto_status = ensure_sakura_lhm_lib_available()
            dll_path = auto_dll or _find_sakura_lhm_dll()
        if dll_path is None:
            self.status = auto_status or "LibreHardwareMonitor: DLL not found"
            return

        try:
            dll_dir = dll_path.parent
            for dependency in sorted(dll_dir.glob("*.dll")):
                if dependency.name.lower() == "librehardwaremonitorlib.dll":
                    continue
                try:
                    clr.AddReference(str(dependency))
                except Exception:  # noqa: BLE001
                    pass

            clr.AddReference(str(dll_path))
            from LibreHardwareMonitor.Hardware import Computer, SensorType  # type: ignore

            self._sensor_type_power = SensorType.Power
            self._sensor_type_load = SensorType.Load
            self._sensor_type_clock = SensorType.Clock
            self._sensor_type_data = SensorType.Data
            self._sensor_type_small_data = SensorType.SmallData
            self._sensor_type_temperature = SensorType.Temperature

            computer = Computer()
            computer.IsCpuEnabled = True
            computer.IsGpuEnabled = True
            computer.IsMemoryEnabled = True
            computer.IsMotherboardEnabled = True
            computer.IsControllerEnabled = True
            computer.Open()

            self._computer = computer
            self.available = True
            if auto_status and auto_dll and dll_path.resolve() == auto_dll.resolve():
                self.status = f"{auto_status}; connected ({dll_path.name})"
            else:
                self.status = f"LibreHardwareMonitor: connected ({dll_path.name})"
        except Exception as exc:  # noqa: BLE001
            self.status = f"LibreHardwareMonitor: load failed ({exc})"

    def _iter_hardware(self):
        if self._computer is None:
            return

        queue = [item for item in self._computer.Hardware]
        while queue:
            hw = queue.pop(0)
            yield hw
            for sub in hw.SubHardware:
                queue.append(sub)

    def collect_cpu_ram_telemetry(self) -> tuple[float | None, float | None, float | None, float | None]:
        if not self.available:
            return None, None, None, None

        cpu_candidates: list[float] = []
        ram_candidates: list[float] = []
        cpu_temp_candidates: list[float] = []
        ram_temp_candidates: list[float] = []

        try:
            for hw in self._iter_hardware() or []:
                hw.Update()
                hw_type = str(hw.HardwareType).lower()

                for sensor in hw.Sensors:
                    if sensor.Value is None:
                        continue

                    sensor_name = str(sensor.Name).lower()
                    value = float(sensor.Value)

                    if sensor.SensorType == self._sensor_type_power:
                        if "cpu" in hw_type:
                            if any(keyword in sensor_name for keyword in ("package", "total", "cpu", "ppt")):
                                cpu_candidates.append(value)
                        elif "memory" in hw_type or "ram" in hw_type:
                            ram_candidates.append(value)

                        if any(keyword in sensor_name for keyword in ("dram", "dimm", "ram", "memory")):
                            ram_candidates.append(value)

                    elif sensor.SensorType == self._sensor_type_temperature:
                        if "cpu" in hw_type:
                            if any(keyword in sensor_name for keyword in ("package", "cpu", "die", "tdie", "tctl")):
                                cpu_temp_candidates.append(value)
                        elif "memory" in hw_type or "ram" in hw_type:
                            ram_temp_candidates.append(value)

                        if any(keyword in sensor_name for keyword in ("dram", "dimm", "ram", "memory")):
                            ram_temp_candidates.append(value)
        except Exception:  # noqa: BLE001
            return None, None, None, None

        cpu_power = max(cpu_candidates) if cpu_candidates else None
        ram_power = max(ram_candidates) if ram_candidates else None
        cpu_temp = max(cpu_temp_candidates) if cpu_temp_candidates else None
        ram_temp = max(ram_temp_candidates) if ram_temp_candidates else None
        return cpu_power, ram_power, cpu_temp, ram_temp

    @staticmethod
    def _data_value_to_mib(value: float, sensor_type: Any) -> int:
        if str(sensor_type) == "SmallData":
            return int(round(value))
        if value <= 512:
            return int(round(value * 1024))
        return int(round(value))

    def collect_gpu_stats(self) -> list[SakuraGPUStats]:
        if not self.available:
            return []

        stats: list[SakuraGPUStats] = []
        try:
            for hw in self._iter_hardware() or []:
                hw.Update()
                hw_type = str(hw.HardwareType).lower()
                if "gpu" not in hw_type:
                    continue

                util_percent: float | None = None
                core_clock_mhz: int | None = None
                power_watts: float | None = None
                temperature_c: float | None = None
                mem_used_mib: int | None = None
                mem_total_mib: int | None = None
                mem_free_mib: int | None = None
                dedicated_used_mib: int | None = None
                dedicated_total_mib: int | None = None
                dedicated_free_mib: int | None = None
                shared_used_mib: int | None = None
                shared_total_mib: int | None = None
                shared_free_mib: int | None = None

                for sensor in hw.Sensors:
                    if sensor.Value is None:
                        continue

                    sensor_name = str(sensor.Name).lower()
                    value = float(sensor.Value)
                    sensor_type = sensor.SensorType

                    if sensor_type == self._sensor_type_load:
                        if "core" in sensor_name or "gpu" in sensor_name:
                            util_percent = value if util_percent is None else max(util_percent, value)

                    elif sensor_type == self._sensor_type_clock:
                        if "core" in sensor_name or "graphics" in sensor_name:
                            mhz = int(round(value))
                            core_clock_mhz = mhz if core_clock_mhz is None else max(core_clock_mhz, mhz)

                    elif sensor_type == self._sensor_type_power:
                        if "total" in sensor_name or "package" in sensor_name or "gpu" in sensor_name:
                            power_watts = value if power_watts is None else max(power_watts, value)

                    elif sensor_type == self._sensor_type_temperature:
                        if "core" in sensor_name or "gpu" in sensor_name or "hotspot" in sensor_name:
                            temperature_c = value if temperature_c is None else max(temperature_c, value)

                    elif sensor_type in (self._sensor_type_data, self._sensor_type_small_data):
                        if "d3d dedicated memory used" in sensor_name:
                            dedicated_used_mib = self._data_value_to_mib(value, sensor_type)
                        elif "d3d dedicated memory total" in sensor_name:
                            dedicated_total_mib = self._data_value_to_mib(value, sensor_type)
                        elif "d3d dedicated memory free" in sensor_name:
                            dedicated_free_mib = self._data_value_to_mib(value, sensor_type)
                        elif "d3d shared memory used" in sensor_name:
                            shared_used_mib = self._data_value_to_mib(value, sensor_type)
                        elif "d3d shared memory total" in sensor_name:
                            shared_total_mib = self._data_value_to_mib(value, sensor_type)
                        elif "d3d shared memory free" in sensor_name:
                            shared_free_mib = self._data_value_to_mib(value, sensor_type)
                        elif "memory used" in sensor_name:
                            mem_used_mib = self._data_value_to_mib(value, sensor_type)
                        elif "memory total" in sensor_name:
                            mem_total_mib = self._data_value_to_mib(value, sensor_type)
                        elif "memory free" in sensor_name:
                            mem_free_mib = self._data_value_to_mib(value, sensor_type)

                if dedicated_total_mib is None and dedicated_used_mib is not None and dedicated_free_mib is not None:
                    dedicated_total_mib = dedicated_used_mib + dedicated_free_mib
                if shared_total_mib is None and shared_used_mib is not None and shared_free_mib is not None:
                    shared_total_mib = shared_used_mib + shared_free_mib
                if mem_total_mib is None and mem_used_mib is not None and mem_free_mib is not None:
                    mem_total_mib = mem_used_mib + mem_free_mib

                vram_total_mib = dedicated_total_mib if dedicated_total_mib is not None else (mem_total_mib or 0)
                vram_used_mib = dedicated_used_mib if dedicated_used_mib is not None else (mem_used_mib or 0)
                vram_percent = (vram_used_mib / vram_total_mib * 100.0) if vram_total_mib else 0.0
                shared_total = shared_total_mib or 0
                shared_used = shared_used_mib or 0
                shared_percent = (shared_used / shared_total * 100.0) if shared_total else 0.0

                stats.append(
                    SakuraGPUStats(
                        name=str(hw.Name),
                        vram_total_mib=vram_total_mib,
                        vram_used_mib=vram_used_mib,
                        vram_percent=vram_percent,
                        shared_total_mib=shared_total,
                        shared_used_mib=shared_used,
                        shared_percent=shared_percent,
                        core_clock_mhz=core_clock_mhz,
                        util_percent=util_percent,
                        power_watts=power_watts,
                        temperature_c=temperature_c,
                    )
                )
        except Exception:  # noqa: BLE001
            return []

        return stats

    def shutdown(self) -> None:
        if self._computer is None:
            return
        try:
            self._computer.Close()
        except Exception:  # noqa: BLE001
            pass


class SakuraMetricRow(QWidget):
    def __init__(self, title: str, with_bar: bool = True) -> None:
        super().__init__()
        self.with_bar = with_bar
        self.title = QLabel(title)
        self.title.setObjectName("SakuraMetricTitle")
        self.value = QLabel("-")
        self.value.setObjectName("SakuraMetricValue")
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.bar = QProgressBar()
        self.bar.setObjectName("SakuraBar")
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(14)
        if not with_bar:
            self.bar.hide()

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.title)
        top.addWidget(self.value)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(top)
        layout.addWidget(self.bar)
        if with_bar:
            self._apply_bar_color(0.0)

    @staticmethod
    def temperature_to_percent(temp_c: float) -> float:
        return max(0.0, min(100.0, temp_c))

    @staticmethod
    def _blend_channel(start: int, end: int, ratio: float) -> int:
        return int(round(start + (end - start) * ratio))

    @classmethod
    def _interpolate_color(cls, percent: float) -> QColor:
        p = max(0.0, min(100.0, percent))
        if p <= 30.0:
            return QColor("#A8E4FF")
        if p <= 45.0:
            ratio = (p - 30.0) / 15.0
            start = QColor("#A8E4FF")
            end = QColor("#FF8FC9")
        elif p <= 80.0:
            ratio = (p - 45.0) / 35.0
            start = QColor("#FF8FC9")
            end = QColor("#FF4D4D")
        else:
            ratio = (p - 80.0) / 20.0
            start = QColor("#FF4D4D")
            end = QColor("#D7263D")

        return QColor(
            cls._blend_channel(start.red(), end.red(), ratio),
            cls._blend_channel(start.green(), end.green(), ratio),
            cls._blend_channel(start.blue(), end.blue(), ratio),
        )

    def _apply_bar_color(self, percent: float) -> None:
        base = self._interpolate_color(percent)
        glow = base.lighter(125)
        self.bar.setStyleSheet(
            f"""
            QProgressBar#SakuraBar {{
                border: 1px solid #DDB2C1;
                border-radius: 7px;
                background-color: #FFEAF1;
            }}
            QProgressBar#SakuraBar::chunk {{
                border-radius: 7px;
                background-color: qlineargradient(
                    spread: pad,
                    x1: 0,
                    y1: 0.5,
                    x2: 1,
                    y2: 0.5,
                    stop: 0 {glow.name()},
                    stop: 1 {base.name()}
                );
            }}
            """
        )

    def set_percent(self, percent: float, text: str) -> None:
        bounded = max(0, min(100, int(round(percent))))
        self.bar.setValue(bounded)
        self._apply_bar_color(float(bounded))
        self.value.setText(text)

    def set_text(self, text: str) -> None:
        self.value.setText(text)


class SakuraGPUCard(QGroupBox):
    def __init__(self, gpu_index: int, gpu_name: str) -> None:
        super().__init__(f"GPU {gpu_index}: {gpu_name}")
        self.setObjectName("SakuraDeviceCard")
        self.vram_row = SakuraMetricRow("GPU Memory")
        self.shared_row = SakuraMetricRow("Shared Memory")
        self.util_row = SakuraMetricRow("GPU Utilization")
        self.temp_row = SakuraMetricRow("Temperature")
        self.clock_row = SakuraMetricRow("Core Clock", with_bar=False)
        self.power_row = SakuraMetricRow("Power Draw", with_bar=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(12)
        layout.addWidget(self.vram_row)
        layout.addWidget(self.shared_row)
        layout.addWidget(self.util_row)
        layout.addWidget(self.temp_row)

        details_row = QHBoxLayout()
        details_row.setContentsMargins(0, 0, 0, 0)
        details_row.setSpacing(12)
        details_row.addWidget(self.clock_row)
        details_row.addWidget(self.power_row)
        layout.addLayout(details_row)

    def apply_stats(self, stats: SakuraGPUStats) -> None:
        if stats.vram_total_mib <= 0:
            self.vram_row.set_percent(0, "N/A")
        else:
            self.vram_row.set_percent(
                stats.vram_percent,
                f"{stats.vram_used_mib:,} / {stats.vram_total_mib:,} MiB ({stats.vram_percent:.1f}%)",
            )

        if stats.shared_total_mib <= 0:
            self.shared_row.set_percent(0, "N/A")
        else:
            self.shared_row.set_percent(
                stats.shared_percent,
                f"{stats.shared_used_mib:,} / {stats.shared_total_mib:,} MiB ({stats.shared_percent:.1f}%)",
            )

        if stats.util_percent is None:
            self.util_row.set_percent(0, "N/A")
        else:
            self.util_row.set_percent(stats.util_percent, f"{stats.util_percent:.1f}%")

        if stats.temperature_c is None:
            self.temp_row.set_percent(0, "N/A")
        else:
            self.temp_row.set_percent(
                SakuraMetricRow.temperature_to_percent(stats.temperature_c),
                f"{stats.temperature_c:.1f} C",
            )

        if stats.core_clock_mhz is None:
            self.clock_row.set_text("N/A")
        else:
            self.clock_row.set_text(f"{stats.core_clock_mhz:,} MHz")

        if stats.power_watts is None:
            self.power_row.set_text("N/A")
        else:
            self.power_row.set_text(f"{stats.power_watts:.1f} W")


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
    cache_type_k_input: QComboBox
    cache_type_v_input: QComboBox
    flash_attn_checkbox: QCheckBox
    no_cache_prompt_checkbox: QCheckBox
    auto_start_checkbox: QCheckBox
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
        model_weight_extensions = {".safetensors", ".bin", ".pt", ".pth"}
        has_model_weights = False
        conversion_files: list[tuple[str, int]] = []
        conversion_config_exts = {".json", ".model", ".txt", ".tiktoken"}
        all_siblings = list(info.siblings or [])

        for sibling in all_siblings:
            name = getattr(sibling, "rfilename", "")
            size_value = getattr(sibling, "size", None) or 0
            lower_name = name.lower()
            if lower_name.endswith(".gguf"):
                files.append(
                    RepoFile(
                        name=name,
                        size_text=self._format_size(size_value if size_value else None),
                        size_bytes=size_value if size_value else None,
                    )
                )
            elif any(lower_name.endswith(ext) for ext in model_weight_extensions):
                has_model_weights = True
                conversion_files.append((name, int(size_value)))

        if has_model_weights:
            existing_names = {n for n, _ in conversion_files}
            for sibling in all_siblings:
                name = getattr(sibling, "rfilename", "")
                size_value = getattr(sibling, "size", None) or 0
                if name in existing_names:
                    continue
                ext = Path(name).suffix.lower()
                # Include all config / tokenizer files at the repo root
                if ext in conversion_config_exts and "/" not in name:
                    conversion_files.append((name, int(size_value)))
                    existing_names.add(name)

        return (
            details,
            sorted(files, key=lambda item: item.name.lower()),
            has_model_weights,
            conversion_files,
        )

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


class ConversionProgressDialog(QDialog):
    """Modal dialog showing download + conversion progress with ETA."""

    cancelled = Signal()

    def __init__(self, parent: QWidget, repo_id: str) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Convert to GGUF \u2014 {repo_id}")
        self.setMinimumWidth(560)
        self.setMinimumHeight(360)
        self.setModal(True)
        self._cancelled = False
        self._phase_start = 0.0

        layout = QVBoxLayout(self)

        self.phase_label = QLabel("Preparing\u2026")
        font = self.phase_label.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        self.phase_label.setFont(font)
        layout.addWidget(self.phase_label)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.eta_label = QLabel("")
        layout.addWidget(self.eta_label)

        self.detail_output = QPlainTextEdit()
        self.detail_output.setReadOnly(True)
        self.detail_output.setMaximumHeight(150)
        self.detail_output.setPlaceholderText("Conversion output will appear here\u2026")
        layout.addWidget(self.detail_output)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._handle_cancel)
        btn_layout.addWidget(self.cancel_button)
        layout.addLayout(btn_layout)

    # -- public helpers ------------------------------------------------ #

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def set_phase(self, phase: str) -> None:
        self.phase_label.setText(phase)
        self._phase_start = time.monotonic()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.eta_label.setText("Estimating time remaining\u2026")

    def set_progress(self, percent: int, status: str = "") -> None:
        self.progress_bar.setValue(max(0, min(100, percent)))
        if status:
            self.status_label.setText(status)
        if percent > 0 and self._phase_start > 0:
            elapsed = time.monotonic() - self._phase_start
            estimated_total = elapsed / (percent / 100.0)
            remaining = max(0.0, estimated_total - elapsed)
            self.eta_label.setText(self._format_eta(remaining))

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_eta(self, seconds: float) -> None:
        self.eta_label.setText(self._format_eta(seconds))

    def append_log(self, text: str) -> None:
        self.detail_output.appendPlainText(text)

    def mark_complete(self, output_path: str) -> None:
        self.progress_bar.setValue(100)
        self.phase_label.setText("Conversion Complete")
        self.status_label.setText(f"GGUF saved to:\n{output_path}")
        self.eta_label.setText("")
        self.cancel_button.setText("Close")
        self.cancel_button.setEnabled(True)
        try:
            self.cancel_button.clicked.disconnect()
        except RuntimeError:
            pass
        self.cancel_button.clicked.connect(self.accept)

    def mark_failed(self, message: str) -> None:
        self.phase_label.setText("Conversion Failed")
        self.status_label.setText(message)
        self.eta_label.setText("")
        self.cancel_button.setText("Close")
        self.cancel_button.setEnabled(True)
        try:
            self.cancel_button.clicked.disconnect()
        except RuntimeError:
            pass
        self.cancel_button.clicked.connect(self.reject)

    # -- internals ----------------------------------------------------- #

    def _handle_cancel(self) -> None:
        if self._cancelled:
            self.reject()
            return
        self._cancelled = True
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Cancelling\u2026")
        self.cancelled.emit()

    @staticmethod
    def _format_eta(seconds: float) -> str:
        if seconds < 1:
            return "Almost done\u2026"
        if seconds < 60:
            return f"Estimated time remaining: {int(seconds)}s"
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        if mins >= 60:
            hours = mins // 60
            mins = mins % 60
            return f"Estimated time remaining: {hours}h {mins}m"
        return f"Estimated time remaining: {mins}m {secs}s"

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if not self._cancelled and self.cancel_button.text() != "Close":
            event.ignore()
            self._handle_cancel()
        else:
            super().closeEvent(event)


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

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
        lines: list[str] = []
        for item in messages:
            role = str(item.get("role", "user") or "user")
            content = str(item.get("content", "") or "")
            lines.append(f"{role}: {content}")
        lines.append("assistant:")
        return "\n".join(lines).strip()

    @staticmethod
    def _build_openai_payload(
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        messages: list[dict[str, str]] | None = None,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "stream": False,
        }
        if messages is not None:
            payload["messages"] = messages
        if prompt is not None:
            payload["prompt"] = prompt
        # Upstream OpenAI-compatible servers may reject negative max_tokens.
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens
        return payload

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
        upstream_host = normalize_connect_host(str(slot.get("host", "127.0.0.1") or "127.0.0.1"))
        base_url = f"http://{upstream_host}:{int(slot.get('port', 8080))}"
        chat_payload = self._build_openai_payload(
            model="local-model",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        chat_text: str | None = None
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{base_url.rstrip('/')}/v1/chat/completions",
                    json=chat_payload,
                    timeout=300,
                )
                response.raise_for_status()
                chat_text = self._extract_text_from_chat_choice(response.json())
                if chat_text:
                    return chat_text
                # Got 200 but empty content — log and fall through to completions
                break
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else 0
                if status_code in {404, 405, 501}:
                    break
                if status_code == 503 and attempt < 2:
                    time.sleep(1.0)
                    continue
                raise
            except requests.RequestException:
                if attempt < 2:
                    time.sleep(1.0)
                    continue
                raise

        # Fallback: /v1/completions with messages flattened to a prompt
        prompt = self._messages_to_prompt(messages)
        completion_response = requests.post(
            f"{base_url.rstrip('/')}/v1/completions",
            json=self._build_openai_payload(
                model="local-model",
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            timeout=300,
        )
        completion_response.raise_for_status()
        text = self._extract_text_from_completion(completion_response.json())
        if text:
            return text

        # Both endpoints returned empty — surface as a visible error
        raise RuntimeError(
            "Upstream server returned empty content from both /v1/chat/completions and /v1/completions. "
            "If using a Qwen3 thinking model, try adding '/no_think' at the start of your message, "
            "or set extra args '--no-reasoning' in the server slot."
        )

    def _forward_completion(
        self,
        slot: dict[str, Any],
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        upstream_host = normalize_connect_host(str(slot.get("host", "127.0.0.1") or "127.0.0.1"))
        base_url = f"http://{upstream_host}:{int(slot.get('port', 8080))}"
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/completions",
            json=self._build_openai_payload(
                model="local-model",
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            timeout=300,
        )
        response.raise_for_status()
        text = self._extract_text_from_completion(response.json())
        if text:
            return text

        chat_response = requests.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json=self._build_openai_payload(
                model="local-model",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            timeout=300,
        )
        chat_response.raise_for_status()
        chat_text = self._extract_text_from_chat_choice(chat_response.json())
        if chat_text:
            return chat_text

        raise RuntimeError(
            "Upstream server returned empty content from both /v1/completions and /v1/chat/completions. "
            "If using a Qwen3 thinking model, try adding '/no_think' at the start of your message, "
            "or set extra args '--no-reasoning' in the server slot."
        )

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
        snapshot = self._get_snapshot()
        default_num_predict = int(snapshot.get("proxy_num_predict", -1) or -1)
        temperature = float(options.get("temperature", payload.get("temperature", 0.7)) or 0.7)
        num_predict_val = options.get("num_predict", payload.get("num_predict", None))
        max_tokens = int(num_predict_val) if num_predict_val is not None else default_num_predict

        try:
            text = self._forward_completion(slot, prompt, temperature, max_tokens)
        except requests.RequestException as exc:
            self._send_error(handler, f"Upstream llama-server request failed: {exc}", 502)
            return
        except RuntimeError as exc:
            self._send_error(handler, str(exc), 502)
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
        snapshot = self._get_snapshot()
        default_num_predict = int(snapshot.get("proxy_num_predict", -1) or -1)
        temperature = float(options.get("temperature", payload.get("temperature", 0.7)) or 0.7)
        num_predict_val = options.get("num_predict", payload.get("num_predict", None))
        max_tokens = int(num_predict_val) if num_predict_val is not None else default_num_predict

        try:
            text = self._forward_chat(slot, messages, temperature, max_tokens)
        except requests.RequestException as exc:
            self._send_error(handler, f"Upstream llama-server request failed: {exc}", 502)
            return
        except RuntimeError as exc:
            self._send_error(handler, str(exc), 502)
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
        self.conversion_files: list[tuple[str, int]] = []
        self.has_model_weights: bool = False
        self._conversion_cancelled = False
        self._conversion_dialog: ConversionProgressDialog | None = None
        self._conversion_current_phase = ""
        self.chat_history: list[dict[str, str]] = []
        self.pending_user_message = ""
        self.available_devices: list[GPUDevice] = self._detect_available_devices()
        self.server_slots: list[ServerSlot] = []
        self.server_running_states = [False, False, False, False]
        self._model_compat_warning_shown: set[int] = set()
        self.sakura_monitor = SakuraNvidiaMonitor()
        self.sakura_lhm = SakuraLibreHardwareMonitorBridge()
        self.sakura_expected_runtime = PROJECT_ROOT / ".venv311" / "Scripts" / "python.exe"
        self.sakura_timer = QTimer(self)
        self.sakura_timer.setInterval(1000)
        self.sakura_timer.timeout.connect(self._refresh_sakura_panel)
        self.sakura_gpu_cards: list[SakuraGPUCard] = []
        self._snapshot_lock = threading.Lock()
        self._ollama_snapshot: dict[str, Any] = {
            "default_server": 0,
            "proxy_num_predict": -1,
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

        # ---- System tray ----
        self._build_tray_icon()

        # ---- Auto-start (deferred until event loop is running) ----
        QTimer.singleShot(500, self._auto_start_servers)

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

            QGroupBox#SakuraDeviceCard {{
                background: rgba(255, 247, 251, 236);
                border: 1px solid #E5BAC8;
                border-radius: 14px;
                margin-top: 8px;
                padding-top: 8px;
            }}

            QGroupBox#SakuraDeviceCard::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
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

            QLabel#SakuraMetricTitle {{
                font-weight: 600;
                color: #6A2A3F;
            }}

            QLabel#SakuraMetricValue {{
                color: #7A3A4E;
                font-weight: 600;
            }}

            QProgressBar#SakuraBar {{
                border: 1px solid #DDB2C1;
                border-radius: 7px;
                background: #FFEAF1;
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
        tabs.addTab(self._build_ollama_tab(), "Ollama Models")
        tabs.addTab(self._build_global_settings_tab(), "Global Settings")
        tabs.addTab(self._build_server_slots_tab(), "Server Slots")
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

        convert_row = QHBoxLayout()
        self.convert_to_gguf_button = QPushButton("Download && Convert to GGUF")
        self.convert_to_gguf_button.clicked.connect(self.start_hf_conversion)
        self.convert_to_gguf_button.setEnabled(False)
        self.convert_to_gguf_button.setToolTip(
            "Download non-GGUF model weights (SafeTensors, etc.) and convert to GGUF format.\n"
            "Requires: gguf, numpy, torch, safetensors, transformers (pip packages)."
        )
        convert_row.addWidget(self.convert_to_gguf_button)

        self.download_progress = QProgressBar()
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self.download_status = QLabel("No download started.")

        files_layout.addWidget(self.files_list)
        files_layout.addLayout(download_row)
        files_layout.addLayout(convert_row)
        files_layout.addWidget(self.download_progress)
        files_layout.addWidget(self.download_status)

        layout.addWidget(search_box)
        layout.addWidget(files_box)
        return container

    def _build_right_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._build_global_settings_tab())
        layout.addWidget(self._build_repo_details_panel())
        return container

    def _build_global_settings_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        hf_box = QGroupBox("Hugging Face")
        hf_layout = QFormLayout(hf_box)

        self.hf_token_input = QLineEdit()
        self.hf_token_input.setPlaceholderText("Optional Hugging Face token for gated/private repos")
        self.hf_token_input.setEchoMode(QLineEdit.Password)
        hf_layout.addRow("HF Token", self.hf_token_input)

        self.model_target_server_selector = QComboBox()
        self.model_target_server_selector.addItems([f"Server {index + 1}" for index in range(4)])
        hf_layout.addRow("HF model target", self.model_target_server_selector)

        ollama_box = QGroupBox("Ollama Proxy")
        ollama_layout = QFormLayout(ollama_box)

        self.ollama_default_server_selector = QComboBox()
        self.ollama_default_server_selector.addItems([f"Server {index + 1}" for index in range(4)])
        ollama_layout.addRow("Default server", self.ollama_default_server_selector)

        ollama_proxy_layout = QGridLayout()
        self.ollama_host_input = QLineEdit("127.0.0.1")
        self.ollama_port_input = QSpinBox()
        self.ollama_port_input.setRange(1, 65535)
        self.ollama_port_input.setValue(11434)
        ollama_proxy_layout.addWidget(QLabel("Host"), 0, 0)
        ollama_proxy_layout.addWidget(self.ollama_host_input, 0, 1)
        ollama_proxy_layout.addWidget(QLabel("Base Port"), 0, 2)
        ollama_proxy_layout.addWidget(self.ollama_port_input, 0, 3)
        ollama_layout.addRow("Listen", self._wrap_layout(ollama_proxy_layout))
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
        ollama_layout.addRow("Proxy Control", self._wrap_layout(ollama_controls))

        self.proxy_num_predict_input = QSpinBox()
        self.proxy_num_predict_input.setRange(-1, 131072)
        self.proxy_num_predict_input.setSingleStep(256)
        self.proxy_num_predict_input.setSpecialValueText("-1 (unlimited)")
        self.proxy_num_predict_input.setValue(-1)
        self.proxy_num_predict_input.setToolTip(
            "Default max tokens for the Ollama proxy when clients don't specify num_predict.\n"
            "Set to -1 for unlimited (let the model generate until it naturally stops or hits context).\n"
            "Thinking models (e.g. Qwen3) need high values because reasoning tokens count towards this limit."
        )
        self.proxy_num_predict_input.valueChanged.connect(self._save_config)
        ollama_layout.addRow("Default Max Tokens", self.proxy_num_predict_input)

        self.ollama_proxy_status = QLabel("Ollama API proxy is stopped.")
        ollama_layout.addRow("Proxy Status", self.ollama_proxy_status)
        self.ollama_proxy_test_status = QLabel("Proxy test idle.")
        ollama_layout.addRow("Proxy Test", self.ollama_proxy_test_status)

        chat_box = QGroupBox("Chat Defaults")
        chat_layout = QFormLayout(chat_box)

        generation_layout = QGridLayout()
        self.temperature_input = QSpinBox()
        self.temperature_input.setRange(0, 200)
        self.temperature_input.setValue(70)
        self.chat_max_tokens_input = QSpinBox()
        self.chat_max_tokens_input.setRange(1, 131072)
        self.chat_max_tokens_input.setValue(512)
        generation_layout.addWidget(QLabel("Temp x100"), 0, 0)
        generation_layout.addWidget(self.temperature_input, 0, 1)
        generation_layout.addWidget(QLabel("Reply Tokens"), 0, 2)
        generation_layout.addWidget(self.chat_max_tokens_input, 0, 3)
        chat_layout.addRow("Defaults", self._wrap_layout(generation_layout))

        layout.addWidget(hf_box)
        layout.addWidget(ollama_box)
        layout.addWidget(chat_box)
        layout.addStretch(1)
        return container

    def _build_server_slots_tab(self) -> QWidget:
        self.server_slots = []
        slots_wrapper = QWidget()
        slots_layout = QVBoxLayout(slots_wrapper)
        for index in range(4):
            slots_layout.addWidget(self._build_server_slot(index))
        slots_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(slots_wrapper)
        return scroll

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

        kv_cache_types = ["f16", "q8_0", "q4_0", "q4_1"]
        cache_type_k_input = QComboBox()
        cache_type_k_input.addItems(kv_cache_types)
        cache_type_v_input = QComboBox()
        cache_type_v_input.addItems(kv_cache_types)
        kv_row = QHBoxLayout()
        kv_row.addWidget(QLabel("K:"))
        kv_row.addWidget(cache_type_k_input)
        kv_row.addWidget(QLabel("V:"))
        kv_row.addWidget(cache_type_v_input)
        form.addRow("KV Cache Type", self._wrap_layout(kv_row))

        flash_attn_checkbox = QCheckBox("Enable (requires Pascal / CC 6.0+ GPU)")
        flash_attn_checkbox.setToolTip(
            "Flash Attention reduces memory during long-context inference.\n"
            "Requires NVIDIA Pascal (GTX 10xx / CC 6.0) or newer.\n"
            "Maxwell GPUs (GTX Titan X, 980, etc.) do NOT support this — the server will crash."
        )
        form.addRow("Flash Attention", flash_attn_checkbox)

        no_cache_prompt_checkbox = QCheckBox("Disable prompt caching")
        no_cache_prompt_checkbox.setToolTip("Pass --no-cache-prompt to llama-server.")
        form.addRow("No Cache Prompt", no_cache_prompt_checkbox)

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

        auto_start_checkbox = QCheckBox("Launch this server when the app starts")
        form.addRow("Auto-Start", auto_start_checkbox)

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
        process.finished.connect(
            lambda exit_code, exit_status, i=index: self._handle_process_finished(i, exit_code, exit_status)
        )

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
                cache_type_k_input=cache_type_k_input,
                cache_type_v_input=cache_type_v_input,
                flash_attn_checkbox=flash_attn_checkbox,
                no_cache_prompt_checkbox=no_cache_prompt_checkbox,
                auto_start_checkbox=auto_start_checkbox,
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

    # ------------------------------------------------------------------ #
    #  Ollama Models tab                                                   #
    # ------------------------------------------------------------------ #

    def _build_ollama_tab(self) -> QWidget:
        container = QWidget()
        outer = QVBoxLayout(container)

        # ── Pull Model ────────────────────────────────────────────────
        pull_box = QGroupBox("Pull Model from Ollama Registry")
        pull_layout = QVBoxLayout(pull_box)

        pull_row = QHBoxLayout()
        self.ollama_pull_input = QLineEdit()
        self.ollama_pull_input.setPlaceholderText(
            "Model name, e.g.  llama3:8b  or  qwen3:0.6b  or  gemma3:4b-it-qat"
        )
        self.ollama_pull_input.returnPressed.connect(self._ollama_pull_model)
        self.ollama_pull_button = QPushButton("Pull")
        self.ollama_pull_button.clicked.connect(self._ollama_pull_model)
        pull_row.addWidget(self.ollama_pull_input)
        pull_row.addWidget(self.ollama_pull_button)

        self.ollama_pull_progress = QProgressBar()
        self.ollama_pull_progress.setRange(0, 100)
        self.ollama_pull_progress.setValue(0)
        self.ollama_pull_status = QLabel("Enter a model name and click Pull.")

        pull_layout.addLayout(pull_row)
        pull_layout.addWidget(self.ollama_pull_progress)
        pull_layout.addWidget(self.ollama_pull_status)

        # ── Extract GGUF ──────────────────────────────────────────────
        extract_box = QGroupBox("Extract GGUF from Local Ollama Models")
        extract_layout = QVBoxLayout(extract_box)

        extract_top_row = QHBoxLayout()
        self.ollama_refresh_button = QPushButton("Refresh Model List")
        self.ollama_refresh_button.clicked.connect(self._ollama_refresh_models)
        extract_top_row.addWidget(self.ollama_refresh_button)
        extract_top_row.addStretch()

        self.ollama_models_table = QTableWidget(0, 4)
        self.ollama_models_table.setHorizontalHeaderLabels(["Name", "ID", "Size", "Modified"])
        self.ollama_models_table.horizontalHeader().setStretchLastSection(True)
        self.ollama_models_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ollama_models_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.ollama_models_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        extract_btn_row = QHBoxLayout()
        self.ollama_extract_button = QPushButton("Extract GGUF to models/")
        self.ollama_extract_button.clicked.connect(self._ollama_extract_gguf)
        self.ollama_extract_button.setEnabled(False)
        self.ollama_extract_button.setToolTip(
            "Copy the GGUF blob from Ollama's storage into the models/ folder\n"
            "and set it as the model path on the selected target server slot."
        )
        extract_btn_row.addWidget(self.ollama_extract_button)
        extract_btn_row.addStretch()

        self.ollama_extract_status = QLabel("Click Refresh to list locally-pulled Ollama models.")

        self.ollama_models_table.itemSelectionChanged.connect(
            lambda: self.ollama_extract_button.setEnabled(
                self.ollama_models_table.currentRow() >= 0
            )
        )

        extract_layout.addLayout(extract_top_row)
        extract_layout.addWidget(self.ollama_models_table)
        extract_layout.addLayout(extract_btn_row)
        extract_layout.addWidget(self.ollama_extract_status)

        outer.addWidget(pull_box)
        outer.addWidget(extract_box)
        return container

    def _build_log_panel(self) -> QWidget:
        panel = QWidget()
        root = QVBoxLayout(panel)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        sakura_box = QGroupBox("Sakura Load Monitor")
        sakura_layout = QVBoxLayout(sakura_box)

        self.sakura_updated_label = QLabel("Last update: --")
        sakura_layout.addWidget(self.sakura_updated_label)

        sakura_scroll = QScrollArea()
        sakura_scroll.setWidgetResizable(True)
        sakura_layout.addWidget(sakura_scroll)

        sakura_body = QWidget()
        self.sakura_body_layout = QVBoxLayout(sakura_body)
        self.sakura_body_layout.setContentsMargins(4, 4, 4, 4)
        self.sakura_body_layout.setSpacing(8)
        sakura_scroll.setWidget(sakura_body)

        self.sakura_cpu_row = SakuraMetricRow("CPU Utilization")
        self.sakura_cpu_temp_row = SakuraMetricRow("CPU Temperature")
        self.sakura_cpu_power_row = SakuraMetricRow("CPU Power", with_bar=False)
        self.sakura_ram_row = SakuraMetricRow("System RAM")
        self.sakura_ram_temp_row = SakuraMetricRow("RAM Temperature")
        self.sakura_ram_power_row = SakuraMetricRow("RAM Power", with_bar=False)
        self.sakura_diag_python_row = SakuraMetricRow("Python", with_bar=False)
        self.sakura_diag_runtime_row = SakuraMetricRow("Expected Runtime", with_bar=False)
        self.sakura_diag_nvml_row = SakuraMetricRow("NVML GPUs", with_bar=False)
        self.sakura_diag_lhm_row = SakuraMetricRow("LHM GPUs", with_bar=False)
        self.sakura_diag_lhm_status_row = SakuraMetricRow("LHM Status", with_bar=False)

        system_box = QGroupBox("CPU + System Memory")
        system_box.setObjectName("SakuraDeviceCard")
        system_layout = QVBoxLayout(system_box)
        system_layout.addWidget(self.sakura_cpu_row)
        system_layout.addWidget(self.sakura_cpu_temp_row)
        system_layout.addWidget(self.sakura_ram_row)
        system_layout.addWidget(self.sakura_ram_temp_row)

        power_row = QHBoxLayout()
        power_row.setContentsMargins(0, 0, 0, 0)
        power_row.setSpacing(12)
        power_row.addWidget(self.sakura_cpu_power_row)
        power_row.addWidget(self.sakura_ram_power_row)
        system_layout.addLayout(power_row)

        diag_box = QGroupBox("Runtime Diagnostics")
        diag_box.setObjectName("SakuraDeviceCard")
        diag_layout = QGridLayout(diag_box)
        diag_layout.setContentsMargins(14, 16, 14, 14)
        diag_layout.setHorizontalSpacing(12)
        diag_layout.setVerticalSpacing(10)
        diag_layout.addWidget(self.sakura_diag_python_row, 0, 0)
        diag_layout.addWidget(self.sakura_diag_runtime_row, 0, 1)
        diag_layout.addWidget(self.sakura_diag_nvml_row, 1, 0)
        diag_layout.addWidget(self.sakura_diag_lhm_row, 1, 1)
        diag_layout.addWidget(self.sakura_diag_lhm_status_row, 2, 0, 1, 2)

        self.sakura_no_gpu_label = QLabel("No GPU telemetry found from NVML or LibreHardwareMonitor backends.")
        self.sakura_no_gpu_label.setWordWrap(True)

        self.sakura_body_layout.addWidget(system_box)
        self.sakura_body_layout.addWidget(diag_box)
        self.sakura_body_layout.addWidget(self.sakura_no_gpu_label)

        self.sakura_gpu_container = QWidget()
        self.sakura_gpu_layout = QVBoxLayout(self.sakura_gpu_container)
        self.sakura_gpu_layout.setContentsMargins(0, 0, 0, 0)
        self.sakura_gpu_layout.setSpacing(8)
        self.sakura_body_layout.addWidget(self.sakura_gpu_container)
        self.sakura_body_layout.addStretch(1)

        log_box = QGroupBox("Server Log")
        log_layout = QVBoxLayout(log_box)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.clicked.connect(self.log_output.clear)
        log_layout.addWidget(self.clear_log_button)
        log_layout.addWidget(self.log_output)

        splitter.addWidget(sakura_box)
        splitter.addWidget(log_box)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([700, 700])

        root.addWidget(splitter)

        if psutil is not None:
            psutil.cpu_percent(interval=None)
        self._refresh_sakura_panel()
        self.sakura_timer.start()
        return panel

    @staticmethod
    def _normalize_sakura_gpu_name(name: str) -> str:
        normalized = "".join(ch.lower() for ch in name if ch.isalnum() or ch.isspace())
        for token in ("nvidia", "amd", "radeon", "geforce"):
            normalized = normalized.replace(token, " ")
        return " ".join(normalized.split())

    @staticmethod
    def _sakura_gpu_name_matches(left: str, right: str) -> bool:
        if left == right:
            return True
        if not left or not right:
            return False
        return left in right or right in left

    @classmethod
    def _merge_sakura_gpu_stats(
        cls,
        nvml_stats: list[SakuraGPUStats],
        lhm_stats: list[SakuraGPUStats],
    ) -> list[SakuraGPUStats]:
        merged = list(nvml_stats)
        lhm_matched: set[int] = set()

        def apply_missing_fields(target: SakuraGPUStats, source: SakuraGPUStats) -> None:
            if target.vram_total_mib <= 0 and source.vram_total_mib > 0:
                target.vram_total_mib = source.vram_total_mib
                target.vram_used_mib = source.vram_used_mib
                target.vram_percent = source.vram_percent
            if target.shared_total_mib <= 0 and source.shared_total_mib > 0:
                target.shared_total_mib = source.shared_total_mib
                target.shared_used_mib = source.shared_used_mib
                target.shared_percent = source.shared_percent
            if target.util_percent is None and source.util_percent is not None:
                target.util_percent = source.util_percent
            if target.core_clock_mhz is None and source.core_clock_mhz is not None:
                target.core_clock_mhz = source.core_clock_mhz
            if target.power_watts is None and source.power_watts is not None:
                target.power_watts = source.power_watts
            if target.temperature_c is None and source.temperature_c is not None:
                target.temperature_c = source.temperature_c

        for nvml_item in merged:
            nvml_key = cls._normalize_sakura_gpu_name(nvml_item.name)
            best_index: int | None = None

            for idx, lhm_item in enumerate(lhm_stats):
                if idx in lhm_matched:
                    continue
                lhm_key = cls._normalize_sakura_gpu_name(lhm_item.name)
                if cls._sakura_gpu_name_matches(nvml_key, lhm_key):
                    best_index = idx
                    break

            if best_index is None:
                continue

            lhm_matched.add(best_index)
            apply_missing_fields(nvml_item, lhm_stats[best_index])

        for idx, lhm_item in enumerate(lhm_stats):
            if idx not in lhm_matched:
                merged.append(lhm_item)

        return merged

    def _collect_sakura_gpu_sources(self) -> tuple[list[SakuraGPUStats], int, int]:
        nvml_stats = self.sakura_monitor.collect()
        lhm_stats = self.sakura_lhm.collect_gpu_stats()
        merged = self._merge_sakura_gpu_stats(nvml_stats, lhm_stats)
        return merged, len(nvml_stats), len(lhm_stats)

    def _collect_sakura_system_stats(self) -> SakuraSystemStats:
        cpu_percent = 0.0
        ram_total_gib = 0.0
        ram_used_gib = 0.0
        ram_percent = 0.0
        if psutil is not None:
            try:
                cpu_percent = float(psutil.cpu_percent(interval=None))
                mem = psutil.virtual_memory()
                ram_total_gib = mem.total / (1024 ** 3)
                ram_used_gib = (mem.total - mem.available) / (1024 ** 3)
                ram_percent = float(mem.percent)
            except Exception:  # noqa: BLE001
                cpu_percent = 0.0

        cpu_power_watts, ram_power_watts, cpu_temp_c, ram_temp_c = self.sakura_lhm.collect_cpu_ram_telemetry()

        return SakuraSystemStats(
            cpu_percent=cpu_percent,
            cpu_power_watts=cpu_power_watts,
            cpu_temp_c=cpu_temp_c,
            ram_total_gib=ram_total_gib,
            ram_used_gib=ram_used_gib,
            ram_percent=ram_percent,
            ram_power_watts=ram_power_watts,
            ram_temp_c=ram_temp_c,
        )

    def _rebuild_sakura_gpu_cards(self, gpu_stats: list[SakuraGPUStats]) -> None:
        for card in self.sakura_gpu_cards:
            card.setParent(None)
        self.sakura_gpu_cards.clear()

        if not gpu_stats:
            self.sakura_no_gpu_label.show()
            return

        self.sakura_no_gpu_label.hide()
        for idx, stats in enumerate(gpu_stats):
            card = SakuraGPUCard(idx, stats.name)
            card.apply_stats(stats)
            self.sakura_gpu_cards.append(card)
            self.sakura_gpu_layout.addWidget(card)

    def _refresh_sakura_panel(self) -> None:
        if not hasattr(self, "sakura_updated_label"):
            return

        system_stats = self._collect_sakura_system_stats()
        self.sakura_cpu_row.set_percent(system_stats.cpu_percent, f"{system_stats.cpu_percent:.1f}%")

        if system_stats.cpu_temp_c is None:
            self.sakura_cpu_temp_row.set_percent(0, "N/A")
        else:
            self.sakura_cpu_temp_row.set_percent(
                SakuraMetricRow.temperature_to_percent(system_stats.cpu_temp_c),
                f"{system_stats.cpu_temp_c:.1f} C",
            )

        if system_stats.cpu_power_watts is None:
            self.sakura_cpu_power_row.set_text("N/A")
        else:
            self.sakura_cpu_power_row.set_text(f"{system_stats.cpu_power_watts:.1f} W")

        if system_stats.ram_total_gib > 0:
            self.sakura_ram_row.set_percent(
                system_stats.ram_percent,
                f"{system_stats.ram_used_gib:.2f} / {system_stats.ram_total_gib:.2f} GiB ({system_stats.ram_percent:.1f}%)",
            )
        else:
            self.sakura_ram_row.set_percent(0, "N/A")

        if system_stats.ram_temp_c is None:
            self.sakura_ram_temp_row.set_percent(0, "N/A")
        else:
            self.sakura_ram_temp_row.set_percent(
                SakuraMetricRow.temperature_to_percent(system_stats.ram_temp_c),
                f"{system_stats.ram_temp_c:.1f} C",
            )

        if system_stats.ram_power_watts is None:
            self.sakura_ram_power_row.set_text("N/A")
        else:
            self.sakura_ram_power_row.set_text(f"{system_stats.ram_power_watts:.1f} W")

        gpu_stats, nvml_count, lhm_count = self._collect_sakura_gpu_sources()
        if len(gpu_stats) != len(self.sakura_gpu_cards):
            self._rebuild_sakura_gpu_cards(gpu_stats)
        for card, stats in zip(self.sakura_gpu_cards, gpu_stats):
            card.apply_stats(stats)

        self.sakura_diag_python_row.set_text(f"{sys.version.split()[0]} ({Path(sys.executable).name})")
        if self.sakura_expected_runtime.exists():
            runtime_text = "OK" if Path(sys.executable).resolve() == self.sakura_expected_runtime.resolve() else "Fallback"
        else:
            runtime_text = "Current"
        self.sakura_diag_runtime_row.set_text(runtime_text)
        self.sakura_diag_nvml_row.set_text(str(nvml_count))
        self.sakura_diag_lhm_row.set_text(str(lhm_count))
        self.sakura_diag_lhm_status_row.set_text(self.sakura_lhm.status)
        self.sakura_updated_label.setText(
            f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

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
        raw_pnp = data.get("proxy_num_predict", -1)
        self.proxy_num_predict_input.setValue(int(raw_pnp) if raw_pnp is not None else -1)

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

            cache_k = str(server_data.get("cache_type_k", "f16") or "f16")
            cache_v = str(server_data.get("cache_type_v", "f16") or "f16")
            k_idx = slot.cache_type_k_input.findText(cache_k)
            v_idx = slot.cache_type_v_input.findText(cache_v)
            if k_idx >= 0:
                slot.cache_type_k_input.setCurrentIndex(k_idx)
            if v_idx >= 0:
                slot.cache_type_v_input.setCurrentIndex(v_idx)
            slot.flash_attn_checkbox.setChecked(bool(server_data.get("flash_attn", False)))
            slot.no_cache_prompt_checkbox.setChecked(bool(server_data.get("no_cache_prompt", False)))
            slot.auto_start_checkbox.setChecked(bool(server_data.get("auto_start", False)))

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
                    "cache_type_k": slot.cache_type_k_input.currentText(),
                    "cache_type_v": slot.cache_type_v_input.currentText(),
                    "flash_attn": slot.flash_attn_checkbox.isChecked(),
                    "no_cache_prompt": slot.no_cache_prompt_checkbox.isChecked(),
                    "auto_start": slot.auto_start_checkbox.isChecked(),
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
            "proxy_num_predict": self.proxy_num_predict_input.value(),
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
        self.convert_to_gguf_button.setEnabled(False)
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
        self.convert_to_gguf_button.setEnabled(False)
        self.repo_title_label.setText(repo_id)
        self.repo_meta_label.setText("Loading metadata...")
        self.repo_summary_text.setPlainText("")

        worker = Worker(self.hf_client.get_repo_details, repo_id, token)
        worker.signals.finished.connect(self._populate_repo_details)
        worker.signals.error.connect(self._show_repo_files_error)
        self.thread_pool.start(worker)

    def _populate_repo_details(self, payload: tuple) -> None:
        details, files, has_model_weights, conversion_files = payload
        self.repo_details = details
        self.repo_files = files
        self.has_model_weights = has_model_weights
        self.conversion_files = conversion_files

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
            if has_model_weights:
                weight_exts: set[str] = set()
                for fname, _ in conversion_files:
                    ext = Path(fname).suffix.lower()
                    if ext in {".safetensors", ".bin", ".pt", ".pth"}:
                        weight_exts.add(ext.lstrip("."))
                total_size = sum(s for _, s in conversion_files)
                ext_text = ", ".join(sorted(weight_exts))
                self.files_list.addItem(
                    f"No GGUF files \u2014 this repo has {ext_text} weights "
                    f"({self._format_bytes(total_size)}) that can be converted to GGUF."
                )
                self.convert_to_gguf_button.setEnabled(True)
            else:
                self.files_list.addItem("No GGUF files were found in this repo.")
                self.convert_to_gguf_button.setEnabled(False)
            self.download_button.setEnabled(False)
            return

        for file_info in files:
            label = file_info.name if not file_info.size_text else f"{file_info.name} ({file_info.size_text})"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, file_info.name)
            self.files_list.addItem(item)

        self.files_list.setCurrentRow(0)
        self.download_button.setEnabled(True)
        self.convert_to_gguf_button.setEnabled(False)

    def _show_repo_files_error(self, message: str) -> None:
        self.files_list.clear()
        self.files_list.addItem("Failed to load files.")
        self._clear_repo_details("Failed to load repository details.")
        self.convert_to_gguf_button.setEnabled(False)
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

    # ------------------------------------------------------------------ #
    #  HuggingFace → GGUF conversion                                      #
    # ------------------------------------------------------------------ #

    def start_hf_conversion(self) -> None:
        """Initiate download and conversion of a non-GGUF HF model to GGUF."""
        repo_id = self._selected_repo_id()
        if not repo_id:
            QMessageBox.warning(self, "No Repository", "Select a repository first.")
            return

        if not self.conversion_files:
            QMessageBox.warning(self, "No Model Files", "No convertible model files found.")
            return

        target_index = self._resolve_server_index(None)
        target_slot = self.server_slots[target_index]
        target_backend = self._infer_backend(self._selected_devices(target_slot)) or "cpu"

        # Locate the convert script
        convert_script = self._find_convert_script(target_backend)
        if not convert_script:
            reply = QMessageBox.question(
                self,
                "Download Converter",
                "convert_hf_to_gguf.py was not found locally.\n\n"
                "Would you like to download it from the llama.cpp repository?\n"
                "(The converter script + matching gguf package will be installed)",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                convert_script = self._download_convert_script()
            if not convert_script:
                return

        self.log_output.appendPlainText(
            f"[CONVERT] Using converter for backend '{target_backend}': {convert_script}"
        )

        # Check for required Python packages
        missing = self._check_conversion_deps()
        if missing:
            reply = QMessageBox.question(
                self,
                "Missing Dependencies",
                f"The following Python packages are required for GGUF conversion:\n\n"
                f"  {', '.join(missing)}\n\n"
                f"Install them now with pip?\n"
                f"(torch may be a large download)",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                if not self._install_conversion_deps(missing):
                    return
                # Re-check after install
                still_missing = self._check_conversion_deps()
                if still_missing:
                    QMessageBox.critical(
                        self,
                        "Dependencies Still Missing",
                        f"These packages could not be installed:\n{', '.join(still_missing)}",
                    )
                    return
            else:
                return

        # Confirm
        total_size = sum(s for _, s in self.conversion_files)
        file_count = len(self.conversion_files)
        reply = QMessageBox.question(
            self,
            "Download & Convert to GGUF",
            f"Repo: {repo_id}\n\n"
            f"This will download {file_count} file(s) ({self._format_bytes(total_size)}) "
            f"and convert them to GGUF (f16) format.\n\n"
            f"The GGUF model will be saved to the models/ folder.\n\n"
            f"Proceed?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # Create and show progress dialog
        self._conversion_cancelled = False
        self._conversion_current_phase = ""
        dialog = ConversionProgressDialog(self, repo_id)
        dialog.cancelled.connect(self._cancel_conversion)
        self._conversion_dialog = dialog

        token = self.hf_token_input.text().strip()

        worker = Worker(
            self._conversion_worker,
            repo_id,
            list(self.conversion_files),
            token,
            "f16",
            str(convert_script),
            use_progress=True,
        )
        worker.signals.progress.connect(self._on_conversion_progress)
        worker.signals.finished.connect(self._on_conversion_complete)
        worker.signals.error.connect(self._on_conversion_error)
        self.thread_pool.start(worker)

        self.convert_to_gguf_button.setEnabled(False)
        dialog.exec()

    def _cancel_conversion(self) -> None:
        self._conversion_cancelled = True

    def _find_convert_script(self, preferred_backend: str = "cpu") -> Path | None:
        """Search for convert_hf_to_gguf.py in common locations."""
        workspace = Path(__file__).resolve().parent
        candidates: list[Path] = [
            workspace / "scripts" / "convert_hf_to_gguf.py",
            workspace / "convert_hf_to_gguf.py",
        ]

        # Check near configured llama-server paths, preferring the backend used
        # by the selected target server slot.
        backend_order = [preferred_backend, "cpu", "cuda", "hip", "vulkan"]
        seen_backends: set[str] = set()
        ordered_backends: list[str] = []
        for backend in backend_order:
            key = (backend or "").strip().lower()
            if not key or key in seen_backends:
                continue
            seen_backends.add(key)
            ordered_backends.append(key)

        for backend in ordered_backends:
            configured = self._llama_path_input_for_backend(backend).text().strip()
            if configured:
                base = Path(configured).parent
                candidates.extend([
                    base / "convert_hf_to_gguf.py",
                    base.parent / "convert_hf_to_gguf.py",
                ])
        # Common source checkout locations
        candidates.extend([
            workspace / "llama.cpp" / "convert_hf_to_gguf.py",
            Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        ])
        for path in candidates:
            if path.exists():
                return path
        return None

    def _download_convert_script(self) -> Path | None:
        """Download convert_hf_to_gguf.py and the matching gguf package from
        the latest stable llama.cpp release tag.  Using a release tag (not
        master) ensures the convert script and the gguf Python package are
        in sync — master often has new model classes before the constants
        are added to gguf-py."""
        dest = Path(__file__).resolve().parent / "scripts" / "convert_hf_to_gguf.py"
        dest.parent.mkdir(parents=True, exist_ok=True)

        # 1. Determine the latest release tag via the GitHub API.
        tag = None
        try:
            api_resp = requests.get(
                "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
                headers={"Accept": "application/vnd.github+json"},
                timeout=15,
            )
            if api_resp.status_code == 200:
                tag = api_resp.json().get("tag_name")
        except Exception:  # noqa: BLE001
            pass

        if not tag:
            # Fallback: try the tags endpoint (cheaper rate-limit wise).
            try:
                tags_resp = requests.get(
                    "https://api.github.com/repos/ggml-org/llama.cpp/tags?per_page=1",
                    headers={"Accept": "application/vnd.github+json"},
                    timeout=15,
                )
                if tags_resp.status_code == 200:
                    tags = tags_resp.json()
                    if tags:
                        tag = tags[0].get("name")
            except Exception:  # noqa: BLE001
                pass

        if not tag:
            QMessageBox.critical(
                self,
                "Release Lookup Failed",
                "Could not determine the latest llama.cpp release tag.\n"
                "Check your internet connection and try again.",
            )
            return None

        self.log_output.appendPlainText(f"[CONVERT] Using llama.cpp release: {tag}")

        # 2. Download the convert script from that tag.
        script_url = (
            f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{tag}/"
            f"convert_hf_to_gguf.py"
        )
        try:
            response = requests.get(script_url, timeout=30)
            response.raise_for_status()
            dest.write_text(response.text, encoding="utf-8")
            self.log_output.appendPlainText(f"[CONVERT] Downloaded convert_hf_to_gguf.py ({tag}) → {dest}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Download Failed",
                f"Failed to download convert_hf_to_gguf.py:\n{exc}",
            )
            return None

        # 3. Install the gguf package from the SAME tag so constants match.
        git_ref = f"gguf@git+https://github.com/ggml-org/llama.cpp@{tag}#subdirectory=gguf-py"
        self.log_output.appendPlainText(
            f"[CONVERT] Installing gguf package from llama.cpp {tag}…"
        )
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "--upgrade", "--no-deps",
                "--quiet", git_ref,
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip()
            self.log_output.appendPlainText(f"[CONVERT] WARNING: gguf install failed: {error}")
            QMessageBox.warning(
                self,
                "gguf Install Warning",
                f"Could not install the gguf package from llama.cpp {tag}.\n\n"
                "The conversion may fail if the installed gguf version is too old.\n\n"
                f"Error: {error[:300]}",
            )
        else:
            self.log_output.appendPlainText(f"[CONVERT] gguf package installed from {tag}.")

        return dest

    @staticmethod
    def _check_conversion_deps() -> list[str]:
        """Return list of pip package names that are missing.

        Note: gguf is NOT checked here because it is installed directly
        from the llama.cpp GitHub repo when the convert script is
        downloaded (see _download_convert_script).  The PyPI version
        is almost always out of date relative to the convert script.
        """
        required = {
            "numpy": "numpy",
            "torch": "torch",
            "safetensors": "safetensors",
            "transformers": "transformers",
            "sentencepiece": "sentencepiece",
        }
        missing: list[str] = []
        for module_name, pip_name in required.items():
            try:
                __import__(module_name)
            except ImportError:
                missing.append(pip_name)
        # Always check gguf is importable (it should have been installed
        # from git by _download_convert_script, but might be missing on
        # first run if the user supplied their own convert script).
        try:
            __import__("gguf")
        except ImportError:
            missing.append("gguf@git+https://github.com/ggml-org/llama.cpp#subdirectory=gguf-py")
        return missing

    def _install_conversion_deps(self, packages: list[str]) -> bool:
        """Install pip packages. Returns True on success."""
        self.log_output.appendPlainText(f"[CONVERT] Installing: {' '.join(packages)}")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "--quiet"] + packages,
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
            if result.returncode != 0:
                error = result.stderr.strip() or result.stdout.strip()
                QMessageBox.critical(self, "Install Failed", f"pip install failed:\n{error}")
                return False
            self.log_output.appendPlainText("[CONVERT] Dependencies installed successfully.")
            return True
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Install Failed", f"pip install error:\n{exc}")
            return False

    def _conversion_worker(
        self,
        repo_id: str,
        files: list[tuple[str, int]],
        token: str,
        output_type: str,
        convert_script_path: str,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> Path:
        """Background worker: downloads HF model files then converts to GGUF."""
        cb = progress_callback or (lambda _: None)

        # ── Phase 1: Download via snapshot_download ────────────────────
        staging_dir = MODELS_DIR / f"_converting_{repo_id.replace('/', '_')}"
        staging_dir.mkdir(parents=True, exist_ok=True)
        total_bytes = sum(size for _, size in files)

        cb({
            "phase": "download",
            "status": (
                f"Downloading {repo_id} "
                f"(~{self._format_bytes(total_bytes)})…"
            ),
            "percent": -1,
        })

        if self._conversion_cancelled:
            raise RuntimeError("Cancelled by user.")

        # Ticker thread sends elapsed-time status updates while
        # snapshot_download blocks the worker thread.
        download_done = threading.Event()
        download_start = time.monotonic()

        def _status_ticker() -> None:
            while not download_done.wait(2.0):
                elapsed = time.monotonic() - download_start
                mins, secs = divmod(int(elapsed), 60)
                time_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
                cb({
                    "phase": "download",
                    "status": (
                        f"Downloading {repo_id} "
                        f"(~{self._format_bytes(total_bytes)}) — "
                        f"{time_str} elapsed"
                    ),
                    "percent": -1,
                })

        ticker = threading.Thread(target=_status_ticker, daemon=True)
        ticker.start()

        try:
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[
                    "*.safetensors", "*.bin", "*.pt", "*.pth",
                    "*.json", "*.model", "*.txt", "*.tiktoken",
                ],
                ignore_patterns=[
                    "*.md", "*.gitattributes", "*.msgpack",
                    "optimizer*", "training_args*", "scheduler*",
                    "onnx/*", "flax_model*", "tf_model*",
                    "coreml/*", "*.ot",
                ],
                token=token or None,
                local_dir=str(staging_dir),
            )
        except Exception as exc:
            raise RuntimeError(f"Download failed: {exc}") from exc
        finally:
            download_done.set()
            ticker.join(timeout=5)

        if self._conversion_cancelled:
            raise RuntimeError("Cancelled by user.")

        cb({"phase": "download", "status": "Download complete.", "percent": 100})

        # ── Phase 2: Convert ───────────────────────────────────────────
        safe_name = repo_id.replace("/", "_")
        output_filename = f"{safe_name}-{output_type}.gguf"
        output_path = MODELS_DIR / output_filename

        cmd = [
            sys.executable,
            convert_script_path,
            str(staging_dir),
            "--outfile", str(output_path),
            "--outtype", output_type,
        ]

        cb({
            "phase": "convert",
            "status": "Starting GGUF conversion…",
            "percent": 0,
            "log": f"> {' '.join(cmd)}",
        })

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines: list[str] = []
        conv_started = time.monotonic()
        while True:
            if self._conversion_cancelled:
                process.kill()
                raise RuntimeError("Cancelled by user.")
            line = process.stdout.readline()  # type: ignore[union-attr]
            if not line and process.poll() is not None:
                break
            if line:
                stripped = line.rstrip()
                output_lines.append(stripped)
                pct = self._parse_convert_progress(stripped, conv_started)
                payload: dict[str, Any] = {"phase": "convert", "log": stripped}
                if pct >= 0:
                    payload["percent"] = pct
                    payload["status"] = stripped
                cb(payload)

        return_code = process.wait()
        if return_code != 0:
            error_tail = "\n".join(output_lines[-30:])
            raise RuntimeError(
                f"Conversion failed (exit code {return_code}):\n\n{error_tail}"
            )

        if not output_path.exists():
            raise RuntimeError(f"Conversion finished but output not found: {output_path}")

        # Clean up staging directory
        try:
            shutil.rmtree(staging_dir)
        except OSError:
            pass

        cb({"phase": "convert", "status": "Conversion complete!", "percent": 100})
        return output_path

    @staticmethod
    def _parse_convert_progress(line: str, _start_time: float) -> int:
        """Try to extract a progress percentage from convert script output."""
        match = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
        if match:
            return min(99, int(float(match.group(1))))
        lower = line.lower()
        if "loading" in lower:
            return 10
        if "converting" in lower or "processing" in lower:
            return 30
        if "writing" in lower:
            return 70
        if "done" in lower or "success" in lower or "complete" in lower:
            return 95
        return -1

    def _on_conversion_progress(self, payload: dict[str, Any]) -> None:
        dialog = self._conversion_dialog
        if not dialog:
            return

        phase = payload.get("phase", "")
        status = payload.get("status", "")
        percent = payload.get("percent", -1)
        log_line = payload.get("log", "")
        eta = payload.get("eta", -1.0)

        if phase and phase != self._conversion_current_phase:
            self._conversion_current_phase = phase
            if phase == "download":
                dialog.set_phase("Phase 1/2 — Downloading Model Files")
            elif phase == "convert":
                dialog.set_phase("Phase 2/2 — Converting to GGUF")

        if percent == -1:
            # Indeterminate / pulsing progress bar (used during snapshot_download)
            if dialog.progress_bar.maximum() != 0:
                dialog.progress_bar.setRange(0, 0)
            if status:
                dialog.set_status(status)
            dialog.eta_label.setText("")
        elif percent >= 0:
            if dialog.progress_bar.maximum() == 0:
                dialog.progress_bar.setRange(0, 100)
            dialog.set_progress(percent, status)
        elif status:
            dialog.set_status(status)

        if eta >= 0:
            dialog.set_eta(eta)

        if log_line:
            dialog.append_log(log_line)

    def _on_conversion_complete(self, output_path: Path) -> None:
        dialog = self._conversion_dialog
        if dialog:
            dialog.mark_complete(str(output_path.resolve()))

        # Set the converted model path in the target server slot
        target_index = self._resolve_server_index(None)
        self.server_slots[target_index].model_path_input.setText(str(output_path.resolve()))
        self._save_config()
        self.log_output.appendPlainText(f"[CONVERT] GGUF saved: {output_path.resolve()}")
        self.convert_to_gguf_button.setEnabled(True)

    def _on_conversion_error(self, message: str) -> None:
        dialog = self._conversion_dialog
        if dialog:
            dialog.mark_failed(message)
        self.log_output.appendPlainText(f"[CONVERT] Failed: {message}")
        self.convert_to_gguf_button.setEnabled(True)

    # ------------------------------------------------------------------ #
    #  Ollama pull / extract                                               #
    # ------------------------------------------------------------------ #

    def _ollama_pull_model(self) -> None:
        """Pull a model from the Ollama registry via `ollama pull`."""
        model_name = self.ollama_pull_input.text().strip()
        if not model_name:
            QMessageBox.warning(self, "No Model Name", "Enter a model name first, e.g. llama3:8b")
            return

        ollama_exe = shutil.which("ollama")
        if not ollama_exe:
            QMessageBox.critical(
                self,
                "Ollama Not Found",
                "Could not find 'ollama' on PATH.\n\n"
                "Install Ollama from https://ollama.com and make sure it's on your PATH.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Pull Ollama Model",
            f"Pull model '{model_name}' from the Ollama registry?\n\n"
            f"This will download the model into Ollama's local storage.\n"
            f"You can then extract the GGUF file to the models/ folder.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.ollama_pull_button.setEnabled(False)
        self.ollama_pull_progress.setRange(0, 0)  # indeterminate
        self.ollama_pull_status.setText(f"Pulling {model_name}…")

        worker = Worker(self._ollama_pull_worker, ollama_exe, model_name, use_progress=True)
        worker.signals.progress.connect(self._on_ollama_pull_progress)
        worker.signals.finished.connect(self._on_ollama_pull_complete)
        worker.signals.error.connect(self._on_ollama_pull_error)
        self.thread_pool.start(worker)

    def _ollama_pull_worker(
        self,
        ollama_exe: str,
        model_name: str,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        """Run `ollama pull <model>` and emit progress."""
        cb = progress_callback or (lambda _: None)
        process = subprocess.Popen(
            [ollama_exe, "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        last_status = ""
        while True:
            line = process.stdout.readline()  # type: ignore[union-attr]
            if not line and process.poll() is not None:
                break
            if line:
                stripped = line.strip()
                if not stripped:
                    continue
                last_status = stripped
                # Try to parse percentage from ollama output, e.g. "pulling abc123… 45%"
                pct_match = re.search(r"(\d+)%", stripped)
                if pct_match:
                    cb({"status": stripped, "percent": int(pct_match.group(1))})
                else:
                    cb({"status": stripped, "percent": -1})

        rc = process.wait()
        if rc != 0:
            raise RuntimeError(f"ollama pull failed (exit {rc}): {last_status}")
        return model_name

    def _on_ollama_pull_progress(self, payload: dict[str, Any]) -> None:
        status = payload.get("status", "")
        percent = payload.get("percent", -1)
        if percent >= 0:
            self.ollama_pull_progress.setRange(0, 100)
            self.ollama_pull_progress.setValue(percent)
        if status:
            self.ollama_pull_status.setText(status)

    def _on_ollama_pull_complete(self, model_name: str) -> None:
        self.ollama_pull_button.setEnabled(True)
        self.ollama_pull_progress.setRange(0, 100)
        self.ollama_pull_progress.setValue(100)
        self.ollama_pull_status.setText(f"Successfully pulled '{model_name}'.")
        self.log_output.appendPlainText(f"[OLLAMA] Pulled model: {model_name}")
        # Auto-refresh the local model list
        self._ollama_refresh_models()

    def _on_ollama_pull_error(self, message: str) -> None:
        self.ollama_pull_button.setEnabled(True)
        self.ollama_pull_progress.setRange(0, 100)
        self.ollama_pull_progress.setValue(0)
        self.ollama_pull_status.setText("Pull failed.")
        QMessageBox.critical(self, "Ollama Pull Failed", message)

    # ── Refresh / list local models ───────────────────────────────────

    def _ollama_refresh_models(self) -> None:
        """Populate the local Ollama models table via `ollama list`."""
        ollama_exe = shutil.which("ollama")
        if not ollama_exe:
            self.ollama_extract_status.setText("'ollama' not found on PATH.")
            return

        self.ollama_refresh_button.setEnabled(False)
        self.ollama_extract_status.setText("Loading model list…")

        worker = Worker(self._ollama_list_worker, ollama_exe)
        worker.signals.finished.connect(self._on_ollama_list_complete)
        worker.signals.error.connect(self._on_ollama_list_error)
        self.thread_pool.start(worker)

    @staticmethod
    def _ollama_list_worker(ollama_exe: str) -> list[dict[str, str]]:
        result = subprocess.run(
            [ollama_exe, "list"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ollama list failed")

        lines = result.stdout.strip().splitlines()
        if not lines:
            return []

        # First line is header: NAME  ID  SIZE  MODIFIED
        models: list[dict[str, str]] = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 4:
                continue
            name = parts[0]
            model_id = parts[1]
            # Size is typically "4.9 GB" (two tokens)
            size_str = f"{parts[2]} {parts[3]}" if len(parts) > 3 else parts[2]
            modified = " ".join(parts[4:]) if len(parts) > 4 else ""
            models.append({
                "name": name,
                "id": model_id,
                "size": size_str,
                "modified": modified,
            })
        return models

    def _on_ollama_list_complete(self, models: list[dict[str, str]]) -> None:
        self.ollama_refresh_button.setEnabled(True)
        self.ollama_models_table.setRowCount(0)
        for row_idx, model in enumerate(models):
            self.ollama_models_table.insertRow(row_idx)
            self.ollama_models_table.setItem(row_idx, 0, QTableWidgetItem(model["name"]))
            self.ollama_models_table.setItem(row_idx, 1, QTableWidgetItem(model["id"]))
            self.ollama_models_table.setItem(row_idx, 2, QTableWidgetItem(model["size"]))
            self.ollama_models_table.setItem(row_idx, 3, QTableWidgetItem(model["modified"]))
        self.ollama_extract_button.setEnabled(False)
        self.ollama_extract_status.setText(f"{len(models)} model(s) found. Select one to extract.")

    def _on_ollama_list_error(self, message: str) -> None:
        self.ollama_refresh_button.setEnabled(True)
        self.ollama_extract_status.setText(f"Failed: {message}")

    # ── Extract GGUF blob ─────────────────────────────────────────────

    def _ollama_extract_gguf(self) -> None:
        """Extract the GGUF blob for the selected Ollama model."""
        row = self.ollama_models_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "No Model Selected", "Select a model from the list first.")
            return

        model_name = self.ollama_models_table.item(row, 0).text()
        ollama_exe = shutil.which("ollama")
        if not ollama_exe:
            QMessageBox.critical(self, "Ollama Not Found", "'ollama' not found on PATH.")
            return

        self.ollama_extract_button.setEnabled(False)
        self.ollama_extract_status.setText(f"Resolving blob hash for {model_name}…")

        worker = Worker(self._ollama_extract_worker, ollama_exe, model_name)
        worker.signals.finished.connect(self._on_ollama_extract_complete)
        worker.signals.error.connect(self._on_ollama_extract_error)
        self.thread_pool.start(worker)

    @staticmethod
    def _ollama_extract_worker(ollama_exe: str, model_name: str) -> dict[str, str]:
        """Resolve blob hash via `ollama show --modelfile`, find blob, copy to models/."""
        result = subprocess.run(
            [ollama_exe, "show", model_name, "--modelfile"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ollama show failed:\n{result.stderr.strip() or result.stdout.strip()}"
            )

        # Parse the FROM line to get either a direct blob path or a blob hash
        blob_hash = None
        blob_path: Path | None = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.upper().startswith("FROM"):
                value = line.split(None, 1)[-1].strip().strip('"')
                candidate_path = Path(value)

                # Value may already be a full blob path, e.g.
                # C:\Users\...\.ollama\models\blobs\sha256-...
                if candidate_path.is_absolute() or candidate_path.exists():
                    blob_path = candidate_path
                    break

                # Otherwise it may be a blob reference like sha256:abc123...
                if value.lower().startswith("sha256:") or value.lower().startswith("sha256-"):
                    blob_hash = value.replace(":", "-")
                    break

        if not blob_hash and blob_path is None:
            raise RuntimeError(
                f"Could not find a FROM sha256:… line in modelfile for '{model_name}'.\n\n"
                f"Output:\n{result.stdout[:500]}"
            )

        # Locate the actual blob file
        if blob_path is None:
            blob_path = OLLAMA_BLOBS_DIR / blob_hash
            if not blob_path.exists():
                # Try with sha256- prefix format
                alt = blob_hash.replace("sha256-", "sha256:")
                alt_path = OLLAMA_BLOBS_DIR / alt
                if alt_path.exists():
                    blob_path = alt_path

        if not blob_path.exists():
            raise RuntimeError(
                f"Blob file not found.\n\n"
                f"Looked for: {blob_path}\n"
                f"Blobs dir: {OLLAMA_BLOBS_DIR}\n\n"
                f"If your OLLAMA_MODELS env var points elsewhere, set it before launching."
            )

        # Copy to models/ with a .gguf name
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", model_name)
        dest = MODELS_DIR / f"{safe_name}.gguf"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # If file already exists, add a suffix
        if dest.exists():
            stem = dest.stem
            for i in range(1, 100):
                candidate = MODELS_DIR / f"{stem}_{i}.gguf"
                if not candidate.exists():
                    dest = candidate
                    break

        shutil.copy2(blob_path, dest)
        return {"model_name": model_name, "path": str(dest.resolve()), "size": blob_path.stat().st_size}

    def _on_ollama_extract_complete(self, result: dict[str, str]) -> None:
        model_name = result["model_name"]
        dest_path = result["path"]
        size = int(result.get("size", 0) or 0)
        size_text = self._format_bytes(size) if size else "unknown size"

        self.ollama_extract_button.setEnabled(True)
        self.ollama_extract_status.setText(
            f"Extracted {model_name} → {dest_path} ({size_text})"
        )

        # Set the extracted model path on the selected target server slot
        target_index = self._resolve_server_index(None)
        self.server_slots[target_index].model_path_input.setText(dest_path)
        self._save_config()
        self.log_output.appendPlainText(
            f"[OLLAMA] Extracted GGUF: {model_name} → {dest_path} ({size_text})"
        )

        QMessageBox.information(
            self,
            "GGUF Extracted",
            f"Model: {model_name}\n"
            f"Saved to: {dest_path}\n"
            f"Size: {size_text}\n\n"
            f"Model path has been set on Server {target_index + 1}.",
        )

    def _on_ollama_extract_error(self, message: str) -> None:
        self.ollama_extract_button.setEnabled(True)
        self.ollama_extract_status.setText("Extraction failed.")
        QMessageBox.critical(self, "Extraction Failed", message)

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
        host = normalize_connect_host(slot.host_input.text().strip() or "127.0.0.1")
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
                    "host": normalize_connect_host(slot.host_input.text().strip() or "127.0.0.1"),
                    "port": slot.port_input.value(),
                    "model_path": slot.model_path_input.text().strip(),
                    "backend": backend or "",
                    "running": bool(self.server_running_states[idx]) if idx < len(self.server_running_states) else False,
                }
            )

        snapshot = {
            "default_server": self.ollama_default_server_selector.currentIndex(),
            "proxy_num_predict": self.proxy_num_predict_input.value(),
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

    # ------------------------------------------------------------------ #
    #  Auto-start on app launch                                           #
    # ------------------------------------------------------------------ #
    def _auto_start_servers(self) -> None:
        """Start any slots marked 'Auto-Start' and the proxy if needed."""
        started_any = False
        for slot in self.server_slots:
            if slot.auto_start_checkbox.isChecked():
                model_path = slot.model_path_input.text().strip()
                if model_path and Path(model_path).exists():
                    self.log_output.appendPlainText(
                        f"[AUTO] Starting Server {slot.index + 1} (auto-start enabled)"
                    )
                    self.start_server(slot.index)
                    started_any = True
                else:
                    self.log_output.appendPlainText(
                        f"[AUTO] Skipped Server {slot.index + 1}: model file not found."
                    )
        if started_any:
            self.log_output.appendPlainText("[AUTO] Starting Ollama proxy …")
            self.start_ollama_proxy()

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
        host = normalize_connect_host(self.ollama_host_input.text().strip() or "127.0.0.1")
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
        if gpu_ids:
            arguments.extend(["--n-gpu-layers", "999"])
        if len(gpu_ids) > 1 and slot.split_mode_input.currentText() == "pooled":
            arguments.extend(["--main-gpu", "0", "--tensor-split", ",".join(["1"] * len(gpu_ids))])

        cache_k = slot.cache_type_k_input.currentText()
        cache_v = slot.cache_type_v_input.currentText()
        if cache_k != "f16":
            arguments.extend(["--cache-type-k", cache_k])
        if cache_v != "f16":
            arguments.extend(["--cache-type-v", cache_v])
        if slot.flash_attn_checkbox.isChecked():
            arguments.append("-fa")
        if slot.no_cache_prompt_checkbox.isChecked():
            arguments.append("--no-cache-prompt")

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
        self._model_compat_warning_shown.discard(index)
        slot.status_label.setText("Starting server…")
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
            self._maybe_report_model_compatibility_issue(index, line)

    def _maybe_report_model_compatibility_issue(self, index: int, line: str) -> None:
        lower = line.lower()
        mismatch = (
            "error loading model hyperparameters" in lower
            or "wrong array length" in lower
            or "qwen35.rope.dimension_sections" in lower
        )
        if not mismatch:
            return

        if index in self._model_compat_warning_shown:
            return
        self._model_compat_warning_shown.add(index)

        slot = self.server_slots[index]
        model_path = slot.model_path_input.text().strip() or "(unknown model path)"
        message = (
            "Model/runtime compatibility issue detected.\n\n"
            "The GGUF appears to use metadata your current llama-server build does not support "
            "(or expects in a different format).\n\n"
            f"Model: {model_path}\n\n"
            "Fix options:\n"
            "1) Update the selected backend runtime to a newer llama.cpp build.\n"
            "2) Re-convert/re-download this model using the same llama.cpp release family as your runtime.\n"
            "3) Try a different GGUF quant/model variant known to load on your runtime."
        )
        slot.status_label.setText("Model/runtime mismatch (see Server Log).")
        self.log_output.appendPlainText(f"[S{index + 1}] compatibility hint: {message.replace(chr(10), ' ')}")
        QMessageBox.warning(self, f"Server {index + 1} Model Compatibility", message)

    def _handle_process_error(self, index: int, _error: QProcess.ProcessError) -> None:
        slot = self.server_slots[index]
        self._append_server_output(index)

        error_names = {
            QProcess.ProcessError.FailedToStart: "FailedToStart",
            QProcess.ProcessError.Crashed: "Crashed",
            QProcess.ProcessError.Timedout: "Timedout",
            QProcess.ProcessError.WriteError: "WriteError",
            QProcess.ProcessError.ReadError: "ReadError",
            QProcess.ProcessError.UnknownError: "UnknownError",
        }
        error_name = error_names.get(_error, str(int(_error)))
        slot.status_label.setText(f"Server process error: {error_name}")
        self.log_output.appendPlainText(
            f"[S{index + 1}] process error: {error_name}"
        )

    def _handle_process_finished(
        self,
        index: int,
        exit_code: int,
        exit_status: QProcess.ExitStatus,
    ) -> None:
        self._append_server_output(index)
        self._set_server_state(index, False)

        status_name = "CrashExit" if exit_status == QProcess.ExitStatus.CrashExit else "NormalExit"
        self.log_output.appendPlainText(
            f"[S{index + 1}] process exited: code={exit_code}, status={status_name}"
        )

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

    # ------------------------------------------------------------------ #
    #  System tray + close-to-tray                                        #
    # ------------------------------------------------------------------ #
    def _build_tray_icon(self) -> None:
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setToolTip(APP_NAME)

        # Use a simple built-in icon; works on all platforms.
        icon = self.style().standardIcon(self.style().StandardPixmap.SP_ComputerIcon)
        self.tray_icon.setIcon(icon)
        self.setWindowIcon(icon)

        tray_menu = QMenu(self)
        show_action = QAction("Show / Restore", self)
        show_action.triggered.connect(self._tray_restore)
        tray_menu.addAction(show_action)

        tray_menu.addSeparator()
        quit_action = QAction("Quit (stop all servers)", self)
        quit_action.triggered.connect(self._tray_quit)
        tray_menu.addAction(quit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self._tray_activated)
        self.tray_icon.show()

    def _tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:  # single-click
            self._tray_restore()

    def _tray_restore(self) -> None:
        self.showNormal()
        self.activateWindow()
        self.raise_()

    def _tray_quit(self) -> None:
        """Actually quit — kill servers, proxy, and exit."""
        self._really_quit = True
        self.close()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        # If the user chose "Quit" from the tray, or no servers are running,
        # do a real exit.  Otherwise minimize to tray.
        any_running = any(
            slot.process.state() != QProcess.NotRunning for slot in self.server_slots
        )
        if getattr(self, "_really_quit", False) or not any_running:
            self.tray_icon.hide()
            self.stop_ollama_proxy()
            self._save_config()
            self.sakura_timer.stop()
            self.sakura_monitor.shutdown()
            self.sakura_lhm.shutdown()
            for slot in self.server_slots:
                if slot.process.state() != QProcess.NotRunning:
                    slot.process.kill()
                    slot.process.waitForFinished(2000)
            super().closeEvent(event)
        else:
            event.ignore()
            self.hide()
            self.tray_icon.showMessage(
                APP_NAME,
                "Servers still running — minimized to tray.\n"
                "Right-click the tray icon to quit.",
                QSystemTrayIcon.MessageIcon.Information,
                3000,
            )


def main() -> int:
    os.makedirs(MODELS_DIR, exist_ok=True)
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # keep alive when minimized to tray
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
