from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

BACKENDS = ("cuda", "hip", "vulkan", "cpu")
GITHUB_RELEASES_PAGE = "https://github.com/ggml-org/llama.cpp/releases"
GITHUB_LATEST_RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

BACKEND_FOLDERS = {
    "cuda": "llama.cpp",
    "hip": "hip-llama",
    "vulkan": "vulkan-llama",
    "cpu": "cpu-llama",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch llama.cpp runtime binaries if missing.")
    parser.add_argument("--backend", required=True, choices=BACKENDS)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--resolve-only", action="store_true", help="Only resolve and print the download URL.")
    return parser.parse_args()


def candidates_for_backend(backend: str, workspace: Path) -> list[Path]:
    folder = workspace / BACKEND_FOLDERS[backend]
    if os.name == "nt":
        names = [
            folder / "llama-server.exe",
            folder / "bin" / "llama-server.exe",
            folder / "build" / "bin" / "Release" / "llama-server.exe",
            folder / "build" / "bin" / "llama-server.exe",
        ]
    else:
        names = [
            folder / "llama-server",
            folder / "bin" / "llama-server",
            folder / "build" / "bin" / "Release" / "llama-server",
            folder / "build" / "bin" / "llama-server",
        ]
    return names


def detect_existing(backend: str, workspace: Path) -> Path | None:
    for candidate in candidates_for_backend(backend, workspace):
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    folder = workspace / BACKEND_FOLDERS[backend]
    if not folder.exists():
        return None

    target = "llama-server.exe" if os.name == "nt" else "llama-server"
    matches = [path for path in folder.rglob(target) if path.is_file()]
    if not matches:
        return None

    newest = max(matches, key=lambda path: path.stat().st_mtime)
    return newest.resolve()


def load_runtime_urls(workspace: Path) -> dict[str, str]:
    urls: dict[str, str] = {}

    config_path = workspace / "frontend_config.json"
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        runtime_urls = data.get("runtime_binary_urls", {})
        if isinstance(runtime_urls, dict):
            for backend in BACKENDS:
                value = runtime_urls.get(backend)
                if isinstance(value, str) and value.strip():
                    urls[backend] = value.strip()

    for backend in BACKENDS:
        env_name = f"DOLLAMA_{backend.upper()}_RUNTIME_URL"
        value = os.environ.get(env_name, "").strip()
        if value:
            urls[backend] = value

    return urls


def _fetch_latest_assets() -> list[dict[str, str]]:
    request = urllib.request.Request(
        GITHUB_LATEST_RELEASE_API,
        headers={
            "User-Agent": "DoLLAMACPP-Frontend",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    assets = payload.get("assets", [])
    result: list[dict[str, str]] = []
    if isinstance(assets, list):
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            name = str(asset.get("name", "")).strip()
            download_url = str(asset.get("browser_download_url", "")).strip()
            if name and download_url:
                result.append({"name": name, "url": download_url})
    return result


def _backend_asset_score(backend: str, name: str) -> int:
    value = name.lower()
    score = 0

    if os.name == "nt":
        if "win" not in value:
            return -1
        score += 20
        if "x64" in value:
            score += 4

    if backend == "cuda":
        if "cuda" not in value:
            return -1
        score += 30
        if re.search(r"cuda[-_.]?12", value):
            score += 50
        if re.search(r"cuda[-_.]?13", value):
            score -= 20
        if "cudart" in value:
            score -= 5
    elif backend == "hip":
        if "hip" not in value and "rocm" not in value:
            return -1
        score += 30
    elif backend == "vulkan":
        if "vulkan" not in value:
            return -1
        score += 30
    elif backend == "cpu":
        if "cpu" not in value:
            return -1
        score += 30
        if any(term in value for term in ("cuda", "hip", "rocm", "vulkan", "opencl", "sycl")):
            score -= 40

    if "llama-b" in value:
        score += 5
    if value.endswith(".zip") or value.endswith(".tar.gz") or value.endswith(".tgz"):
        score += 2

    return score


def _resolve_release_asset_url(backend: str) -> str:
    assets = _fetch_latest_assets()
    best_url = ""
    best_score = -1

    for asset in assets:
        name = asset["name"]
        score = _backend_asset_score(backend, name)
        if score > best_score:
            best_score = score
            best_url = asset["url"]

    return best_url if best_score >= 0 else ""


def _normalize_url_candidate(backend: str, raw_url: str) -> str:
    value = raw_url.strip()
    if not value:
        return ""

    lower = value.lower()
    if lower == GITHUB_RELEASES_PAGE.lower() or lower == GITHUB_LATEST_RELEASE_API.lower() or lower.endswith("/releases"):
        return _resolve_release_asset_url(backend)

    return value


def download_to_temp(url: str) -> Path:
    suffix = Path(url.split("?", 1)[0]).suffix.lower() or ".bin"
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = Path(tmp_file.name)
    tmp_file.close()

    with urllib.request.urlopen(url, timeout=120) as response, tmp_path.open("wb") as output:
        shutil.copyfileobj(response, output)

    return tmp_path


def _strip_archive_suffixes(filename: str) -> str:
    value = filename
    for suffix in (".tar.gz", ".tgz", ".zip", ".tar", ".gz"):
        if value.lower().endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value


def version_folder_name_from_url(url: str, backend: str) -> str:
    name = Path(url.split("?", 1)[0]).name
    stem = _strip_archive_suffixes(name)
    if not stem:
        stem = f"{backend}-runtime"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")
    return safe or f"{backend}-runtime"


def extract_or_copy(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    lower_name = archive_path.name.lower()

    if lower_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(destination)
        return

    if lower_name.endswith(".tar.gz") or lower_name.endswith(".tgz") or lower_name.endswith(".tar"):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(destination)
        return

    target_name = "llama-server.exe" if os.name == "nt" else "llama-server"
    shutil.copy2(archive_path, destination / target_name)


def find_llama_server_recursive(root: Path) -> Path | None:
    target = "llama-server.exe" if os.name == "nt" else "llama-server"
    for path in root.rglob(target):
        if path.is_file():
            return path.resolve()
    return None


def main() -> int:
    args = parse_args()
    backend = args.backend
    workspace = Path(args.workspace).resolve()

    existing = detect_existing(backend, workspace)
    if existing:
        print(str(existing))
        return 0

    urls = load_runtime_urls(workspace)
    configured_url = urls.get(backend, "")
    url = _normalize_url_candidate(backend, configured_url)
    if not url:
        url = _resolve_release_asset_url(backend)

    if args.resolve_only:
        if not url:
            print(f"No matching release asset found for backend '{backend}'.", file=sys.stderr)
            return 4
        print(url)
        return 0

    if not url:
        print(
            f"No runtime URL could be resolved for '{backend}'. "
            "Set frontend_config.json.runtime_binary_urls.<backend> or "
            f"DOLLAMA_{backend.upper()}_RUNTIME_URL, or use {GITHUB_RELEASES_PAGE}.",
            file=sys.stderr,
        )
        return 2

    backend_root = (workspace / BACKEND_FOLDERS[backend]).resolve()
    destination = backend_root / version_folder_name_from_url(url, backend)
    archive_path: Path | None = None

    try:
        archive_path = download_to_temp(url)
        extract_or_copy(archive_path, destination)
        found = find_llama_server_recursive(destination)
        if not found:
            print(f"Downloaded runtime but could not find llama-server for '{backend}'.", file=sys.stderr)
            return 3
        print(str(found))
        return 0
    except Exception as exc:
        print(f"Failed to fetch runtime for '{backend}': {exc}", file=sys.stderr)
        return 1
    finally:
        if archive_path and archive_path.exists():
            try:
                archive_path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
