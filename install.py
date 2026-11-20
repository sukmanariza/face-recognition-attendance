"""One-shot setup for the face recognition app.

    python install.py

What it does, in order:

1. checks the Python version (3.11 or newer is required by the pinned wheels);
2. creates a virtual environment in ``.venv`` (skipped if it already exists);
3. upgrades pip inside it and installs ``requirements.txt``;
4. downloads the InsightFace model pack if it is not on disk yet, with progress;
5. creates ``database/`` and a ``settings.json`` skeleton;
6. runs a quick self-check (imports, database, configuration) and prints the
   commands to start the app.

It is idempotent: running it again only does the steps that are still missing. It
never installs system packages and never touches anything outside this folder.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import venv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
DATABASE_DIR = PROJECT_ROOT / "database"
SETTINGS_FILE = PROJECT_ROOT / "settings.json"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
MIN_PYTHON = (3, 11)
STEP_OK = "[ ok ]"
STEP_SKIP = "[skip]"
STEP_FAIL = "[fail]"


def venv_python() -> Path:
    """Path to the interpreter inside the virtual environment."""
    if sys.platform.startswith("win"):
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def check_python() -> bool:
    if sys.version_info < MIN_PYTHON:
        required = ".".join(str(part) for part in MIN_PYTHON)
        print(
            f"{STEP_FAIL} Python {sys.version_info.major}.{sys.version_info.minor} is too old. "
            f"{required} or newer is required."
        )
        return False
    print(f"{STEP_OK} Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    return True


def ensure_venv() -> bool:
    if venv_python().is_file():
        print(f"{STEP_SKIP} Virtual environment already exists at {VENV_DIR}")
        return True
    print(f"{STEP_OK} Creating a virtual environment at {VENV_DIR}")
    try:
        venv.EnvBuilder(with_pip=True, clear=False, upgrade_deps=False).create(VENV_DIR)
    except OSError as exc:
        print(f"{STEP_FAIL} Could not create the virtual environment: {exc}")
        return False
    return venv_python().is_file()


def run_pip(args: list[str]) -> bool:
    command = [str(venv_python()), "-m", "pip", *args]
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return result.returncode == 0


def install_requirements() -> bool:
    if not REQUIREMENTS_FILE.is_file():
        print(f"{STEP_FAIL} requirements.txt is missing from {PROJECT_ROOT}")
        return False
    print(f"{STEP_OK} Upgrading pip")
    run_pip(["install", "--upgrade", "pip", "--quiet"])
    drop_conflicting_runtime()
    print(f"{STEP_OK} Installing dependencies (this downloads about 200 MB once)")
    if not run_pip(["install", "-r", str(REQUIREMENTS_FILE)]):
        print(f"{STEP_FAIL} Dependency installation failed. Re-run this script to retry.")
        return False
    return True


def drop_conflicting_runtime() -> None:
    """Remove the other onnxruntime build before installing this platform's one.

    Every onnxruntime build installs the same ``onnxruntime`` package directory, so
    having two of them at once leaves a mix of files that fails in strange ways.
    """
    wanted = "onnxruntime-directml" if sys.platform == "win32" else "onnxruntime"
    other = "onnxruntime" if sys.platform == "win32" else "onnxruntime-directml"
    installed = subprocess.run(
        [str(venv_python()), "-m", "pip", "show", "--name-only", other],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if installed.returncode != 0:
        return
    print(f"{STEP_OK} Removing {other} so {wanted} can provide the onnxruntime module")
    run_pip(["uninstall", "-y", other, "--quiet"])


def download_model() -> bool:
    """Fetch the InsightFace pack via the application's own downloader."""
    try:
        from config import load_config
        from src.face_detector import ModelLoadError, ensure_model_available
    except ImportError as exc:
        print(f"{STEP_FAIL} The dependencies are not importable yet: {exc}")
        return False

    config = load_config()
    target = config.model_dir
    if target.is_dir() and any(target.glob("*.onnx")):
        print(f"{STEP_SKIP} Model {config.model.name} is already present at {target}")
        return True

    print(f"{STEP_OK} Downloading model {config.model.name} (about 120 MB, resumable)")

    def progress(downloaded: int, total: int) -> None:
        if total <= 0:
            return
        percent = int(downloaded * 100 / total)
        bar = "#" * (percent // 2)
        sys.stdout.write(f"\r       [{bar:<50}] {percent:3d}%")
        sys.stdout.flush()

    try:
        ensure_model_available(target, config.model.name, progress)
    except ModelLoadError as exc:
        print(f"\n{STEP_FAIL} Model download failed: {exc}")
        print("       Run this script again; the partial download resumes.")
        return False
    print("\n" + STEP_OK + " Model ready")
    return True


def ensure_directories() -> bool:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{STEP_OK} Data folder ready at {DATABASE_DIR}")

    if SETTINGS_FILE.is_file():
        print(f"{STEP_SKIP} settings.json already exists; left untouched")
        return True

    # An empty override file means "use config.yaml plus the hardware profile".
    SETTINGS_FILE.write_text(
        json.dumps({"version": 1}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{STEP_OK} Created {SETTINGS_FILE.name} (edit values in the app's Settings page)")
    return True


def ensure_key() -> bool:
    """Create the database encryption key early, so the first run cannot surprise you."""
    key_file = DATABASE_DIR / "face_key.bin"
    if key_file.is_file():
        print(f"{STEP_SKIP} Database key already exists at {key_file}")
        print("       Back it up now if you have not: without it the faces cannot be read.")
        return True
    try:
        from src.crypto import FieldCipher, passphrase_from_environment
    except ImportError as exc:
        print(f"{STEP_FAIL} The dependencies are not importable yet: {exc}")
        return False

    passphrase = passphrase_from_environment()
    try:
        FieldCipher.from_key_file(key_file, passphrase)
    except Exception as exc:  # noqa: BLE001 - the installer must explain, not trace
        print(f"{STEP_FAIL} Could not create the database key: {exc}")
        return False
    mode = "protected by FACEID_PASSPHRASE" if passphrase else "owner-only file"
    print(f"{STEP_OK} Database key created at {key_file} ({mode})")
    return True


def self_check() -> bool:
    """Import everything, open the database, and report the environment."""
    print(f"{STEP_OK} Running a self-check")
    result = subprocess.run(
        [str(venv_python()), str(PROJECT_ROOT / "main.py"), "status"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0:
        print(f"{STEP_FAIL} Self-check failed. Run `main.py status` to see the error.")
        return False
    return True


def print_next_steps() -> None:
    python = ".\\.venv\\Scripts\\python.exe" if sys.platform.startswith("win") else "./.venv/bin/python"
    print(
        "\nSetup complete.\n"
        "\nNext steps:\n"
        f"  1. Register a face:      {python} main.py register\n"
        f"  2. Open the app:         {python} main.py\n"
        f"  3. Check attendance:     {python} main.py attendance\n"
        f"  4. Back up the database: {python} main.py backup\n"
        "\nThe database key in database/face_key.bin is what makes the stored faces\n"
        "readable. Copy it somewhere safe, separately from the database. Without it,\n"
        "a backup is unreadable and every face has to be registered again.\n"
        "\nOn a Raspberry Pi or similar board, `pip install PySide6-Essentials` may not\n"
        "have a wheel for your system. The CLI commands above work without Qt.\n"
    )


def main() -> int:
    print(f"\nFace ID installer\nProject: {PROJECT_ROOT}\n")
    if not check_python():
        return 2
    if not ensure_venv():
        return 1
    if not install_requirements():
        return 1
    if not ensure_directories():
        return 1
    if not ensure_key():
        return 1
    if not download_model():
        return 1
    if not self_check():
        return 1
    print_next_steps()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
