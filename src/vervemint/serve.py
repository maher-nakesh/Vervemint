"""Run the backend and the web UI in one container.

Docker Compose gives each of them its own container, but a single
container host (Back4App, Render, Railway, Cloud Run, Fly) runs one image
and routes one port to it. This starts:

- the backend on 127.0.0.1:8000, reachable only inside the container
- the web UI on $PORT (8501 if the host does not set one)

and stops both as soon as either exits, so the host notices and restarts
the container instead of serving a half-dead app.

    python -m src.vervemint.serve

The backend needs about 1 GB of memory for the retrieval models, plus
what the answers use: give the container at least 2 GB, or it is killed
while loading and nothing ever listens.
"""

import logging
import os
import signal
import subprocess
import sys
import time

from src.vervemint.logging_config import setup_logging

logger = logging.getLogger(__name__)

API_PORT = "8000"
_POLL_SECONDS = 1


def commands(ui_port: str) -> dict[str, list[str]]:
    """What to run: the backend inside, the web UI facing the host."""
    return {
        "api": [
            sys.executable, "-m", "uvicorn", "src.vervemint.api:app",
            "--host", "127.0.0.1", "--port", API_PORT, "--workers", "1",
            "--no-access-log", "--timeout-graceful-shutdown", "30",
        ],
        "ui": [
            sys.executable, "-m", "streamlit", "run", "ui/app.py",
            f"--server.port={ui_port}", "--server.address=0.0.0.0",
            "--server.headless=true",
        ],
    }


def _stop(running: dict[str, subprocess.Popen]) -> None:
    for name, process in running.items():
        if process.poll() is None:
            logger.info("Stopping %s", name)
            process.terminate()
    for process in running.values():
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> None:
    setup_logging()
    ui_port = os.environ.get("PORT", "8501")
    # The UI talks to the backend inside this container.
    os.environ.setdefault("VERVEMINT_API_URL", f"http://127.0.0.1:{API_PORT}")
    logger.info("Starting Vervemint: web UI on port %s, backend on %s",
                ui_port, API_PORT)

    processes = {name: subprocess.Popen(command)
                 for name, command in commands(ui_port).items()}
    stopping = False

    def _on_signal(*_: object) -> None:
        nonlocal stopping
        stopping = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _on_signal)

    while not stopping:
        for name, process in processes.items():
            code = process.poll()
            if code is not None:
                logger.error("%s exited with code %s; stopping the container",
                             name, code)
                _stop(processes)
                sys.exit(code or 1)
        time.sleep(_POLL_SECONDS)
    _stop(processes)


if __name__ == "__main__":
    main()
