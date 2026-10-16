"""Start the local server detached from the launching terminal."""
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

root = Path(__file__).resolve().parent
port = os.environ.get("PORT", "8000")
with (root / "server.log").open("ab") as log:
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", port, "--workers", "1"],
        cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
for _ in range(40):
    time.sleep(0.25)
    if process.poll() is not None:
        raise SystemExit("Server failed to start. Check server.log.")
    try:
        with urlopen(f"http://127.0.0.1:{port}/", timeout=1) as response:
            if response.status == 200:
                (root / "server.pid").write_text(str(process.pid))
                print(f"Running at http://localhost:{port} (PID {process.pid}); logs: {root / 'server.log'}")
                break
    except OSError:
        pass
else:
    raise SystemExit("Startup did not complete within 10 seconds. Check server.log.")
