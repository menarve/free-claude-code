"""Shared process helpers for installed client CLI launchers."""

import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from free_claude_code.cli.process_registry import (
    kill_pid_tree_best_effort,
    register_pid,
    unregister_pid,
)
from free_claude_code.config.paths import server_log_path

PROXY_PREFLIGHT_PATH = "/health"
PROXY_PREFLIGHT_TIMEOUT_SECONDS = 1.5
SERVER_AUTOSTART_TIMEOUT_SECONDS = 20.0
SERVER_AUTOSTART_POLL_SECONDS = 0.25


def preflight_proxy(proxy_root_url: str) -> str | None:
    """Return an error message when the local proxy health check is unreachable."""

    url = f"{proxy_root_url.rstrip('/')}{PROXY_PREFLIGHT_PATH}"
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=PROXY_PREFLIGHT_TIMEOUT_SECONDS) as response:
            status_code = response.getcode()
    except HTTPError as exc:
        return f"returned HTTP {exc.code}"
    except URLError as exc:
        return str(exc.reason)
    except OSError as exc:
        return str(exc)

    if not 200 <= status_code < 300:
        return f"returned HTTP {status_code}"
    return None


def ensure_server_running(
    proxy_root_url: str,
    *,
    startup_timeout_seconds: float = SERVER_AUTOSTART_TIMEOUT_SECONDS,
) -> str | None:
    """Start `fcc-server` in the background when the local proxy is unreachable.

    Returns an error message if the proxy is still unreachable after trying to
    start it; ``None`` once the health check succeeds. The spawned server is
    detached (its own session) so it keeps running after this client exits,
    matching how a user would run `fcc-server` in another terminal by hand.
    """

    error = preflight_proxy(proxy_root_url)
    if error is None:
        return None

    server_binary = shutil.which("fcc-server")
    if server_binary is None:
        return f"proxy unreachable ({error}) and the fcc-server executable was not found"

    print(
        f"Free Claude Code proxy is not running at {proxy_root_url}; starting fcc-server...",
        file=sys.stderr,
    )

    log_path = server_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        subprocess.Popen(
            [server_binary],
            stdout=log_file,
            stderr=log_file,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    deadline = time.monotonic() + startup_timeout_seconds
    while time.monotonic() < deadline:
        if preflight_proxy(proxy_root_url) is None:
            print(f"fcc-server is up at {proxy_root_url}", file=sys.stderr)
            return None
        time.sleep(SERVER_AUTOSTART_POLL_SECONDS)

    return (
        f"fcc-server did not become ready within {startup_timeout_seconds:g}s "
        f"(see {log_path} for details)"
    )


def resolve_client_binary(
    *,
    binary_name: str,
    display_name: str,
    install_hint: str,
) -> str:
    """Resolve an installed client binary or exit with a user-facing hint."""

    client_command = shutil.which(binary_name)
    if client_command is None:
        print(
            f"Could not find {display_name} command: {binary_name}",
            file=sys.stderr,
        )
        print(install_hint, file=sys.stderr)
        raise SystemExit(127)
    return client_command


def run_client_process(
    *,
    command: list[str],
    env: Mapping[str, str],
    binary_name: str,
    display_name: str,
    install_hint: str,
) -> None:
    """Run a client CLI command and mirror its exit code."""

    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(command, env=dict(env))
        if process.pid:
            register_pid(process.pid)
        return_code = process.wait()
    except FileNotFoundError:
        print(
            f"Could not find {display_name} command: {binary_name}",
            file=sys.stderr,
        )
        print(install_hint, file=sys.stderr)
        raise SystemExit(127) from None
    except KeyboardInterrupt:
        if process is not None and process.pid:
            kill_pid_tree_best_effort(process.pid)
            process.wait()
        raise
    finally:
        if process is not None and process.pid:
            unregister_pid(process.pid)

    raise SystemExit(return_code)
