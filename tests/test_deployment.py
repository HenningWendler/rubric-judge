"""The service as it really runs: a uvicorn process, and the Docker image built from this
repository. The in-process tests in `test_api.py` start neither.

The Docker tests carry the `docker` marker and are deselected by default, because they need a
running Docker daemon and a first build takes about a minute. Run them with
`pytest -m docker`."""

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator

import httpx
import pytest

from tests.conftest import JUDGE_ENVIRONMENT

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_TAG = "rubric-judge:test"
STARTUP_TIMEOUT_SECONDS = 30
BUILD_TIMEOUT_SECONDS = 600


# --- uvicorn, the way the README starts the service ------------------------------------------


def test_uvicorn_refuses_to_start_without_a_configured_judge():
    refused_start = subprocess.run(
        _uvicorn_command(_free_port()),
        env=_environment_without_judge(),
        capture_output=True,
        text=True,
        timeout=STARTUP_TIMEOUT_SECONDS,
    )

    assert refused_start.returncode != 0
    _assert_names_every_missing_variable(refused_start.stderr)


def test_uvicorn_serves_once_the_judge_is_configured():
    port = _free_port()
    with subprocess.Popen(
        _uvicorn_command(port), env=_environment_without_judge() | JUDGE_ENVIRONMENT
    ) as server:
        try:
            assert _health_once_up(f"http://127.0.0.1:{port}/health") == {"status": "ok"}
        finally:
            server.terminate()


# --- the Docker image ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def image() -> str:
    """The image built from this checkout, once for all Docker tests in the module."""
    subprocess.run(
        ["docker", "build", "--tag", IMAGE_TAG, REPOSITORY_ROOT],
        check=True,
        timeout=BUILD_TIMEOUT_SECONDS,
    )
    return IMAGE_TAG


@pytest.fixture
def configured_container(image: str) -> Iterator[str]:
    """A running container configured from `JUDGE_ENVIRONMENT`, removed afterwards."""
    environment_arguments = [
        argument for pair in JUDGE_ENVIRONMENT.items() for argument in ("--env", "=".join(pair))
    ]
    container_id = _docker(
        "run", "--detach", "--rm", "--publish", "127.0.0.1::8000", *environment_arguments, image
    )
    try:
        yield container_id
    finally:
        _docker("stop", container_id)


@pytest.mark.docker
def test_the_container_refuses_to_start_without_a_configured_judge(image):
    refused_start = subprocess.run(
        ["docker", "run", "--rm", image],
        capture_output=True,
        text=True,
        timeout=STARTUP_TIMEOUT_SECONDS,
    )

    assert refused_start.returncode != 0
    _assert_names_every_missing_variable(refused_start.stderr)


@pytest.mark.docker
def test_the_container_reports_healthy_once_configured(configured_container):
    """`healthy` is Docker's own HEALTHCHECK verdict, so this also proves the check in the
    Dockerfile reaches `/health` from inside the image."""
    assert _health_status_once_settled(configured_container) == "healthy"

    published_address = _docker("port", configured_container, "8000")
    assert httpx.get(f"http://{published_address}/health").json() == {"status": "ok"}


@pytest.mark.docker
def test_the_container_does_not_run_as_root(image):
    assert _docker("run", "--rm", "--entrypoint", "id", image, "-u") != "0"


# --- helpers ---------------------------------------------------------------------------------


def _assert_names_every_missing_variable(startup_log: str) -> None:
    """Both ways of starting must say what to fix, not only that something is wrong."""
    for variable in JUDGE_ENVIRONMENT:
        assert f"{variable} is missing" in startup_log


def _uvicorn_command(port: int) -> list[str]:
    """The README's start command, run by the interpreter the tests run on."""
    return [
        sys.executable, "-m", "uvicorn", "rubric_judge.api:app",
        "--host", "127.0.0.1", "--port", str(port),
    ]


def _environment_without_judge() -> dict[str, str]:
    """This process's environment minus any `RUBRIC_JUDGE_*`, so a developer's own exported
    key can neither leak into a test nor make an unconfigured start succeed."""
    return {
        name: value for name, value in os.environ.items() if not name.startswith("RUBRIC_JUDGE_")
    }


def _free_port() -> int:
    """A port nothing listens on right now, so parallel runs and a local server do not clash."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _health_once_up(url: str) -> dict[str, str]:
    """The `/health` body as soon as the server accepts connections, failing after
    `STARTUP_TIMEOUT_SECONDS` rather than hanging the suite."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            return dict(httpx.get(url).json())
        except httpx.ConnectError:
            time.sleep(0.1)
    raise AssertionError(f"nothing answered on {url} within {STARTUP_TIMEOUT_SECONDS}s")


def _health_status_once_settled(container_id: str) -> str:
    """Docker's health verdict once it has left `starting`, or `starting` after
    `STARTUP_TIMEOUT_SECONDS`."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    status = "starting"
    while status == "starting" and time.monotonic() < deadline:
        time.sleep(0.5)
        status = _docker("inspect", "--format", "{{.State.Health.Status}}", container_id)
    return status


def _docker(*arguments: str) -> str:
    """One Docker CLI call whose failure fails the test, with its trimmed standard output."""
    return subprocess.run(
        ["docker", *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=STARTUP_TIMEOUT_SECONDS,
    ).stdout.strip()
