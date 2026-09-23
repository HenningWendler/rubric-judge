"""The service as it really runs: a uvicorn process, and the Docker image built from this
repository. The in-process tests in `test_api.py` start neither.

Both are pointed at a `StubOpenAIServer`, because a service configured from the environment
proves its judge with one real call before it serves.

The Docker tests carry the `docker` marker and are deselected by default, because they need a
running Docker daemon and a first build takes about a minute. Run them with
`pytest -m docker`."""

import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import httpx
import pytest

from tests.conftest import StubOpenAIServer, judge_environment

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_TAG = "rubric-judge:test"
STARTUP_TIMEOUT_SECONDS = 30
BUILD_TIMEOUT_SECONDS = 600
UVICORN_STARTUP_FAILED = 3
"""The exit code uvicorn ends with when the app's startup raised."""

HALF_WRITTEN_CONFIGURATION = {"RUBRIC_JUDGE_MODEL": "stub-model"}
"""A judge was intended, since one variable is set, but its endpoint and key are missing."""

COMPARE_ONLY_WARNING = "starts without a judge"


# --- uvicorn, the way the README starts the service ------------------------------------------


def test_uvicorn_refuses_a_half_written_configuration():
    refused_start = _run_to_exit(_uvicorn_command(_free_port()), HALF_WRITTEN_CONFIGURATION)

    assert refused_start.returncode == UVICORN_STARTUP_FAILED
    _assert_names_missing_variables(refused_start.stderr)


def test_uvicorn_refuses_a_judge_the_endpoint_refuses(stub_openai_server):
    stub_openai_server.status = 401
    environment = judge_environment(stub_openai_server.endpoint_seen_from())

    refused_start = _run_to_exit(_uvicorn_command(_free_port()), environment)

    assert refused_start.returncode == UVICORN_STARTUP_FAILED
    assert "AuthenticationError" in refused_start.stderr


def test_uvicorn_serves_a_proven_judge(stub_openai_server):
    environment = judge_environment(stub_openai_server.endpoint_seen_from())

    with _uvicorn_serving(environment) as (health_url, _):
        assert httpx.get(health_url).json() == {"status": "ok", "judge": "ok"}


def test_uvicorn_without_any_judge_variable_only_compares():
    with _uvicorn_serving({}) as (health_url, read_log):
        assert httpx.get(health_url).json() == {"status": "ok", "judge": "none"}

    assert COMPARE_ONLY_WARNING in read_log()


def test_uvicorn_refuses_an_empty_access_token():
    refused_start = _run_to_exit(
        _uvicorn_command(_free_port()), {"RUBRIC_JUDGE_ACCESS_TOKEN": " "}
    )

    assert refused_start.returncode == UVICORN_STARTUP_FAILED
    assert "RUBRIC_JUDGE_ACCESS_TOKEN is empty" in refused_start.stderr


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
def stub_reachable_from_containers() -> Iterator[StubOpenAIServer]:
    """A stub listening on every interface, because a container reaches the host through
    Docker's gateway and never through the host's loopback."""
    server = StubOpenAIServer(host="0.0.0.0")
    yield server
    server.stop()


@pytest.mark.docker
def test_the_container_refuses_a_half_written_configuration(image):
    refused_start = _run_to_exit(_docker_run_command(image, HALF_WRITTEN_CONFIGURATION), {})

    assert refused_start.returncode == UVICORN_STARTUP_FAILED
    _assert_names_missing_variables(refused_start.stderr)


@pytest.mark.docker
def test_the_container_refuses_a_judge_the_endpoint_refuses(image, stub_reachable_from_containers):
    stub_reachable_from_containers.status = 401
    environment = _judge_environment_inside_containers(stub_reachable_from_containers)

    refused_start = _run_to_exit(_docker_run_command(image, environment), {})

    assert refused_start.returncode == UVICORN_STARTUP_FAILED
    assert "AuthenticationError" in refused_start.stderr


@pytest.mark.docker
def test_the_container_reports_healthy_with_a_proven_judge(image, stub_reachable_from_containers):
    """`healthy` is Docker's own HEALTHCHECK verdict, so this also proves the check in the
    Dockerfile reaches `/health` from inside the image."""
    environment = _judge_environment_inside_containers(stub_reachable_from_containers)

    with _running_container(image, environment) as container_id:
        assert _health_status_once_settled(container_id) == "healthy"
        assert _health_of(container_id) == {"status": "ok", "judge": "ok"}


@pytest.mark.docker
def test_the_container_without_any_judge_variable_only_compares(image):
    with _running_container(image, {}) as container_id:
        assert _health_status_once_settled(container_id) == "healthy"
        assert _health_of(container_id) == {"status": "ok", "judge": "none"}
        assert COMPARE_ONLY_WARNING in _docker("logs", container_id, include_stderr=True)


@pytest.mark.docker
def test_the_container_does_not_run_as_root(image):
    assert _docker("run", "--rm", "--entrypoint", "id", image, "-u") != "0"


# --- helpers ---------------------------------------------------------------------------------


def _assert_names_missing_variables(startup_log: str) -> None:
    """Both ways of starting must say what to fix, not only that something is wrong."""
    for variable in ("RUBRIC_JUDGE_ENDPOINT", "RUBRIC_JUDGE_API_KEY"):
        assert f"{variable} is missing" in startup_log


def _uvicorn_command(port: int) -> list[str]:
    """The README's start command, run by the interpreter the tests run on."""
    return [
        sys.executable, "-m", "uvicorn", "rubric_judge.api:app",
        "--host", "127.0.0.1", "--port", str(port),
    ]


def _environment_with_only(judge_variables: dict[str, str]) -> dict[str, str]:
    """This process's environment with exactly the given `RUBRIC_JUDGE_*` variables, so a
    developer's own exported key can neither leak into a test nor change the mode the
    service starts in."""
    without_judge = {
        name: value for name, value in os.environ.items() if not name.startswith("RUBRIC_JUDGE_")
    }
    return without_judge | judge_variables


def _run_to_exit(
    command: list[str], judge_variables: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """A start that is expected to fail, run until it exits, with its output kept."""
    return subprocess.run(
        command,
        env=_environment_with_only(judge_variables),
        capture_output=True,
        text=True,
        timeout=STARTUP_TIMEOUT_SECONDS,
    )


@contextmanager
def _uvicorn_serving(
    judge_variables: dict[str, str],
) -> Iterator[tuple[str, Callable[[], str]]]:
    """A uvicorn process that has started serving, its `/health` URL, and a way to read its
    log once it is stopped. Stopped when the block ends."""
    port = _free_port()
    health_url = f"http://127.0.0.1:{port}/health"
    server = subprocess.Popen(
        _uvicorn_command(port),
        env=_environment_with_only(judge_variables),
        stderr=subprocess.PIPE,
        text=True,
    )
    log: list[str] = []
    try:
        _wait_until_answering(health_url)
        yield health_url, lambda: "".join(log)
    finally:
        server.terminate()
        log.append(server.communicate(timeout=STARTUP_TIMEOUT_SECONDS)[1])


def _docker_run_command(image: str, environment: dict[str, str]) -> list[str]:
    """`docker run` with `environment` passed in and the host reachable as
    `host.docker.internal`, which Docker Desktop provides and Linux needs spelled out."""
    environment_arguments = [
        argument for pair in environment.items() for argument in ("--env", "=".join(pair))
    ]
    return [
        "docker", "run", "--rm", "--add-host", "host.docker.internal:host-gateway",
        *environment_arguments, image,
    ]


def _judge_environment_inside_containers(server: StubOpenAIServer) -> dict[str, str]:
    return judge_environment(server.endpoint_seen_from("host.docker.internal"))


@contextmanager
def _running_container(image: str, environment: dict[str, str]) -> Iterator[str]:
    """A detached container publishing port 8000 on a free host port, removed afterwards."""
    command = _docker_run_command(image, environment)
    detached = [*command[:2], "--detach", "--publish", "127.0.0.1::8000", *command[2:]]
    container_id = subprocess.run(
        detached, check=True, capture_output=True, text=True, timeout=STARTUP_TIMEOUT_SECONDS
    ).stdout.strip()
    try:
        yield container_id
    finally:
        _docker("stop", container_id)


def _health_of(container_id: str) -> dict[str, str]:
    published_address = _docker("port", container_id, "8000")
    return dict(httpx.get(f"http://{published_address}/health").json())


def _free_port() -> int:
    """A port nothing listens on right now, so parallel runs and a local server do not clash."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_until_answering(url: str) -> None:
    """Return once the server accepts connections, failing after `STARTUP_TIMEOUT_SECONDS`
    rather than hanging the suite."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            httpx.get(url)
            return
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


def _docker(*arguments: str, include_stderr: bool = False) -> str:
    """One Docker CLI call whose failure fails the test, with its trimmed output. Standard
    error is added for `docker logs`, which replays a container's stderr there."""
    completed = subprocess.run(
        ["docker", *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=STARTUP_TIMEOUT_SECONDS,
    )
    return (completed.stdout + (completed.stderr if include_stderr else "")).strip()
