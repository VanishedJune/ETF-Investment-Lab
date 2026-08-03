"""Real-backend native-Python Playwright acceptance for V3.3-20W.

Unlike ``v33_visual_review.py``, this review never installs browser routes or
returns fixture API payloads.  It starts the production FastAPI application,
serves the built ``frontend/dist`` bundle, and exercises the copied production
SQLite data through the real HTTP endpoints and Vue controls.

Safety boundaries:

* ``INVESTMENT_LAB_HOME`` always points at a disposable application directory;
* ``INVESTMENT_LAB_RESOURCE_ROOT`` points at this read-only source tree;
* the source database is opened with SQLite ``mode=ro`` and copied with the
  online backup API;
* no training button or training endpoint is used;
* the source database byte hash and metadata must be unchanged at the end.

The webapp-testing ``with_server.py`` helper is intentionally not used here:
this acceptance must stop FastAPI, start a fresh process against the same
temporary database, and prove that a calendar update survived that restart.
"""

from __future__ import annotations

import argparse
import atexit
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

from playwright.sync_api import (
    APIRequestContext,
    Browser,
    Page,
    Response,
    TimeoutError as PlaywrightTimeoutError,
    expect,
    sync_playwright,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DATABASE = ROOT / "data" / "investment_lab.db"
SOURCE_DATABASE = Path(
    os.environ.get("V33_REVIEW_SOURCE_DATABASE", str(DEFAULT_SOURCE_DATABASE))
).resolve()
FRONTEND_INDEX = ROOT / "frontend" / "dist" / "index.html"
REPORTS = ROOT / "reports"
ACTIVE_MARKETS = ("399006", "159941")
ALLOWED_FUND_ETF_RATIOS = {"7:3", "6:4", "5:5", "4:6", "3:7"}
STARTUP_TIMEOUT_SECONDS = 240.0
ANALYSIS_TIMEOUT_MS = 300_000
TEMP_CLEANUP_TIMEOUT_SECONDS = 30.0
_ACTIVE_SERVERS: set["FastApiServer"] = set()


@dataclass(frozen=True)
class FileFingerprint:
    size: int
    modified_ns: int
    sha256: str


@dataclass
class PageAudit:
    """Browser diagnostics collected without intercepting any request."""

    base_url: str
    console_errors: list[str]
    page_errors: list[str]
    failed_requests: list[str]
    http_errors: list[str]
    api_requests: list[str]

    @classmethod
    def attach(cls, page: Page, base_url: str) -> PageAudit:
        audit = cls(base_url, [], [], [], [], [])

        def on_console(message: Any) -> None:
            if message.type == "error":
                audit.console_errors.append(message.text)

        def on_page_error(error: Any) -> None:
            audit.page_errors.append(str(error))

        def on_request_failed(request: Any) -> None:
            failure = request.failure
            audit.failed_requests.append(
                f"{request.method} {request.url}: {failure or 'unknown failure'}"
            )

        def on_response(response: Response) -> None:
            if response.status >= 400:
                audit.http_errors.append(
                    f"{response.status} {response.request.method} {response.url}"
                )

        def on_request(request: Any) -> None:
            if request.url.startswith(f"{base_url}/api/"):
                path = urlsplit(request.url).path
                audit.api_requests.append(f"{request.method} {path}")

        page.on("console", on_console)
        page.on("pageerror", on_page_error)
        page.on("requestfailed", on_request_failed)
        page.on("response", on_response)
        page.on("request", on_request)
        return audit

    def assert_clean(self, label: str) -> None:
        failures = {
            "console_errors": self.console_errors,
            "page_errors": self.page_errors,
            "failed_requests": self.failed_requests,
            "http_errors": self.http_errors,
        }
        failures = {key: value for key, value in failures.items() if value}
        if failures:
            raise AssertionError(f"{label} browser diagnostics failed: {failures}")

    def assert_no_training_request(self) -> None:
        forbidden = [
            request
            for request in self.api_requests
            if request == "POST /api/v33/model/train"
            or request.startswith("POST /api/v33/training/")
        ]
        if forbidden:
            raise AssertionError(f"acceptance invoked a training endpoint: {forbidden}")


class FastApiServer:
    """A production-ASGI subprocess bound to one ephemeral localhost port."""

    def __init__(self, application_home: Path, sequence: int) -> None:
        self.application_home = application_home
        self.sequence = sequence
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.log_path = application_home / f"fastapi-{sequence}.log"
        self._log_handle: Any | None = None
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> FastApiServer:
        if self.process is not None:
            raise RuntimeError("FastAPI server has already been started")
        environment = os.environ.copy()
        environment["INVESTMENT_LAB_HOME"] = str(self.application_home)
        environment["INVESTMENT_LAB_RESOURCE_ROOT"] = str(ROOT)
        environment["PYTHONUNBUFFERED"] = "1"
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.web:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--log-level",
                "warning",
            ],
            cwd=ROOT,
            env=environment,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creation_flags,
        )
        _ACTIVE_SERVERS.add(self)
        self._wait_until_ready()
        return self

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        last_error = "server did not answer"
        while time.monotonic() < deadline:
            if self.process is None:
                raise RuntimeError("FastAPI process was not created")
            return_code = self.process.poll()
            if return_code is not None:
                self._close_log()
                raise AssertionError(
                    f"FastAPI exited during startup with code {return_code}:\n"
                    f"{self.read_log()}"
                )
            try:
                payload = _url_json(f"{self.base_url}/api/health", timeout=2.0)
                if payload.get("status") == "ok":
                    return
                last_error = f"unexpected health payload: {payload!r}"
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
                last_error = str(error)
            time.sleep(0.2)
        raise AssertionError(
            f"FastAPI was not ready after {STARTUP_TIMEOUT_SECONDS:.0f}s: {last_error}\n"
            f"{self.read_log()}"
        )

    def stop(self) -> None:
        _ACTIVE_SERVERS.discard(self)
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        self._close_log()

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def read_log(self) -> str:
        if self._log_handle is not None:
            self._log_handle.flush()
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return ""

    def __enter__(self) -> FastApiServer:
        return self.start()

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.stop()


def _remove_tree_with_retry(path: Path) -> None:
    """Remove a disposable app home after transient Windows handles close."""

    deadline = time.monotonic() + TEMP_CLEANUP_TIMEOUT_SECONDS
    last_error: OSError | None = None
    while path.exists():
        try:
            shutil.rmtree(path)
        except OSError as error:
            last_error = error
            if time.monotonic() >= deadline:
                break
            time.sleep(0.25)
        else:
            break
    if path.exists():
        raise AssertionError(
            f"temporary application home remained after cleanup retries: {path}; "
            f"last_error={last_error!r}"
        )


@contextmanager
def _temporary_application_home(prefix: str) -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=ROOT / ".pytest-tmp"))
    try:
        yield path
    finally:
        _remove_tree_with_retry(path)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


def _stop_active_servers() -> None:
    for server in tuple(_ACTIVE_SERVERS):
        try:
            server.stop()
        except Exception:
            # Process cleanup must not mask the original failure or interrupt.
            pass


def _install_process_cleanup() -> None:
    atexit.register(_stop_active_servers)

    def stop_on_signal(signum: int, _frame: Any) -> None:
        _stop_active_servers()
        raise SystemExit(128 + signum)

    for signal_name in ("SIGINT", "SIGTERM"):
        value = getattr(signal, signal_name, None)
        if value is not None:
            signal.signal(value, stop_on_signal)


def _file_fingerprint(path: Path) -> FileFingerprint:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return FileFingerprint(stat.st_size, stat.st_mtime_ns, digest.hexdigest())


def _read_only_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _copy_database_read_only(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise AssertionError(f"temporary destination already exists: {destination}")
    with closing(_read_only_connection(source)) as source_connection:
        with closing(sqlite3.connect(destination, timeout=30)) as destination_connection:
            source_connection.backup(destination_connection, pages=8_192, sleep=0.01)
            result = destination_connection.execute("PRAGMA quick_check").fetchone()
            if result != ("ok",):
                raise AssertionError(f"temporary SQLite backup failed quick_check: {result}")


def _hash_rows(connection: sqlite3.Connection, query: str, values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    cursor = connection.execute(query, tuple(values))
    while True:
        rows = cursor.fetchmany(64)
        if not rows:
            break
        for row in rows:
            digest.update(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            )
            digest.update(b"\n")
    return digest.hexdigest()


def _database_training_signature(database: Path) -> dict[str, dict[str, Any]]:
    """Mirror the training boundary fingerprint without hashing analysis rows."""

    result: dict[str, dict[str, Any]] = {}
    with closing(_read_only_connection(database)) as connection:
        for market in ACTIVE_MARKETS:
            iteration_hash = _hash_rows(
                connection,
                """
                SELECT id, training_run_id, feature_snapshot_id, iteration_number,
                       parent_iteration_number, week_key, cutoff_date, status,
                       maturity_status, champion_model_version,
                       challenger_model_version, promoted, forecast_json,
                       evaluation_json, optimizer_state_json, composite_loss,
                       wis_loss, price_turn_error_days, dif_zero_error_days,
                       completed_at, audit_json
                FROM v33_training_iterations
                WHERE model_market = ?
                ORDER BY iteration_number
                """,
                (market,),
            )
            model_hash = _hash_rows(
                connection,
                """
                SELECT id, version, parent_version, trained_through_date,
                       artifact_hash, status
                FROM v33_model_versions
                WHERE model_market = ?
                ORDER BY created_at, id
                """,
                (market,),
            )
            optimizer_hash = _hash_rows(
                connection,
                """
                SELECT id, model_market, iteration_number,
                       last_training_week_key, champion_model_version,
                       optimizer_memory_json, last_successful_training_at,
                       next_training_eligible_at, state_hash, updated_at
                FROM v33_optimizer_states
                WHERE model_market = ?
                ORDER BY id
                """,
                (market,),
            )
            champion_hash = _hash_rows(
                connection,
                """
                SELECT version, parent_version, feature_version,
                       methodology_version, trained_through_date, random_seed,
                       artifact_hash, parameters_json, metrics_json, status,
                       created_at
                FROM v33_model_versions
                WHERE model_market = ? AND status = 'champion'
                ORDER BY created_at, id
                """,
                (market,),
            )
            iteration_count = int(
                connection.execute(
                    "SELECT count(*) FROM v33_training_iterations WHERE model_market = ?",
                    (market,),
                ).fetchone()[0]
            )
            result[market] = {
                "iteration_count": iteration_count,
                "iteration_registry_hash": iteration_hash,
                "model_registry_hash": model_hash,
                "optimizer_hash": optimizer_hash,
                "champion_hash": champion_hash,
            }
    return result


def _url_json(url: str, *, timeout: float = 30.0) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_v33_idle(base_url: str) -> dict[str, Any]:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    latest: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        latest = _url_json(f"{base_url}/api/v33/model/status")
        markets = latest.get("markets", {})
        if all(
            markets.get(market, {}).get("bootstrapped")
            and not markets.get(market, {}).get("is_training")
            for market in ACTIVE_MARKETS
        ):
            for market in ACTIVE_MARKETS:
                status = markets[market]
                if int(status.get("iteration_count") or 0) <= 0:
                    raise AssertionError(f"{market} has no formal V3.3 iterations")
            return latest
        time.sleep(0.5)
    raise AssertionError(f"V3.3 did not reach idle bootstrapped state: {latest!r}")


def _api_json(
    request_context: APIRequestContext,
    base_url: str,
    path: str,
) -> dict[str, Any]:
    response = request_context.get(f"{base_url}{path}", timeout=120_000)
    if not response.ok:
        raise AssertionError(
            f"GET {path} failed with {response.status}: {response.text()}"
        )
    value = response.json()
    if not isinstance(value, dict):
        raise AssertionError(f"GET {path} returned a non-object payload")
    return value


def _http_training_signature(
    request_context: APIRequestContext,
    base_url: str,
) -> dict[str, Any]:
    statuses = _api_json(request_context, base_url, "/api/v33/model/status")
    signature: dict[str, Any] = {}
    for market in ACTIVE_MARKETS:
        status = statuses["markets"][market]
        champion = _api_json(
            request_context,
            base_url,
            f"/api/v33/models/{market}/champion",
        )
        signature[market] = {
            "status": {
                key: status.get(key)
                for key in (
                    "bootstrapped",
                    "iteration_count",
                    "champion_version",
                    "last_training_week_key",
                    "last_successful_training_at",
                    "next_training_eligible_at",
                    "pending_count",
                )
            },
            "champion": champion,
        }
    return signature


def _response_path(response: Response) -> str:
    return urlsplit(response.url).path


def _wait_for_page(page: Page, base_url: str) -> None:
    page.goto(base_url, wait_until="networkidle", timeout=180_000)
    page.wait_for_load_state("networkidle", timeout=120_000)
    page.locator("h1").wait_for(state="visible", timeout=60_000)
    page.wait_for_selector('[data-testid="unified-market-chart"] canvas', timeout=60_000)
    page.wait_for_selector('[data-testid="v33-curve-399006"] canvas', timeout=60_000)
    page.wait_for_selector('[data-testid="v33-curve-159941"] canvas', timeout=60_000)
    module_source = page.locator('script[type="module"]').get_attribute("src")
    if not module_source or not module_source.startswith("/assets/"):
        raise AssertionError(
            f"page is not using the production frontend bundle: {module_source!r}"
        )
    _wait_for_optional_analysis_chart(page)


def _wait_for_optional_analysis_chart(page: Page) -> None:
    """Stabilise a persisted result before a market switch replaces its DOM."""

    result = page.get_by_test_id("v33-analysis-result")
    if not result.count():
        return
    result.wait_for(state="visible", timeout=60_000)
    page.wait_for_selector(
        '[data-testid="v33-path-chart"] canvas', timeout=120_000
    )


def _assert_layout(page: Page, *, mobile: bool) -> dict[str, Any]:
    metrics = page.evaluate(
        """
        () => ({
          horizontalOverflow:
            document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
          bodyFont: parseFloat(getComputedStyle(document.body).fontSize),
          headingFont: parseFloat(getComputedStyle(document.querySelector('h1')).fontSize),
          canvasCount: document.querySelectorAll('canvas').length,
        })
        """
    )
    if metrics["horizontalOverflow"]:
        raise AssertionError("page has horizontal overflow")
    if metrics["bodyFont"] < 14:
        raise AssertionError(f"body font is too small: {metrics['bodyFont']}")
    minimum_heading = 30 if mobile else 40
    if metrics["headingFont"] < minimum_heading:
        raise AssertionError(f"heading font is too small: {metrics['headingFont']}")
    if metrics["canvasCount"] < 3:
        raise AssertionError(f"expected real ECharts canvases, got {metrics['canvasCount']}")
    return metrics


def _validate_market_payload(
    request_context: APIRequestContext,
    base_url: str,
    market: str,
) -> dict[str, Any]:
    prices = _api_json(
        request_context,
        base_url,
        f"/api/market/{market}/prices?timeframe=weekly",
    )
    rows = prices.get("rows")
    if not isinstance(rows, list) or not rows:
        raise AssertionError(f"{market} has no real weekly price rows")
    recent = rows[-20:]
    if not any(float(row.get("volume") or 0) > 0 for row in recent):
        raise AssertionError(f"{market} recent weekly rows have no real volume")
    indicators_response = request_context.get(
        f"{base_url}/api/indicators/{market}?timeframe=weekly",
        timeout=120_000,
    )
    if not indicators_response.ok:
        raise AssertionError(
            f"{market} indicators failed with {indicators_response.status}: "
            f"{indicators_response.text()}"
        )
    indicators = indicators_response.json()
    if not isinstance(indicators, list) or not indicators:
        raise AssertionError(f"{market} has no real weekly indicator rows")
    recent_indicators = indicators[-20:]
    for key in ("dif", "dea", "macd_histogram"):
        if not any(row.get(key) is not None for row in recent_indicators):
            raise AssertionError(f"{market} recent weekly indicators miss {key}")
    return {
        "row_count": len(rows),
        "indicator_count": len(indicators),
        "latest_date": rows[-1]["date"],
        "latest_volume": rows[-1].get("volume"),
        "latest_source": rows[-1].get("source"),
    }


def _switch_market(page: Page, market: str) -> None:
    button = page.locator(".market-switch button").filter(has_text=market)
    if button.count() != 1:
        raise AssertionError(f"could not identify market switch for {market}")
    with page.expect_response(
        lambda response: (
            response.request.method == "GET"
            and _response_path(response) == f"/api/market/{market}/prices"
        ),
        timeout=120_000,
    ) as response_info:
        button.click()
    if not response_info.value.ok:
        raise AssertionError(
            f"market switch {market} failed with {response_info.value.status}"
        )
    page.wait_for_load_state("networkidle", timeout=120_000)
    page.wait_for_selector('[data-testid="unified-market-chart"] canvas', timeout=60_000)
    _wait_for_optional_analysis_chart(page)


def _expect_test_id_text(page: Page, test_id: str, expected: str) -> None:
    """Wait for Vue's post-response render, not merely the HTTP body."""

    expect(page.get_by_test_id(test_id)).to_have_text(expected, timeout=60_000)


def _calendar_create_and_update(
    page: Page,
    *,
    baseline_position: int,
    marker: str,
) -> dict[str, Any]:
    if not 0 <= baseline_position <= 100:
        raise AssertionError(f"invalid baseline position: {baseline_position}")
    direction = "increase" if baseline_position <= 90 else "decrease"
    sign = 1 if direction == "increase" else -1
    operation_date = date.today().isoformat()
    page.get_by_test_id("position-direction").select_option(direction)
    page.get_by_test_id("position-date").fill(operation_date)
    page.get_by_test_id("position-change").fill("5")
    page.locator(".position-form textarea").fill(f"{marker}-created")
    with page.expect_response(
        lambda response: (
            response.request.method == "POST"
            and _response_path(response) == "/api/v2/position-events"
        ),
        timeout=120_000,
    ) as create_info:
        page.get_by_test_id("position-submit").click()
    create_response = create_info.value
    if not create_response.ok:
        raise AssertionError(
            f"calendar create failed with {create_response.status}: {create_response.text()}"
        )
    created = create_response.json()
    event_id = int(created["id"])
    event = page.get_by_test_id(f"position-event-{event_id}")
    event.wait_for(state="visible", timeout=60_000)
    page.get_by_test_id("current-position").wait_for(state="visible")
    expected_created = baseline_position + sign * 5
    _expect_test_id_text(page, "current-position", f"{expected_created}%")

    event.locator("button").first.click()
    page.get_by_test_id("position-change").fill("10")
    page.locator(".position-form textarea").fill(f"{marker}-updated")
    with page.expect_response(
        lambda response: (
            response.request.method == "PATCH"
            and _response_path(response) == f"/api/v2/position-events/{event_id}"
        ),
        timeout=120_000,
    ) as update_info:
        page.get_by_test_id("position-submit").click()
    update_response = update_info.value
    if not update_response.ok:
        raise AssertionError(
            f"calendar update failed with {update_response.status}: {update_response.text()}"
        )
    updated = update_response.json()
    expected_updated = baseline_position + sign * 10
    event = page.get_by_test_id(f"position-event-{event_id}")
    event.wait_for(state="visible", timeout=60_000)
    if f"{marker}-updated" not in event.inner_text():
        raise AssertionError("calendar update was not rendered in the real ledger")
    _expect_test_id_text(page, "current-position", f"{expected_updated}%")
    return {
        "event_id": event_id,
        "operation_date": operation_date,
        "direction": direction,
        "baseline_position": baseline_position,
        "persisted_position": expected_updated,
        "marker": marker,
        "api_position_after": int(updated["position_after"]),
    }


def _validate_analysis(payload: dict[str, Any], expected_position: int) -> dict[str, Any]:
    if payload.get("implementation_revision") != "V3.3-20W":
        raise AssertionError("analysis is not the V3.3-20W implementation")
    model = payload["model"]
    if int(model["horizon_weeks"]) != 20 or int(model["daily_window_sessions"]) != 100:
        raise AssertionError(f"analysis model windows are invalid: {model}")
    path = payload["path"]
    for key in ("weeks", "p10", "p50", "p90", "expected", "weekly_base", "daily_correction"):
        if len(path.get(key, [])) != 20:
            raise AssertionError(f"analysis path {key} is not exactly 20 weeks")
    if any(abs(float(value)) > 1e-12 for value in path["daily_correction"][4:]):
        raise AssertionError("100-day daily correction leaked beyond the first four weeks")
    position = payload["position"]
    if int(position["current"]) != expected_position:
        raise AssertionError(
            f"analysis did not read the local calendar position: {position['current']} != "
            f"{expected_position}"
        )
    for key in ("current", "weekly_base_target", "target", "change"):
        if int(position[key]) % 5:
            raise AssertionError(f"analysis position field {key} is off the 5% grid")
    advice = payload["advice"]
    if advice["fund_etf_ratio"] not in ALLOWED_FUND_ETF_RATIOS:
        raise AssertionError(f"invalid fund/ETF ratio: {advice['fund_etf_ratio']}")
    batches = advice["batches"]
    if len(batches) > 4:
        raise AssertionError(f"analysis returned too many batches: {len(batches)}")
    for batch in batches:
        percent = int(batch["percentage_points"])
        if percent <= 0 or percent % 5:
            raise AssertionError(f"invalid execution batch percentage: {batch}")
        if not batch.get("expected_date") or not batch.get("window_start") or not batch.get("window_end"):
            raise AssertionError(f"execution batch has no real date window: {batch}")
    if sum(int(batch["percentage_points"]) for batch in batches) != abs(int(position["change"])):
        raise AssertionError("execution batch sum does not conserve the position change")
    if payload.get("training_identity_before") != payload.get("training_identity_after"):
        raise AssertionError("analysis response reports a changed training identity")
    if not payload.get("training_identity_unchanged"):
        raise AssertionError("analysis response did not confirm training identity isolation")
    if payload.get("training_fingerprint_before") != payload.get("training_fingerprint_after"):
        raise AssertionError("analysis response reports a changed training fingerprint")
    if not payload.get("training_fingerprint_unchanged"):
        raise AssertionError("analysis response did not confirm full fingerprint isolation")
    return {
        "run_id": payload["run_id"],
        "market": payload["market"],
        "model_version": model["version"],
        "model_iteration": model["iteration_number"],
        "current_position": position["current"],
        "target_position": position["target"],
        "batch_count": len(batches),
        "batch_percentages": [batch["percentage_points"] for batch in batches],
        "fund_etf_ratio": advice["fund_etf_ratio"],
        "training_identity_unchanged": True,
        "training_fingerprint_unchanged": True,
    }


def _run_analysis(
    page: Page,
    expected_position: int,
    audit: PageAudit,
) -> dict[str, Any]:
    button = page.get_by_test_id("v33-analyze")
    button.wait_for(state="visible", timeout=60_000)
    if button.is_disabled():
        raise AssertionError("data-analysis button is disabled for a bootstrapped idle model")
    with page.expect_response(
        lambda response: (
            response.request.method == "POST"
            and _response_path(response) == "/api/v33/model/analysis"
        ),
        timeout=ANALYSIS_TIMEOUT_MS,
    ) as analysis_info:
        button.click()
    response = analysis_info.value
    if not response.ok:
        raise AssertionError(
            f"real V3.3 analysis failed with {response.status}: {response.text()}"
        )
    payload = response.json()
    page.get_by_test_id("v33-analysis-result").wait_for(
        state="visible", timeout=120_000
    )
    try:
        page.wait_for_selector(
            '[data-testid="v33-path-chart"] canvas', timeout=60_000
        )
    except PlaywrightTimeoutError as error:
        diagnostic = REPORTS / "v33-real-backend-analysis-error.png"
        page.screenshot(path=str(diagnostic), full_page=True)
        path = page.get_by_test_id("v33-path-chart")
        details = {
            "path_element_count": path.count(),
            "path_html": path.inner_html()[:2_000] if path.count() else None,
            "path_error": (
                page.locator(".path-error").inner_text()
                if page.locator(".path-error").count()
                else None
            ),
            "console_errors": audit.console_errors,
            "page_errors": audit.page_errors,
            "failed_requests": audit.failed_requests,
            "http_errors": audit.http_errors,
            "screenshot": str(diagnostic),
        }
        raise AssertionError(
            "real analysis rendered its summary but not its 20-week chart: "
            f"{details}"
        ) from error
    return _validate_analysis(payload, expected_position)


def _delete_calendar_event_after_restart(
    page: Page,
    calendar: dict[str, Any],
) -> None:
    event_id = int(calendar["event_id"])
    event = page.get_by_test_id(f"position-event-{event_id}")
    event.wait_for(state="visible", timeout=60_000)
    text = event.inner_text()
    if calendar["marker"] not in text or "10%" not in text:
        raise AssertionError("updated calendar event did not survive FastAPI restart")
    current = page.get_by_test_id("current-position").inner_text().strip()
    if current != f"{calendar['persisted_position']}%":
        raise AssertionError(
            f"position did not survive restart: {current} != {calendar['persisted_position']}%"
        )
    delete_button = event.locator("button").last
    delete_button.click()
    with page.expect_response(
        lambda response: (
            response.request.method == "DELETE"
            and _response_path(response) == f"/api/v2/position-events/{event_id}"
        ),
        timeout=120_000,
    ) as delete_info:
        delete_button.click()
    if delete_info.value.status != 200:
        raise AssertionError(
            f"calendar delete failed with {delete_info.value.status}: "
            f"{delete_info.value.text()}"
        )
    if delete_info.value.json() != {"deleted": True, "id": event_id}:
        raise AssertionError(
            f"calendar delete returned an invalid payload: {delete_info.value.text()}"
        )
    event.wait_for(state="detached", timeout=60_000)
    expected = f"{calendar['baseline_position']}%"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if page.get_by_test_id("current-position").inner_text().strip() == expected:
            return
        time.sleep(0.1)
    raise AssertionError("calendar delete did not restore the replayed baseline position")


def _new_page(browser: Browser, base_url: str, *, mobile: bool) -> tuple[Any, Page, PageAudit]:
    viewport = {"width": 412, "height": 915} if mobile else {"width": 1500, "height": 1000}
    context = browser.new_context(viewport=viewport, device_scale_factor=1)
    page = context.new_page()
    page.set_default_timeout(60_000)
    audit = PageAudit.attach(page, base_url)
    return context, page, audit


def main() -> None:
    if not SOURCE_DATABASE.is_file():
        raise FileNotFoundError(f"production database does not exist: {SOURCE_DATABASE}")
    if not FRONTEND_INDEX.is_file():
        raise FileNotFoundError(
            f"production frontend is missing: {FRONTEND_INDEX}; run npm.cmd --prefix frontend run build"
        )
    REPORTS.mkdir(parents=True, exist_ok=True)
    (ROOT / ".pytest-tmp").mkdir(parents=True, exist_ok=True)
    source_before = _file_fingerprint(SOURCE_DATABASE)
    result: dict[str, Any] = {
        "status": "FAIL",
        "source_database": str(SOURCE_DATABASE),
        "source_database_before": asdict(source_before),
        "training_endpoint_requests": 0,
    }
    diagnostic_logs: list[str] = []
    try:
        with _temporary_application_home("v33-real-backend-") as application_home:
            temporary_database = application_home / "data" / "investment_lab.db"
            (application_home / "data" / "weekly_analysis_v2").mkdir(parents=True)
            if (ROOT / "config").is_dir():
                shutil.copytree(ROOT / "config", application_home / "config")
            _copy_database_read_only(SOURCE_DATABASE, temporary_database)
            training_before_start = _database_training_signature(temporary_database)

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                server_one = FastApiServer(application_home, 1)
                try:
                    server_one.start()
                    statuses = _wait_for_v33_idle(server_one.base_url)
                    training_after_start = _database_training_signature(temporary_database)
                    if training_before_start != training_after_start:
                        raise AssertionError(
                            "starting the acceptance server changed V3.3 training state"
                        )

                    desktop_context, desktop, desktop_audit = _new_page(
                        browser, server_one.base_url, mobile=False
                    )
                    _wait_for_page(desktop, server_one.base_url)
                    market_results = {
                        market: _validate_market_payload(
                            desktop.request, server_one.base_url, market
                        )
                        for market in ACTIVE_MARKETS
                    }
                    _switch_market(desktop, "159941")
                    _expect_test_id_text(
                        desktop,
                        "data-cutoff",
                        market_results["159941"]["latest_date"],
                    )
                    _switch_market(desktop, "399006")
                    _expect_test_id_text(
                        desktop,
                        "data-cutoff",
                        market_results["399006"]["latest_date"],
                    )

                    positions = _api_json(
                        desktop.request,
                        server_one.base_url,
                        "/api/investment-calendar/current-positions",
                    )
                    baseline_position = int(positions.get("399006") or 0)
                    marker = f"PLAYWRIGHT-{uuid4().hex[:12]}"
                    calendar = _calendar_create_and_update(
                        desktop,
                        baseline_position=baseline_position,
                        marker=marker,
                    )
                    if calendar["api_position_after"] != calendar["persisted_position"]:
                        raise AssertionError("calendar update API returned an inconsistent position")

                    http_identity_before = _http_training_signature(
                        desktop.request, server_one.base_url
                    )
                    database_identity_before = _database_training_signature(temporary_database)
                    analysis = _run_analysis(
                        desktop,
                        calendar["persisted_position"],
                        desktop_audit,
                    )
                    http_identity_after = _http_training_signature(
                        desktop.request, server_one.base_url
                    )
                    database_identity_after = _database_training_signature(temporary_database)
                    if http_identity_before != http_identity_after:
                        raise AssertionError("analysis changed the HTTP training identity")
                    if database_identity_before != database_identity_after:
                        raise AssertionError("analysis changed the SQLite training identity")
                    desktop_metrics = _assert_layout(desktop, mobile=False)
                    desktop_screenshot = REPORTS / "v33-real-backend-desktop.png"
                    desktop.screenshot(path=str(desktop_screenshot), full_page=True)
                    desktop_audit.assert_no_training_request()
                    desktop_audit.assert_clean("desktop")

                    mobile_context, mobile, mobile_audit = _new_page(
                        browser, server_one.base_url, mobile=True
                    )
                    _wait_for_page(mobile, server_one.base_url)
                    mobile.get_by_test_id("v33-analysis-result").wait_for(
                        state="visible", timeout=120_000
                    )
                    _expect_test_id_text(
                        mobile,
                        "current-position",
                        f"{calendar['persisted_position']}%",
                    )
                    mobile_metrics = _assert_layout(mobile, mobile=True)
                    mobile_screenshot = REPORTS / "v33-real-backend-mobile.png"
                    mobile.screenshot(path=str(mobile_screenshot), full_page=True)
                    mobile_audit.assert_no_training_request()
                    mobile_audit.assert_clean("mobile")
                    result["training_endpoint_requests"] += sum(
                        1
                        for audit in (desktop_audit, mobile_audit)
                        for request in audit.api_requests
                        if request == "POST /api/v33/model/train"
                        or request.startswith("POST /api/v33/training/")
                    )
                    mobile_context.close()
                    desktop_context.close()
                finally:
                    server_one.stop()
                    diagnostic_logs.append(server_one.read_log())

                server_two = FastApiServer(application_home, 2)
                try:
                    server_two.start()
                    _wait_for_v33_idle(server_two.base_url)
                    restart_context, restart_page, restart_audit = _new_page(
                        browser, server_two.base_url, mobile=False
                    )
                    _wait_for_page(restart_page, server_two.base_url)
                    _delete_calendar_event_after_restart(restart_page, calendar)
                    restart_screenshot = REPORTS / "v33-real-backend-restart.png"
                    restart_page.screenshot(path=str(restart_screenshot), full_page=True)
                    restart_audit.assert_no_training_request()
                    restart_audit.assert_clean("restart")
                    result["training_endpoint_requests"] += sum(
                        1
                        for request in restart_audit.api_requests
                        if request == "POST /api/v33/model/train"
                        or request.startswith("POST /api/v33/training/")
                    )
                    restart_context.close()
                finally:
                    server_two.stop()
                    diagnostic_logs.append(server_two.read_log())
                    browser.close()

            if _database_training_signature(temporary_database) != training_after_start:
                raise AssertionError("acceptance changed V3.3 training state after restart")
            with closing(_read_only_connection(temporary_database)) as connection:
                if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
                    raise AssertionError("temporary database failed final quick_check")
                remaining = int(
                    connection.execute(
                        "SELECT count(*) FROM v2_position_events WHERE note LIKE ?",
                        (f"{marker}%",),
                    ).fetchone()[0]
                )
                if remaining:
                    raise AssertionError("calendar delete left the acceptance event behind")

            result.update(
                {
                    "status": "PASS",
                    "markets": market_results,
                    "model_status": {
                        market: {
                            "iteration_count": statuses["markets"][market]["iteration_count"],
                            "champion_version": statuses["markets"][market]["champion_version"],
                            "pending_count": statuses["markets"][market]["pending_count"],
                        }
                        for market in ACTIVE_MARKETS
                    },
                    "analysis": analysis,
                    "calendar": {
                        **calendar,
                        "survived_server_restart": True,
                        "deleted_after_restart": True,
                    },
                    "desktop": {
                        **desktop_metrics,
                        "screenshot": str(desktop_screenshot),
                    },
                    "mobile": {
                        **mobile_metrics,
                        "screenshot": str(mobile_screenshot),
                    },
                    "restart_screenshot": str(restart_screenshot),
                    "temporary_database_quick_check": "ok",
                }
            )
    except Exception:
        diagnostic_path = REPORTS / "v33-real-backend-server.log"
        diagnostic_path.write_text("\n\n===== RESTART =====\n\n".join(diagnostic_logs), encoding="utf-8")
        raise
    finally:
        source_after = _file_fingerprint(SOURCE_DATABASE)
        result["source_database_after"] = asdict(source_after)
        if source_before != source_after:
            raise AssertionError(
                "read-only source database changed during isolated Playwright acceptance: "
                f"before={source_before}, after={source_after}"
            )
    if result["training_endpoint_requests"] != 0:
        raise AssertionError("acceptance used a V3.3 training endpoint")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def inspect_existing_analysis() -> None:
    """Fast diagnostic for the latest persisted analysis; performs no writes."""

    if not SOURCE_DATABASE.is_file():
        raise FileNotFoundError(f"source database does not exist: {SOURCE_DATABASE}")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (ROOT / ".pytest-tmp").mkdir(parents=True, exist_ok=True)
    before = _file_fingerprint(SOURCE_DATABASE)
    diagnostics: dict[str, Any] = {}
    with _temporary_application_home("v33-real-inspect-") as application_home:
        database = application_home / "data" / "investment_lab.db"
        (application_home / "data" / "weekly_analysis_v2").mkdir(parents=True)
        if (ROOT / "config").is_dir():
            shutil.copytree(ROOT / "config", application_home / "config")
        _copy_database_read_only(SOURCE_DATABASE, database)
        with FastApiServer(application_home, 1) as server:
            _wait_for_v33_idle(server.base_url)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                context, page, audit = _new_page(
                    browser, server.base_url, mobile=False
                )
                try:
                    _wait_for_page(page, server.base_url)
                    result = page.get_by_test_id("v33-analysis-result")
                    result.wait_for(state="visible", timeout=60_000)
                    chart = page.locator('[data-testid="v33-path-chart"] canvas')
                    try:
                        chart.wait_for(state="visible", timeout=30_000)
                    except PlaywrightTimeoutError:
                        pass
                    screenshot = REPORTS / "v33-real-backend-existing-analysis.png"
                    page.screenshot(path=str(screenshot), full_page=True)
                    path = page.get_by_test_id("v33-path-chart")
                    diagnostics = {
                        "status": "PASS" if chart.count() else "FAIL",
                        "path_canvas_count": chart.count(),
                        "path_element_count": path.count(),
                        "path_html": path.inner_html()[:2_000] if path.count() else None,
                        "path_error": (
                            page.locator(".path-error").inner_text()
                            if page.locator(".path-error").count()
                            else None
                        ),
                        "console_errors": audit.console_errors,
                        "page_errors": audit.page_errors,
                        "failed_requests": audit.failed_requests,
                        "http_errors": audit.http_errors,
                        "screenshot": str(screenshot),
                    }
                    audit.assert_no_training_request()
                finally:
                    context.close()
                    browser.close()
    after = _file_fingerprint(SOURCE_DATABASE)
    if before != after:
        raise AssertionError("read-only source changed during existing-analysis inspection")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    if diagnostics.get("status") != "PASS":
        raise AssertionError(f"persisted real analysis did not render: {diagnostics}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run native Python Playwright against real FastAPI and a disposable "
            "copy of the V3.3 SQLite database."
        )
    )
    parser.add_argument(
        "--source-database",
        type=Path,
        default=SOURCE_DATABASE,
        help=(
            "read-only SQLite source copied into the disposable app home "
            "(default: V33_REVIEW_SOURCE_DATABASE or data/investment_lab.db)"
        ),
    )
    parser.add_argument(
        "--inspect-existing",
        action="store_true",
        help="inspect the latest persisted real analysis without CRUD or a new analysis",
    )
    return parser.parse_args()


def _run_cli() -> None:
    global SOURCE_DATABASE
    arguments = _parse_args()
    SOURCE_DATABASE = arguments.source_database.resolve()
    _install_process_cleanup()
    if arguments.inspect_existing:
        inspect_existing_analysis()
    else:
        main()


if __name__ == "__main__":
    _run_cli()
