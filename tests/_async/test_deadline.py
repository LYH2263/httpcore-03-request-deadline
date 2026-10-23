"""
Tests for the absolute `deadline` request timeout extension.

The deadline is an absolute point in time on the `time.monotonic()` clock.
When set, every connect/read/write/pool wait must be tightened against the
remaining budget, instead of each phase independently using its full
configured timeout.

These tests use mock network backends that record the timeout values which
reach the actual I/O exits, together with a fake monotonic clock that makes
the consumed-budget scenarios fully deterministic.
"""

import time
import typing

import pytest

import httpcore
from httpcore import (
    AsyncConnectionPool,
    AsyncHTTPConnection,
    AsyncMockBackend,
    AsyncMockStream,
    AsyncNetworkStream,
    ConnectError,
    ConnectTimeout,
    Origin,
    PoolTimeout,
    Proxy,
    ReadTimeout,
    TimeoutException,
    WriteTimeout,
)


class FakeClock:
    """A controllable stand-in for the `time.monotonic()` clock."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


HTTP_RESPONSE = [
    b"HTTP/1.1 200 OK\r\n",
    b"Content-Type: plain/text\r\n",
    b"Content-Length: 13\r\n",
    b"\r\n",
    b"Hello, world!",
]


class RecordingReadStream(AsyncMockStream):
    """
    Simulates a server that never responds: the very first read records the
    timeout it was given and then raises a ReadTimeout.
    """

    def __init__(self, record: dict[str, typing.Any]) -> None:
        super().__init__([])
        self._record = record

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        self._record.setdefault("read", []).append(timeout)
        raise ReadTimeout()


class SlowConnectBackend(AsyncMockBackend):
    """
    `connect_tcp` consumes `connect_delay` of the fake clock budget, then
    returns a stream whose reads block/timeout.
    """

    def __init__(
        self, record: dict[str, typing.Any], clock: FakeClock, connect_delay: float
    ) -> None:
        super().__init__([])
        self._record = record
        self._clock = clock
        self._connect_delay = connect_delay

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:
        self._record.setdefault("connect", []).append(timeout)
        self._clock.advance(self._connect_delay)
        return RecordingReadStream(self._record)


class WriteTimeoutStream(AsyncMockStream):
    """Records the write timeout then simulates a blocked/timed-out write."""

    def __init__(self, record: dict[str, typing.Any]) -> None:
        super().__init__([])
        self._record = record

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        if buffer:
            self._record.setdefault("write", []).append(timeout)
            raise WriteTimeout()


class WriteTimeoutBackend(AsyncMockBackend):
    def __init__(self, record: dict[str, typing.Any]) -> None:
        super().__init__([])
        self._record = record

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:
        return WriteTimeoutStream(self._record)


class FlakyConnectBackend(AsyncMockBackend):
    """
    Fails `connect_tcp` a number of times, recording every connect timeout
    and backoff sleep duration. Sleeps advance the fake clock, so that later
    attempts observe a reduced remaining deadline budget.
    """

    def __init__(
        self, failures: int, record: dict[str, typing.Any], clock: FakeClock
    ) -> None:
        super().__init__(HTTP_RESPONSE)
        self._failures = failures
        self._record = record
        self._clock = clock

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:
        self._record.setdefault("connect", []).append(timeout)
        if self._failures > 0:
            self._failures -= 1
            raise ConnectError()
        return AsyncMockStream(list(self._buffer))

    async def sleep(self, seconds: float) -> None:
        self._record.setdefault("sleep", []).append(seconds)
        self._clock.advance(seconds)


class NeverConnectBackend(AsyncMockBackend):
    """Records attempts; fails the test if a connection is actually opened."""

    def __init__(self, record: dict[str, typing.Any]) -> None:
        super().__init__([])
        self._record = record

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:
        self._record.setdefault("connect", []).append(timeout)
        raise AssertionError("connect_tcp must not be called once deadline expired")


class H2ReadTimeoutStream(AsyncMockStream):
    """HTTP/2 stream that simulates a read timeout on the first read."""

    def __init__(self, record: dict[str, typing.Any]) -> None:
        super().__init__([], http2=True)
        self._record = record

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        self._record.setdefault("read", []).append(timeout)
        raise ReadTimeout()


class H2ReadTimeoutBackend(AsyncMockBackend):
    def __init__(self, record: dict[str, typing.Any]) -> None:
        super().__init__([], http2=True)
        self._record = record

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:
        return H2ReadTimeoutStream(self._record)


class TunnelRecordingStream(AsyncMockStream):
    """Records the timeout passed to the tunnel `start_tls` handshake."""

    def __init__(
        self, buffer: typing.List[bytes], record: dict[str, typing.Any]
    ) -> None:
        super().__init__(buffer)
        self._record = record

    async def start_tls(
        self,
        ssl_context: typing.Any,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> AsyncNetworkStream:
        self._record.setdefault("start_tls", []).append(timeout)
        raise ConnectTimeout()


class TunnelBackend(AsyncMockBackend):
    def __init__(
        self, record: dict[str, typing.Any], clock: FakeClock, connect_delay: float
    ) -> None:
        super().__init__([b"HTTP/1.1 200 Connection Established\r\n\r\n"])
        self._record = record
        self._clock = clock
        self._connect_delay = connect_delay

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:
        self._clock.advance(self._connect_delay)
        return TunnelRecordingStream(list(self._buffer), self._record)


@pytest.mark.anyio
async def test_deadline_tightens_read_timeout(monkeypatch):
    """
    Most of the deadline budget is spent establishing the connection.
    The read phase must then use the remaining budget, not its original
    (much larger) phase timeout.
    """
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock.monotonic)

    record: dict[str, typing.Any] = {}
    network_backend = SlowConnectBackend(record=record, clock=clock, connect_delay=0.25)
    origin = Origin(b"http", b"example.com", 80)
    extensions = {"timeout": {"read": 30.0, "deadline": clock.monotonic() + 0.3}}

    async with AsyncHTTPConnection(
        origin=origin, network_backend=network_backend
    ) as conn:
        with pytest.raises(ReadTimeout):
            await conn.request("GET", "http://example.com/", extensions=extensions)

    # The read I/O was entered with the tightened remaining budget.
    assert record["read"]
    effective_read = record["read"][0]
    assert effective_read is not None
    assert effective_read < 0.1
    assert effective_read != 30.0


@pytest.mark.anyio
async def test_phase_timeout_without_deadline_is_not_tightened(monkeypatch):
    """
    A small/large phase timeout on its own does not shrink as time passes.
    Only the absolute deadline tightens it.
    """
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock.monotonic)

    record: dict[str, typing.Any] = {}
    network_backend = SlowConnectBackend(record=record, clock=clock, connect_delay=0.25)
    origin = Origin(b"http", b"example.com", 80)
    extensions = {"timeout": {"read": 30.0}}

    async with AsyncHTTPConnection(
        origin=origin, network_backend=network_backend
    ) as conn:
        with pytest.raises(ReadTimeout):
            await conn.request("GET", "http://example.com/", extensions=extensions)

    assert record["read"][0] == 30.0


@pytest.mark.anyio
async def test_explicit_none_deadline_is_disabled(monkeypatch):
    """An explicit `None` deadline must behave exactly as if absent."""
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock.monotonic)

    record: dict[str, typing.Any] = {}
    network_backend = SlowConnectBackend(record=record, clock=clock, connect_delay=0.25)
    origin = Origin(b"http", b"example.com", 80)
    extensions = {"timeout": {"read": 0.05, "deadline": None}}

    async with AsyncHTTPConnection(
        origin=origin, network_backend=network_backend
    ) as conn:
        with pytest.raises(ReadTimeout):
            await conn.request("GET", "http://example.com/", extensions=extensions)

    assert record["read"][0] == 0.05


@pytest.mark.anyio
async def test_deadline_without_phase_timeout_uses_remaining(monkeypatch):
    """If the phase timeout is `None`, the remaining budget is the timeout."""
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock.monotonic)

    record: dict[str, typing.Any] = {}
    network_backend = SlowConnectBackend(record=record, clock=clock, connect_delay=0.0)
    origin = Origin(b"http", b"example.com", 80)
    extensions = {"timeout": {"deadline": clock.monotonic() + 0.3}}

    async with AsyncHTTPConnection(
        origin=origin, network_backend=network_backend
    ) as conn:
        with pytest.raises(ReadTimeout):
            await conn.request("GET", "http://example.com/", extensions=extensions)

    assert record["read"][0] is not None
    assert 0.0 < record["read"][0] <= 0.3


@pytest.mark.anyio
async def test_deadline_tightens_write_timeout(monkeypatch):
    """
    The write phase must be tightened by the deadline, and raise
    WriteTimeout once the budget has elapsed.
    """
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock.monotonic)

    record: dict[str, typing.Any] = {}
    network_backend = WriteTimeoutBackend(record)
    origin = Origin(b"http", b"example.com", 80)
    extensions = {"timeout": {"write": 30.0, "deadline": clock.monotonic() + 0.2}}

    async with AsyncHTTPConnection(
        origin=origin, network_backend=network_backend
    ) as conn:
        with pytest.raises(WriteTimeout):
            await conn.request(
                "POST",
                "http://example.com/",
                content=b"abc",
                extensions=extensions,
            )

    assert record["write"]
    effective_write = record["write"][0]
    assert effective_write is not None
    assert effective_write < 1.0
    assert effective_write != 30.0


@pytest.mark.anyio
async def test_connect_retries_sleep_truncated_and_budget_shrinks(
    monkeypatch,
):
    """
    On connection retries the backoff sleep must be truncated by the
    remaining budget, and each subsequent connect attempt must observe a
    reduced timeout, rather than reusing the original connect timeout and
    full backoff interval.
    """
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock.monotonic)

    record: dict[str, typing.Any] = {}
    network_backend = FlakyConnectBackend(failures=3, record=record, clock=clock)
    origin = Origin(b"http", b"example.com", 80)
    extensions = {"timeout": {"connect": 30.0, "deadline": clock.monotonic() + 1.0}}

    async with AsyncHTTPConnection(
        origin=origin, network_backend=network_backend, retries=3
    ) as conn:
        with pytest.raises(ConnectTimeout):
            await conn.request("GET", "http://example.com/", extensions=extensions)

    # connect_tcp was called for the three failed attempts. The fourth
    # attempt was never entered because the deadline had elapsed.
    assert len(record["connect"]) == 3
    assert record["connect"][0] == pytest.approx(1.0)
    assert record["connect"][1] == pytest.approx(1.0)
    # After the truncated sleeps the third attempt gets far less than the
    # original 30 second connect timeout.
    assert record["connect"][2] < 0.6
    assert record["connect"][2] < record["connect"][0]

    # Backoff sequence is 0, 0.5, 1.0... Both non-zero sleeps are truncated
    # to the remaining ~0.5s budget rather than running their full interval.
    assert record["sleep"][0] == 0.0
    assert record["sleep"][1] == pytest.approx(0.5)
    # The nominal third delay would be 1.0s but only 0.5s remained.
    assert record["sleep"][2] < 1.0
    assert record["sleep"][2] == pytest.approx(0.5)


@pytest.mark.anyio
async def test_expired_deadline_does_not_enter_connect(monkeypatch):
    """
    If the deadline has already expired before connecting, connect_tcp must
    never be entered and ConnectTimeout is raised immediately.
    """
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock.monotonic)

    record: dict[str, typing.Any] = {}
    network_backend = NeverConnectBackend(record)
    origin = Origin(b"http", b"example.com", 80)
    extensions = {"timeout": {"connect": 30.0, "deadline": clock.monotonic() - 1.0}}

    async with AsyncHTTPConnection(
        origin=origin, network_backend=network_backend
    ) as conn:
        with pytest.raises(ConnectTimeout):
            await conn.request("GET", "http://example.com/", extensions=extensions)

    assert record.get("connect", []) == []


@pytest.mark.anyio
async def test_expired_deadline_pool_wait_raises_immediately(monkeypatch):
    """
    An expired deadline while waiting for a pool connection raises
    PoolTimeout immediately, without entering the underlying wait.
    """
    # Keep a reference to the real clock so we can measure wall time while
    # the module level `time.monotonic` is patched.
    real_monotonic = time.monotonic
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock.monotonic)

    network_backend = AsyncMockBackend(HTTP_RESPONSE)

    extensions = {"timeout": {"pool": 30.0, "deadline": clock.monotonic() - 1.0}}

    started = real_monotonic()
    async with AsyncConnectionPool(
        network_backend=network_backend, max_connections=1
    ) as pool:
        # Occupy the single connection with an in-flight request.
        async with pool.stream("GET", "https://example.com/"):
            with pytest.raises(PoolTimeout):
                await pool.request("GET", "https://example.com/", extensions=extensions)
    # No blocking occurred despite the generous 30 second pool timeout.
    assert real_monotonic() - started < 5.0


@pytest.mark.anyio
async def test_deadline_is_per_request_on_shared_connection():
    """
    Deadlines belong to individual request extensions: a request whose
    deadline elapses must not poison later requests on the same pool.
    """
    network_backend = AsyncMockBackend(HTTP_RESPONSE)

    async with AsyncConnectionPool(network_backend=network_backend) as pool:
        # First request completes normally.
        response = await pool.request("GET", "http://example.com/")
        assert response.status == 200
        assert response.content == b"Hello, world!"

        # Second request carries an already expired deadline and fails.
        expired = {"timeout": {"deadline": time.monotonic() - 1.0}}
        with pytest.raises(TimeoutException):
            await pool.request("GET", "http://example.com/", extensions=expired)

        # A third request without a deadline is unaffected and succeeds,
        # even though the pool saw an expired deadline moments earlier.
        response = await pool.request("GET", "http://example.com/")
        assert response.status == 200
        assert response.content == b"Hello, world!"


@pytest.mark.anyio
async def test_http2_deadline_tightens_read_timeout(monkeypatch):
    """The HTTP/2 receive path must honour the deadline as well."""
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock.monotonic)

    record: dict[str, typing.Any] = {}
    network_backend = H2ReadTimeoutBackend(record)
    origin = Origin(b"https", b"example.com", 443)
    extensions = {"timeout": {"read": 30.0, "deadline": clock.monotonic() + 0.3}}

    async with AsyncHTTPConnection(
        origin=origin, network_backend=network_backend, http2=True
    ) as conn:
        with pytest.raises(ReadTimeout):
            await conn.request("GET", "https://example.com/", extensions=extensions)

    assert record["read"]
    effective_read = record["read"][0]
    assert effective_read is not None
    assert effective_read < 1.0
    assert effective_read != 30.0


class H2WriteTimeoutStream(AsyncMockStream):
    """HTTP/2 stream that simulates a write timeout on the first write."""

    def __init__(self, record: dict[str, typing.Any]) -> None:
        super().__init__([], http2=True)
        self._record = record

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._record.setdefault("write", []).append(timeout)
        raise WriteTimeout()


class H2WriteTimeoutBackend(AsyncMockBackend):
    def __init__(self, record: dict[str, typing.Any]) -> None:
        super().__init__([], http2=True)
        self._record = record

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:
        return H2WriteTimeoutStream(self._record)


@pytest.mark.anyio
async def test_http2_deadline_tightens_write_timeout(monkeypatch):
    """The HTTP/2 send path must honour the deadline as well."""
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock.monotonic)

    record: dict[str, typing.Any] = {}
    network_backend = H2WriteTimeoutBackend(record)
    origin = Origin(b"https", b"example.com", 443)
    extensions = {"timeout": {"write": 30.0, "deadline": clock.monotonic() + 0.2}}

    async with AsyncHTTPConnection(
        origin=origin, network_backend=network_backend, http2=True
    ) as conn:
        with pytest.raises(WriteTimeout):
            await conn.request(
                "POST",
                "https://example.com/",
                content=b"abc",
                extensions=extensions,
            )

    assert record["write"]
    effective_write = record["write"][0]
    assert effective_write is not None
    assert effective_write < 1.0
    assert effective_write != 30.0


@pytest.mark.anyio
async def test_socks_proxy_expired_deadline_connect_timeout(monkeypatch):
    """
    Through a SOCKS proxy an expired deadline must raise ConnectTimeout
    before even opening the TCP connection to the proxy.
    """
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock.monotonic)

    record: dict[str, typing.Any] = {}
    network_backend = NeverConnectBackend(record)

    extensions = {"timeout": {"connect": 30.0, "deadline": clock.monotonic() - 1.0}}

    async with AsyncConnectionPool(
        proxy=Proxy("socks5://localhost:8000/"),
        network_backend=network_backend,
    ) as pool:
        with pytest.raises(ConnectTimeout):
            await pool.request("GET", "http://example.com/", extensions=extensions)

    assert record.get("connect", []) == []


@pytest.mark.anyio
async def test_tunnel_start_tls_tightened_by_deadline(monkeypatch):
    """
    The tunnel `start_tls` after an HTTP CONNECT must be tightened against
    the budget consumed while connecting to the proxy.
    """
    clock = FakeClock()
    monkeypatch.setattr(time, "monotonic", clock.monotonic)

    record: dict[str, typing.Any] = {}
    network_backend = TunnelBackend(record=record, clock=clock, connect_delay=0.3)

    extensions = {"timeout": {"connect": 30.0, "deadline": clock.monotonic() + 0.5}}

    async with AsyncConnectionPool(
        proxy=Proxy("http://localhost:8080/"),
        network_backend=network_backend,
    ) as pool:
        with pytest.raises(ConnectTimeout):
            await pool.request("GET", "https://example.com/", extensions=extensions)

    # The TLS handshake received the remaining budget (~0.2s), tightened
    # from the configured 30 second connect timeout.
    assert record["start_tls"]
    effective_tls = record["start_tls"][0]
    assert effective_tls is not None
    assert effective_tls < 0.3
    assert effective_tls != 30.0
