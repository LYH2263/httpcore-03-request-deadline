from __future__ import annotations

import select
import socket
import sys
import time
import typing

from ._exceptions import (
    ConnectTimeout,
    PoolTimeout,
    ReadTimeout,
    WriteTimeout,
)

# The timeout exception that applies when the absolute deadline has elapsed
# for each of the four configurable timeout phases.
_DEADLINE_TIMEOUT_EXCEPTIONS = {
    "pool": PoolTimeout,
    "connect": ConnectTimeout,
    "read": ReadTimeout,
    "write": WriteTimeout,
}


def get_timeout(
    timeouts: typing.Mapping[str, typing.Optional[float]], phase: str
) -> typing.Optional[float]:
    """
    Return the effective timeout in seconds for a given phase.

    `timeouts` is the request's `"timeout"` extension mapping, which may
    include an optional `"deadline"` key, holding an absolute point in time
    on the `time.monotonic()` clock.

    * When no deadline is configured, the phase timeout is returned
      unchanged, preserving the existing behaviour.
    * Otherwise the remaining budget (`deadline - time.monotonic()`) is used
      to tighten the phase timeout. If the phase timeout is `None`, then the
      remaining budget is used. Otherwise the smaller of the two is used.
    * If the remaining budget is zero or negative, then the matching timeout
      exception for the phase is raised immediately, without entering the
      wait or I/O operation.
    """
    timeout = timeouts.get(phase)
    deadline = timeouts.get("deadline")
    if deadline is None:
        return timeout

    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        raise _DEADLINE_TIMEOUT_EXCEPTIONS[phase]()
    if timeout is None:
        return remaining
    return min(timeout, remaining)


def is_socket_readable(sock: socket.socket | None) -> bool:
    """
    Return whether a socket, as identifed by its file descriptor, is readable.
    "A socket is readable" means that the read buffer isn't empty, i.e. that calling
    .recv() on it would immediately return some data.
    """
    # NOTE: we want check for readability without actually attempting to read, because
    # we don't want to block forever if it's not readable.

    # In the case that the socket no longer exists, or cannot return a file
    # descriptor, we treat it as being readable, as if it the next read operation
    # on it is ready to return the terminating `b""`.
    sock_fd = None if sock is None else sock.fileno()
    if sock_fd is None or sock_fd < 0:  # pragma: nocover
        return True

    # The implementation below was stolen from:
    # https://github.com/python-trio/trio/blob/20ee2b1b7376db637435d80e266212a35837ddcc/trio/_socket.py#L471-L478
    # See also: https://github.com/encode/httpcore/pull/193#issuecomment-703129316

    # Use select.select on Windows, and when poll is unavailable and select.poll
    # everywhere else. (E.g. When eventlet is in use. See #327)
    if (
        sys.platform == "win32" or getattr(select, "poll", None) is None
    ):  # pragma: nocover
        rready, _, _ = select.select([sock_fd], [], [], 0)
        return bool(rready)
    p = select.poll()
    p.register(sock_fd, select.POLLIN)
    return bool(p.poll(0))
