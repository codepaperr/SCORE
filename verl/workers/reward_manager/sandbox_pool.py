# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Persistent out-of-Ray subprocess pool for reward sandbox execution.

Why this exists: the code reward (feedback/code.py) fork()s one child per
test case. When compute_score runs inside a Ray worker -- a multi-threaded
process holding Plasma/Raylet sockets -- those high-frequency forks corrupt
the Plasma connection and take the node down ("Invalid Plasma message
type", "Raylet is terminated", "ActorDiedError: Owner's node has crashed";
observed on the H200 forkexit queue 2026-07-04/05, all 38 runs dead before
save_freq).

Fix: run compute_score in a small pool of PLAIN subprocesses launched with
subprocess.Popen (this file re-executed with --worker). Unlike
multiprocessing spawn/fork, the children are fresh interpreters that never
re-import the parent's __main__ (which under Ray is default_worker.py!) and
inherit zero Ray state. The per-test fork() inside feedback/code.py then
happens in Ray-free children: exactly as fast as before, and safe.
(forkserver for the inner layer was tried on the H200 and is far too slow.)

Protocol (parent <-> worker, over the worker's stdin/stdout):
    request : 4-byte big-endian length + pickle(call_kwargs dict)
    response: 4-byte big-endian length + pickle(("ok", result) | ("err", msg))
The worker dups its real stdout for the protocol and points fd 1 at stderr,
so stray print()s from reward code land in the log instead of corrupting
the pickle stream. One OS thread per worker in the parent does blocking
request/response with a deadline; a hung or dead worker is killed,
restarted, and the item scores None (caller maps that to 0).
"""

import base64
import os
import pickle
import select
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

__all__ = ["get_pool", "reset_pool", "SandboxPool"]

_HDR = struct.Struct("!I")


# --------------------------------------------------------------------------
# Worker side (this file re-executed as: python sandbox_pool.py --worker ...)
# --------------------------------------------------------------------------
def _worker_main(module_path: str, fn_name: str, reward_kwargs_b64: str) -> None:
    os.environ["PYTHONNOUSERSITE"] = "1"
    # Claim the protocol channel, then send fd 1 to stderr so reward-code
    # prints can't corrupt the pickle stream.
    proto_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    proto_in = os.fdopen(os.dup(sys.stdin.fileno()), "rb")

    import importlib.util

    spec = importlib.util.spec_from_file_location("sandbox_reward_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    score_fn = getattr(module, fn_name)
    reward_kwargs = pickle.loads(base64.b64decode(reward_kwargs_b64))

    while True:
        header = proto_in.read(_HDR.size)
        if len(header) < _HDR.size:  # parent closed stdin -> exit
            return
        (n,) = _HDR.unpack(header)
        call_kwargs = pickle.loads(proto_in.read(n))
        try:
            resp = ("ok", score_fn(**{**reward_kwargs, **call_kwargs}))
        except BaseException as e:  # noqa: BLE001 - report, don't die
            resp = ("err", f"{type(e).__name__}: {e}")
        payload = pickle.dumps(resp, protocol=pickle.HIGHEST_PROTOCOL)
        proto_out.write(_HDR.pack(len(payload)))
        proto_out.write(payload)
        proto_out.flush()


# --------------------------------------------------------------------------
# Parent side
# --------------------------------------------------------------------------
class _Worker:
    """One persistent sandbox subprocess + its request/response channel."""

    def __init__(self, cmd: list[str]):
        self._cmd = cmd
        self._proc: subprocess.Popen | None = None
        self._spawn()

    def _spawn(self) -> None:
        self._proc = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # worker prints go straight to the parent's log
        )

    def _read_exact(self, n: int, deadline: float) -> bytes | None:
        fd = self._proc.stdout.fileno()
        buf = b""
        while len(buf) < n:
            import time

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                return None
            chunk = os.read(fd, n - len(buf))
            if not chunk:  # EOF: worker died
                return None
            buf += chunk
        return buf

    def request(self, call_kwargs: dict, timeout: float):
        """Score one item; None on worker death/timeout (worker is restarted)."""
        import time

        deadline = time.monotonic() + timeout
        try:
            payload = pickle.dumps(call_kwargs, protocol=pickle.HIGHEST_PROTOCOL)
            self._proc.stdin.write(_HDR.pack(len(payload)))
            self._proc.stdin.write(payload)
            self._proc.stdin.flush()
            header = self._read_exact(_HDR.size, deadline)
            if header is None:
                raise TimeoutError("sandbox worker timed out or died")
            (n,) = _HDR.unpack(header)
            body = self._read_exact(n, deadline)
            if body is None:
                raise TimeoutError("sandbox worker timed out mid-response")
            status, value = pickle.loads(body)
            if status == "ok":
                return value
            print(f"[Sandbox] reward fn raised in worker: {value}", flush=True)
            return None
        except Exception as e:
            print(f"[Sandbox] worker failure ({type(e).__name__}: {e}); restarting worker.", flush=True)
            self.close()
            self._spawn()
            return None

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.kill()
            self._proc.wait(timeout=5)
        except Exception:
            pass
        self._proc = None


class SandboxPool:
    """N persistent workers; one parent thread per worker feeds items."""

    def __init__(self, module_path: str, fn_name: str, num_workers: int, reward_kwargs: dict | None = None):
        rkw_b64 = base64.b64encode(pickle.dumps(reward_kwargs or {})).decode()
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", module_path, fn_name, rkw_b64]
        self._workers = [_Worker(cmd) for _ in range(num_workers)]
        self._free: list[_Worker] = list(self._workers)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="sandbox")

    def _score_one(self, call_kwargs: dict, timeout: float):
        with self._lock:
            worker = self._free.pop()
        try:
            return worker.request(call_kwargs, timeout)
        finally:
            with self._lock:
                self._free.append(worker)

    def score_batch(self, batch: list[dict], timeout: float, progress_desc: str | None = None) -> list:
        """Score a batch; entry is None where the item failed/timed out."""
        futures = [self._executor.submit(self._score_one, kw, timeout) for kw in batch]
        if progress_desc:
            from tqdm import tqdm

            results = []
            with tqdm(total=len(batch), desc=progress_desc, leave=False) as pbar:
                for fut in futures:
                    results.append(fut.result())
                    pbar.update(1)
            return results
        return [fut.result() for fut in futures]

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        for w in self._workers:
            w.close()


_POOL: SandboxPool | None = None
_POOL_KEY = None


def get_pool(module_path: str, fn_name: str, num_workers: int, reward_kwargs: dict | None = None) -> SandboxPool:
    """Lazily create (and cache) the pool; persists across training steps."""
    global _POOL, _POOL_KEY
    key = (module_path, fn_name, num_workers)
    if _POOL is not None and _POOL_KEY == key:
        return _POOL
    if _POOL is not None:
        _POOL.close()
    _POOL = SandboxPool(module_path, fn_name, num_workers, reward_kwargs)
    _POOL_KEY = key
    return _POOL


def reset_pool() -> None:
    global _POOL, _POOL_KEY
    if _POOL is not None:
        _POOL.close()
    _POOL = None
    _POOL_KEY = None


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit("usage: sandbox_pool.py --worker <module_path> <fn_name> <reward_kwargs_b64>")
