"""GPU-accelerated DuckDB connection for v2 pipeline data operations.

Wraps the vector-gpu-engine''s ``duckdb_gpu`` binary (gpu_shell/build/duckdb_gpu)
in its persistent ``--serve`` mode.

Architecture -- hand-off exclusive ownership
--------------------------------------------
DuckDB only allows ONE writer per database file at a time.  The previous
design opened the same ``pipeline.duckdb`` from BOTH the duckdb-python
process AND the duckdb_gpu subprocess simultaneously, causing:

  1. IO Error: Could not set lock on file ... another process has it open
  2. Thread over-subscription (threads=N on each → 2N total)
  3. Memory over-subscription (memory_limit=M on each → 2M committed)

The fix: ``HybridConnection`` maintains exactly one active DuckDB process at
a time and hands off file ownership between them as the SQL workload shifts:

  CPU mode  -- duckdb-python holds the file; GPU subprocess is not running.
               Used for: all SQL, UDF registration, data imports, key building,
               feature extraction (large fetchmany streaming).

  GPU mode  -- duckdb_gpu subprocess holds the file; CPU connection is closed.
               Used for: Jaro-Winkler candidate scoring (INSERT INTO candidates)
               where the GPU optimizer offloads the projection arithmetic.

  CPU → GPU: CHECKPOINT, close CPU connection, launch duckdb_gpu with db_path.
  GPU → CPU: CHECKPOINT (via SQL), close subprocess, reopen duckdb-python.

Because the two processes are never simultaneously open, the full thread /
memory budget is available to whichever is active -- no splitting needed.

Precision contract (GPU mode)
-----------------------------
- ``gpu_fast_math = false``           No NVRTC --use_fast_math; IEEE-754 compliant.
- ``gpu_precision_relaxation = false`` No float32 downcast; strict FP64 end-to-end.
- ``gpu_stream_pipeline = true``       GPU-resident column cache enabled (Session 49+).
Environment:
- ``VECTOR_GPU_WARMUP=1``  CUDA context init overlaps DuckDB file open.
- ``VECTOR_GPU_STRICT_FP=1`` C++ engine also rejects any FP32 code path.

Fallback
--------
If ``duckdb_gpu`` is absent, ``connect()`` returns a plain ``duckdb.connect()``
so the pipeline runs unmodified on CPU.

Usage
-----
>>> import gpu_connection
>>> con = gpu_connection.connect(db_path, threads=4, memory_limit="1536MB",
...                              spill_dir=some_path)
>>> con.execute("SELECT 1").fetchone()
(1,)
>>> con.close()
"""

from __future__ import annotations

import os
import subprocess
import textwrap
import threading
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Locate the duckdb_gpu binary
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).resolve().parent
# …/noobs_on_duty_amazon_ML_challange
_REPO_ROOT = _SRC_DIR.parents[2]
# scaling-broccoli lives next to the challenge repo inside CR/
_GPU_ENGINE_ROOT = _REPO_ROOT.parent / "scaling-broccoli" / "vector-gpu-engine"
_GPU_BINARY = _GPU_ENGINE_ROOT / "gpu_shell" / "build" / "duckdb_gpu"
if not _GPU_BINARY.exists():
    _candidate = _GPU_BINARY.with_suffix(".exe")
    if _candidate.exists():
        _GPU_BINARY = _candidate

GPU_AVAILABLE: bool = _GPU_BINARY.exists()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IEEE-754 / pipeline-cache pragmas sent to the GPU subprocess right after it
# opens the database.  Sent as a batch so they take effect before any user SQL.
_GPU_PRECISION_INIT_SQL = textwrap.dedent("""\
    SET gpu_fast_math = false;
    SET gpu_precision_relaxation = false;
    SET gpu_stream_pipeline = true;
""")

# Environment variables forwarded to the duckdb_gpu child process.
_GPU_ENV_EXTRAS: dict[str, str] = {
    # Start CUDA context on a background thread concurrently with DB open.
    # Saves ~155 ms on the first offloadable query (docs/TODO.md P1 item 3).
    "VECTOR_GPU_WARMUP": "1",
    # Belt-and-suspenders: C++ engine rejects any FP32 code path at compile time.
    "VECTOR_GPU_STRICT_FP": "1",
}

# SQL patterns whose presence signals that the GPU optimizer should be used.
# Covers the Jaro-Winkler candidate-scoring INSERT and its dependent queries.
_GPU_PATTERNS = (
    "INSERT INTO candidates",
    "jaro_winkler_similarity",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _substitute_params(sql: str, parameters: list) -> str:
    """Replace positional ``?`` placeholders with literal values."""
    it = iter(parameters)
    parts = sql.split("?")
    rendered = []
    for i, part in enumerate(parts):
        rendered.append(part)
        if i < len(parts) - 1:
            val = next(it, None)
            if val is None:
                rendered.append("NULL")
            elif isinstance(val, str):
                rendered.append("'" + val.replace("'", "''") + "'")
            else:
                rendered.append(str(val))
    return "".join(rendered)


def _parse_text_table(text: str) -> list[tuple]:
    """Parse DuckDB text-mode table output into a list of tuples.

    Handles both the Unicode box-drawing format and the ASCII pipe format.
    Returns ``[]`` for DDL / SET output (nothing to parse).
    """
    rows: list[tuple] = []
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return rows
        
    # Check for gpu_shell --serve format:
    # Line 0: headers
    # Line 1: types
    # Line 2: [ Rows: N]
    # Line 3+: values (tab-separated)
    if len(lines) >= 3 and lines[2].startswith("[ Rows:"):
        headers = [h for h in lines[0].split("\t") if h]
        num_cols = len(headers)
        for dl in lines[3:]:
            parts = dl.split("\t")
            if len(parts) > num_cols and parts[-1] == "":
                parts = parts[:-1]
            rows.append(tuple(parts))
        return rows

    BOX = "\u2502"  # │
    if any(BOX in l for l in lines):
        data_lines = [
            l for l in lines
            if BOX in l
            and "\u2500" not in l   # ─
            and "\u250c" not in l   # ┌
            and "\u2514" not in l   # └
            and "\u251c" not in l   # ├
        ]
        for dl in data_lines[1:]:   # skip the header row
            parts = [p.strip() for p in dl.strip().strip(BOX).split(BOX)]
            rows.append(tuple(parts))
    elif len(lines) >= 2 and set(lines[1]).issubset(set("-+ \t")):
        for dl in lines[2:]:
            parts = [p.strip() for p in dl.split("|")]
            if parts:
                rows.append(tuple(parts))
    else:
        for l in lines:
            rows.append((l.strip(),))
    return rows


class _Cursor:
    """Minimal cursor-like wrapper over a list of rows."""
    __slots__ = ("_rows", "_pos")

    def __init__(self, rows: list) -> None:
        self._rows = rows
        self._pos = 0

    def fetchone(self):
        if self._pos >= len(self._rows):
            return None
        row = self._rows[self._pos]
        self._pos += 1
        return row

    def fetchmany(self, size: int = 1):
        chunk = self._rows[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk

    def fetchall(self):
        rest = self._rows[self._pos :]
        self._pos = len(self._rows)
        return rest

    def __iter__(self):
        return iter(self._rows[self._pos :])


# ---------------------------------------------------------------------------
# HybridConnection -- hand-off ownership between CPU and GPU
# ---------------------------------------------------------------------------

class HybridConnection:
    """Single-file DuckDB connection that alternates between CPU and GPU modes.

    At any moment exactly ONE DuckDB process holds the database file:

      CPU mode  duckdb-python holds the file; the GPU subprocess is not running.
      GPU mode  duckdb_gpu subprocess holds the file; the CPU connection is closed.

    Transitions are triggered lazily by ``execute()``:
      CPU → GPU  on the first SQL matching a GPU pattern.
      GPU → CPU  on the first SQL NOT matching a GPU pattern (or ``cursor()``).

    Because the two processes are never simultaneously open:
    - The file lock conflict cannot occur.
    - The full thread / memory budget is available to whichever is active.
      There is no need to split resources; neither connection is starved.
    """

    def __init__(
        self,
        db_path: str | Path | None,
        threads: int,
        memory_limit: str,
        spill_dir: str | Path | None,
        max_temp_directory_size: str,
    ) -> None:
        import duckdb as _duckdb
        self._duckdb = _duckdb

        self._db_path = db_path
        self._threads = threads
        self._memory_limit = memory_limit
        self._spill_dir = spill_dir
        self._max_temp_directory_size = max_temp_directory_size

        # UDF registry: re-applied every time the CPU connection is (re)opened.
        self._udfs: list[tuple] = []

        self._mode = "cpu"
        self._cpu_con = self._new_cpu_con()
        self._gpu_proc: subprocess.Popen | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # CPU connection management
    # ------------------------------------------------------------------

    def _new_cpu_con(self):
        """Open a fresh duckdb-python connection with the configured settings."""
        con = self._duckdb.connect(
            str(self._db_path) if self._db_path else ":memory:"
        )
        con.execute(f"PRAGMA threads = {self._threads}")
        con.execute(f"SET memory_limit='{self._memory_limit}'")
        if self._spill_dir:
            spill_posix = (
                Path(self._spill_dir).resolve().as_posix().replace("'", "''")
            )
            con.execute(f"SET temp_directory='{spill_posix}'")
            con.execute(
                f"SET max_temp_directory_size='{self._max_temp_directory_size}'"
            )
        for name, func, arg_types, return_type in self._udfs:
            con.create_function(name, func, arg_types, return_type)
        return con

    # ------------------------------------------------------------------
    # Mode transitions
    # ------------------------------------------------------------------

    def _switch_to_gpu(self) -> None:
        """Hand the database file to the GPU subprocess.

        Steps:
        1. CHECKPOINT -- flush all WAL changes to the main database file.
        2. Close the CPU connection -- releases the exclusive file lock.
        3. Launch duckdb_gpu with the same db_path -- acquires the lock.
        4. Send init pragmas (threads, memory, IEEE FP64 settings).
        """
        if self._mode == "gpu":
            return

        # 1. Checkpoint and release the file lock.
        try:
            if self._cpu_con is not None:
                self._cpu_con.execute("CHECKPOINT")
                self._cpu_con.close()
                self._cpu_con = None
        except Exception as exc:
            print(
                f"[gpu_connection] WARNING: checkpoint before GPU switch failed: {exc}",
                flush=True,
            )

        # 2. Launch the GPU subprocess.
        db_arg = str(self._db_path) if self._db_path else ""
        cmd = [str(_GPU_BINARY)]
        if db_arg:
            cmd.append(db_arg)
        cmd += ["--serve", "--gpu-trace"]

        child_env = dict(os.environ)
        child_env.update(_GPU_ENV_EXTRAS)

        try:
            self._gpu_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=child_env,
            )
        except OSError as exc:
            # GPU subprocess failed to start (e.g. missing CUDA driver).
            # Reopen CPU connection and stay in CPU mode.
            print(
                f"[gpu_connection] GPU subprocess failed to start ({exc}). "
                "Falling back to CPU for this SQL.",
                flush=True,
            )
            self._cpu_con = self._new_cpu_con()
            return

        # 3. Send per-session init SQL (threads, memory, FP64 pragmas).
        init_parts = [
            f"PRAGMA threads = {self._threads};",
            f"SET memory_limit='{self._memory_limit}';",
        ]
        if self._spill_dir:
            spill_posix = (
                Path(self._spill_dir).resolve().as_posix().replace("'", "''")
            )
            init_parts += [
                f"SET temp_directory='{spill_posix}';",
                f"SET max_temp_directory_size='{self._max_temp_directory_size}';",
            ]
        init_parts.append(_GPU_PRECISION_INIT_SQL)
        try:
            self._gpu_send_raw("\n".join(init_parts))
        except Exception as exc:
            print(
                f"[gpu_connection] WARNING: GPU init SQL failed ({exc}). "
                "Falling back to CPU.",
                flush=True,
            )
            self._gpu_proc.kill()
            self._gpu_proc = None
            self._cpu_con = self._new_cpu_con()
            return

        self._mode = "gpu"
        print(
            f"[gpu_connection] CPU → GPU hand-off (PID {self._gpu_proc.pid})",
            flush=True,
        )

    def _switch_to_cpu(self) -> None:
        """Return file ownership from the GPU subprocess back to duckdb-python.

        Steps:
        1. Send CHECKPOINT to the GPU subprocess.
        2. Close stdin -- the serve loop exits, DuckDB checkpoints on close.
        3. Wait for the process to exit -- releases the file lock.
        4. Reopen the CPU connection (with UDF re-registration).
        """
        if self._mode == "cpu":
            return

        # 1. Ask the subprocess to checkpoint before we take the lock back.
        try:
            self._gpu_send_raw("CHECKPOINT;")
        except Exception:
            pass

        # 2–3. Shut down the subprocess.
        try:
            if self._gpu_proc is not None:
                if self._gpu_proc.stdin:
                    self._gpu_proc.stdin.close()
                if self._gpu_proc.stdout:
                    self._gpu_proc.stdout.close()
                if self._gpu_proc.stderr:
                    self._gpu_proc.stderr.close()
                self._gpu_proc.wait(timeout=30)
        except Exception as exc:
            print(
                f"[gpu_connection] WARNING: GPU shutdown error ({exc})", flush=True
            )
            if self._gpu_proc is not None:
                self._gpu_proc.kill()
        self._gpu_proc = None

        # 4. Reopen CPU connection (re-registers UDFs automatically).
        self._cpu_con = self._new_cpu_con()
        self._mode = "cpu"
        print("[gpu_connection] GPU → CPU hand-off", flush=True)

    # ------------------------------------------------------------------
    # GPU subprocess I/O (--serve protocol from gpu_shell/main.cpp)
    # ------------------------------------------------------------------

    def _gpu_send_raw(self, sql: str) -> list[tuple]:
        """Send ``sql`` over the --serve pipe; return parsed rows."""
        proc = self._gpu_proc
        if proc is None:
            raise RuntimeError("GPU subprocess is not running")
        proc.stdin.write(sql.rstrip("\n") + "\n--END--\n")
        proc.stdin.flush()
        out_lines: list[str] = []
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError(
                    "duckdb_gpu process terminated unexpectedly while reading"
                )
            line = line.rstrip("\r\n")
            if line == "--DONE--":
                break
            out_lines.append(line)
        return _parse_text_table("\n".join(out_lines))

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _should_gpu(self, sql: str) -> bool:
        """True if this SQL should be routed to the GPU subprocess."""
        if not GPU_AVAILABLE:
            return False
        sql_u = sql.upper()
        return any(p.upper() in sql_u for p in _GPU_PATTERNS)

    # ------------------------------------------------------------------
    # Public API (duckdb.DuckDBPyConnection-compatible subset)
    # ------------------------------------------------------------------

    def execute(self, sql: str, parameters=None):
        """Execute ``sql``, routing to GPU or CPU based on SQL content."""
        if self._closed:
            raise RuntimeError("HybridConnection is closed")

        if self._should_gpu(sql):
            if self._mode == "cpu":
                self._switch_to_gpu()
            # If _switch_to_gpu() fell back to CPU (e.g. no CUDA driver),
            # _mode is still "cpu" -- run on CPU normally.
            if self._mode == "gpu":
                if parameters:
                    sql = _substitute_params(sql, parameters)
                rows = self._gpu_send_raw(sql)
                return _Cursor(rows)
        else:
            if self._mode == "gpu":
                self._switch_to_cpu()

        # CPU path (default and fallback)
        return self._cpu_con.execute(sql, parameters or [])

    def cursor(self):
        """Return a CPU cursor (forces GPU→CPU switch; needed for fetchmany streaming)."""
        if self._mode == "gpu":
            self._switch_to_cpu()
        return self._cpu_con.cursor()

    def create_function(self, name: str, func, arg_types, return_type) -> None:
        """Register a Python UDF on the CPU connection.

        The UDF is stored in ``_udfs`` so it is re-registered automatically
        every time the CPU connection is (re)opened after a GPU→CPU hand-off.
        """
        if self._mode == "gpu":
            # GPU subprocess cannot accept Python UDFs; hand off first.
            self._switch_to_cpu()
        self._udfs.append((name, func, arg_types, return_type))
        self._cpu_con.create_function(name, func, arg_types, return_type)

    def close(self) -> None:
        """Checkpoint, close whichever connection is active, and clean up."""
        if self._closed:
            return
        self._closed = True
        if self._mode == "gpu" and self._gpu_proc is not None:
            try:
                self._gpu_send_raw("CHECKPOINT;")
            except Exception:
                pass
            try:
                if self._gpu_proc.stdin:
                    self._gpu_proc.stdin.close()
                if self._gpu_proc.stdout:
                    self._gpu_proc.stdout.close()
                if self._gpu_proc.stderr:
                    self._gpu_proc.stderr.close()
                self._gpu_proc.wait(timeout=15)
            except Exception:
                self._gpu_proc.kill()
            self._gpu_proc = None
        elif self._cpu_con is not None:
            try:
                self._cpu_con.close()
            except Exception:
                pass
        self._cpu_con = None


# ---------------------------------------------------------------------------
# Public factory -- drop-in for duckdb.connect()
# ---------------------------------------------------------------------------

def connect(
    db_path: str | Path | None = None,
    threads: int = 4,
    memory_limit: str = "1536MB",
    spill_dir: str | Path | None = None,
    max_temp_directory_size: str = "20GB",
):
    """Open a DuckDB connection, GPU-accelerated when the binary is present.

    Resource handling
    -----------------
    The returned ``HybridConnection`` ensures exactly ONE DuckDB process holds
    the database file at a time, so ``threads`` and ``memory_limit`` are given
    in full to whichever process is currently active.  There is no splitting or
    doubling -- the values you pass here are what DuckDB actually uses.

    GPU precision settings (applied automatically when GPU is active)
    -----------------------------------------------------------------
    - gpu_fast_math = false           (IEEE-754; no NVRTC --use_fast_math)
    - gpu_precision_relaxation = false (strict FP64; no float32 downcast)
    - gpu_stream_pipeline = true      (GPU-resident column cache enabled)
    """
    if not GPU_AVAILABLE:
        print(
            "[gpu_connection] duckdb_gpu not found at "
            f"{_GPU_BINARY} -- using plain duckdb.",
            flush=True,
        )
        import duckdb
        con = duckdb.connect(str(db_path) if db_path else ":memory:")
        con.execute(f"PRAGMA threads = {threads}")
        con.execute(f"SET memory_limit='{memory_limit}'")
        if spill_dir:
            spill_posix = (
                Path(spill_dir).resolve().as_posix().replace("'", "''")
            )
            con.execute(f"SET temp_directory='{spill_posix}'")
            con.execute(f"SET max_temp_directory_size='{max_temp_directory_size}'")
        return con

    return HybridConnection(
        db_path=db_path,
        threads=threads,
        memory_limit=memory_limit,
        spill_dir=spill_dir,
        max_temp_directory_size=max_temp_directory_size,
    )
