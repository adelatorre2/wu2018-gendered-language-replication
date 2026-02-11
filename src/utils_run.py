from __future__ import annotations

from pathlib import Path
import shlex
import subprocess


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def raw_pkg_dir() -> Path:
    return repo_root() / "data" / "raw" / "openicpsr_wu2018_replication-pkg"


def ensure_output_dirs() -> None:
    root = repo_root() / "output"
    for name in ["logs", "figures", "tables", "intermediate"]:
        (root / name).mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], log_path: Path, cwd: Path | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {shlex.join(cmd)}\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if proc.stdout is None:
            raise RuntimeError("Failed to capture subprocess output")
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"Command failed with exit code {proc.returncode}. See log: {log_path}"
            )
