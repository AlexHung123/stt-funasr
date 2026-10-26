"""Disk-backed transcription queue.

One job runs at a time. A few more may wait on disk. Audio stays under
``root/<job_id>/`` until the worker finishes, then only the status and result
JSON are kept.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

MAX_WAITING_DEFAULT = 3
QUEUE_FULL_MESSAGE = (
    "Someone else is transcribing. Try again in a little while."
)
RESTART_MESSAGE = "Transcription service restarted before this job finished."

Runner = Callable[["JobRecord"], Awaitable[dict[str, Any]]]


class QueueFull(Exception):
    def __init__(self) -> None:
        super().__init__(QUEUE_FULL_MESSAGE)


@dataclass
class JobRecord:
    id: str
    status: str
    filename: str
    language: Optional[str]
    spk: bool
    source_path: str
    error: Optional[str] = None
    result: Optional[dict[str, Any]] = None


class TranscriptionQueue:
    def __init__(self, root: Path, max_waiting: int = MAX_WAITING_DEFAULT) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_waiting = max_waiting
        self._jobs: dict[str, JobRecord] = {}
        # uploading and queued jobs occupy a slot. Running jobs do not.
        self._occupying: list[str] = []
        self._cond = asyncio.Condition()

    def waiting_count(self) -> int:
        return len(self._occupying)

    def begin_upload(
        self,
        filename: str,
        language: Optional[str],
        spk: bool,
    ) -> JobRecord:
        if len(self._occupying) >= self.max_waiting:
            raise QueueFull()
        job_id = uuid.uuid4().hex
        safe_name = Path(filename).name[:200] or "audio"
        ext = Path(safe_name).suffix.lower()
        if len(ext) < 2 or len(ext) > 8 or not ext[1:].isalnum():
            ext = ".bin"
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        source = job_dir / f"source{ext}"
        job = JobRecord(
            id=job_id,
            status="uploading",
            filename=safe_name,
            language=(language or "").strip() or None,
            spk=spk,
            source_path=str(source),
        )
        self._jobs[job_id] = job
        self._occupying.append(job_id)
        self._write_status(job)
        return job

    def abandon(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is None or job.status != "uploading":
            return
        self._occupying = [item for item in self._occupying if item != job_id]
        job.status = "failed"
        job.error = "Upload did not finish."
        self._write_status(job)
        self._delete_media(job)

    def mark_queued(self, job_id: str) -> JobRecord:
        job = self._jobs[job_id]
        if job.status != "uploading":
            raise RuntimeError(f"job {job_id} is {job.status}")
        job.status = "queued"
        self._write_status(job)
        return job

    async def notify(self) -> None:
        async with self._cond:
            self._cond.notify_all()

    def get(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    def public_view(self, job_id: str) -> Optional[dict[str, Any]]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        body: dict[str, Any] = {
            "id": job.id,
            "status": job.status,
            "filename": job.filename,
            "error": job.error,
        }
        if job.status == "succeeded":
            body["result"] = job.result
        return body

    def cancel(self, job_id: str) -> Optional[str]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.status == "queued":
            job.status = "cancelled"
            self._occupying = [item for item in self._occupying if item != job_id]
            self._write_status(job)
            self._delete_media(job)
            return "cancelled"
        return job.status

    def load_restarted(self) -> None:
        """Fail jobs left behind by a process that exited mid-upload or mid-run."""
        if not self.root.exists():
            return
        for job_dir in self.root.iterdir():
            if not job_dir.is_dir():
                continue
            status_path = job_dir / "status.json"
            if not status_path.is_file():
                continue
            try:
                data = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            job_id = str(data.get("id") or job_dir.name)
            if job_id in self._jobs:
                continue
            job = JobRecord(
                id=job_id,
                status=str(data.get("status") or "failed"),
                filename=str(data.get("filename") or "audio"),
                language=data.get("language") if isinstance(data.get("language"), str) else None,
                spk=bool(data.get("spk", True)),
                source_path=str(job_dir / "source.bin"),
                error=data.get("error") if isinstance(data.get("error"), str) else None,
            )
            result_path = job_dir / "result.json"
            if result_path.is_file():
                try:
                    loaded = json.loads(result_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        job.result = loaded
                except (OSError, json.JSONDecodeError):
                    job.result = None
            if job.status in ("uploading", "queued", "running"):
                job.status = "failed"
                job.error = RESTART_MESSAGE
                self._delete_dir_media(job_dir)
                self._write_status(job)
            self._jobs[job.id] = job

    async def serve(self, runner: Runner) -> None:
        while True:
            job = await self._next_queued()
            job.status = "running"
            self._write_status(job)
            try:
                job.result = await runner(job)
                job.status = "succeeded"
                job.error = None
            except asyncio.CancelledError:
                job.status = "failed"
                job.error = RESTART_MESSAGE
                raise
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc) or "Transcription failed"
            finally:
                self._write_status(job)
                self._delete_media(job)

    async def _next_queued(self) -> JobRecord:
        async with self._cond:
            while True:
                chosen: Optional[JobRecord] = None
                for job_id in list(self._occupying):
                    job = self._jobs.get(job_id)
                    if job is not None and job.status == "queued":
                        chosen = job
                        break
                if chosen is not None:
                    self._occupying = [
                        item for item in self._occupying if item != chosen.id
                    ]
                    return chosen
                await self._cond.wait()

    def _write_status(self, job: JobRecord) -> None:
        job_dir = self.root / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": job.id,
            "status": job.status,
            "filename": job.filename,
            "language": job.language,
            "spk": job.spk,
            "error": job.error,
        }
        (job_dir / "status.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        if job.result is not None:
            (job_dir / "result.json").write_text(
                json.dumps(job.result),
                encoding="utf-8",
            )

    def _delete_media(self, job: JobRecord) -> None:
        self._delete_dir_media(self.root / job.id)

    def _delete_dir_media(self, job_dir: Path) -> None:
        if not job_dir.is_dir():
            return
        for child in job_dir.iterdir():
            if child.name in ("status.json", "result.json"):
                continue
            if child.is_file():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
