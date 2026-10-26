#!/usr/bin/env python3
"""
Local FunASR / SenseVoice HTTP server (OpenAI-compatible).

Defaults are tuned for this Mac (Apple Silicon): SenseVoice on MPS when available,
otherwise CPU. Wraps FunASR's built-in FastAPI app and patches VAD / generate
kwargs for long-form audio/video (e.g. ~1 hour).

Video containers (mp4/mkv/webm/mov) are converted in-process with ffmpeg to
16 kHz mono PCM WAV before ASR. Common audio (wav/mp3/...) keeps FunASR's
original path. No HTTP loopback — conversion then calls the in-app handler.

Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/audio/transcriptions   (OpenAI-compatible; video + audio)
  POST /asr                       (FunASR REST; video + audio)
  GET  /docs                      (Swagger UI)

Examples:
  python server.py
  python server.py --device cpu --port 8000
  python server.py --model sensevoice --device mps
  python server.py --vad-max-single-segment-time 60000 --batch-size-s 60

  curl -X POST http://127.0.0.1:8002/v1/audio/transcriptions \\
    -F file=@clip.mp4 -F model=sensevoice -F response_format=verbose_json \\
    -o result.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from jobs import QUEUE_FULL_MESSAGE, QueueFull, TranscriptionQueue

# Must be module-level: with `from __future__ import annotations`, route param
# annotations become ForwardRefs resolved against module globals — not the
# local namespace of install_video_input_support(). Local-only imports leave
# UploadFile undefined and crash Pydantic with "not fully defined".
from fastapi import File, Form, HTTPException, Request, UploadFile

# FunASR AutoModel (SenseVoice) is not concurrency-safe: shared kwargs / VAD state
# and temp-file lifecycles race under concurrent requests. Serialize prepare + ASR.
_asr_lock = asyncio.Lock()
_original_transcribe: Optional[Callable] = None


def detect_device() -> str:
    """Prefer MPS on Apple Silicon, then CUDA, else CPU."""
    try:
        import torch

        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# Video containers that need an ffmpeg extract step before STT.
VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".webm", ".mov"})


def _file_suffix(filename: Optional[str]) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower()


def _require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it to transcribe video "
            "(e.g. brew install ffmpeg)."
        )
    return path


def ffmpeg_extract_wav(src_path: str, dst_wav_path: str) -> None:
    """
    Extract audio to 16 kHz mono PCM WAV for ASR.

      ffmpeg -y -i input -ar 16000 -ac 1 -c:a pcm_s16le /tmp/meeting.wav
    """
    ffmpeg = _require_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        src_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        dst_wav_path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg timed out while extracting audio from video") from exc
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        tail = err[-800:] if err else str(exc)
        raise RuntimeError(f"ffmpeg failed to extract audio: {tail}") from exc
    if not os.path.isfile(dst_wav_path) or os.path.getsize(dst_wav_path) == 0:
        raise RuntimeError("ffmpeg produced an empty or missing WAV file")
    # Keep stderr available for debugging on quiet success paths if needed.
    _ = proc


def _unlink_quiet(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


@dataclass
class PreparedMedia:
    """Upload handed to FunASR after optional video→wav conversion."""

    upload: Any
    cleanup_paths: list[str] = field(default_factory=list)
    _open_handles: list[Any] = field(default_factory=list)

    def cleanup(self) -> None:
        for handle in self._open_handles:
            try:
                handle.close()
            except OSError:
                pass
        self._open_handles.clear()
        for path in self.cleanup_paths:
            _unlink_quiet(path)
        self.cleanup_paths.clear()


class _AsyncFileUpload:
    """Minimal UploadFile-compatible object for FunASR handlers (async read + filename)."""

    def __init__(self, filename: str, fileobj: Any):
        self.filename = filename
        self.file = fileobj

    async def read(self, size: int = -1) -> bytes:
        return self.file.read(size)

    async def seek(self, offset: int) -> None:
        self.file.seek(offset)

    async def close(self) -> None:
        self.file.close()


async def prepare_media_for_asr(file: Any) -> PreparedMedia:
    """
    1) Receive upload → temp input (video only)
    2) mp4/mkv/webm/mov → ffmpeg 16k mono wav; else keep original bytes/path logic
    3) Caller runs in-process FunASR handler
    4) Caller must .cleanup() temp files
    """
    filename = getattr(file, "filename", None) or "audio.bin"
    suffix = _file_suffix(filename)

    # Non-video (wav/mp3/flac/...): original FunASR path — do not re-encode.
    if suffix not in VIDEO_EXTENSIONS:
        if hasattr(file, "seek"):
            await file.seek(0)
        return PreparedMedia(upload=file)

    content = await file.read()
    if not content:
        raise ValueError("Empty upload")

    cleanup: list[str] = []
    fd_in, path_in = tempfile.mkstemp(prefix="stt_in_", suffix=suffix)
    os.close(fd_in)
    cleanup.append(path_in)
    try:
        with open(path_in, "wb") as f:
            f.write(content)

        fd_out, path_wav = tempfile.mkstemp(prefix="stt_wav_", suffix=".wav")
        os.close(fd_out)
        cleanup.append(path_wav)

        print(f"  video-input: ffmpeg extract {suffix} → 16k mono wav ({filename})")
        ffmpeg_extract_wav(path_in, path_wav)

        # FunASR handlers re-read the upload; feed a fresh WAV-compatible object.
        wav_file = open(path_wav, "rb")
        upload = _AsyncFileUpload(filename="meeting.wav", fileobj=wav_file)
        return PreparedMedia(
            upload=upload,
            cleanup_paths=cleanup,
            _open_handles=[wav_file],
        )
    except Exception:
        for path in cleanup:
            _unlink_quiet(path)
        raise


def prepare_media_from_path(source_path: str, filename: str) -> PreparedMedia:
    """Open a file already stored on disk. Video is extracted to wav beside it."""
    suffix = _file_suffix(filename) or _file_suffix(source_path)
    if suffix not in VIDEO_EXTENSIONS:
        handle = open(source_path, "rb")
        upload = _AsyncFileUpload(
            filename=Path(filename).name or "audio.bin",
            fileobj=handle,
        )
        return PreparedMedia(upload=upload, _open_handles=[handle])

    wav_path = str(Path(source_path).with_suffix(".asr.wav"))
    print(f"  video-input: ffmpeg extract {suffix} → 16k mono wav ({filename})")
    ffmpeg_extract_wav(source_path, wav_path)
    handle = open(wav_path, "rb")
    upload = _AsyncFileUpload(filename="meeting.wav", fileobj=handle)
    return PreparedMedia(
        upload=upload,
        cleanup_paths=[wav_path],
        _open_handles=[handle],
    )


def _remove_post_routes(app: Any, paths: set[str]) -> dict[str, Callable]:
    """Drop POST routes for the given paths; return path → original endpoint."""
    found: dict[str, Callable] = {}
    keep = []
    for route in app.router.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or set()
        if path in paths and "POST" in methods:
            found[path] = route.endpoint
        else:
            keep.append(route)
    app.router.routes[:] = keep
    return found


def install_video_input_support(app: Any) -> None:
    """
    Wrap FunASR transcription routes so video is converted in-process before ASR.

    Flow (serialized under ``_asr_lock``):
      file → temp → (optional ffmpeg) → original handler → cleanup → JSON.

    Concurrent requests queue on the lock (FIFO within one process). Audio
    formats (mp3/wav/...) still call the original handler, but under the same
    lock so SenseVoice is never shared across in-flight jobs.
    """
    originals = _remove_post_routes(
        app, {"/v1/audio/transcriptions", "/asr"}
    )
    if not originals:
        print("  video-input: warning — no transcription routes found to wrap")
        return

    original_transcribe = originals.get("/v1/audio/transcriptions")
    original_asr = originals.get("/asr")
    global _original_transcribe
    _original_transcribe = original_transcribe

    if original_transcribe is not None:

        @app.post("/v1/audio/transcriptions")
        async def transcribe_with_video(  # type: ignore[no-redef]
            file: UploadFile = File(...),
            model: str = Form(default="sensevoice"),
            language: Optional[str] = Form(default=None),
            response_format: Optional[str] = Form(default="json"),
            spk: bool = Form(default=False),
        ):
            # One ASR job at a time (shared AutoModel + temp-file lifecycle).
            async with _asr_lock:
                try:
                    prepared = await prepare_media_for_asr(file)
                except HTTPException:
                    raise
                except (RuntimeError, ValueError) as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                except Exception as exc:
                    raise HTTPException(
                        status_code=500, detail=f"media prepare failed: {exc}"
                    ) from exc
                try:
                    return await original_transcribe(
                        file=prepared.upload,
                        model=model,
                        language=language,
                        response_format=response_format,
                        spk=spk,
                    )
                finally:
                    prepared.cleanup()

        print("  video-input: wrapped POST /v1/audio/transcriptions (serialized)")

    if original_asr is not None:

        @app.post("/asr")
        async def asr_with_video(  # type: ignore[no-redef]
            file: UploadFile = File(...),
            language: Optional[str] = Form(default=None),
            hotwords: str = Form(default=""),
            spk: bool = Form(default=False),
        ):
            async with _asr_lock:
                try:
                    prepared = await prepare_media_for_asr(file)
                except HTTPException:
                    raise
                except (RuntimeError, ValueError) as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                except Exception as exc:
                    raise HTTPException(
                        status_code=500, detail=f"media prepare failed: {exc}"
                    ) from exc
                try:
                    return await original_asr(
                        file=prepared.upload,
                        language=language,
                        hotwords=hotwords,
                        spk=spk,
                    )
                finally:
                    prepared.cleanup()

        print("  video-input: wrapped POST /asr (serialized)")


def _payload_from_response(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    body = getattr(response, "body", None)
    if isinstance(body, memoryview):
        body = body.tobytes()
    if isinstance(body, bytearray):
        body = bytes(body)
    if isinstance(body, bytes):
        parsed = json.loads(body.decode("utf-8"))
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, str):
            return {"text": parsed}
    raise RuntimeError("transcription handler returned an unreadable response")


def install_transcription_jobs(
    app: Any,
    *,
    jobs_dir: Path,
    max_waiting: int,
    max_bytes: int,
) -> TranscriptionQueue:
    """Accept a streamed upload, keep it on disk, and transcribe it one job at a time.

    Speaker diarization stays on unless the caller sends ``X-Spk: false``.
    """
    queue = TranscriptionQueue(jobs_dir, max_waiting=max_waiting)
    queue.load_restarted()

    async def run_job(job: Any) -> dict[str, Any]:
        if _original_transcribe is None:
            raise RuntimeError("transcription route is not installed")
        async with _asr_lock:
            prepared = await asyncio.to_thread(
                prepare_media_from_path,
                job.source_path,
                job.filename,
            )
            try:
                try:
                    response = await _original_transcribe(
                        file=prepared.upload,
                        model="sensevoice",
                        language=job.language,
                        response_format="verbose_json",
                        spk=job.spk,
                    )
                except HTTPException as exc:
                    detail = exc.detail if isinstance(exc.detail, str) else "Transcription failed"
                    raise RuntimeError(detail) from exc
                return _payload_from_response(response)
            finally:
                prepared.cleanup()

    previous_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _jobs_lifespan(app_):
        task = asyncio.create_task(queue.serve(run_job))
        app_.state.transcription_worker = task
        try:
            async with previous_lifespan(app_):
                yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app.router.lifespan_context = _jobs_lifespan

    @app.post("/v1/audio/jobs")
    async def create_transcription_job(request: Request):
        filename = (request.headers.get("x-filename") or "audio.bin").strip() or "audio.bin"
        language = (request.headers.get("x-language") or "").strip() or None
        spk_header = (request.headers.get("x-spk") or "true").strip().lower()
        spk = spk_header not in ("0", "false", "no")
        try:
            job = queue.begin_upload(filename, language, spk)
        except QueueFull as exc:
            raise HTTPException(status_code=429, detail=QUEUE_FULL_MESSAGE) from exc

        try:
            size = 0
            with open(job.source_path, "wb") as out:
                async for chunk in request.stream():
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File must be {max_bytes} bytes or smaller",
                        )
                    out.write(chunk)
            if size <= 0:
                raise HTTPException(status_code=400, detail="This file is empty.")
            queue.mark_queued(job.id)
            await queue.notify()
        except HTTPException:
            queue.abandon(job.id)
            raise
        except Exception as exc:
            queue.abandon(job.id)
            raise HTTPException(status_code=400, detail="Upload did not finish.") from exc
        return {"id": job.id, "status": "queued"}

    @app.get("/v1/audio/jobs/{job_id}")
    async def read_transcription_job(job_id: str):
        body = queue.public_view(job_id)
        if body is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return body

    @app.delete("/v1/audio/jobs/{job_id}")
    async def cancel_transcription_job(job_id: str):
        status = queue.cancel(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"id": job_id, "status": status}

    print(
        f"  jobs: dir={jobs_dir} waiting={max_waiting} "
        f"max_bytes={max_bytes} spk default=on"
    )
    return queue


def _ensure_python_runtime_ok() -> None:
    """Refuse to start if this process cannot import stdlib multiprocessing.dummy.

    Homebrew replacing python@3.13 while a long-lived server is still running
    leaves the process bound to a deleted prefix. ASR can keep going from
    already-loaded modules; speaker clustering then dies on a lazy import of
    ``multiprocessing.dummy`` (joblib ThreadPool → UMAP/HDBSCAN).
    """
    prefix = sys.base_prefix or sys.prefix
    if not os.path.isdir(prefix):
        print(f"ERROR: Python prefix is missing on disk: {prefix}")
        print("Homebrew likely upgraded Python while an old server process was running.")
        print("Stop that process and start again with .venv (see start_server.sh).")
        sys.exit(1)
    try:
        import multiprocessing.dummy  # noqa: F401
    except ModuleNotFoundError:
        print("ERROR: stdlib module multiprocessing.dummy is missing.")
        print(f"  interpreter: {sys.executable}")
        print(f"  version:     {sys.version.split()[0]}")
        print(f"  prefix:      {prefix}")
        print("This usually means the Homebrew Python that created this process was uninstalled.")
        print("Restart the server. If it still fails, recreate the venv:")
        print("  rm -rf .venv && python3 -m venv .venv && source .venv/bin/activate")
        print("  pip install -r requirements.txt")
        sys.exit(1)


def _spectral_labels_for_embeddings(cluster: Any, X: Any, oracle_num: Any) -> Any:
    """Spectral clustering, subsampled when N is large (O(N²) otherwise)."""
    import numpy as np

    x_np = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
    n = int(x_np.shape[0])
    if n < 20:
        return np.zeros(n, dtype="int")

    cap = 2047
    if n <= cap:
        labels = cluster.spectral_cluster(x_np, oracle_num)
    else:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n, size=cap, replace=False))
        sub_labels = cluster.spectral_cluster(x_np[idx], oracle_num)
        n_spk = int(sub_labels.max()) + 1
        centers = []
        for i in range(n_spk):
            members = x_np[idx][sub_labels == i]
            if members.size == 0:
                centers.append(np.zeros(x_np.shape[1], dtype=x_np.dtype))
            else:
                centers.append(members.mean(0))
        centers = np.stack(centers, axis=0)
        x_n = x_np / np.clip(np.linalg.norm(x_np, axis=1, keepdims=True), 1e-8, None)
        c_n = centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8, None)
        labels = (x_n @ c_n.T).argmax(axis=1)

    if oracle_num is None and "merge_thr" in getattr(cluster, "model_config", {}):
        labels = cluster.merge_by_cos(labels, x_np, cluster.model_config["merge_thr"])
    return labels


def install_speaker_cluster_safety() -> None:
    """
    Speaker diarization (spk=true) runs after ASR. On long media FunASR uses
    UMAP+HDBSCAN, which pulls in joblib ThreadPool → multiprocessing.dummy.

    - Force UMAP onto one thread so joblib does not need a ThreadPool.
    - If UMAP still fails, fall back to spectral clustering (subsampled).
    - If labeling still fails, return the transcript without speaker IDs
      instead of turning a finished ASR job into HTTP 500.
    """
    import funasr.bin._server_app as funasr_server_app
    import umap.umap_ as umap
    from funasr.models.campplus.cluster_backend import ClusterBackend

    original_umap_init = umap.UMAP.__init__

    def umap_init_single_thread(self, *args: Any, **kwargs: Any):
        kwargs["n_jobs"] = 1
        return original_umap_init(self, *args, **kwargs)

    umap.UMAP.__init__ = umap_init_single_thread  # type: ignore[method-assign]
    print("  speaker-cluster: UMAP n_jobs=1 (avoid joblib ThreadPool)")

    original_forward = ClusterBackend.forward

    def forward_with_fallback(self, X, **params):
        try:
            return original_forward(self, X, **params)
        except Exception as exc:
            print(
                f"  speaker-cluster: UMAP/HDBSCAN failed "
                f"({type(exc).__name__}: {exc}); spectral fallback"
            )
            oracle_num = params.get("oracle_num")
            return _spectral_labels_for_embeddings(self, X, oracle_num)

    ClusterBackend.forward = forward_with_fallback  # type: ignore[method-assign]

    original_attach = funasr_server_app.attach_speaker_labels

    def attach_speaker_labels_safe(audio_data, sr, segments, speaker_model, device):
        try:
            return original_attach(audio_data, sr, segments, speaker_model, device)
        except Exception as exc:
            print(
                f"  speaker-labels: failed ({type(exc).__name__}: {exc}); "
                "returning transcript without speaker IDs"
            )
            return segments

    funasr_server_app.attach_speaker_labels = attach_speaker_labels_safe
    print("  speaker-cluster: ASR result kept if diarization fails")


def build_parser() -> argparse.ArgumentParser:
    default_device = detect_device()
    parser = argparse.ArgumentParser(
        description="FunASR SenseVoice HTTP server (OpenAI-compatible, long-audio VAD)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python server.py
  python server.py --device cpu --port 9000
  python server.py --model sensevoice --hub ms
  # Long lecture / 1h video (defaults already target this):
  python server.py --vad-max-single-segment-time 60000 --merge-vad --batch-size-s 60

Test with curl:
  curl -X POST http://127.0.0.1:8002/v1/audio/transcriptions \\
    -F file=@long.wav \\
    -F model=sensevoice \\
    -F response_format=verbose_json \\
    -o result.json

  # Video (mp4/mkv/webm/mov) — server extracts 16k mono wav via ffmpeg:
  curl -X POST http://127.0.0.1:8002/v1/audio/transcriptions \\
    -F file=@clip.mp4 \\
    -F model=sensevoice \\
    -F response_format=verbose_json \\
    -o result.json
""",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8002, help="Port (default: 8002)")
    parser.add_argument(
        "--device",
        default=default_device,
        help=f"Device: cuda, cpu, mps (default: auto → {default_device})",
    )
    parser.add_argument(
        "--model",
        default="sensevoice",
        help="Pre-load model: sensevoice, paraformer, fun-asr-nano, auto (default: sensevoice)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local model path or hub model ID (overrides --model)",
    )
    parser.add_argument(
        "--hub",
        default="ms",
        choices=["ms", "hf"],
        help="Model hub: ms (ModelScope) or hf (HuggingFace) (default: ms)",
    )
    parser.add_argument(
        "--spk-model",
        default="cam++",
        help="Speaker model for spk=true requests (default: cam++)",
    )
    parser.add_argument(
        "--jobs-dir",
        default=str(Path(__file__).resolve().parent / "stt-jobs"),
        help="Directory for queued audio on disk (default: ./stt-jobs next to server.py)",
    )
    parser.add_argument(
        "--max-waiting-jobs",
        type=int,
        default=3,
        help="How many uploads may wait while one job is transcribing (default: 3)",
    )
    parser.add_argument(
        "--max-upload-bytes",
        type=int,
        default=2 * 1024 * 1024 * 1024,
        help="Largest accepted upload in bytes (default: 2 GiB)",
    )
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=None,
        metavar="ORIGIN",
        help="Trusted browser origin for CORS; repeat for multiple origins",
    )

    # --- Long-form audio / video VAD & batching ---
    # FunASR stock server uses max_single_segment_time=30000 and does NOT pass
    # merge_vad / batch_size_s. For ~1h media, VAD still chunks speech; we tune
    # segment length, silence cut, merge of short clips, and ASR batch seconds.
    parser.add_argument(
        "--vad-max-single-segment-time",
        type=int,
        default=60000,
        metavar="MS",
        help=(
            "VAD: max continuous speech segment before forced cut, in ms "
            "(default: 60000 = 60s). Keep 30–90s; do NOT set to full file length."
        ),
    )
    parser.add_argument(
        "--vad-max-end-silence-time",
        type=int,
        default=800,
        metavar="MS",
        help=(
            "VAD: silence duration that ends a speech segment, in ms "
            "(default: 800). Raise to ~1200–1500 for sparse lecture pauses."
        ),
    )
    parser.add_argument(
        "--merge-vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge adjacent short VAD clips before ASR (default: on; SenseVoice long-form)",
    )
    parser.add_argument(
        "--merge-length-s",
        type=int,
        default=15,
        metavar="SEC",
        help="Max merged VAD clip length in seconds when --merge-vad (default: 15)",
    )
    parser.add_argument(
        "--batch-size-s",
        type=int,
        default=60,
        metavar="SEC",
        help=(
            "ASR dynamic batch budget in seconds of speech (default: 60; "
            "safer for long files on MPS/CPU; raise to 120–300 if you have RAM)"
        ),
    )
    return parser


def _set_vad_opt(vad_model: Any, name: str, value: Any) -> bool:
    """Set a field on the live VAD model if present."""
    opts = getattr(vad_model, "vad_opts", None)
    if opts is None or not hasattr(opts, name):
        return False
    setattr(opts, name, value)
    return True


def _wrap_generate(model: Any, generate_defaults: dict[str, Any]) -> None:
    """Inject long-form defaults into every model.generate() call."""
    original: Callable = model.generate

    def generate_with_long_audio(*args: Any, **kwargs: Any):
        for key, value in generate_defaults.items():
            kwargs.setdefault(key, value)
        return original(*args, **kwargs)

    model.generate = generate_with_long_audio  # type: ignore[method-assign]


def apply_long_audio_settings(
    app: Any,
    *,
    vad_max_single_segment_time: int,
    vad_max_end_silence_time: int,
    merge_vad: bool,
    merge_length_s: int,
    batch_size_s: int,
) -> None:
    """
    Patch FunASR's stock app for hour-scale media.

    Stock create_app hardcodes vad max segment 30s and generate(batch_size=1)
    without merge_vad. We:
      1) update live VAD opts (segment / silence)
      2) wrap AutoModel.generate with merge_vad + batch_size_s defaults
    """
    generate_defaults: dict[str, Any] = {
        "merge_vad": merge_vad,
        "merge_length_s": merge_length_s,
        "batch_size_s": batch_size_s,
        # Prefer duration-based batching over stock batch_size=1
        "batch_size": 1,
    }

    patched = 0

    # AutoModel fallbacks (SenseVoice / Paraformer / custom)
    fallbacks = getattr(app.state, "fallback_models", {}) or {}
    for name, model in list(fallbacks.items()):
        vad = getattr(model, "vad_model", None)
        if vad is not None:
            _set_vad_opt(vad, "max_single_segment_time", vad_max_single_segment_time)
            _set_vad_opt(vad, "max_end_silence_time", vad_max_end_silence_time)
            if hasattr(model, "vad_kwargs") and isinstance(model.vad_kwargs, dict):
                model.vad_kwargs["max_single_segment_time"] = vad_max_single_segment_time
                model.vad_kwargs["max_end_silence_time"] = vad_max_end_silence_time
        _wrap_generate(model, generate_defaults)
        patched += 1
        print(f"  long-audio: patched fallback model '{name}'")

    # Standalone VAD used by Fun-ASR-Nano vLLM path (AutoModel("fsmn-vad"))
    standalone_vad = getattr(app.state, "vad_model", None)
    if standalone_vad is not None:
        # Prefer nested nn.Module with vad_opts, else the object itself
        candidates = [
            getattr(standalone_vad, "model", None),
            getattr(standalone_vad, "vad_model", None),
            standalone_vad,
        ]
        for cand in candidates:
            if cand is None:
                continue
            if _set_vad_opt(cand, "max_single_segment_time", vad_max_single_segment_time):
                _set_vad_opt(cand, "max_end_silence_time", vad_max_end_silence_time)
                break
        if hasattr(standalone_vad, "generate"):
            # vLLM path only runs VAD.generate; merge_vad is handled in ASR path
            _wrap_generate(
                standalone_vad,
                {"max_end_silence_time": vad_max_end_silence_time},
            )
        patched += 1
        print("  long-audio: patched standalone VAD model")

    if patched == 0:
        print("  long-audio: warning — no models found to patch yet")


def main() -> None:
    args = build_parser().parse_args()
    _ensure_python_runtime_ok()

    try:
        import uvicorn
        from funasr.bin._server_app import create_app
        from funasr.bin.server import server_version_label
    except ImportError as exc:
        print("Missing dependency:", exc)
        print("Activate the venv and install deps:")
        print("  source .venv/bin/activate")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    print("Installing speaker-cluster safety (UMAP n_jobs=1, keep ASR on failure)...")
    install_speaker_cluster_safety()

    app = create_app(
        device=args.device,
        preload_model=args.model,
        model_path=args.model_path,
        hub=args.hub,
        spk_model=args.spk_model,
        cors_origins=args.cors_origin,
    )

    print("Applying long-audio VAD / batch settings...")
    apply_long_audio_settings(
        app,
        vad_max_single_segment_time=args.vad_max_single_segment_time,
        vad_max_end_silence_time=args.vad_max_end_silence_time,
        merge_vad=args.merge_vad,
        merge_length_s=args.merge_length_s,
        batch_size_s=args.batch_size_s,
    )

    print("Installing video input support (ffmpeg → 16k mono wav)...")
    install_video_input_support(app)
    print("Installing transcription job queue...")
    install_transcription_jobs(
        app,
        jobs_dir=Path(args.jobs_dir).expanduser(),
        max_waiting=max(1, args.max_waiting_jobs),
        max_bytes=args.max_upload_bytes,
    )
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    if not ffmpeg_ok:
        print("  video-input: WARNING — ffmpeg not on PATH; video uploads will fail")

    print("╔══════════════════════════════════════════════╗")
    print(f"║  {server_version_label():<44}║")
    print(f"║  Device: {args.device:<35}║")
    print(f"║  Model:  {args.model:<35}║")
    if args.model_path:
        print(f"║  Path:   {str(args.model_path)[:35]:<35}║")
        print(f"║  Hub:    {args.hub:<35}║")
    print(f"║  VAD max seg: {args.vad_max_single_segment_time} ms{' ' * max(0, 22 - len(str(args.vad_max_single_segment_time)))}║")
    print(f"║  VAD end sil: {args.vad_max_end_silence_time} ms{' ' * max(0, 22 - len(str(args.vad_max_end_silence_time)))}║")
    merge_label = f"on/{args.merge_length_s}s" if args.merge_vad else "off"
    print(f"║  merge_vad: {merge_label:<32}║")
    print(f"║  batch_size_s: {args.batch_size_s:<29}║")
    video_label = "on (need ffmpeg)" if ffmpeg_ok else "on (ffmpeg MISSING)"
    print(f"║  video mp4/mkv/webm/mov: {video_label:<19}║")
    print(f"║  URL:    http://{args.host}:{args.port}/v1{' ' * max(0, 20 - len(str(args.port)))}║")
    print(f"║  Docs:   http://{args.host}:{args.port}/docs{' ' * max(0, 18 - len(str(args.port)))}║")
    print("╚══════════════════════════════════════════════╝")
    print("Long audio: VAD splits speech into chunks; full file length is OK (~1h+).")
    print("Video: mp4/mkv/webm/mov → ffmpeg 16k mono wav in-process, then SenseVoice.")
    print("Audio: wav/mp3/... uses FunASR stock path (no re-encode).")
    print("Concurrency: one transcription runs at a time. Extra uploads wait on disk.")
    print("Jobs: POST /v1/audio/jobs then poll GET /v1/audio/jobs/{id}. Speaker labels stay on.")
    print("First request may download SenseVoice weights from ModelScope.")

    uvicorn.run(app, host=args.host, port=args.port, timeout_keep_alive=600)


if __name__ == "__main__":
    main()
