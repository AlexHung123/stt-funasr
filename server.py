#!/usr/bin/env python3
"""
Local FunASR / SenseVoice HTTP server (OpenAI-compatible).

Defaults are tuned for this Mac (Apple Silicon): SenseVoice on MPS when available,
otherwise CPU. Wraps FunASR's built-in FastAPI app and patches VAD / generate
kwargs for long-form audio/video (e.g. ~1 hour).

Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/audio/transcriptions   (OpenAI-compatible)
  POST /asr                       (FunASR REST)
  GET  /docs                      (Swagger UI)

Examples:
  python server.py
  python server.py --device cpu --port 8000
  python server.py --model sensevoice --device mps
  python server.py --vad-max-single-segment-time 60000 --batch-size-s 60
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable


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
    -F response_format=verbose_json
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
    print(f"║  URL:    http://{args.host}:{args.port}/v1{' ' * max(0, 20 - len(str(args.port)))}║")
    print(f"║  Docs:   http://{args.host}:{args.port}/docs{' ' * max(0, 18 - len(str(args.port)))}║")
    print("╚══════════════════════════════════════════════╝")
    print("Long audio: VAD splits speech into chunks; full file length is OK (~1h+).")
    print("First request may download SenseVoice weights from ModelScope.")

    uvicorn.run(app, host=args.host, port=args.port, timeout_keep_alive=600)


if __name__ == "__main__":
    main()
