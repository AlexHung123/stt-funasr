# FunASR + SenseVoice STT Server

Local speech-to-text stack using [FunASR](https://github.com/modelscope/FunASR) and **SenseVoice**, with an OpenAI-compatible HTTP API.

- Audio: `wav` / `mp3` / `flac` / … go straight to FunASR
- Video: `mp4` / `mkv` / `webm` / `mov` are converted in-process with **ffmpeg** to 16 kHz mono PCM WAV, then transcribed
- Long-form: VAD splits speech so ~1 hour files are fine
- Concurrency: ASR jobs are serialized (one at a time) in a single process

## Setup

### 1. Prerequisites

| Tool | Notes |
|------|--------|
| Python 3.10+ | 3.13 is tested on this project |
| `ffmpeg` | Required for video uploads (`brew install ffmpeg`) |
| Git | To clone the repo |
| Apple Silicon / NVIDIA / CPU | Device is auto-detected: `mps` → `cuda` → `cpu` |

### 2. Clone

```bash
git clone https://github.com/AlexHung123/stt-funasr.git
cd stt-funasr
```

### 3. Create a virtualenv and install Python deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

| Package | Role |
|---------|------|
| `funasr` | ASR toolkit + SenseVoice pipeline |
| `torch` / `torchaudio` | Inference backend (MPS on Apple Silicon) |
| `fastapi` + `uvicorn` + `python-multipart` | HTTP API |
| `modelscope` | Model download (SenseVoice weights) |

### 4. Install ffmpeg (video input)

```bash
brew install ffmpeg
ffmpeg -version
```

Audio-only transcription works without ffmpeg. Video uploads (`mp4` / `mkv` / `webm` / `mov`) fail until ffmpeg is on `PATH`.

### 5. Start the server

```bash
source .venv/bin/activate
python server.py
# or, on Apple Silicon (forces MPS):
./start_server.sh
```

Defaults:

| Setting | Value |
|---------|--------|
| Model | `sensevoice` (`iic/SenseVoiceSmall`) |
| Device | auto (`mps` if available, else `cuda`, else `cpu`) |
| Port | `8002` |
| Host | `0.0.0.0` |
| Long-form VAD | on (`max_single_segment_time=60s`, `merge_vad`, `batch_size_s=60`) |

The first start downloads SenseVoice (and VAD) weights from ModelScope. That can take a few minutes. Weights cache under `~/.cache/modelscope`.

OpenAPI docs: [http://127.0.0.1:8002/docs](http://127.0.0.1:8002/docs)

Useful flags:

```bash
python server.py --device cpu --port 9000
python server.py --model sensevoice --device mps
python server.py --cors-origin http://localhost:3000
```

`./start_server.sh` always passes `--device mps`. Use `python server.py` if you want auto-detect or CPU.

### 6. Verify

```bash
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8002/v1/models
```

Sample audio:

```bash
curl -L "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav" \
  -o sample.wav

curl -X POST http://127.0.0.1:8002/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

## Optional: nginx reverse proxy

If you already front local APIs with Homebrew nginx (this machine listens on **8005**, same server also binds 443 without TLS), add a `/transcribe/` location so FunASR is reachable without colliding with LM Studio’s `/v1/`.

Do **not** use 443 unless a remote client cannot reach 8005. Port 443 in this config is plain HTTP, not HTTPS.

Edit `/opt/homebrew/etc/nginx/nginx.conf` inside the existing `server { ... }` block:

```nginx
# stt-funasr (SenseVoice): /transcribe/* -> http://127.0.0.1:8002/*
# e.g. POST /transcribe/v1/audio/transcriptions
#      -> POST http://127.0.0.1:8002/v1/audio/transcriptions
location ^~ /transcribe/ {
    client_max_body_size 2g;
    proxy_connect_timeout 60s;
    proxy_send_timeout 3600s;
    proxy_read_timeout 3600s;

    proxy_pass http://127.0.0.1:8002/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Connection "";
    proxy_request_buffering off;
    proxy_buffering off;
}

location = /transcribe {
    return 301 /transcribe/;
}
```

Apply and start:

```bash
nginx -t
brew services start nginx
# if nginx is already running:
nginx -s reload
```

Gateway examples (port **8005**):

```bash
curl http://127.0.0.1:8005/transcribe/health

curl -X POST http://127.0.0.1:8005/transcribe/v1/audio/transcriptions \
  -F file=@clip.mp4 \
  -F model=sensevoice \
  -F response_format=verbose_json
```

OpenAI SDK through nginx:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8005/transcribe/v1", api_key="not-needed")
```

Leave nginx’s existing `/v1/` location pointed at LM Studio. FunASR is only under `/transcribe/`.

## API usage

Endpoints:

| Method | Path | Notes |
|--------|------|--------|
| `GET` | `/health` | Liveness |
| `GET` | `/v1/models` | OpenAI-compatible model list |
| `POST` | `/v1/audio/transcriptions` | OpenAI-compatible; audio + video |
| `POST` | `/asr` | FunASR REST; audio + video |
| `GET` | `/docs` | Swagger UI |

### cURL (audio)

```bash
curl -X POST http://127.0.0.1:8002/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

### cURL (video)

```bash
curl -X POST http://127.0.0.1:8002/v1/audio/transcriptions \
  -F file=@clip.mp4 \
  -F model=sensevoice \
  -F response_format=verbose_json \
  -o result.json
```

Supported video containers: `mp4`, `mkv`, `webm`, `mov`. The server runs:

```text
ffmpeg -y -i input -ar 16000 -ac 1 -c:a pcm_s16le /tmp/meeting.wav
```

then calls FunASR in-process (no HTTP loopback). Temp files are deleted after the request.

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8002/v1", api_key="not-needed")

with open("sample.wav", "rb") as audio:
    result = client.audio.transcriptions.create(
        model="sensevoice",
        file=audio,
        response_format="verbose_json",
    )
print(result.text)
```

## Long audio / video (~1 hour)

VAD splits speech into short chunks so full file length is fine. This wrapper applies SenseVoice-friendly long-form defaults:

| Flag | Default | Meaning |
|------|---------|---------|
| `--vad-max-single-segment-time` | `60000` (ms) | Max continuous speech before forced cut |
| `--vad-max-end-silence-time` | `800` (ms) | Silence that ends a segment |
| `--merge-vad` / `--no-merge-vad` | on | Merge short VAD clips before ASR |
| `--merge-length-s` | `15` | Max length of merged clips (seconds) |
| `--batch-size-s` | `60` | ASR batch budget in seconds of speech |

```bash
# defaults already target long media
python server.py

# longer silence for lectures; larger ASR batch if you have RAM
python server.py --vad-max-end-silence-time 1500 --batch-size-s 120

# shorter forced cuts (safer on low memory)
python server.py --vad-max-single-segment-time 30000 --batch-size-s 30
```

Do **not** set `--vad-max-single-segment-time` to the full hour — that defeats VAD and can OOM. Keep it roughly 30–90 seconds.

## SenseVoice features

- Multilingual ASR: Chinese, Cantonese, English, Japanese, Korean
- Language ID, emotion tags, audio events
- Good CPU / Apple Silicon performance for local deployment

## Notes

- On Apple Silicon, `mps` is selected automatically when available. If you hit a model-specific MPS issue, fall back with `--device cpu`.
- `./start_server.sh` hardcodes `--device mps`. Use `python server.py` for auto-detect.
- vLLM acceleration is **not** used here (GPU Linux path); this setup uses FunASR `AutoModel` for SenseVoice.
- Models cache under `~/.cache/modelscope` by default.
- FunASR `AutoModel` is not concurrency-safe; overlapping requests queue on one process-wide lock.
- Official CLI (no long-audio patch, no video wrapper): `funasr-server --model sensevoice --device mps --port 8000`
