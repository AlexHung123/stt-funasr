# FunASR + SenseVoice STT Server

Local speech-to-text stack using [FunASR](https://github.com/modelscope/FunASR) and **SenseVoice**, with an OpenAI-compatible HTTP API.

## Setup (already done in this folder)

```bash
cd /Users/admin/tools/stt-funasr
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Packages installed:

| Package | Role |
|---------|------|
| `funasr` | ASR toolkit + SenseVoice pipeline |
| `torch` / `torchaudio` | Inference backend (MPS on Apple Silicon) |
| `fastapi` + `uvicorn` + `python-multipart` | HTTP API |
| `modelscope` | Model download (SenseVoice weights) |

## Start the server

```bash
source .venv/bin/activate
python server.py
# or
./start_server.sh
```

Defaults on this Mac:

- **Model:** `sensevoice` (`iic/SenseVoiceSmall`)
- **Device:** auto (`mps` if available, else `cpu`)
- **Port:** `8002`
- **Host:** `0.0.0.0`
- **Long-form VAD:** on (`max_single_segment_time=60s`, `merge_vad`, `batch_size_s=60`)

Options:

```bash
python server.py --device cpu --port 9000
python server.py --model sensevoice --device mps
python server.py --cors-origin http://localhost:3000
funasr-server --model sensevoice --device mps --port 8000   # official CLI (no long-audio patch)
```

### Long audio / video (~1 hour)

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

OpenAPI docs: [http://127.0.0.1:8002/docs](http://127.0.0.1:8002/docs)

> The first start downloads SenseVoice (and VAD) weights from ModelScope. That can take a few minutes.

## API usage

### cURL

```bash
# health
curl http://127.0.0.1:8002/health

# list models
curl http://127.0.0.1:8002/v1/models

# transcribe
curl -X POST http://127.0.0.1:8002/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

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

### Sample audio

```bash
curl -L "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav" \
  -o sample.wav
```

## SenseVoice features

- Multilingual ASR: Chinese, Cantonese, English, Japanese, Korean
- Language ID, emotion tags, audio events
- Good CPU / Apple Silicon performance for local deployment

## Notes

- On Apple Silicon, `mps` is selected automatically when available. If you hit a model-specific MPS issue, fall back with `--device cpu`.
- vLLM acceleration is **not** used here (GPU Linux path); this setup uses FunASR `AutoModel` for SenseVoice.
- Models cache under `~/.cache/modelscope` by default.
