# speaker-transcribe

Extract audio from a video and produce a **speaker-labeled transcript** ("who said
what"). Fully local, fully free.

Pipeline: `ffmpeg` (audio extract) → **WhisperX** (Whisper transcription +
pyannote speaker diarization + word alignment) → clean grouped transcript.

## Output

```
[00:00] SPEAKER_00: Hey, thanks for jumping on the call.
[00:04] SPEAKER_01: Of course. So where did we land on the pricing?
[00:09] SPEAKER_00: Right, so I ran the numbers again...
```

## Install

Needs Python 3.9–3.12, `ffmpeg`, and a (free) Hugging Face token.

```bash
cd ~/projects/speaker-transcribe

# 1. system deps
sudo apt install ffmpeg          # or: brew install ffmpeg

# 2. python deps (in a venv)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Hugging Face token (for pyannote diarization models)
#    - make a free account at https://huggingface.co
#    - create a token: https://huggingface.co/settings/tokens
#    - ACCEPT the model licenses (one click each):
#        https://huggingface.co/pyannote/speaker-diarization-3.1
#        https://huggingface.co/pyannote/segmentation-3.0
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

> Diarization models are gated. If you skip the license-accept step you'll get a
> 401/403 when whisperx tries to download them.

## Usage

```bash
# auto-detect speaker count
./transcribe.py interview.mp4

# tell it exactly how many speakers (more accurate)
./transcribe.py interview.mp4 --speakers 2

# bound the range instead
./transcribe.py panel.mp4 --min-speakers 3 --max-speakers 5

# already have audio? it skips ffmpeg
./transcribe.py call.wav --speakers 2 -o call.txt

# faster / smaller model, force language, no timestamps
./transcribe.py clip.mp4 --model small --language en --no-timestamps
```

### Windows files (WSL)

Running under WSL? Windows drives mount under `/mnt/`. Translate the path:
`G:\video\...` → `/mnt/g/video/...` (backslashes become slashes) and **quote it**
if it contains spaces:

```bash
./transcribe.py "/mnt/g/video/2026-08-20 15-15-07.mp4" --speakers 2
```

The transcript lands next to the video by default (`...transcript.txt` on `G:`).
Send it to your Linux home instead with `-o ~/transcript.txt`.

### Flags

| Flag | Meaning |
|------|---------|
| `--speakers N` | exact speaker count (pins min=max=N) |
| `--min-speakers N` / `--max-speakers N` | bound the range |
| `--model` | whisper model: `tiny`..`large-v3` (default `large-v2`) |
| `--language` | force language (`en`,`es`,`pt`,...); default auto-detect |
| `--device` | `auto` / `cuda` / `cpu` (auto = cuda if NVIDIA GPU seen) |
| `--hf-token` | HF token (or set `HF_TOKEN`) |
| `-o, --output` | output path (default `<input>.transcript.txt`) |
| `--no-timestamps` | drop `[mm:ss]` prefixes |
| `--keep-audio` | keep the extracted `.wav` |

## Speed notes

- **GPU (CUDA):** minutes. Uses `float16`.
- **CPU only:** works but slow; uses `int8`. Drop to `--model small` or `base`
  to keep it usable.

## Optional: polish labels with an LLM

WhisperX gives you `SPEAKER_00` / `SPEAKER_01`. To turn those into real names,
pipe the transcript to Claude/GPT with a prompt like:
*"Relabel the speakers using context; SPEAKER_00 is the interviewer."*
Diarization (who spoke) is an audio problem the LLM can't do — but relabeling and
cleanup is exactly what it's good at.
