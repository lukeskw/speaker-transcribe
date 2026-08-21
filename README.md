# speaker-transcribe

Extract audio from a video and produce a **speaker-labeled transcript** ("who said
what"). Fully local, fully free.

Pipeline: `ffmpeg` (audio extract + normalization) → **openai-whisper** (transcription) +
**pyannote** (speaker diarization) → clean grouped transcript.

## Output

```
[00:00] SPEAKER_00: Hey, thanks for jumping on the call.
[00:04] SPEAKER_01: Of course. So where did we land on the pricing?
[00:09] SPEAKER_00: Right, so I ran the numbers again...
```

## Install (AMD GPU on WSL2 — ROCm)

Needs: Windows AMD Adrenalin driver >= 26.1.1, WSL2 Ubuntu 24.04, Python 3.12,
`ffmpeg`, and a (free) Hugging Face token.

```bash
# 1. system deps
sudo apt install ffmpeg

# 2. ROCm 7.2 WSL runtime (one-time)
cd /tmp
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
sudo apt install -y ./amdgpu-install_7.2.70200-1_all.deb
sudo amdgpu-install --usecase=wsl,rocm --no-dkms -y
sudo usermod -a -G render,video $LOGNAME
# then, in Windows PowerShell:  wsl --shutdown   (and reopen)

# 3. venv + ROCm torch (from AMD's repo, NOT PyPI)
cd ~/projects/speaker-transcribe
python3.12 -m venv .venv-rocm
.venv-rocm/bin/pip install -U pip wheel
.venv-rocm/bin/pip install \
  --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/ \
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.6.0+rocm7.2.0.gitba5c1517-cp312-cp312-linux_x86_64.whl" \
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.10.0+rocm7.2.0.lw.gitb6ee5fde-cp312-cp312-linux_x86_64.whl" \
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchaudio-2.10.0+rocm7.2.0.git5047768f-cp312-cp312-linux_x86_64.whl"

# 4. app deps
.venv-rocm/bin/pip install -r requirements.txt

# 5. verify the GPU is visible
.venv-rocm/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> True AMD Radeon RX 9060 XT
```

### Hugging Face token (pyannote diarization)

```bash
# free account + token: https://huggingface.co/settings/tokens
# ACCEPT the licenses (one click each) with the account that owns the token:
#   https://huggingface.co/pyannote/speaker-diarization-3.1
#   https://huggingface.co/pyannote/segmentation-3.0
#   https://huggingface.co/pyannote/speaker-diarization-community-1
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

> pyannote 4.x pulls all three gated repos. Missing any license accept gives a
> 401/403 when the models download. `speaker-diarization-community-1` is the one
> people miss — it's new in pyannote 4.x.

### CPU / NVIDIA

The tool is device-agnostic — `--device cpu` works anywhere, and on an NVIDIA
box install the normal CUDA torch instead of the ROCm wheels. `--device auto`
picks the GPU when torch sees one.

## Usage

```bash
# auto-detect speaker count
./transcribe.py interview.mp4

# tell it exactly how many speakers (more accurate)
./transcribe.py interview.mp4 --speakers 2

# bound the range instead
./transcribe.py panel.mp4 --min-speakers 3 --max-speakers 5

# audio files are also normalized via ffmpeg
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
| `--device` | `auto` / `cuda` / `cpu` (auto = GPU if torch sees one — CUDA or ROCm — else cpu) |
| `--hf-token` | HF token (or set `HF_TOKEN`) |
| `-o, --output` | output path (default `<input>.transcript.txt`) |
| `--no-timestamps` | drop `[mm:ss]` prefixes |
| `--keep-audio` | keep the extracted `.wav` |

## Speed notes

- **GPU (ROCm/CUDA):** first run pays a one-time Triton kernel-compile cost,
  then warm runs are several times faster than CPU (large-v2 ~7x on an RX 9060 XT).
- **CPU only:** slow; drop to `--model small` or `base`.

## Optional: polish labels with an LLM

WhisperX gives you `SPEAKER_00` / `SPEAKER_01`. To turn those into real names,
pipe the transcript to Claude/GPT with a prompt like:
*"Relabel the speakers using context; SPEAKER_00 is the interviewer."*
Diarization (who spoke) is an audio problem the LLM can't do — but relabeling and
cleanup is exactly what it's good at.
