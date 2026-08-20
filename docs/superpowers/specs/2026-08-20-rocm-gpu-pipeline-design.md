# speaker-transcribe — ROCm GPU pipeline (engine swap)

**Date:** 2026-08-20
**Status:** approved design, pre-implementation

## Problem

`transcribe.py` runs on CPU on this machine. Root cause chain:

1. `detect_device` picks the GPU only when `nvidia-smi` is on PATH. The GPU here
   is an AMD Radeon RX 9060 XT (RDNA4) — no `nvidia-smi` — so `auto` always fell
   back to `cpu`.
2. Even forced to `cuda`, the current engine (**WhisperX → faster-whisper →
   CTranslate2**) is CUDA-only. CTranslate2 has no ROCm or Vulkan backend, so it
   can never use this GPU.

A Vulkan-via-dzn route (whisper.cpp) was tried and rejected: dzn exposes only
32 KB compute shared memory (D3D12 limit), which crashes whisper `medium`/`large`.

## Solution

Run the GPU through **ROCm**, which now officially supports gfx1200 (RX 9060 XT)
under WSL2 (ROCm 7.2, torch 2.10+rocm7.2). Verified working: `torch.cuda.is_available()`
is true, device reports "AMD Radeon RX 9060 XT", large-v2 transcribes in ~2.5 s
warm vs ~17 s CPU (≈7× faster).

Because CTranslate2 has no ROCm path, **replace WhisperX with a two-library
pipeline** that does run on ROCm torch:

- **openai-whisper** — transcription (pure PyTorch; runs on CPU/CUDA/ROCm).
- **pyannote.audio** — speaker diarization (PyTorch; runs on the same devices).

Speaker assignment uses openai-whisper's own `word_timestamps=True` (no wav2vec
forced-alignment stage). Good enough for interviews/calls; fewer deps, no
per-language alignment model.

The tool stays **device-agnostic**: openai-whisper + pyannote run on CPU and
NVIDIA too, so this is an engine swap, not an AMD lock-in.

## Architecture

```
input (video/audio)
  └─ ffmpeg → 16 kHz mono wav                              [unchanged]
       ├─ transcribe_words()  openai-whisper, word_timestamps=True   → [(start,end,text)]   [GPU]
       └─ diarize()           pyannote speaker-diarization-3.1        → [(start,end,label)]  [GPU]
                    │
             assign_speakers(): each word → speaker turn of max time-overlap
                    │
             build_transcript(): group consecutive same-speaker words
                    │
             write <input>.transcript.txt                  [unchanged format]
```

Single file (`transcribe.py`), internals split into focused, testable functions.

### Components

| Function | Responsibility | Depends on |
|----------|----------------|------------|
| `detect_device` | **FIX**: `auto` → `torch.cuda.is_available()` (true on ROCm), not `nvidia-smi`. Returns `"cuda"`/`"cpu"`. | torch |
| `extract_audio` | ffmpeg → 16 kHz mono wav. | ffmpeg (kept as-is) |
| `transcribe_words` | Load whisper model once; `transcribe(fp16=(device=="cuda"), word_timestamps=True, language=…)`. Flatten to word list. | whisper, torch |
| `diarize` | `Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token).to(torch.device(device))`; apply speaker-count bounds. Return turns. | pyannote.audio, torch |
| `assign_speakers` | For each word pick the turn with greatest overlap; fallback to nearest turn. Pure function. | — |
| `build_transcript` | Reuse existing grouping/formatting on merged word→speaker stream. | — |

### Speaker-count controls

Map existing flags to pyannote:
- `--speakers N` → `num_speakers=N`.
- `--min-speakers` / `--max-speakers` → `min_speakers` / `max_speakers`.
- none → pyannote auto-estimates.

Keep the existing mutual-exclusion check (`--speakers` vs `--min/--max`).

## CLI & output — unchanged

Same flags: `input`, `-o/--output`, `--speakers`, `--min-speakers`,
`--max-speakers`, `--model` (default `large-v2`), `--language`,
`--device {auto,cuda,cpu}`, `--hf-token`/`HF_TOKEN`, `--no-timestamps`,
`--keep-audio`.

Same output:
```
[00:00] SPEAKER_00: Hey, thanks for jumping on the call.
[00:04] SPEAKER_01: Of course. So where did we land on the pricing?
```

## Error handling

Keep `die()`. Add clear messages for:
- missing HF token (existing).
- pyannote 401/403 → model licenses not accepted (link the two gated models).
- CUDA/ROCm out-of-memory → suggest a smaller `--model`.
- empty transcript / no speech (existing).

## Dependencies & environment

- **Primary venv:** `.venv-rocm` (already built + verified; kept as-is, named in README).
- **Delete** old `.venv` (cu128 + whisperx) and `vendor/whisper.cpp` (dzn dead end).
- **`requirements.txt`:** drop `whisperx`; add `openai-whisper`, `pyannote.audio`,
  `numpy`. torch/torchaudio are **not on PyPI** for ROCm — documented as a separate
  install step (AMD repo wheels via `--find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/`).
- **README:** replace WhisperX/CUDA instructions with the ROCm-on-WSL setup
  (Adrenalin ≥26.1.1, `amdgpu-install --usecase=wsl,rocm --no-dkms`, `wsl --shutdown`,
  torch 2.10+rocm7.2 wheels), plus the unchanged HF-token/gated-model steps.

## Testing

- **Unit (no GPU/model):**
  - `assign_speakers` — overlap picks correct speaker; boundary/no-overlap fallback.
  - `build_transcript` — grouping, timestamps on/off, speaker changes.
- **Smoke:** short sample wav → non-empty speaker-labeled transcript; assert runs
  on `cuda` when available. (Keep a small sample clip before deleting `vendor/`.)

## Out of scope (YAGNI)

- wav2vec forced alignment (word-timestamp path chosen instead).
- LLM speaker relabeling (README already documents doing it externally).
- Non-AMD-specific perf tuning.
