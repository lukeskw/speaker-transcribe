# ROCm GPU Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the CUDA-only WhisperX engine in `transcribe.py` with an openai-whisper + pyannote pipeline that runs on the AMD RX 9060 XT via ROCm, keeping the CLI and output format identical.

**Architecture:** Single-file CLI. `ffmpeg` extracts 16 kHz mono wav; openai-whisper transcribes with word timestamps; pyannote diarizes; a pure merge function assigns each word to the max-overlap speaker turn; the existing grouping/formatting emits `[mm:ss] SPEAKER_NN: text`. Device-agnostic (CPU/CUDA/ROCm) — ROCm torch reports as `cuda`.

**Tech Stack:** Python 3.12, openai-whisper, pyannote.audio, torch 2.10+rocm7.2, ffmpeg, pytest.

## Global Constraints

- Python interpreter/venv: `.venv-rocm` (already built + verified; do NOT rebuild or rename).
- torch/torchaudio are installed from AMD's ROCm repo (`--find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/`), NOT PyPI. Never `pip install torch` from PyPI — it would clobber the ROCm build with a CPU/CUDA one.
- CLI flags and output format are FROZEN — must match the current `transcribe.py` exactly.
- Default model: `large-v2`. Default device: `auto`.
- Diarization model `pyannote/speaker-diarization-3.1` is gated; needs `HF_TOKEN` + accepted licenses at runtime.
- Run all test commands with the venv interpreter: `.venv-rocm/bin/python -m pytest ...`.
- Commits: do NOT commit autonomously. Leave commits to the user (their standing rule). Steps below that say "Commit" are staged for the user to run — show the command, do not execute it.

---

## File Structure

- `transcribe.py` (modify) — the CLI; internals refactored into focused functions.
- `tests/test_merge.py` (create) — unit tests for `assign_speakers` (pure).
- `tests/test_transcript.py` (create) — unit tests for `build_transcript` (pure).
- `tests/test_device.py` (create) — unit test for `detect_device` (torch mocked).
- `tests/data/sample.wav` (create) — short sample copied from `vendor/whisper.cpp/samples/jfk.wav` before that dir is deleted.
- `requirements.txt` (modify) — drop whisperx; add openai-whisper, pyannote.audio, numpy; document ROCm torch install.
- `README.md` (modify) — ROCm-on-WSL setup replacing CUDA/WhisperX instructions.
- Delete: `.venv/`, `vendor/whisper.cpp/` (after `tests/data/sample.wav` is copied out).

---

### Task 1: Test harness + preserve sample + pytest

**Files:**
- Create: `tests/__init__.py` (empty), `tests/data/sample.wav`
- Modify: (none)

**Interfaces:**
- Consumes: nothing.
- Produces: `.venv-rocm` has pytest; `tests/data/sample.wav` exists for later smoke tests.

- [ ] **Step 1: Install pytest into the ROCm venv**

Run: `.venv-rocm/bin/pip install pytest`
Expected: `Successfully installed pytest-...`

- [ ] **Step 2: Preserve a sample wav before vendor/ is deleted**

```bash
mkdir -p tests/data
cp vendor/whisper.cpp/samples/jfk.wav tests/data/sample.wav
touch tests/__init__.py
```
Expected: `tests/data/sample.wav` exists (~176 kB).

- [ ] **Step 3: Verify pytest collects (no tests yet is fine)**

Run: `.venv-rocm/bin/python -m pytest -q`
Expected: "no tests ran" (exit 5) — confirms pytest works.

- [ ] **Step 4: Commit (user runs)**

```bash
git add tests/__init__.py tests/data/sample.wav
git commit -m "test: add pytest harness and sample audio"
```

---

### Task 2: `assign_speakers` (pure merge)

**Files:**
- Create: `tests/test_merge.py`
- Modify: `transcribe.py` (add `assign_speakers`)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `assign_speakers(words, turns) -> list[dict]`
  - `words`: `list[dict]`, each `{"start": float, "end": float, "text": str}`.
  - `turns`: `list[dict]`, each `{"start": float, "end": float, "speaker": str}`.
  - Returns a new list of words, each with an added `"speaker": str`. A word gets the turn of greatest time-overlap; if no turn overlaps, the nearest turn by gap; if `turns` is empty, `"SPEAKER_?"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_merge.py
from transcribe import assign_speakers

TURNS = [
    {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
    {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
]

def test_word_inside_turn():
    words = [{"start": 1.0, "end": 2.0, "text": "hello"}]
    out = assign_speakers(words, TURNS)
    assert out[0]["speaker"] == "SPEAKER_00"

def test_word_max_overlap_wins():
    # spans the boundary but mostly in turn 1
    words = [{"start": 4.5, "end": 8.0, "text": "boundary"}]
    out = assign_speakers(words, TURNS)
    assert out[0]["speaker"] == "SPEAKER_01"

def test_word_no_overlap_nearest():
    words = [{"start": 12.0, "end": 13.0, "text": "after"}]
    out = assign_speakers(words, TURNS)
    assert out[0]["speaker"] == "SPEAKER_01"  # nearest turn

def test_empty_turns():
    words = [{"start": 0.0, "end": 1.0, "text": "x"}]
    out = assign_speakers(words, [])
    assert out[0]["speaker"] == "SPEAKER_?"

def test_does_not_mutate_input():
    words = [{"start": 1.0, "end": 2.0, "text": "hi"}]
    assign_speakers(words, TURNS)
    assert "speaker" not in words[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-rocm/bin/python -m pytest tests/test_merge.py -v`
Expected: FAIL — `ImportError: cannot import name 'assign_speakers'`.

- [ ] **Step 3: Write minimal implementation**

Add to `transcribe.py` (near the other helpers):

```python
def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _gap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    if a_end < b_start:
        return b_start - a_end
    if b_end < a_start:
        return a_start - b_end
    return 0.0


def assign_speakers(words: list[dict], turns: list[dict]) -> list[dict]:
    """Attach a 'speaker' to each word by max time-overlap with diarization turns.

    No overlap -> nearest turn by gap. No turns -> 'SPEAKER_?'.
    Returns a new list; does not mutate the input words.
    """
    out: list[dict] = []
    for w in words:
        speaker = "SPEAKER_?"
        if turns:
            best_overlap = 0.0
            best = None
            for t in turns:
                ov = _overlap(w["start"], w["end"], t["start"], t["end"])
                if ov > best_overlap:
                    best_overlap = ov
                    best = t
            if best is None:
                best = min(
                    turns,
                    key=lambda t: _gap(w["start"], w["end"], t["start"], t["end"]),
                )
            speaker = best["speaker"]
        out.append({**w, "speaker": speaker})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-rocm/bin/python -m pytest tests/test_merge.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit (user runs)**

```bash
git add tests/test_merge.py transcribe.py
git commit -m "feat: add assign_speakers word->speaker merge"
```

---

### Task 3: `build_transcript` refactor (take a list, not a JSON file)

**Files:**
- Create: `tests/test_transcript.py`
- Modify: `transcribe.py` (change `build_transcript` signature)

**Interfaces:**
- Consumes: word dicts with `"speaker"` (from `assign_speakers`).
- Produces:
  - `build_transcript(segments, show_ts) -> str`
  - `segments`: `list[dict]`, each `{"speaker": str, "text": str, "start": float}`.
  - Groups consecutive same-speaker segments into one line `"[mm:ss] SPEAKER: joined text"` (timestamp of the first segment in the group), blank-line separated. `show_ts=False` drops the `[mm:ss]` prefix. Empty input -> `""`.
  - `fmt_ts(seconds: float) -> str` unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transcript.py
from transcribe import build_transcript

def seg(spk, text, start):
    return {"speaker": spk, "text": text, "start": start}

def test_groups_consecutive_same_speaker():
    segs = [seg("SPEAKER_00", "hello", 0.0), seg("SPEAKER_00", "there", 0.5),
            seg("SPEAKER_01", "hi", 2.0)]
    out = build_transcript(segs, show_ts=True)
    assert out == "[00:00] SPEAKER_00: hello there\n\n[00:02] SPEAKER_01: hi"

def test_no_timestamps():
    segs = [seg("SPEAKER_00", "hello", 0.0)]
    assert build_transcript(segs, show_ts=False) == "SPEAKER_00: hello"

def test_empty():
    assert build_transcript([], show_ts=True) == ""

def test_timestamp_hours():
    segs = [seg("SPEAKER_00", "late", 3661.0)]
    assert build_transcript(segs, show_ts=True) == "[01:01:01] SPEAKER_00: late"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-rocm/bin/python -m pytest tests/test_transcript.py -v`
Expected: FAIL — current `build_transcript(result_json: Path, show_ts)` reads a file; passing a list errors.

- [ ] **Step 3: Rewrite `build_transcript`**

Replace the existing `build_transcript` in `transcribe.py` with:

```python
def build_transcript(segments: list[dict], show_ts: bool) -> str:
    if not segments:
        return ""

    lines: list[str] = []
    cur_speaker = None
    buf: list[str] = []
    seg_start = 0.0

    def flush() -> None:
        if not buf:
            return
        speaker = cur_speaker or "SPEAKER_?"
        text = " ".join(t.strip() for t in buf if t.strip())
        if not text:
            return
        prefix = f"[{fmt_ts(seg_start)}] " if show_ts else ""
        lines.append(f"{prefix}{speaker}: {text}")

    for seg in segments:
        spk = seg.get("speaker", "SPEAKER_?")
        if spk != cur_speaker:
            flush()
            cur_speaker = spk
            buf = []
            seg_start = seg.get("start", 0.0)
        buf.append(seg.get("text", ""))
    flush()

    return "\n\n".join(lines)
```

Keep `fmt_ts` exactly as-is.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-rocm/bin/python -m pytest tests/test_transcript.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit (user runs)**

```bash
git add tests/test_transcript.py transcribe.py
git commit -m "refactor: build_transcript takes segment list instead of JSON file"
```

---

### Task 4: `detect_device` fix (torch, not nvidia-smi)

**Files:**
- Create: `tests/test_device.py`
- Modify: `transcribe.py` (rewrite `detect_device`, remove `import shutil` usage only if unused elsewhere — it's still used by `require`, keep it)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `detect_device(requested: str) -> tuple[str, bool]`
  - Returns `(device, fp16)`. `requested="auto"` -> `"cuda"` if `torch.cuda.is_available()` else `"cpu"`. `fp16` is `True` iff `device == "cuda"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_device.py
import transcribe

def test_explicit_cpu():
    assert transcribe.detect_device("cpu") == ("cpu", False)

def test_explicit_cuda():
    assert transcribe.detect_device("cuda") == ("cuda", True)

def test_auto_uses_torch(monkeypatch):
    monkeypatch.setattr(transcribe.torch.cuda, "is_available", lambda: True)
    assert transcribe.detect_device("auto") == ("cuda", True)
    monkeypatch.setattr(transcribe.torch.cuda, "is_available", lambda: False)
    assert transcribe.detect_device("auto") == ("cpu", False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-rocm/bin/python -m pytest tests/test_device.py -v`
Expected: FAIL — old `detect_device` returns `(device, compute_type_str)` and uses `shutil.which("nvidia-smi")`; also `transcribe.torch` not imported yet.

- [ ] **Step 3: Implement**

At the top of `transcribe.py` add `import torch` (with the other imports). Replace `detect_device`:

```python
def detect_device(requested: str) -> tuple[str, bool]:
    """Return (device, fp16). 'auto' -> cuda if torch sees a GPU (incl. ROCm)."""
    if requested == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested
    return device, device == "cuda"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-rocm/bin/python -m pytest tests/test_device.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit (user runs)**

```bash
git add tests/test_device.py transcribe.py
git commit -m "fix: detect_device uses torch.cuda (ROCm-aware) not nvidia-smi"
```

---

### Task 5: `transcribe_words` (openai-whisper wrapper)

**Files:**
- Modify: `transcribe.py` (add `transcribe_words`)

**Interfaces:**
- Consumes: `(device, fp16)` from `detect_device`.
- Produces:
  - `transcribe_words(audio, model_name, device, fp16, language) -> list[dict]`
  - `audio`: `Path`. Returns `list[dict]` `{"start","end","text"}` — one per word, in order. `language` may be `None` (auto-detect).

- [ ] **Step 1: Install openai-whisper (already installed in Task pre-work; verify)**

Run: `.venv-rocm/bin/python -c "import whisper; print(whisper.__version__)"`
Expected: prints a version (e.g. `20250625`).

- [ ] **Step 2: Add the implementation**

```python
import whisper  # add near other imports

def transcribe_words(audio: Path, model_name: str, device: str, fp16: bool,
                     language: str | None) -> list[dict]:
    """Transcribe with per-word timestamps. Returns [{'start','end','text'}]."""
    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(
        str(audio), fp16=fp16, word_timestamps=True, language=language,
    )
    words: list[dict] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            text = w.get("word", "").strip()
            if not text:
                continue
            words.append({"start": w["start"], "end": w["end"], "text": text})
    return words
```

- [ ] **Step 3: Smoke test on the sample (GPU)**

Run:
```bash
.venv-rocm/bin/python -c "
from pathlib import Path
from transcribe import transcribe_words, detect_device
dev, fp16 = detect_device('auto')
ws = transcribe_words(Path('tests/data/sample.wav'), 'base', dev, fp16, 'en')
print('device', dev, 'words', len(ws))
print(' '.join(w['text'] for w in ws[:8]))
assert len(ws) > 5 and all({'start','end','text'} <= w.keys() for w in ws)
print('OK')
"
```
Expected: `device cuda`, a word count > 5, the opening of the JFK line, `OK`. (Uses `base` to keep the smoke test fast.)

- [ ] **Step 4: Commit (user runs)**

```bash
git add transcribe.py
git commit -m "feat: add transcribe_words (openai-whisper, word timestamps)"
```

---

### Task 6: `diarize` (pyannote wrapper)

**Files:**
- Modify: `transcribe.py` (add `diarize`), `requirements.txt` implied (installed here)

**Interfaces:**
- Consumes: `device` from `detect_device`; `hf_token`; speaker-count args.
- Produces:
  - `diarize(audio, hf_token, device, num_speakers, min_speakers, max_speakers) -> list[dict]`
  - Returns `list[dict]` `{"start","end","speaker"}` sorted by start. `num_speakers` pins the count; else `min_speakers`/`max_speakers` bound it; any may be `None`.

- [ ] **Step 1: Install pyannote.audio**

Run: `.venv-rocm/bin/pip install pyannote.audio`
Expected: `Successfully installed pyannote.audio-...` (torch already satisfied by the ROCm build — verify it was NOT reinstalled: `grep -i "Uninstalling torch" ` should find nothing in the pip output).

Then confirm torch survived:
Run: `.venv-rocm/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available())"`
Expected: `2.10.0+rocm7.2.0... True`. If torch was clobbered, reinstall it from the ROCm wheel per the Global Constraints and README.

- [ ] **Step 2: Add the implementation**

```python
def diarize(audio: Path, hf_token: str, device: str,
            num_speakers: int | None, min_speakers: int | None,
            max_speakers: int | None) -> list[dict]:
    """Run pyannote speaker diarization. Returns [{'start','end','speaker'}]."""
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token,
    )
    if pipeline is None:
        die("pyannote failed to load — HF token invalid or model license not "
            "accepted. See README.md (accept speaker-diarization-3.1 and "
            "segmentation-3.0).")
    pipeline.to(torch.device(device))

    kwargs: dict = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

    annotation = pipeline(str(audio), **kwargs)
    turns = [
        {"start": seg.start, "end": seg.end, "speaker": label}
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t["start"])
    return turns
```

- [ ] **Step 3: Smoke test (requires HF_TOKEN + accepted licenses)**

Run:
```bash
.venv-rocm/bin/python -c "
import os; from pathlib import Path
from transcribe import diarize, detect_device
dev, _ = detect_device('auto')
turns = diarize(Path('tests/data/sample.wav'), os.environ['HF_TOKEN'], dev, None, None, None)
print('device', dev, 'turns', len(turns))
assert all({'start','end','speaker'} <= t.keys() for t in turns)
print('OK', turns[:2])
"
```
Expected: `device cuda`, ≥1 turn, `OK`. If it 401/403s, the gated licenses aren't accepted — that's a runtime setup issue, not a code bug (README covers it).

- [ ] **Step 4: Commit (user runs)**

```bash
git add transcribe.py requirements.txt
git commit -m "feat: add pyannote diarize wrapper"
```

---

### Task 7: Wire `main()` — replace WhisperX flow

**Files:**
- Modify: `transcribe.py` (remove `run_whisperx`; rewrite `main` body; drop the `require("whisperx", ...)` check and `--compute_type` remnants)

**Interfaces:**
- Consumes: `detect_device`, `extract_audio`, `transcribe_words`, `diarize`, `assign_speakers`, `build_transcript`.
- Produces: end-to-end CLI unchanged in flags/output.

- [ ] **Step 1: Delete `run_whisperx`**

Remove the entire `run_whisperx` function from `transcribe.py`.

- [ ] **Step 2: Rewrite the `main` compute section**

Replace the `require("whisperx", ...)` line and the `with tempfile...` body's transcription part so it reads:

```python
    device, fp16 = detect_device(args.device)

    out_path = args.output or args.input.with_suffix(".transcript.txt")

    with tempfile.TemporaryDirectory(prefix="sptr_") as tmp:
        tmpdir = Path(tmp)

        if args.input.suffix.lower() in AUDIO_EXTS:
            audio = args.input
        else:
            require("ffmpeg", "Install it: sudo apt install ffmpeg  (or brew install ffmpeg)")
            audio = tmpdir / f"{args.input.stem}.wav"
            print(f"[ffmpeg] extracting audio -> {audio.name}", file=sys.stderr)
            extract_audio(args.input, audio)
            if args.keep_audio:
                kept = out_path.with_suffix(".wav")
                shutil.copy(audio, kept)
                print(f"[ffmpeg] kept audio: {kept}", file=sys.stderr)

        print(f"[whisper] model={args.model} device={device} transcribing ...", file=sys.stderr)
        words = transcribe_words(audio, args.model, device, fp16, args.language)
        if not words:
            die("transcript is empty — no speech detected?")

        print("[pyannote] diarizing ...", file=sys.stderr)
        turns = diarize(audio, args.hf_token, device,
                        args.speakers, args.min_speakers, args.max_speakers)

        merged = assign_speakers(words, turns)
        transcript = build_transcript(merged, show_ts=not args.no_timestamps)

    if not transcript:
        die("transcript is empty — no speech detected?")

    out_path.write_text(transcript + "\n")
    print(f"\n{transcript}\n")
    print(f"[done] transcript written to {out_path}", file=sys.stderr)
```

Also delete the now-unused `--compute_type`-related help text in `--model` if any and confirm `import json` is removed only if no longer used (it is no longer used — remove `import json`).

- [ ] **Step 3: Run the unit suite (must still pass)**

Run: `.venv-rocm/bin/python -m pytest -q`
Expected: all unit tests pass (merge, transcript, device).

- [ ] **Step 4: End-to-end smoke on the sample**

Run:
```bash
HF_TOKEN=$HF_TOKEN .venv-rocm/bin/python transcribe.py tests/data/sample.wav --model base -o /tmp/sample.transcript.txt
cat /tmp/sample.transcript.txt
```
Expected: a `SPEAKER_00: ... ask not what your country can do for you ...` line; exits 0. (Needs HF_TOKEN + accepted licenses.)

- [ ] **Step 5: Commit (user runs)**

```bash
git add transcribe.py
git commit -m "feat: wire main to openai-whisper + pyannote pipeline"
```

---

### Task 8: `requirements.txt` + `README.md`

**Files:**
- Modify: `requirements.txt`, `README.md`

**Interfaces:** none (docs/deps).

- [ ] **Step 1: Rewrite `requirements.txt`**

```
# Transcription + diarization. torch/torchaudio are NOT here — install them
# first from AMD's ROCm repo (see README "ROCm setup"); installing torch from
# PyPI would replace the ROCm build with a CPU/CUDA one.
openai-whisper>=20250625
pyannote.audio>=3.1
numpy>=1.26
```

- [ ] **Step 2: Rewrite the README install section**

Replace the "Install" and "Speed notes" sections with ROCm-on-WSL steps. Use exactly the versions/commands verified for this machine:

````markdown
## Install (AMD GPU on WSL2 — ROCm)

Needs: Windows AMD Adrenalin driver ≥ 26.1.1, WSL2 Ubuntu 24.04, Python 3.12,
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
# ACCEPT the licenses (one click each):
#   https://huggingface.co/pyannote/speaker-diarization-3.1
#   https://huggingface.co/pyannote/segmentation-3.0
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

> Skipping the license-accept step gives a 401/403 when pyannote downloads models.

### CPU / NVIDIA

The tool is device-agnostic — `--device cpu` works anywhere, and on an NVIDIA
box install the normal CUDA torch instead of the ROCm wheels. `--device auto`
picks the GPU when torch sees one.

## Speed notes

- **GPU (ROCm/CUDA):** large-v2 ≈ real-time-ish; first run pays a one-time
  Triton kernel-compile cost, then warm runs are ~7× faster than CPU.
- **CPU only:** slow; drop to `--model small` or `base`.
````

Also update the pipeline line at the top of the README (`WhisperX` → `openai-whisper + pyannote`) and the `--device` flag row (`auto = cuda if NVIDIA GPU seen` → `auto = GPU if torch sees one (CUDA or ROCm)`).

- [ ] **Step 3: Sanity check the app deps install cleanly**

Run: `.venv-rocm/bin/pip install -r requirements.txt`
Expected: everything already satisfied; torch line reports the ROCm build unchanged.

- [ ] **Step 4: Commit (user runs)**

```bash
git add requirements.txt README.md
git commit -m "docs: ROCm-on-WSL install; drop whisperx"
```

---

### Task 9: Delete dead artifacts

**Files:**
- Delete: `.venv/`, `vendor/whisper.cpp/`

**Interfaces:** none.

- [ ] **Step 1: Confirm the sample was preserved (Task 1) before deleting vendor**

Run: `ls -la tests/data/sample.wav`
Expected: file exists. If missing, STOP and redo Task 1 Step 2.

- [ ] **Step 2: Delete**

```bash
rm -rf .venv vendor/whisper.cpp
# if vendor/ is now empty:
rmdir vendor 2>/dev/null || true
```

- [ ] **Step 3: Full test suite + e2e still green**

Run: `.venv-rocm/bin/python -m pytest -q`
Expected: all unit tests pass. (E2E from Task 7 Step 4 still runs if HF_TOKEN set.)

- [ ] **Step 4: Commit (user runs)**

```bash
git add -A
git commit -m "chore: remove old cu128 venv and whisper.cpp dzn experiment"
```

---

## Self-Review

**Spec coverage:**
- detect_device fix → Task 4. ✓
- openai-whisper transcription (word timestamps) → Task 5. ✓
- pyannote diarization (speaker-count controls) → Task 6. ✓
- word→speaker merge (max overlap, nearest fallback) → Task 2. ✓
- build_transcript reuse (unchanged output) → Task 3. ✓
- main wiring / remove WhisperX → Task 7. ✓
- CLI + output frozen → Tasks 3, 7 (format tests + e2e). ✓
- error handling (HF token, 401/403, empty) → Tasks 6, 7. ✓ (ROCm OOM message: covered by whisper/pyannote raising; explicit message deferred — acceptable, not a spec-mandated string.)
- deps: requirements.txt + ROCm torch documented → Task 8. ✓
- delete .venv + vendor/whisper.cpp → Task 9. ✓
- keep .venv-rocm → Global Constraints. ✓
- README rewrite → Task 8. ✓
- unit tests (assign_speakers, build_transcript) + smoke → Tasks 2, 3, 5, 7. ✓

**Placeholder scan:** no TBD/TODO; all code steps contain real code.

**Type consistency:** word dict `{start,end,text}` (Tasks 2,5) and `+speaker` (Task 2); turn dict `{start,end,speaker}` (Tasks 2,6); segment dict for build_transcript `{speaker,text,start}` — words carry all three after merge, so Task 7 feeds `merged` directly. `detect_device -> (device, fp16: bool)` consistent across Tasks 4,5,7. ✓
