#!/usr/bin/env python3
"""
speaker-transcribe — extract audio from a video and produce a speaker-labeled
transcript ("who said what") using ffmpeg + WhisperX (Whisper + pyannote diarization).

All local, all free. Requires a (free) Hugging Face token to download the
pyannote diarization models — see README.md.

Usage:
    ./transcribe.py input.mp4
    ./transcribe.py input.mp4 --speakers 2
    ./transcribe.py call.wav --min-speakers 2 --max-speakers 4 -o out.txt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def require(cmd: str, hint: str) -> None:
    if shutil.which(cmd) is None:
        die(f"'{cmd}' not found on PATH. {hint}")


def detect_device(requested: str) -> tuple[str, str]:
    """Return (device, compute_type). 'auto' -> cuda if an NVIDIA GPU is visible."""
    if requested == "auto":
        device = "cuda" if shutil.which("nvidia-smi") else "cpu"
    else:
        device = requested
    compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def extract_audio(src: Path, dst: Path) -> None:
    """Extract mono 16 kHz WAV — the format Whisper wants."""
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ar", "16000", "-ac", "1", "-vn",
        "-loglevel", "error",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        die(f"ffmpeg failed:\n{proc.stderr.strip()}")


def run_whisperx(audio: Path, outdir: Path, args, device: str, compute_type: str) -> Path:
    cmd = [
        "whisperx", str(audio),
        "--model", args.model,
        "--device", device,
        "--compute_type", compute_type,
        "--output_dir", str(outdir),
        "--output_format", "json",
        "--diarize",
        "--hf_token", args.hf_token,
    ]
    if args.language:
        cmd += ["--language", args.language]

    # Speaker-count controls. --speakers N pins both bounds.
    if args.speakers is not None:
        cmd += ["--min_speakers", str(args.speakers), "--max_speakers", str(args.speakers)]
    else:
        if args.min_speakers is not None:
            cmd += ["--min_speakers", str(args.min_speakers)]
        if args.max_speakers is not None:
            cmd += ["--max_speakers", str(args.max_speakers)]

    print(f"[whisperx] model={args.model} device={device} ...", file=sys.stderr)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        die("whisperx failed (see output above)")

    # whisperx names output after the audio file's stem.
    result = outdir / f"{audio.stem}.json"
    if not result.exists():
        cands = list(outdir.glob("*.json"))
        if not cands:
            die("whisperx produced no JSON output")
        result = cands[0]
    return result


def fmt_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def build_transcript(result_json: Path, show_ts: bool) -> str:
    data = json.loads(result_json.read_text())
    segments = data.get("segments", [])
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
        prefix = f"[{fmt_ts(seg_start)}] " if show_ts else ""
        lines.append(f"{prefix}{speaker}: {text}")

    for seg in segments:
        spk = seg.get("speaker", "SPEAKER_?")
        text = seg.get("text", "")
        if spk != cur_speaker:
            flush()
            cur_speaker = spk
            buf = []
            seg_start = seg.get("start", 0.0)
        buf.append(text)
    flush()

    return "\n\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="speaker-transcribe",
        description="Extract audio from a video and produce a speaker-labeled transcript.",
    )
    p.add_argument("input", type=Path, help="video or audio file")
    p.add_argument("-o", "--output", type=Path,
                   help="output transcript path (default: <input>.transcript.txt)")

    g = p.add_argument_group("speaker count (optional)")
    g.add_argument("--speakers", type=int, metavar="N",
                   help="exact number of speakers (pins min=max=N)")
    g.add_argument("--min-speakers", type=int, metavar="N", help="minimum speakers")
    g.add_argument("--max-speakers", type=int, metavar="N", help="maximum speakers")

    p.add_argument("--model", default="large-v2",
                   help="whisper model (tiny|base|small|medium|large-v2|large-v3, default: large-v2)")
    p.add_argument("--language", help="language code, e.g. en, es, pt (default: auto-detect)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="compute device (default: auto)")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="Hugging Face token (or set HF_TOKEN env var)")
    p.add_argument("--no-timestamps", action="store_true",
                   help="omit [mm:ss] timestamps in output")
    p.add_argument("--keep-audio", action="store_true",
                   help="keep the extracted .wav next to the output")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    if not args.input.exists():
        die(f"input not found: {args.input}")
    if not args.hf_token:
        die("no Hugging Face token. Pass --hf-token or set HF_TOKEN "
            "(needed for pyannote diarization models). See README.md")
    if args.speakers is not None and (args.min_speakers or args.max_speakers):
        die("--speakers is mutually exclusive with --min-speakers/--max-speakers")

    require("whisperx", "Install it: pip install whisperx  (see README.md)")

    device, compute_type = detect_device(args.device)

    out_path = args.output or args.input.with_suffix(".transcript.txt")

    with tempfile.TemporaryDirectory(prefix="sptr_") as tmp:
        tmpdir = Path(tmp)

        # Skip extraction if the input is already audio.
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

        result_json = run_whisperx(audio, tmpdir, args, device, compute_type)
        transcript = build_transcript(result_json, show_ts=not args.no_timestamps)

    if not transcript:
        die("transcript is empty — no speech detected?")

    out_path.write_text(transcript + "\n")
    print(f"\n{transcript}\n")
    print(f"[done] transcript written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
