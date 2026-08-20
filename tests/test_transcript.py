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
