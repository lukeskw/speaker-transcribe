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
