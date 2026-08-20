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
