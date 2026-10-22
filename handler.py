#!/usr/bin/env python3
"""Runpod serverless queue worker - F5-TTS voice-clone chunk renderer with ASR self-check.

Job input contract:
  {"chunks": [{"id": "s1_c01", "text": "..."}], "ref_text": "<exact reference transcript>"}
The reference audio is read ONLY from /runpod-volume/voice/ref_jason.wav - never from the
payload. Model weights cache on the network volume (HF_HOME=/runpod-volume/hf): the first
worker to boot downloads the F5-TTS and faster-whisper weights there; later workers mount
the cache instead of re-downloading.

Returns:
  {"results": [{"id", "audio_b64", "duration_s", "asr_ratio", "attempts"}],
   "worker_report": {...}}
Logs carry chunk ids and scores only - never script text, never audio. Rendered audio is
written to container-local scratch only, never to the volume.
"""
import base64
import os
import re
import tempfile
import time
import wave
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

VOLUME = Path("/runpod-volume")
REF_PATH = VOLUME / "voice" / "ref_jason.wav"
HF_HOME = os.environ.get("HF_HOME", "/runpod-volume/hf")
os.environ["HF_HOME"] = HF_HOME
ASR_THRESHOLD = 0.97
SEEDS = [42, 1077, 1154]  # take 1 = local-standard seed 42; retries mirror the local
                          # gen_segments.py re-render formula (1000 + att*77, att = 1, 2)

# --- ASR scoring normalization, copied from ~/workspace/voice-clone/verify_segments.py ---
ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
        'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
        'seventeen', 'eighteen', 'nineteen']
TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']


def int_words(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + (' ' + ONES[n % 10] if n % 10 else '')
    if n < 1000:
        return ONES[n // 100] + ' hundred' + (' ' + int_words(n % 100) if n % 100 else '')
    if n < 10000:
        return int_words(n // 1000) + ' thousand' + (' ' + int_words(n % 1000) if n % 1000 else '')
    return str(n)


def norm_tokens(s):
    s = s.lower().replace('%', ' percent ')
    s = s.replace("'", '')
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    toks = []
    for t in s.split():
        if t.isdigit():
            toks.extend(int_words(int(t)).split())
        else:
            toks.append(t)
    return toks


_state = {}


def _cache_present():
    hub = Path(HF_HOME) / "hub"
    f5 = hub.exists() and any(hub.glob("models--SWivid--F5-TTS"))
    wh = hub.exists() and any(hub.glob("models--Systran--faster-whisper-small"))
    return bool(f5), bool(wh)


def _init():
    if "tts" in _state:
        return _state
    t0 = time.time()
    f5_before, wh_before = _cache_present()
    import torch
    from f5_tts.api import F5TTS
    from faster_whisper import WhisperModel
    tts = F5TTS(model="F5TTS_v1_Base", device="cuda")
    asr = WhisperModel("small", device="cuda", compute_type="float16")
    f5_after, wh_after = _cache_present()
    _state.update(tts=tts, asr=asr, report={
        "gpu": torch.cuda.get_device_name(0),
        "f5_cache_present_before_load": f5_before,
        "whisper_cache_present_before_load": wh_before,
        "weights_downloaded_this_boot": bool((not f5_before and f5_after)
                                             or (not wh_before and wh_after)),
        "model_load_s": round(time.time() - t0, 2),
        "ref_bytes": REF_PATH.stat().st_size if REF_PATH.exists() else None,
    })
    r = _state["report"]
    print(f"init: gpu={r['gpu']} load_s={r['model_load_s']} "
          f"f5_cache_before={f5_before} whisper_cache_before={wh_before}", flush=True)
    return _state


def _load_audio_16k(path):
    """Return (float32 mono 16 kHz array for ASR, duration_s of the original file)."""
    import torchaudio
    wav, sr = torchaudio.load(path)
    with wave.open(str(path)) as w:
        duration = w.getnframes() / w.getframerate()
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0).numpy().astype(np.float32), duration


def _score(asr, audio16k, text):
    segs, _info = asr.transcribe(audio16k, language="en", beam_size=1)
    heard = " ".join(s.text.strip() for s in segs)
    return SequenceMatcher(None, norm_tokens(text), norm_tokens(heard)).ratio()


def handler(job):
    st = _init()
    job_input = job.get("input") or {}
    chunks = job_input["chunks"]
    ref_text = job_input["ref_text"]
    if not REF_PATH.exists():
        raise RuntimeError(f"reference audio missing at {REF_PATH}")
    results = []
    first_pass = 0
    retries = 0
    for c in chunks:
        cid, text = c["id"], c["text"]
        best = None  # (ratio, attempts, audio_b64, duration_s)
        for take, seed in enumerate(SEEDS, start=1):
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                st["tts"].infer(ref_file=str(REF_PATH), ref_text=ref_text, gen_text=text,
                                file_wave=tmp, remove_silence=False, seed=seed)
                audio16k, duration = _load_audio_16k(tmp)
                ratio = _score(st["asr"], audio16k, text)
                b64 = base64.b64encode(Path(tmp).read_bytes()).decode()
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            print(f"chunk {cid} take {take} seed {seed} asr_ratio {ratio:.4f}", flush=True)
            if best is None or ratio > best[0]:
                best = (ratio, take, b64, duration)
            if ratio >= ASR_THRESHOLD:
                break
        ratio, attempts, b64, duration = best
        if attempts == 1:
            first_pass += 1
        retries += attempts - 1
        results.append({"id": cid, "audio_b64": b64, "duration_s": round(duration, 3),
                        "asr_ratio": round(ratio, 4), "attempts": attempts})
    report = dict(_state["report"])
    report.update({"chunks": len(chunks), "first_pass": first_pass, "retries": retries})
    return {"results": results, "worker_report": report}


import runpod

runpod.serverless.start({"handler": handler})
