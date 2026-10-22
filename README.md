# runpod-f5-worker

Runpod Serverless (queue-based) worker image: renders F5-TTS voice-clone narration
chunks on a GPU and self-verifies each chunk with faster-whisper ASR before returning it.

## Why this repo is public - and safe to be

The image contains only code and public model configuration:

- **No voice data.** The reference audio is never in this repo or image. It lives on a
  private Runpod network volume (`/runpod-volume/voice/ref_jason.wav`) and is read from
  there at runtime.
- **No keys or credentials.** The worker needs no API keys; Runpod invokes it through the
  endpoint's queue.
- **Nothing proprietary.** Pinned open-source deps only: `runpod`, `f5-tts`,
  `faster-whisper` on an official `runpod/pytorch` base image. Model weights
  (F5-TTS, whisper-small) are public and are downloaded at runtime into a cache on the
  network volume, not baked into the image.

## Contract

Job input: `{"chunks": [{"id": "s1_c01", "text": "..."}], "ref_text": "<reference transcript>"}`

Output: `{"results": [{"id", "audio_b64", "duration_s", "asr_ratio", "attempts"}], "worker_report": {...}}`

A chunk passes at ASR ratio >= 0.97 (same normalization/scoring as the local pipeline);
on a miss the worker retries with a new seed, max 3 takes, and keeps the best take.
Worker logs carry chunk ids and scores only - never script text, never audio.

## Build

GitHub Actions builds on push to `main` and pushes
`ghcr.io/unclesam906/runpod-f5-worker:<commit-sha>` and `:v1`.
