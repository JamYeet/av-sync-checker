"""Evaluate the sync checker on GRID clips with synthetic audio offsets.

Usage (from repo root):  PYTHONPATH=src python scripts/evaluate.py
"""
import csv
import tempfile
from pathlib import Path

from scipy.io import wavfile

from avsync.extract import extract_audio, extract_frames, crop_faces
from avsync.sync import (load_syncnet, crops_to_tensor, audio_to_mfcc,
                         embed_windows, find_offset, shift_audio)

CLIP_DIR = Path("samples/grid/s1")
N_CLIPS = 20
OFFSETS_MS = [-200, -120, -80, -40, 0, 40, 80, 120, 200]
FRAME_MS = 40
RESULTS_DIR = Path("results")


def main():
    clips = sorted(CLIP_DIR.glob("*.mpg"))[:N_CLIPS]
    model = load_syncnet()
    rows = []

    with tempfile.TemporaryDirectory() as tmp:
        for clip in clips:
            # video side: done once per clip
            frames, fps = extract_frames(clip)
            try:
                video = crops_to_tensor(crop_faces(frames, fps))
            except ValueError:
                print(f"skipping {clip.name}: no face found")
                continue

            # audio side: read once, then shifted many times
            wav = extract_audio(clip, out_path=Path(tmp) / "a.wav")
            sample_rate, audio = wavfile.read(wav)

            for true_ms in OFFSETS_MS:
                shifted = shift_audio(audio, sample_rate, true_ms)
                mfcc = audio_to_mfcc(shifted, sample_rate)
                lip_embs, aud_embs = embed_windows(model, video, mfcc)
                result = find_offset(lip_embs, aud_embs)

                detected_ms = result["offset_ms"]
                error_ms = abs(detected_ms - true_ms)
                rows.append({
                    "clip": clip.name,
                    "true_ms": true_ms,
                    "detected_ms": detected_ms,
                    "error_ms": error_ms,
                    "confidence": round(result["confidence"], 3),
                })
            print(f"{clip.name} done")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "eval_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summarise(rows)


def summarise(rows):
    n = len(rows)
    mae = sum(r["error_ms"] for r in rows) / n
    within_1 = sum(1 for r in rows if r["error_ms"] <= FRAME_MS) / n          # BLANK 2: error at most one frame
    exact = sum(1 for r in rows if r["error_ms"] == 0) / n             # BLANK 3: no error at all

    lines = [
        f"Clips: {n // len(OFFSETS_MS)}, test cases: {n}",
        f"Mean absolute error: {mae:.1f} ms",
        f"Accuracy within ±1 frame ({FRAME_MS} ms): {within_1:.1%}",
        f"Exact accuracy: {exact:.1%}",
        "",
        "Per offset (true ms → within ±1 frame):",
    ]
    for off in OFFSETS_MS:
        subset = [r for r in rows if r["true_ms"] == off]
        acc = sum(1 for r in subset if r["error_ms"] <= FRAME_MS) / len(subset)
        lines.append(f"  {off:+5d} ms → {acc:.0%}")

    summary = "\n".join(lines)
    print(summary)
    (RESULTS_DIR / "summary.txt").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()