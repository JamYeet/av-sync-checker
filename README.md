# AV Sync Checker

![tests](https://github.com/JamYeet/av-sync-checker/actions/workflows/tests.yml/badge.svg)

Finds how far a video's audio is out of sync with the speaker's lips, and tells you whether it's in sync, out of sync, or uncertain.

https://github.com/user-attachments/assets/34b9b400-3b0d-4919-b48a-5081b75ca0be

## How it works

1. **Extract** the audio (16 kHz mono) and video frames, normalised to 25 fps.
2. **Crop the face** in every frame with MediaPipe's face detector.
3. **Fingerprint** every 0.2-second window of lip movement and audio with SyncNet, a pre-trained audio-visual model.
4. **Shift** the audio backwards and forwards up to 15 frames (±600 ms). The shift where the lip and audio fingerprints match best is the offset.
5. **Score confidence** by how clearly the best shift stands out from the rest. Low confidence gives an "uncertain" verdict instead of a guess.

The hardest part was the timing: video runs at 25 frames a second but the audio features come at 100 steps a second, so every frame matches 4 audio steps, and the windows have to be sliced so they line up exactly.

## Results

I shifted the audio of 20 in-sync clips from the GRID corpus by known amounts (−200 to +200 ms, 9 offsets each) and checked the detected offset.

| Metric | Result |
| --- | --- |
| Test cases | 180 |
| Within one frame (±40 ms) | 100% |
| Exact frame | 75.6% |
| Mean absolute error | 9.8 ms |

## Quick start

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org/) on your PATH. Model weights download automatically on first run.

```bash
git clone https://github.com/JamYeet/av-sync-checker.git
cd av-sync-checker
pip install -e ".[dev]"
```

Check a video from the command line:

```bash
avsync check examples/audio_200ms_late.mp4
```

Or run the web demo and open http://127.0.0.1:7860:

```bash
python app.py
```

Run the tests:

```bash
pytest -v
```

## Limitations

- **Clean input.** Tested on GRID, which is studio footage: one speaker, facing the camera, good lighting. Head turns, multiple speakers and background noise haven't been tested yet.
- **Needs a visible face.** If there's no face in the first frame, it stops with an error.
- **Limited range.** It searches up to 15 frames either way (±600 ms); anything further won't be caught.
- **Short clips.** If a clip is too short, the confidence score isn't reliable, so it says uncertain.

## Credits

- **SyncNet**: Chung & Zisserman, "Out of time: automated lip sync in the wild" (ACCV 2016). Model code and weights from [joonson/syncnet_python](https://github.com/joonson/syncnet_python), MIT licence.
- **Face detection**: [MediaPipe](https://ai.google.dev/edge/mediapipe) BlazeFace.
- **GRID corpus**: Cooke, Barker, Cunningham & Shao (2006), used under CC BY 4.0. Example clips are derived from GRID.
