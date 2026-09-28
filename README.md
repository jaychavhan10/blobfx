# BlobFX

BlobFX is a playful hand-tracked visual effect that turns a detected hand into a hand-drawn blob animation. The effect starts with a red blob, adds a blue counterpart, merges them into a magenta form, creates a small dot marker, and then triggers a cinematic "boom" explosion when the hand flicks or releases the gesture.

This project is implemented in [bloom.py](bloom.py) and uses OpenCV, NumPy, and MediaPipe.

## What it does

- Tracks a single hand with MediaPipe Hands
- Detects finger states and pinch/flick gestures
- Animate red + blue blobs into a magenta merge
- Generates a small dot and then emits an explosion effect
- Supports webcam input or video file input
- Can save output to a video file

## Visual flow

1. Red blob appears when one finger is raised.
2. Blue blob appears when two fingers are raised and separated.
3. Fingers come together to merge the blobs into magenta.
4. A dot appears near the fingertip.
5. A flick/release triggers a burst effect.
6. Screen shake, shockwave, confetti, and glow animations play.

## Controls

- `q`: quit the preview window
- `b`: trigger a manual boom effect for testing

## Requirements

Install the Python dependencies:

```bash
pip install opencv-python numpy mediapipe
```

## Run with webcam

```bash
python bloom.py
```

## Run without showing the preview window

```bash
python bloom.py --src 0 --no-show
```

## Notes

- The script treats `--src 0` as the default webcam source.
- `--src` can also be a path to an input video.
- `--out` writes the processed frames to an MP4 file.
- The file in this workspace is named `bloom.py`, while the embedded header in the script refers to `blobfx.py`; both describe the same project.

## Project structure

```text
blobfx/
├── README.md
├── bloom.py
```

## Technical summary

The project is driven by a small animation state machine:

- `RED_ONLY`
- `VS`
- `MERGE`
- `MAGENTA`
- `DOT`
- `BOOM`
- `GONE`

Each state updates blob position, motion, colour interpolation, and effect timing using spring-like smoothing and easing functions.

## Troubleshooting

- If the animation is not triggering, make sure your camera is well lit and the hand is clearly visible.
- If the boom fires too early or too late, tune the gesture thresholds near the top of the script:
  - `DOT_TIP`
  - `FLICK_CLOSED`
  - `FLICK_OPEN`
  - `FLICK_FRAMES`
  - `FLICK_SPEED`

## License

This project does not appear to include an explicit license file. Treat it as a personal or educational project unless otherwise specified by the author.
