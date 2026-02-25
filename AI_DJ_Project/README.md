# AI_DJ_Project

## Coursework-ready demo (HF Spaces + Gradio, Phase A/B)

This repo now includes a **Hugging Face Spaces** demo in `app.py`:

- Upload **Song A** and **Song B**.
- Pick a transition style plugin + text instruction.
- Build a rough seam (`A_tail + B_head`) with BPM-aware stretching.
- Run **ACE-Step repaint** on the seam window.
- Output two artifacts:
  - transition-only clip
  - stitched clip (`Song A up to cue + transition + Song B continuation`, seam is replaced not inserted)

### Deterministic transition API (Phase A)

Core reusable pipeline lives in:
- `pipeline/audio_utils.py`
- `pipeline/transition_generator.py`

Run via command:

```shell
python -m pipeline.transition_generator \
  --song-a /path/to/song_a.mp3 \
  --song-b /path/to/song_b.mp3 \
  --plugin "Smooth Blend" \
  --instruction "smooth, rising energy, no vocals" \
  --seed 42 \
  --output-dir outputs
```

This writes:
- `*_transition.wav`
- `*_stitched.wav`

### Deploy to Hugging Face Spaces (ZeroGPU)

Create a new Space with:
- **SDK**: Gradio
- **Hardware**: ZeroGPU

Upload these files from this folder:
- `app.py`
- `requirements.txt`
- `packages.txt` (installs `ffmpeg` + `libsndfile1` for audio decoding/runtime)

Important: **Do not upload copyrighted songs** into the Space repo. The demo is designed for **user uploads**.

### Repo hygiene

- The coursework spec notebook at repo root is intentionally git-ignored:
  `(0) 70113_Generative_AI_README_for_Coursework.ipynb`

### ACE-Step backend (required)

This coursework pipeline uses ACE-Step as the generation method.

```shell
pip install git+https://github.com/ACE-Step/ACE-Step-1.5.git
```

Then run with environment vars as needed:

```shell
export AI_DJ_ACESTEP_MODEL_CONFIG=acestep-v15-turbo
# optional persistent root for checkpoints:
export AI_DJ_ACESTEP_PROJECT_ROOT=/data/acestep_runtime
```

Notes:
- ACE-Step currently targets Python 3.11.
- ACE-Step first run can take time due to checkpoint download.

### Optional: Demucs stem-aware cue scoring

Cuepoint scoring can optionally run Demucs on the **analysis windows only** (A tail window + B head window), derive stem-aware mixability signals (`vocals`, `drums`, `bass`, accompaniment density), and penalize overlap risk (vocal-vocal and bass-bass clashes).

Transition generation can also use Demucs for:
- drum-led phase locking,
- one-bassline handoff shaping in `src_audio`,
- accompaniment-only `reference_audio`,
- post-repaint stem correction near transition boundaries.

Environment toggles:

```shell
# disable Demucs analysis entirely
export AI_DJ_ENABLE_DEMUCS_ANALYSIS=0

# disable Demucs transition refinements entirely
export AI_DJ_ENABLE_DEMUCS_TRANSITION=0

# choose analysis device when enabled (default: cuda if available)
export AI_DJ_DEMUCS_DEVICE=cpu

# choose reference period type passed into ACE-Step reference_audio
# values: accompaniment-only (default) | full-period-a
export AI_DJ_REFERENCE_AUDIO_MODE=accompaniment-only
```
