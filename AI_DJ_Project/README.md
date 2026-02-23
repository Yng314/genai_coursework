# AI_DJ_Project

## Coursework-ready demo (HF Spaces + Gradio, Phase A/B)

This repo now includes a **Hugging Face Spaces** demo in `app.py`:

- Upload **Song A** and **Song B**.
- Pick a transition style plugin + text instruction.
- Build a rough seam (`A_tail + B_head`) with BPM-aware stretching.
- Run **ACE-Step repaint** on the seam window.
- If ACE-Step fails, fallback to deterministic crossfade.
- Output two artifacts:
  - transition-only clip
  - stitched clip (`A context + transition + B context`)

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
- `packages.txt` (installs `ffmpeg` for MP3 decoding)

Important: **Do not upload copyrighted songs** into the Space repo. The demo is designed for **user uploads**.

### Optional: Enable ACE-Step backend

`ACE-Step` is not required for basic usage. To enable it:

```shell
pip install git+https://github.com/ACE-Step/ACE-Step-1.5.git
```

Then run with optional env vars:

```shell
export AI_DJ_DEFAULT_BACKEND=acestep
export AI_DJ_ACESTEP_MODEL_CONFIG=acestep-v15-base
# optional persistent root for checkpoints:
export AI_DJ_ACESTEP_PROJECT_ROOT=/data/acestep_runtime
```

Notes:
- ACE-Step currently targets Python 3.11.
- ACE-Step first run can take time due to checkpoint download.
- Crossfade fallback is automatically used when ACE-Step repaint fails.

## Downloading Allin1 (legacy / local experiments)
 
[https://github.com/mir-aidj/all-in-one?tab=readme-ov-file#usage-for-python](https://github.com/mir-aidj/all-in-one?tab=readme-ov-file#usage-for-python)

### 1. Install PyTorch

Visit [PyTorch](https://pytorch.org/) and install the appropriate version for your system.

### 2. Install NATTEN (Required for Linux and Windows; macOS will auto-install)

* **Linux**: Download from [NATTEN website](https://www.shi-labs.com/natten/)
* **macOS**: Auto-installs with `allin1`.
* **Windows**: Build from source:
  
```shell
pip install ninja # Recommended, not required
git clone https://github.com/SHI-Labs/NATTEN
cd NATTEN
make
```

### 3. Install the package

```shell
pip install git+https://github.com/CPJKU/madmom  # install the latest madmom directly from GitHub
pip install allin1  # install this package
```

### 4. (Optional) Install FFmpeg for MP3 support

For ubuntu:

```shell
sudo apt install ffmpeg
```

For macOS:

```shell
brew install ffmpeg
```