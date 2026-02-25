import logging
import os
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

LOGGER = logging.getLogger(__name__)


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def ensure_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y
    return np.mean(y, axis=1)


def ffprobe_duration_sec(path: str) -> Optional[float]:
    if not shutil.which("ffprobe"):
        return None

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return float(out)
    except Exception:
        return None


def decode_segment(path: str, start_sec: float, duration_sec: float, sr: int, max_decode_sec: float = 120.0) -> Tuple[np.ndarray, int]:
    start_sec = max(0.0, float(start_sec))
    duration_sec = max(0.0, float(duration_sec))
    duration_sec = min(duration_sec, max_decode_sec)

    if duration_sec <= 0:
        return np.zeros((0,), dtype=np.float32), sr

    if shutil.which("ffmpeg"):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-ss",
                str(start_sec),
                "-t",
                str(duration_sec),
                "-i",
                path,
                "-ac",
                "1",
                "-ar",
                str(sr),
                tmp_path,
            ]
            subprocess.run(cmd, check=True)
            y, read_sr = sf.read(tmp_path, dtype="float32", always_2d=False)
            y = ensure_mono(np.asarray(y))
            return y.astype(np.float32), int(read_sr)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    y, read_sr = librosa.load(path, sr=sr, mono=True, offset=start_sec, duration=duration_sec)
    return y.astype(np.float32), int(read_sr)


def estimate_bpm_and_beats(y: np.ndarray, sr: int) -> Tuple[Optional[float], np.ndarray]:
    if y.size < sr:
        return None, np.array([], dtype=np.float32)

    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo_f = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).astype(np.float32)
        if not (40.0 <= tempo_f <= 220.0):
            tempo_f = None
        return tempo_f, beat_times
    except Exception:
        return None, np.array([], dtype=np.float32)


def choose_nearest_beat(beat_times: np.ndarray, target_sec: float) -> float:
    if beat_times.size == 0:
        return float(target_sec)
    idx = int(np.argmin(np.abs(beat_times - float(target_sec))))
    return float(beat_times[idx])


def choose_first_beat_after(beat_times: np.ndarray, target_sec: float) -> float:
    if beat_times.size == 0:
        return float(target_sec)
    for bt in beat_times:
        if float(bt) >= float(target_sec):
            return float(bt)
    return float(beat_times[-1])


def linear_fade(n: int, fade_in: bool) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    if fade_in:
        return np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.linspace(1.0, 0.0, n, dtype=np.float32)


def normalize_peak(y: np.ndarray, peak: float = 0.98) -> np.ndarray:
    if y.size == 0:
        return y.astype(np.float32)
    maximum = float(np.max(np.abs(y)))
    if maximum <= 1e-9:
        return y.astype(np.float32)
    if maximum <= peak:
        return y.astype(np.float32)
    return (y * (peak / maximum)).astype(np.float32)


def apply_edge_fades(y: np.ndarray, sr: int, fade_ms: float = 30.0) -> np.ndarray:
    n = y.size
    fade_n = int(sr * (fade_ms / 1000.0))
    fade_n = min(fade_n, n // 2)
    if fade_n <= 0:
        return y
    y2 = y.copy()
    y2[:fade_n] *= linear_fade(fade_n, fade_in=True)
    y2[-fade_n:] *= linear_fade(fade_n, fade_in=False)
    return y2


def ensure_length(y: np.ndarray, target_n: int) -> np.ndarray:
    target_n = int(max(0, target_n))
    if y.size < target_n:
        return np.pad(y, (0, target_n - y.size), mode="constant")
    return y[:target_n]


def safe_time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    rate = float(rate)
    if y.size == 0:
        return y.astype(np.float32)
    if abs(rate - 1.0) < 1e-6:
        return y.astype(np.float32)
    try:
        return librosa.effects.time_stretch(y, rate=rate).astype(np.float32)
    except Exception as exc:
        LOGGER.warning("Time-stretch failed (%s); using original audio.", exc)
        return y.astype(np.float32)


def resample_if_needed(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if int(orig_sr) == int(target_sr):
        return y.astype(np.float32)
    return librosa.resample(y, orig_sr=int(orig_sr), target_sr=int(target_sr)).astype(np.float32)


def crossfade_equal_length(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = min(a.size, b.size)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    a = a[:n]
    b = b[:n]
    fade_in = linear_fade(n, fade_in=True)
    fade_out = 1.0 - fade_in
    return (a * fade_out + b * fade_in).astype(np.float32)


def write_wav(path: str, y: np.ndarray, sr: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y.astype(np.float32), int(sr))

