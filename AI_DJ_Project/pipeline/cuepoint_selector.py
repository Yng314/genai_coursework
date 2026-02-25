import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import librosa  # type: ignore[reportMissingImports]
import numpy as np

from .audio_utils import choose_first_beat_after, choose_nearest_beat, decode_segment, ensure_length

LOGGER = logging.getLogger(__name__)

_ANALYSIS_HOP = 512
_STRUCT_SR = 22050
_DEMUCS_ENABLED = os.getenv("AI_DJ_ENABLE_DEMUCS_ANALYSIS", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_DEMUCS_MODEL_NAME = os.getenv("AI_DJ_DEMUCS_MODEL", "htdemucs").strip() or "htdemucs"
_DEMUCS_DEVICE_PREF = os.getenv("AI_DJ_DEMUCS_DEVICE", "cuda").strip().lower()
_DEMUCS_SEGMENT_SEC = 7.0
_DEMUCS_MIN_WINDOW_SEC = 6.0

_PROFILE_CACHE: Dict[Tuple[str, int], Optional["_TrackProfiles"]] = {}
_LIBROSA_STRUCT_CACHE: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
_DEMUCS_MODEL: Any = None
_DEMUCS_TORCH: Any = None
_DEMUCS_DEVICE = "cpu"
_DEMUCS_LOAD_ATTEMPTED = False
_DEMUCS_LOAD_ERROR: Optional[str] = None


@dataclass
class CueSelectionResult:
    cue_a_sec: float
    cue_b_sec: float
    method: str
    debug: Dict[str, object]


@dataclass
class _CueCandidate:
    time_sec: float
    beat_idx: int
    phrase: float
    energy: float
    onset: float
    chroma: np.ndarray
    vocal_ratio: float
    vocal_onset: float
    vocal_phrase_score: float
    drum_anchor: float
    bass_energy: float
    bass_stability: float
    instrumental_density: float
    density_score: float
    period_vocal_ratio: float
    period_vocal_phrase_score: float
    period_drum_anchor: float
    period_bass_energy: float
    period_bass_stability: float
    period_density_score: float
    period_coverage: float
    period_vocal_curve: np.ndarray
    period_bass_curve: np.ndarray


@dataclass
class _TrackProfiles:
    rms: np.ndarray
    rms_times: np.ndarray
    onset: np.ndarray
    onset_times: np.ndarray
    chroma: np.ndarray
    chroma_times: np.ndarray


@dataclass
class _VocalActivityProfile:
    vocal_ratio: np.ndarray
    vocal_onset: np.ndarray
    drum_onset: np.ndarray
    bass_rms: np.ndarray
    instrumental_rms: np.ndarray
    times: np.ndarray
    method: str
    has_drums: bool
    has_bass: bool


@dataclass
class _StructuredCandidate:
    cue: _CueCandidate
    label: str
    label_score: float
    edge_score: float
    position_score: float


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _mean_1d(values: np.ndarray, times: np.ndarray, start: float, end: float) -> float:
    if values.size == 0 or times.size == 0:
        return 0.0
    lo = float(min(start, end))
    hi = float(max(start, end))
    mask = (times >= lo) & (times <= hi)
    if np.any(mask):
        return float(np.mean(values[mask]))
    idx = int(np.argmin(np.abs(times - ((lo + hi) * 0.5))))
    return float(values[idx])


def _std_1d(values: np.ndarray, times: np.ndarray, start: float, end: float) -> float:
    if values.size == 0 or times.size == 0:
        return 0.0
    lo = float(min(start, end))
    hi = float(max(start, end))
    mask = (times >= lo) & (times <= hi)
    if np.any(mask):
        return float(np.std(values[mask]))
    idx = int(np.argmin(np.abs(times - ((lo + hi) * 0.5))))
    return 0.0 if idx < 0 or idx >= values.size else 0.0


def _smooth_1d(values: np.ndarray, kernel_size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros((1,), dtype=np.float32)
    k = int(max(1, kernel_size))
    if k == 1 or arr.size < k:
        return arr.astype(np.float32)
    kernel = np.ones((k,), dtype=np.float32) / float(k)
    return np.convolve(arr, kernel, mode="same").astype(np.float32)


def _normalize_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros((1,), dtype=np.float32)
    lo = float(np.percentile(arr, 5))
    hi = float(np.percentile(arr, 95))
    if hi - lo > 1e-6:
        out = (arr - lo) / (hi - lo)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    mx = float(np.max(np.abs(arr)))
    if mx > 1e-6:
        out = arr / mx
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    return np.zeros_like(arr, dtype=np.float32)


def _align_series_min_length(series: List[np.ndarray]) -> List[np.ndarray]:
    clean = [np.asarray(x, dtype=np.float32).reshape(-1) for x in series]
    if not clean:
        return []
    min_len = min((x.size for x in clean if x.size > 0), default=0)
    if min_len <= 0:
        return [np.zeros((1,), dtype=np.float32) for _ in clean]
    return [x[:min_len].astype(np.float32) if x.size >= min_len else np.pad(x, (0, min_len - x.size)).astype(np.float32) for x in clean]


def _mean_2d(values: np.ndarray, times: np.ndarray, start: float, end: float) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] == 0 or times.size == 0:
        return np.zeros((12,), dtype=np.float32)
    lo = float(min(start, end))
    hi = float(max(start, end))
    mask = (times >= lo) & (times <= hi)
    if np.any(mask):
        vec = np.mean(values[:, mask], axis=1).astype(np.float32)
    else:
        idx = int(np.argmin(np.abs(times - ((lo + hi) * 0.5))))
        vec = values[:, idx].astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-9:
        vec = vec / norm
    return vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def _phrase_score(beat_idx: int) -> float:
    if beat_idx < 0:
        return 0.5
    mod4 = beat_idx % 4
    mod8 = beat_idx % 8
    dist4 = min(mod4, 4 - mod4)
    dist8 = min(mod8, 8 - mod8)
    score4 = 1.0 - (dist4 / 2.0)
    score8 = 1.0 - (dist8 / 4.0)
    return _clamp((0.65 * score4) + (0.35 * score8), 0.0, 1.0)


def _target_position_score(x: float, target: float, spread: float) -> float:
    spread = max(1e-3, float(spread))
    return float(np.exp(-abs(float(x) - float(target)) / spread))


def _edge_score(x: float, duration_sec: float) -> float:
    if duration_sec <= 1e-6:
        return 0.0
    ratio = float(x / duration_sec)
    return _clamp(min(ratio / 0.16, (1.0 - ratio) / 0.16), 0.0, 1.0)


def _resolve_demucs_device(torch_mod: Any) -> str:
    pref = (_DEMUCS_DEVICE_PREF or "").strip().lower()
    if pref in {"cpu"}:
        return "cpu"
    if pref in {"cuda", "gpu"}:
        return "cuda" if bool(torch_mod.cuda.is_available()) else "cpu"
    return "cuda" if bool(torch_mod.cuda.is_available()) else "cpu"


def _get_demucs_model() -> Tuple[Optional[Any], Optional[Any], str, Optional[str]]:
    global _DEMUCS_MODEL, _DEMUCS_TORCH, _DEMUCS_DEVICE, _DEMUCS_LOAD_ATTEMPTED, _DEMUCS_LOAD_ERROR

    if not _DEMUCS_ENABLED:
        return None, None, "disabled", "AI_DJ_ENABLE_DEMUCS_ANALYSIS=0"

    if _DEMUCS_LOAD_ATTEMPTED:
        if _DEMUCS_MODEL is None:
            return None, _DEMUCS_TORCH, "unavailable", _DEMUCS_LOAD_ERROR
        return _DEMUCS_MODEL, _DEMUCS_TORCH, "ready", None

    _DEMUCS_LOAD_ATTEMPTED = True
    try:
        import torch  # type: ignore[reportMissingImports]
        from demucs.pretrained import get_model  # type: ignore[reportMissingImports]

        model = get_model(_DEMUCS_MODEL_NAME)
        model.eval()
        _DEMUCS_DEVICE = _resolve_demucs_device(torch)
        model.to(_DEMUCS_DEVICE)
        _DEMUCS_MODEL = model
        _DEMUCS_TORCH = torch
        _DEMUCS_LOAD_ERROR = None
        return _DEMUCS_MODEL, _DEMUCS_TORCH, "ready", None
    except Exception as exc:
        _DEMUCS_MODEL = None
        _DEMUCS_TORCH = None
        _DEMUCS_LOAD_ERROR = str(exc)
        LOGGER.warning(
            "Demucs vocal analysis unavailable (%s). Cue selection continues without vocal penalty.",
            exc,
        )
        return None, None, "unavailable", _DEMUCS_LOAD_ERROR


def _vocal_score_from_ratio(vocal_ratio: float) -> float:
    ratio = _clamp(float(vocal_ratio), 0.0, 1.0)
    # Penalize clearly vocal-dominant bars while leaving mixed bars mostly neutral.
    return 1.0 - _clamp((ratio - 0.32) / 0.5, 0.0, 1.0)


def _lookup_stem_mixability(profile: Optional[_VocalActivityProfile], time_sec: float) -> Dict[str, float]:
    neutral = {
        "vocal_ratio": 0.0,
        "vocal_onset": 0.0,
        "vocal_phrase_score": 0.5,
        "drum_anchor": 0.5,
        "bass_energy": 0.5,
        "bass_stability": 0.5,
        "instrumental_density": 0.5,
        "density_score": 0.5,
    }
    if profile is None or profile.times.size == 0:
        return neutral
    t_min = float(np.min(profile.times))
    t_max = float(np.max(profile.times))
    if float(time_sec) < (t_min - 0.6) or float(time_sec) > (t_max + 0.6):
        return neutral

    ratio = _clamp(_mean_1d(profile.vocal_ratio, profile.times, time_sec - 1.2, time_sec + 1.2), 0.0, 1.0)
    vocal_onset = _clamp(_mean_1d(profile.vocal_onset, profile.times, time_sec - 0.2, time_sec + 0.3), 0.0, 1.0)
    vocal_before = _clamp(_mean_1d(profile.vocal_ratio, profile.times, time_sec - 1.8, time_sec - 0.25), 0.0, 1.0)
    vocal_after = _clamp(_mean_1d(profile.vocal_ratio, profile.times, time_sec + 0.25, time_sec + 1.8), 0.0, 1.0)
    ending_score = _clamp((vocal_before - vocal_after + 0.05) / 0.35, 0.0, 1.0)
    low_vocal_score = _vocal_score_from_ratio(ratio)
    onset_quiet_score = 1.0 - vocal_onset
    vocal_phrase_score = _clamp(
        (0.52 * low_vocal_score) + (0.30 * ending_score) + (0.18 * onset_quiet_score),
        0.0,
        1.0,
    )

    if profile.has_drums:
        drum_hit = _clamp(_mean_1d(profile.drum_onset, profile.times, time_sec - 0.1, time_sec + 0.24), 0.0, 1.0)
        drum_bg = _clamp(_mean_1d(profile.drum_onset, profile.times, time_sec - 1.0, time_sec + 1.0), 0.0, 1.0)
        drum_anchor = _clamp((0.72 * drum_hit) + (0.28 * _clamp(drum_hit - drum_bg + 0.22, 0.0, 1.0)), 0.0, 1.0)
    else:
        drum_anchor = 0.5

    if profile.has_bass:
        bass_energy = _clamp(_mean_1d(profile.bass_rms, profile.times, time_sec - 1.4, time_sec + 1.4), 0.0, 1.0)
        bass_std = _clamp(_std_1d(profile.bass_rms, profile.times, time_sec - 1.8, time_sec + 1.8), 0.0, 1.0)
        bass_cv = bass_std / max(1e-4, bass_energy + 0.08)
        bass_stability = 1.0 - _clamp((bass_cv - 0.18) / 0.85, 0.0, 1.0)
    else:
        bass_energy = 0.5
        bass_stability = 0.5

    instrumental_density = _clamp(
        _mean_1d(profile.instrumental_rms, profile.times, time_sec - 1.4, time_sec + 1.4),
        0.0,
        1.0,
    )
    density_score = _target_position_score(instrumental_density, target=0.56, spread=0.24)

    return {
        "vocal_ratio": float(ratio),
        "vocal_onset": float(vocal_onset),
        "vocal_phrase_score": float(vocal_phrase_score),
        "drum_anchor": float(drum_anchor),
        "bass_energy": float(bass_energy),
        "bass_stability": float(_clamp(bass_stability, 0.0, 1.0)),
        "instrumental_density": float(instrumental_density),
        "density_score": float(_clamp(density_score, 0.0, 1.0)),
    }


def _range_coverage_ratio(times: np.ndarray, start: float, end: float) -> float:
    if times.size == 0:
        return 0.0
    lo = float(min(start, end))
    hi = float(max(start, end))
    if hi - lo <= 1e-6:
        return 0.0
    t_min = float(np.min(times))
    t_max = float(np.max(times))
    overlap = max(0.0, min(hi, t_max) - max(lo, t_min))
    return _clamp(overlap / max(1e-6, (hi - lo)), 0.0, 1.0)


def _sample_curve(values: np.ndarray, times: np.ndarray, start: float, end: float, samples: int = 16) -> np.ndarray:
    n = int(max(4, samples))
    if values.size == 0 or times.size == 0:
        return np.zeros((n,), dtype=np.float32)
    lo = float(min(start, end))
    hi = float(max(start, end))
    if hi - lo <= 1e-6:
        base = float(_mean_1d(values, times, lo - 0.25, hi + 0.25))
        return np.full((n,), _clamp(base, 0.0, 1.0), dtype=np.float32)
    ts = np.linspace(lo, hi, n, dtype=np.float32)
    if times.size < 2:
        base = float(_mean_1d(values, times, lo, hi))
        return np.full((n,), _clamp(base, 0.0, 1.0), dtype=np.float32)
    curve = np.interp(
        ts.astype(np.float64),
        times.astype(np.float64),
        values.astype(np.float64),
        left=float(values[0]),
        right=float(values[-1]),
    ).astype(np.float32)
    return np.clip(curve, 0.0, 1.0).astype(np.float32)


def _lookup_period_mixability(
    profile: Optional[_VocalActivityProfile],
    start_sec: float,
    end_sec: float,
    incoming: bool,
) -> Dict[str, Any]:
    neutral_curve = np.full((16,), 0.5, dtype=np.float32)
    neutral = {
        "coverage": 0.0,
        "period_vocal_ratio": 0.0,
        "period_vocal_phrase_score": 0.5,
        "period_drum_anchor": 0.5,
        "period_bass_energy": 0.5,
        "period_bass_stability": 0.5,
        "period_density_score": 0.5,
        "period_vocal_curve": neutral_curve.copy(),
        "period_bass_curve": neutral_curve.copy(),
    }
    if profile is None or profile.times.size == 0:
        return neutral

    lo = float(min(start_sec, end_sec))
    hi = float(max(start_sec, end_sec))
    span = max(1e-4, hi - lo)
    coverage = _range_coverage_ratio(profile.times, lo, hi)
    if coverage <= 0.03:
        return neutral

    ratio_mean = _clamp(_mean_1d(profile.vocal_ratio, profile.times, lo, hi), 0.0, 1.0)
    vocal_curve = _sample_curve(profile.vocal_ratio, profile.times, lo, hi, samples=16)
    bass_curve = _sample_curve(profile.bass_rms, profile.times, lo, hi, samples=16)
    first_cut = lo + (0.35 * span)
    last_cut = hi - (0.35 * span)
    vocal_start = _clamp(_mean_1d(profile.vocal_ratio, profile.times, lo, first_cut), 0.0, 1.0)
    vocal_end = _clamp(_mean_1d(profile.vocal_ratio, profile.times, last_cut, hi), 0.0, 1.0)
    boundary_t = lo if incoming else hi
    vocal_onset_boundary = _clamp(
        _mean_1d(profile.vocal_onset, profile.times, boundary_t - 0.16, boundary_t + 0.26),
        0.0,
        1.0,
    )
    low_vocal_score = _vocal_score_from_ratio(ratio_mean)
    onset_quiet = 1.0 - vocal_onset_boundary
    if incoming:
        start_quiet = _clamp(1.0 - ((vocal_start - 0.22) / 0.58), 0.0, 1.0)
        rise_ok = _clamp((vocal_end - vocal_start + 0.08) / 0.38, 0.0, 1.0)
        trend_score = _clamp((0.72 * start_quiet) + (0.28 * rise_ok), 0.0, 1.0)
    else:
        ending = _clamp((vocal_start - vocal_end + 0.05) / 0.35, 0.0, 1.0)
        trend_score = ending
    vocal_phrase = _clamp((0.50 * low_vocal_score) + (0.30 * trend_score) + (0.20 * onset_quiet), 0.0, 1.0)

    if profile.has_drums:
        drum_mean = _clamp(_mean_1d(profile.drum_onset, profile.times, lo, hi), 0.0, 1.0)
        drum_std = _clamp(_std_1d(profile.drum_onset, profile.times, lo, hi), 0.0, 1.0)
        drum_boundary = _clamp(
            _mean_1d(profile.drum_onset, profile.times, boundary_t - 0.12, boundary_t + 0.20),
            0.0,
            1.0,
        )
        drum_anchor = _clamp(
            (0.45 * drum_boundary)
            + (0.35 * drum_mean)
            + (0.20 * (1.0 - _clamp(drum_std / 0.35, 0.0, 1.0))),
            0.0,
            1.0,
        )
    else:
        drum_anchor = 0.5

    if profile.has_bass:
        bass_mean = _clamp(_mean_1d(profile.bass_rms, profile.times, lo, hi), 0.0, 1.0)
        bass_std = _clamp(_std_1d(profile.bass_rms, profile.times, lo, hi), 0.0, 1.0)
        bass_cv = bass_std / max(0.08, bass_mean)
        bass_stability = 1.0 - _clamp((bass_cv - 0.20) / 0.95, 0.0, 1.0)
    else:
        bass_mean = 0.5
        bass_stability = 0.5

    density_mean = _clamp(_mean_1d(profile.instrumental_rms, profile.times, lo, hi), 0.0, 1.0)
    density_std = _clamp(_std_1d(profile.instrumental_rms, profile.times, lo, hi), 0.0, 1.0)
    density_target = _target_position_score(density_mean, target=0.56, spread=0.22)
    density_stability = 1.0 - _clamp(density_std / 0.32, 0.0, 1.0)
    density_score = _clamp((0.75 * density_target) + (0.25 * density_stability), 0.0, 1.0)

    return {
        "coverage": float(coverage),
        "period_vocal_ratio": float(ratio_mean),
        "period_vocal_phrase_score": float(vocal_phrase),
        "period_drum_anchor": float(drum_anchor),
        "period_bass_energy": float(bass_mean),
        "period_bass_stability": float(_clamp(bass_stability, 0.0, 1.0)),
        "period_density_score": float(density_score),
        "period_vocal_curve": vocal_curve.astype(np.float32),
        "period_bass_curve": bass_curve.astype(np.float32),
    }


def _period_overlap_clash(cand_a: _CueCandidate, cand_b: _CueCandidate) -> Tuple[float, float, float]:
    n = int(
        max(
            4,
            min(
                int(cand_a.period_vocal_curve.size),
                int(cand_b.period_vocal_curve.size),
                int(cand_a.period_bass_curve.size),
                int(cand_b.period_bass_curve.size),
            ),
        )
    )
    if n <= 0:
        vocal = _clamp(cand_a.period_vocal_ratio * cand_b.period_vocal_ratio, 0.0, 1.0)
        bass = _clamp(cand_a.period_bass_energy * cand_b.period_bass_energy, 0.0, 1.0)
        cov = 0.5 * (cand_a.period_coverage + cand_b.period_coverage)
        return vocal, bass, cov

    a_v = ensure_length(cand_a.period_vocal_curve.astype(np.float32), n)
    b_v = ensure_length(cand_b.period_vocal_curve.astype(np.float32), n)
    a_b = ensure_length(cand_a.period_bass_curve.astype(np.float32), n)
    b_b = ensure_length(cand_b.period_bass_curve.astype(np.float32), n)
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)

    w_a_v = 1.0 - x
    w_b_v = x
    vocal_risk = float(np.mean((a_v * w_a_v) * (b_v * w_b_v)))
    vocal_risk = _clamp(vocal_risk * 4.0, 0.0, 1.0)

    w_b_b = np.clip((x - 0.60) / 0.28, 0.0, 1.0).astype(np.float32)
    w_a_b = 1.0 - w_b_b
    center_bass_shape = (0.35 + (0.65 * np.abs((2.0 * x) - 1.0))).astype(np.float32)
    bass_risk = float(np.mean((a_b * w_a_b * center_bass_shape) * (b_b * w_b_b * center_bass_shape)))
    bass_risk = _clamp(bass_risk * 6.0, 0.0, 1.0)

    coverage = _clamp(0.5 * (cand_a.period_coverage + cand_b.period_coverage), 0.0, 1.0)
    if coverage < 0.35:
        fallback_v = _clamp(cand_a.period_vocal_ratio * cand_b.period_vocal_ratio, 0.0, 1.0)
        fallback_b = _clamp(cand_a.period_bass_energy * cand_b.period_bass_energy, 0.0, 1.0)
        alpha = _clamp((0.35 - coverage) / 0.35, 0.0, 1.0)
        vocal_risk = (1.0 - alpha) * vocal_risk + (alpha * fallback_v)
        bass_risk = (1.0 - alpha) * bass_risk + (alpha * fallback_b)

    return float(vocal_risk), float(bass_risk), float(coverage)


def _extract_vocal_profile_demucs(
    y: np.ndarray,
    sr: int,
    window_start_sec: float,
    track_label: str,
) -> Tuple[Optional[_VocalActivityProfile], Dict[str, object]]:
    global _DEMUCS_DEVICE

    info: Dict[str, object] = {
        "track": track_label,
        "enabled": bool(_DEMUCS_ENABLED),
        "model": _DEMUCS_MODEL_NAME,
    }
    if y.size < int(max(1, sr) * _DEMUCS_MIN_WINDOW_SEC):
        info["status"] = "skipped-short-window"
        return None, info

    model, torch_mod, status, reason = _get_demucs_model()
    info["status"] = status
    if reason:
        info["reason"] = reason
    if model is None or torch_mod is None:
        return None, info

    try:
        from demucs.apply import apply_model  # type: ignore[reportMissingImports]

        mono = np.asarray(y, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            info["status"] = "empty-window"
            return None, info
        peak = float(np.max(np.abs(mono)))
        if peak > 1e-9:
            mono = mono / peak

        demucs_sr = int(getattr(model, "samplerate", 44100))
        if int(sr) != demucs_sr:
            mono = librosa.resample(mono, orig_sr=int(sr), target_sr=demucs_sr).astype(np.float32)
        if mono.size < int(demucs_sr * _DEMUCS_MIN_WINDOW_SEC):
            info["status"] = "skipped-short-window"
            return None, info

        stereo = np.stack([mono, mono], axis=0)
        mix = torch_mod.from_numpy(stereo).unsqueeze(0).to(_DEMUCS_DEVICE)
        audio_sec = float(mono.size / max(1, demucs_sr))
        segment_limit = float(_DEMUCS_SEGMENT_SEC)
        if audio_sec <= (segment_limit + 0.02):
            use_split = False
            segment_sec = None
        else:
            use_split = True
            segment_sec = segment_limit

        try:
            with torch_mod.no_grad():
                estimates = apply_model(
                    model,
                    mix,
                    shifts=1,
                    split=use_split,
                    overlap=0.25,
                    progress=False,
                    device=_DEMUCS_DEVICE,
                    segment=segment_sec,
                )
        except Exception as exc:
            if _DEMUCS_DEVICE == "cuda":
                model.to("cpu")
                _DEMUCS_DEVICE = "cpu"
                mix = mix.to("cpu")
                with torch_mod.no_grad():
                    estimates = apply_model(
                        model,
                        mix,
                        shifts=1,
                        split=use_split,
                        overlap=0.25,
                        progress=False,
                        device="cpu",
                        segment=segment_sec,
                    )
                info["device_fallback"] = f"cuda->cpu ({exc})"
            else:
                raise

        estimates = estimates.detach().cpu()
        est = estimates[0] if estimates.ndim == 4 else estimates
        if est.ndim != 3:
            raise RuntimeError(f"Unexpected demucs output ndim: {est.ndim}")

        source_names = [str(s) for s in getattr(model, "sources", [])]
        if not source_names:
            raise RuntimeError("Demucs model returned no source labels.")
        if est.shape[0] != len(source_names):
            if est.shape[1] == len(source_names):
                est = est.permute(1, 0, 2)
            else:
                raise RuntimeError(
                    f"Demucs output/source mismatch ({tuple(est.shape)} vs {len(source_names)} sources)."
                )
        if "vocals" not in source_names:
            raise RuntimeError("Demucs model does not expose a 'vocals' stem.")

        vocal_idx = source_names.index("vocals")
        vocals = est[vocal_idx]
        has_drums = "drums" in source_names
        has_bass = "bass" in source_names
        drums = est[source_names.index("drums")] if has_drums else torch_mod.zeros_like(vocals)
        bass = est[source_names.index("bass")] if has_bass else torch_mod.zeros_like(vocals)
        non_vocal_idxs = [i for i in range(len(source_names)) if i != vocal_idx]
        if non_vocal_idxs:
            accompaniment = est[non_vocal_idxs].sum(dim=0)
        else:
            accompaniment = torch_mod.zeros_like(vocals)

        vocals_mono = vocals.mean(dim=0).numpy().astype(np.float32)
        drums_mono = drums.mean(dim=0).numpy().astype(np.float32)
        bass_mono = bass.mean(dim=0).numpy().astype(np.float32)
        accompaniment_mono = accompaniment.mean(dim=0).numpy().astype(np.float32)

        vocal_rms = librosa.feature.rms(y=vocals_mono, frame_length=2048, hop_length=_ANALYSIS_HOP)[0].astype(np.float32)
        acc_rms = librosa.feature.rms(y=accompaniment_mono, frame_length=2048, hop_length=_ANALYSIS_HOP)[0].astype(np.float32)
        bass_rms = librosa.feature.rms(y=bass_mono, frame_length=2048, hop_length=_ANALYSIS_HOP)[0].astype(np.float32)
        inst_rms = librosa.feature.rms(y=accompaniment_mono, frame_length=2048, hop_length=_ANALYSIS_HOP)[0].astype(np.float32)
        vocal_onset = librosa.onset.onset_strength(y=vocals_mono, sr=demucs_sr, hop_length=_ANALYSIS_HOP).astype(np.float32)
        drum_onset = librosa.onset.onset_strength(y=drums_mono, sr=demucs_sr, hop_length=_ANALYSIS_HOP).astype(np.float32)

        ratio_raw = vocal_rms / np.maximum(vocal_rms + acc_rms, 1e-6)
        ratio_raw = np.clip(ratio_raw, 0.0, 1.0).astype(np.float32)
        aligned = _align_series_min_length([ratio_raw, vocal_onset, drum_onset, bass_rms, inst_rms])
        if not aligned:
            raise RuntimeError("Demucs profile alignment failed.")
        ratio, vocal_onset_n, drum_onset_n, bass_rms_n, inst_rms_n = aligned

        ratio = _smooth_1d(ratio, kernel_size=5)
        vocal_onset_n = _normalize_1d(_smooth_1d(vocal_onset_n, kernel_size=3))
        drum_onset_n = _normalize_1d(_smooth_1d(drum_onset_n, kernel_size=3))
        bass_rms_n = _normalize_1d(_smooth_1d(bass_rms_n, kernel_size=5))
        inst_rms_n = _normalize_1d(_smooth_1d(inst_rms_n, kernel_size=5))

        times = librosa.frames_to_time(np.arange(ratio.size), sr=demucs_sr, hop_length=_ANALYSIS_HOP).astype(np.float32)
        times = times + float(window_start_sec)

        info.update(
            {
                "status": "ready",
                "device": _DEMUCS_DEVICE,
                "method": "demucs-stem-mixability",
                "has_drums": bool(has_drums),
                "has_bass": bool(has_bass),
                "split_mode": "chunked" if use_split else "full-window",
                "window_start_sec": round(float(window_start_sec), 3),
                "window_duration_sec": round(float(mono.size / max(1, demucs_sr)), 3),
            }
        )
        return _VocalActivityProfile(
            vocal_ratio=ratio,
            vocal_onset=vocal_onset_n,
            drum_onset=drum_onset_n,
            bass_rms=bass_rms_n,
            instrumental_rms=inst_rms_n,
            times=times,
            method="demucs-stem-mixability",
            has_drums=bool(has_drums),
            has_bass=bool(has_bass),
        ), info
    except Exception as exc:
        LOGGER.warning("Demucs vocal analysis failed for %s (%s). Continuing without vocal penalty.", track_label, exc)
        info["status"] = "error"
        info["reason"] = str(exc)
        return None, info


def _label_weight(label: str, outgoing: bool) -> float:
    label_l = (label or "").strip().lower()
    if outgoing:
        table = [
            ("outro", 1.00),
            ("break", 0.95),
            ("bridge", 0.90),
            ("verse", 0.82),
            ("chorus", 0.66),
            ("intro", 0.20),
            ("start", 0.10),
            ("end", 0.05),
        ]
    else:
        table = [
            ("verse", 0.95),
            ("break", 0.90),
            ("bridge", 0.84),
            ("chorus", 0.80),
            ("intro", 0.74),
            ("outro", 0.20),
            ("start", 0.10),
            ("end", 0.05),
        ]
    for token, score in table:
        if token in label_l:
            return float(score)
    return 0.60


def _compute_profiles(y: np.ndarray, sr: int) -> _TrackProfiles:
    if y.size == 0:
        zero = np.zeros((1,), dtype=np.float32)
        return _TrackProfiles(
            rms=zero,
            rms_times=zero.copy(),
            onset=zero.copy(),
            onset_times=zero.copy(),
            chroma=np.zeros((12, 1), dtype=np.float32),
            chroma_times=zero.copy(),
        )

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=_ANALYSIS_HOP)[0].astype(np.float32)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_ANALYSIS_HOP).astype(np.float32)
    try:
        harmonic = librosa.effects.harmonic(y)
        chroma = librosa.feature.chroma_cens(y=harmonic, sr=sr, hop_length=_ANALYSIS_HOP).astype(np.float32)
    except Exception as exc:
        LOGGER.warning("Harmonic chroma extraction failed (%s); falling back to raw chroma.", exc)
        chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=_ANALYSIS_HOP).astype(np.float32)

    if rms.size == 0:
        rms = np.zeros((1,), dtype=np.float32)
    if onset.size == 0:
        onset = np.zeros((1,), dtype=np.float32)
    if chroma.ndim != 2 or chroma.shape[1] == 0:
        chroma = np.zeros((12, 1), dtype=np.float32)

    max_rms = float(np.max(rms))
    if max_rms > 1e-9:
        rms = rms / max_rms
    max_onset = float(np.max(onset))
    if max_onset > 1e-9:
        onset = onset / max_onset

    rms_times = librosa.frames_to_time(np.arange(rms.size), sr=sr, hop_length=_ANALYSIS_HOP).astype(np.float32)
    onset_times = librosa.frames_to_time(np.arange(onset.size), sr=sr, hop_length=_ANALYSIS_HOP).astype(np.float32)
    chroma_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=_ANALYSIS_HOP).astype(np.float32)
    return _TrackProfiles(
        rms=rms,
        rms_times=rms_times,
        onset=onset,
        onset_times=onset_times,
        chroma=chroma,
        chroma_times=chroma_times,
    )


def _build_candidates(
    beat_times: np.ndarray,
    min_sec: float,
    max_sec: float,
    prefer_tail: bool,
    limit: int,
) -> List[Tuple[float, int]]:
    if beat_times.size == 0:
        return []
    idxs = [idx for idx, t in enumerate(beat_times) if float(min_sec) <= float(t) <= float(max_sec)]
    if not idxs:
        return []
    idxs = idxs[-limit:] if prefer_tail else idxs[:limit]
    return [(float(beat_times[idx]), int(idx)) for idx in idxs]


def _make_candidate(
    time_sec: float,
    beat_idx: int,
    profiles: _TrackProfiles,
    incoming: bool,
    seam_sec: float,
    vocal_profile: Optional[_VocalActivityProfile] = None,
    vocal_time_sec: Optional[float] = None,
) -> _CueCandidate:
    if incoming:
        energy = _mean_1d(profiles.rms, profiles.rms_times, time_sec, time_sec + 1.0)
        onset = _mean_1d(profiles.onset, profiles.onset_times, time_sec - 0.1, time_sec + 0.5)
    else:
        energy = _mean_1d(profiles.rms, profiles.rms_times, time_sec - 1.0, time_sec)
        onset = _mean_1d(profiles.onset, profiles.onset_times, time_sec - 0.5, time_sec + 0.1)
    chroma = _mean_2d(profiles.chroma, profiles.chroma_times, time_sec - 2.0, time_sec + 2.0)
    vocal_lookup_sec = float(vocal_time_sec) if vocal_time_sec is not None else float(time_sec)
    stem_mix = _lookup_stem_mixability(vocal_profile, vocal_lookup_sec)
    seam = max(1e-3, float(seam_sec))
    period_start = vocal_lookup_sec if incoming else (vocal_lookup_sec - seam)
    period_end = (vocal_lookup_sec + seam) if incoming else vocal_lookup_sec
    period_mix = _lookup_period_mixability(
        profile=vocal_profile,
        start_sec=period_start,
        end_sec=period_end,
        incoming=incoming,
    )
    return _CueCandidate(
        time_sec=float(time_sec),
        beat_idx=int(beat_idx),
        phrase=_phrase_score(int(beat_idx)),
        energy=float(_clamp(energy, 0.0, 1.0)),
        onset=float(_clamp(onset, 0.0, 1.0)),
        chroma=chroma,
        vocal_ratio=float(stem_mix["vocal_ratio"]),
        vocal_onset=float(stem_mix["vocal_onset"]),
        vocal_phrase_score=float(stem_mix["vocal_phrase_score"]),
        drum_anchor=float(stem_mix["drum_anchor"]),
        bass_energy=float(stem_mix["bass_energy"]),
        bass_stability=float(stem_mix["bass_stability"]),
        instrumental_density=float(stem_mix["instrumental_density"]),
        density_score=float(stem_mix["density_score"]),
        period_vocal_ratio=float(period_mix["period_vocal_ratio"]),
        period_vocal_phrase_score=float(period_mix["period_vocal_phrase_score"]),
        period_drum_anchor=float(period_mix["period_drum_anchor"]),
        period_bass_energy=float(period_mix["period_bass_energy"]),
        period_bass_stability=float(period_mix["period_bass_stability"]),
        period_density_score=float(period_mix["period_density_score"]),
        period_coverage=float(period_mix["coverage"]),
        period_vocal_curve=np.asarray(period_mix["period_vocal_curve"], dtype=np.float32),
        period_bass_curve=np.asarray(period_mix["period_bass_curve"], dtype=np.float32),
    )


def _score_pair(
    cand_a: _CueCandidate,
    cand_b: _CueCandidate,
    target_a: float,
    target_b: float,
) -> Tuple[float, Dict[str, float]]:
    energy_match = 1.0 - min(1.0, abs(cand_a.energy - cand_b.energy))
    phrase_match = 0.5 * (cand_a.phrase + cand_b.phrase)
    key_match = _clamp(_cosine_similarity(cand_a.chroma, cand_b.chroma), 0.0, 1.0)
    onset_match = (0.35 * cand_a.onset) + (0.65 * cand_b.onset)
    position_match = 0.5 * (
        _target_position_score(cand_a.time_sec, target_a, spread=3.0)
        + _target_position_score(cand_b.time_sec, target_b, spread=3.0)
    )
    vocal_phrase_match = 0.5 * (cand_a.vocal_phrase_score + cand_b.vocal_phrase_score)
    drum_anchor_match = 0.5 * (cand_a.drum_anchor + cand_b.drum_anchor)
    bass_stability_match = 0.5 * (cand_a.bass_stability + cand_b.bass_stability)
    density_match = 0.5 * (cand_a.density_score + cand_b.density_score)
    period_vocal_phrase_match = 0.5 * (cand_a.period_vocal_phrase_score + cand_b.period_vocal_phrase_score)
    period_drum_anchor_match = 0.5 * (cand_a.period_drum_anchor + cand_b.period_drum_anchor)
    period_bass_stability_match = 0.5 * (cand_a.period_bass_stability + cand_b.period_bass_stability)
    period_density_match = 0.5 * (cand_a.period_density_score + cand_b.period_density_score)
    vocal_clash_risk, bass_clash_risk, period_coverage = _period_overlap_clash(cand_a, cand_b)
    clash_avoidance = 1.0 - _clamp((0.67 * vocal_clash_risk) + (0.33 * bass_clash_risk), 0.0, 1.0)

    total = (
        (0.07 * energy_match)
        + (0.06 * phrase_match)
        + (0.08 * key_match)
        + (0.04 * onset_match)
        + (0.03 * position_match)
        + (0.05 * vocal_phrase_match)
        + (0.04 * drum_anchor_match)
        + (0.03 * bass_stability_match)
        + (0.02 * density_match)
        + (0.14 * period_vocal_phrase_match)
        + (0.10 * period_drum_anchor_match)
        + (0.09 * period_bass_stability_match)
        + (0.07 * period_density_match)
        + (0.16 * clash_avoidance)
        + (0.02 * period_coverage)
    )
    components = {
        "energy_match": float(energy_match),
        "phrase_match": float(phrase_match),
        "key_match": float(key_match),
        "onset_match": float(onset_match),
        "position_match": float(position_match),
        "vocal_phrase_match": float(vocal_phrase_match),
        "drum_anchor_match": float(drum_anchor_match),
        "bass_stability_match": float(bass_stability_match),
        "density_match": float(density_match),
        "period_vocal_phrase_match": float(period_vocal_phrase_match),
        "period_drum_anchor_match": float(period_drum_anchor_match),
        "period_bass_stability_match": float(period_bass_stability_match),
        "period_density_match": float(period_density_match),
        "period_coverage": float(period_coverage),
        "vocal_clash_risk": float(vocal_clash_risk),
        "bass_clash_risk": float(bass_clash_risk),
        "clash_avoidance": float(clash_avoidance),
        "total": float(total),
    }
    return float(total), components


def _segments_from_boundaries(boundaries: np.ndarray, duration_sec: float) -> List[Dict[str, object]]:
    clean = [0.0]
    for t in np.asarray(boundaries, dtype=np.float32):
        x = float(t)
        if 0.0 < x < float(duration_sec):
            clean.append(x)
    clean.append(float(duration_sec))
    clean = sorted(set(round(x, 3) for x in clean))
    segs: List[Dict[str, object]] = []
    for idx in range(len(clean) - 1):
        start = float(clean[idx])
        end = float(clean[idx + 1])
        if end - start < 4.0:
            continue
        segs.append({"start": start, "end": end, "label": f"section_{idx + 1}"})
    return segs


def _try_get_librosa_structure(path: str, duration_sec: float) -> Optional[Dict[str, np.ndarray]]:
    if path in _LIBROSA_STRUCT_CACHE:
        return _LIBROSA_STRUCT_CACHE[path]

    decode_sec = _clamp(float(duration_sec), 15.0, 600.0)
    try:
        y, _ = decode_segment(
            path,
            start_sec=0.0,
            duration_sec=decode_sec,
            sr=_STRUCT_SR,
            max_decode_sec=max(600.0, decode_sec + 3.0),
        )
    except Exception as exc:
        LOGGER.warning("librosa full-track decode failed for %s (%s).", path, exc)
        _LIBROSA_STRUCT_CACHE[path] = None
        return None

    if y.size < _STRUCT_SR:
        _LIBROSA_STRUCT_CACHE[path] = None
        return None

    try:
        _, beat_frames = librosa.beat.beat_track(y=y, sr=_STRUCT_SR, trim=False)
        beat_times = librosa.frames_to_time(np.asarray(beat_frames), sr=_STRUCT_SR).astype(np.float32)
        downbeats = beat_times[::4] if beat_times.size > 0 else np.array([], dtype=np.float32)

        onset_env = librosa.onset.onset_strength(y=y, sr=_STRUCT_SR, hop_length=_ANALYSIS_HOP).astype(np.float32)
        boundary_frames = librosa.util.peak_pick(
            onset_env,
            pre_max=8,
            post_max=8,
            pre_avg=24,
            post_avg=24,
            delta=0.06,
            wait=18,
        )
        boundaries = librosa.frames_to_time(
            np.asarray(boundary_frames),
            sr=_STRUCT_SR,
            hop_length=_ANALYSIS_HOP,
        ).astype(np.float32)
        payload: Dict[str, np.ndarray] = {"downbeats": downbeats, "boundaries": boundaries}
        _LIBROSA_STRUCT_CACHE[path] = payload
        return payload
    except Exception as exc:
        LOGGER.warning("librosa structure extraction failed for %s (%s).", path, exc)
        _LIBROSA_STRUCT_CACHE[path] = None
        return None


def _get_or_build_profiles_for_track(path: str, duration_sec: float, sr: int) -> Optional[_TrackProfiles]:
    key = (path, int(sr))
    if key in _PROFILE_CACHE:
        return _PROFILE_CACHE[key]

    decode_sec = _clamp(float(duration_sec), 15.0, 600.0)
    try:
        y, _ = decode_segment(
            path,
            start_sec=0.0,
            duration_sec=decode_sec,
            sr=int(sr),
            max_decode_sec=max(600.0, decode_sec + 3.0),
        )
    except Exception as exc:
        LOGGER.warning("Full-track decode failed for %s (%s).", path, exc)
        _PROFILE_CACHE[key] = None
        return None

    if y.size < int(sr):
        _PROFILE_CACHE[key] = None
        return None

    profiles = _compute_profiles(y, int(sr))
    _PROFILE_CACHE[key] = profiles
    return profiles


def _label_for_time(segments: List[Dict[str, object]], t: float) -> str:
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        if start <= float(t) < end:
            return str(seg.get("label", "unknown"))
    return "unknown"


def _dedupe_times(times: List[float], min_gap_sec: float) -> List[float]:
    if not times:
        return []
    sorted_times = sorted(float(t) for t in times)
    out: List[float] = [sorted_times[0]]
    for t in sorted_times[1:]:
        if (t - out[-1]) >= float(min_gap_sec):
            out.append(t)
    return out


def _build_structured_candidates(
    downbeats: np.ndarray,
    segments: List[Dict[str, object]],
    profiles: _TrackProfiles,
    vocal_profile: Optional[_VocalActivityProfile],
    seam_sec: float,
    duration_sec: float,
    min_sec: float,
    max_sec: float,
    incoming: bool,
    target_ratio: float,
    limit: int = 20,
) -> List[_StructuredCandidate]:
    if max_sec <= min_sec:
        return []

    raw_times: List[float] = []
    if downbeats.size > 0:
        raw_times.extend([float(t) for t in downbeats if min_sec <= float(t) <= max_sec])

    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        if min_sec <= start <= max_sec:
            raw_times.append(start)
        if min_sec <= end <= max_sec:
            raw_times.append(end)

    if not raw_times and downbeats.size > 0:
        raw_times.extend([float(t) for t in downbeats])

    if not raw_times:
        return []

    snapped: List[float] = []
    for t in raw_times:
        if downbeats.size > 0:
            idx = int(np.argmin(np.abs(downbeats - float(t))))
            snapped_t = float(downbeats[idx])
        else:
            snapped_t = float(t)
        if min_sec <= snapped_t <= max_sec:
            snapped.append(snapped_t)

    snapped = _dedupe_times(snapped, min_gap_sec=1.2)
    if not snapped:
        return []

    if incoming:
        snapped = snapped[:limit]
    else:
        snapped = snapped[-limit:]

    target_sec = float(target_ratio * duration_sec)
    spread = max(4.0, 0.15 * duration_sec)

    built: List[_StructuredCandidate] = []
    for i, t in enumerate(snapped):
        label = _label_for_time(segments, t)
        cue = _make_candidate(
            time_sec=t,
            beat_idx=(i * 4),
            profiles=profiles,
            incoming=incoming,
            seam_sec=seam_sec,
            vocal_profile=vocal_profile,
            vocal_time_sec=t,
        )
        built.append(
            _StructuredCandidate(
                cue=cue,
                label=label,
                label_score=_label_weight(label, outgoing=(not incoming)),
                edge_score=_edge_score(t, duration_sec),
                position_score=_target_position_score(t, target=target_sec, spread=spread),
            )
        )
    return built


def _score_structured_pair(
    cand_a: _StructuredCandidate,
    cand_b: _StructuredCandidate,
) -> Tuple[float, Dict[str, float]]:
    energy_match = 1.0 - min(1.0, abs(cand_a.cue.energy - cand_b.cue.energy))
    phrase_match = 0.5 * (cand_a.cue.phrase + cand_b.cue.phrase)
    key_match = _clamp(_cosine_similarity(cand_a.cue.chroma, cand_b.cue.chroma), 0.0, 1.0)
    onset_match = (0.40 * cand_a.cue.onset) + (0.60 * cand_b.cue.onset)
    label_match = 0.5 * (cand_a.label_score + cand_b.label_score)
    position_match = 0.5 * (cand_a.position_score + cand_b.position_score)
    edge_match = 0.5 * (cand_a.edge_score + cand_b.edge_score)
    vocal_phrase_match = 0.5 * (cand_a.cue.vocal_phrase_score + cand_b.cue.vocal_phrase_score)
    drum_anchor_match = 0.5 * (cand_a.cue.drum_anchor + cand_b.cue.drum_anchor)
    bass_stability_match = 0.5 * (cand_a.cue.bass_stability + cand_b.cue.bass_stability)
    density_match = 0.5 * (cand_a.cue.density_score + cand_b.cue.density_score)
    period_vocal_phrase_match = 0.5 * (cand_a.cue.period_vocal_phrase_score + cand_b.cue.period_vocal_phrase_score)
    period_drum_anchor_match = 0.5 * (cand_a.cue.period_drum_anchor + cand_b.cue.period_drum_anchor)
    period_bass_stability_match = 0.5 * (cand_a.cue.period_bass_stability + cand_b.cue.period_bass_stability)
    period_density_match = 0.5 * (cand_a.cue.period_density_score + cand_b.cue.period_density_score)
    vocal_clash_risk, bass_clash_risk, period_coverage = _period_overlap_clash(cand_a.cue, cand_b.cue)
    clash_avoidance = 1.0 - _clamp((0.67 * vocal_clash_risk) + (0.33 * bass_clash_risk), 0.0, 1.0)

    total = (
        (0.08 * energy_match)
        + (0.09 * key_match)
        + (0.06 * onset_match)
        + (0.05 * phrase_match)
        + (0.10 * label_match)
        + (0.05 * position_match)
        + (0.04 * edge_match)
        + (0.06 * vocal_phrase_match)
        + (0.04 * drum_anchor_match)
        + (0.03 * bass_stability_match)
        + (0.02 * density_match)
        + (0.12 * period_vocal_phrase_match)
        + (0.08 * period_drum_anchor_match)
        + (0.07 * period_bass_stability_match)
        + (0.05 * period_density_match)
        + (0.06 * clash_avoidance)
        + (0.01 * period_coverage)
    )
    components = {
        "energy_match": float(energy_match),
        "key_match": float(key_match),
        "onset_match": float(onset_match),
        "phrase_match": float(phrase_match),
        "label_match": float(label_match),
        "position_match": float(position_match),
        "edge_match": float(edge_match),
        "vocal_phrase_match": float(vocal_phrase_match),
        "drum_anchor_match": float(drum_anchor_match),
        "bass_stability_match": float(bass_stability_match),
        "density_match": float(density_match),
        "period_vocal_phrase_match": float(period_vocal_phrase_match),
        "period_drum_anchor_match": float(period_drum_anchor_match),
        "period_bass_stability_match": float(period_bass_stability_match),
        "period_density_match": float(period_density_match),
        "period_coverage": float(period_coverage),
        "vocal_clash_risk": float(vocal_clash_risk),
        "bass_clash_risk": float(bass_clash_risk),
        "clash_avoidance": float(clash_avoidance),
        "total": float(total),
    }
    return float(total), components


def _try_structure_aware_selection(
    song_a_path: Optional[str],
    song_b_path: Optional[str],
    song_a_duration_sec: Optional[float],
    song_b_duration_sec: Optional[float],
    pre_sec: float,
    seam_sec: float,
    post_sec: float,
    vocal_profile_a: Optional[_VocalActivityProfile],
    vocal_profile_b: Optional[_VocalActivityProfile],
) -> Optional[CueSelectionResult]:
    if not song_a_path or not song_b_path:
        return None
    if song_a_duration_sec is None or song_b_duration_sec is None:
        return None

    dur_a = float(song_a_duration_sec)
    dur_b = float(song_b_duration_sec)

    min_a = max(seam_sec + 2.0, pre_sec + 2.0, 0.30 * dur_a)
    max_a = min(dur_a - seam_sec - 2.0, 0.88 * dur_a)
    min_b = max(4.0, 0.10 * dur_b)
    max_b = min(dur_b - (seam_sec + post_sec + 2.0), 0.72 * dur_b)

    if max_a <= min_a or max_b <= min_b:
        return None

    source = "librosa"
    lib_a = _try_get_librosa_structure(song_a_path, dur_a)
    lib_b = _try_get_librosa_structure(song_b_path, dur_b)
    if lib_a is None or lib_b is None:
        return None

    downbeats_a = np.asarray(lib_a.get("downbeats", []), dtype=np.float32)
    downbeats_b = np.asarray(lib_b.get("downbeats", []), dtype=np.float32)
    segments_a: List[Dict[str, object]] = _segments_from_boundaries(
        np.asarray(lib_a.get("boundaries", []), dtype=np.float32),
        duration_sec=dur_a,
    )
    segments_b: List[Dict[str, object]] = _segments_from_boundaries(
        np.asarray(lib_b.get("boundaries", []), dtype=np.float32),
        duration_sec=dur_b,
    )

    if downbeats_a.size < 4 or downbeats_b.size < 4:
        return None

    profiles_a = _get_or_build_profiles_for_track(song_a_path, dur_a, sr=_STRUCT_SR)
    profiles_b = _get_or_build_profiles_for_track(song_b_path, dur_b, sr=_STRUCT_SR)
    if profiles_a is None or profiles_b is None:
        return None

    cands_a = _build_structured_candidates(
        downbeats=downbeats_a,
        segments=segments_a,
        profiles=profiles_a,
        vocal_profile=vocal_profile_a,
        seam_sec=seam_sec,
        duration_sec=dur_a,
        min_sec=min_a,
        max_sec=max_a,
        incoming=False,
        target_ratio=0.63,
        limit=22,
    )
    cands_b = _build_structured_candidates(
        downbeats=downbeats_b,
        segments=segments_b,
        profiles=profiles_b,
        vocal_profile=vocal_profile_b,
        seam_sec=seam_sec,
        duration_sec=dur_b,
        min_sec=min_b,
        max_sec=max_b,
        incoming=True,
        target_ratio=0.27,
        limit=22,
    )
    if not cands_a or not cands_b:
        return None

    best_score = -1.0
    best_a: Optional[_StructuredCandidate] = None
    best_b: Optional[_StructuredCandidate] = None
    ranked: List[Dict[str, object]] = []
    for ca in cands_a:
        for cb in cands_b:
            score, comps = _score_structured_pair(ca, cb)
            ranked.append(
                {
                    "score": float(score),
                    "song_a_sec": float(ca.cue.time_sec),
                    "song_b_sec": float(cb.cue.time_sec),
                    "song_a_label": ca.label,
                    "song_b_label": cb.label,
                    "song_a_vocal_ratio": float(ca.cue.vocal_ratio),
                    "song_b_vocal_ratio": float(cb.cue.vocal_ratio),
                    "song_a_period_vocal_phrase": float(ca.cue.period_vocal_phrase_score),
                    "song_b_period_vocal_phrase": float(cb.cue.period_vocal_phrase_score),
                    "song_a_period_drum_anchor": float(ca.cue.period_drum_anchor),
                    "song_b_period_drum_anchor": float(cb.cue.period_drum_anchor),
                    "song_a_period_bass_stability": float(ca.cue.period_bass_stability),
                    "song_b_period_bass_stability": float(cb.cue.period_bass_stability),
                    "song_a_period_density": float(ca.cue.period_density_score),
                    "song_b_period_density": float(cb.cue.period_density_score),
                    "song_a_period_coverage": float(ca.cue.period_coverage),
                    "song_b_period_coverage": float(cb.cue.period_coverage),
                    "components": comps,
                }
            )
            if score > best_score:
                best_score = float(score)
                best_a = ca
                best_b = cb

    if best_a is None or best_b is None:
        return None

    ranked = sorted(ranked, key=lambda x: float(x["score"]), reverse=True)
    top_pairs = []
    for item in ranked[:3]:
        top_pairs.append(
            {
                "score": round(float(item["score"]), 4),
                "song_a_sec": round(float(item["song_a_sec"]), 3),
                "song_b_sec": round(float(item["song_b_sec"]), 3),
                "song_a_label": str(item["song_a_label"]),
                "song_b_label": str(item["song_b_label"]),
                "song_a_vocal_ratio": round(float(item["song_a_vocal_ratio"]), 4),
                "song_b_vocal_ratio": round(float(item["song_b_vocal_ratio"]), 4),
                "song_a_period_vocal_phrase": round(float(item["song_a_period_vocal_phrase"]), 4),
                "song_b_period_vocal_phrase": round(float(item["song_b_period_vocal_phrase"]), 4),
                "song_a_period_drum_anchor": round(float(item["song_a_period_drum_anchor"]), 4),
                "song_b_period_drum_anchor": round(float(item["song_b_period_drum_anchor"]), 4),
                "song_a_period_bass_stability": round(float(item["song_a_period_bass_stability"]), 4),
                "song_b_period_bass_stability": round(float(item["song_b_period_bass_stability"]), 4),
                "song_a_period_density": round(float(item["song_a_period_density"]), 4),
                "song_b_period_density": round(float(item["song_b_period_density"]), 4),
                "song_a_period_coverage": round(float(item["song_a_period_coverage"]), 4),
                "song_b_period_coverage": round(float(item["song_b_period_coverage"]), 4),
                "components": {k: round(float(v), 4) for k, v in item["components"].items()},
            }
        )

    principles = [
        "phrase/downbeat alignment",
        "section boundary awareness",
        "energy continuity",
        "harmonic/chroma compatibility",
    ]
    if vocal_profile_a is not None or vocal_profile_b is not None:
        principles.extend(
            [
                "vocal phrase-safe cueing (low or ending vocals)",
                "drum-anchor confidence",
                "bassline stability control",
                "instrumental density targeting",
                "clash-risk precheck (vocal+bass overlap)",
            ]
        )

    return CueSelectionResult(
        cue_a_sec=float(best_a.cue.time_sec),
        cue_b_sec=float(best_b.cue.time_sec),
        method=f"{source}-structure-aware",
        debug={
            "source": source,
            "candidate_ranges_sec": {
                "song_a": [round(min_a, 3), round(max_a, 3)],
                "song_b": [round(min_b, 3), round(max_b, 3)],
            },
            "transition_period_sec": round(float(seam_sec), 3),
            "candidate_counts": {"song_a": len(cands_a), "song_b": len(cands_b)},
            "selected_sec": {"song_a": round(float(best_a.cue.time_sec), 3), "song_b": round(float(best_b.cue.time_sec), 3)},
            "selected_labels": {"song_a": best_a.label, "song_b": best_b.label},
            "selected_mixability": {
                "song_a_ratio": round(float(best_a.cue.vocal_ratio), 4),
                "song_b_ratio": round(float(best_b.cue.vocal_ratio), 4),
                "song_a_vocal_onset": round(float(best_a.cue.vocal_onset), 4),
                "song_b_vocal_onset": round(float(best_b.cue.vocal_onset), 4),
                "song_a_vocal_phrase": round(float(best_a.cue.vocal_phrase_score), 4),
                "song_b_vocal_phrase": round(float(best_b.cue.vocal_phrase_score), 4),
                "song_a_drum_anchor": round(float(best_a.cue.drum_anchor), 4),
                "song_b_drum_anchor": round(float(best_b.cue.drum_anchor), 4),
                "song_a_bass_stability": round(float(best_a.cue.bass_stability), 4),
                "song_b_bass_stability": round(float(best_b.cue.bass_stability), 4),
                "song_a_density_score": round(float(best_a.cue.density_score), 4),
                "song_b_density_score": round(float(best_b.cue.density_score), 4),
                "song_a_period_vocal_phrase": round(float(best_a.cue.period_vocal_phrase_score), 4),
                "song_b_period_vocal_phrase": round(float(best_b.cue.period_vocal_phrase_score), 4),
                "song_a_period_drum_anchor": round(float(best_a.cue.period_drum_anchor), 4),
                "song_b_period_drum_anchor": round(float(best_b.cue.period_drum_anchor), 4),
                "song_a_period_bass_stability": round(float(best_a.cue.period_bass_stability), 4),
                "song_b_period_bass_stability": round(float(best_b.cue.period_bass_stability), 4),
                "song_a_period_density_score": round(float(best_a.cue.period_density_score), 4),
                "song_b_period_density_score": round(float(best_b.cue.period_density_score), 4),
                "song_a_period_coverage": round(float(best_a.cue.period_coverage), 4),
                "song_b_period_coverage": round(float(best_b.cue.period_coverage), 4),
            },
            "top_pairs": top_pairs,
            "period_scoring": {
                "enabled": True,
                "window_def": {"song_a": "[cue-seam, cue]", "song_b": "[cue, cue+seam]"},
                "overlap_simulation": "weighted vocal/bass clash precheck",
            },
            "dj_principles": principles,
        },
    )


def select_mix_cuepoints(
    y_a_analysis: np.ndarray,
    y_b_analysis: np.ndarray,
    sr: int,
    analysis_sec: float,
    pre_sec: float,
    seam_sec: float,
    post_sec: float,
    a_analysis_start_sec: float,
    beats_a: np.ndarray,
    beats_b: np.ndarray,
    cue_a_override_sec: Optional[float] = None,
    cue_b_override_sec: Optional[float] = None,
    song_a_path: Optional[str] = None,
    song_b_path: Optional[str] = None,
    song_a_duration_sec: Optional[float] = None,
    song_b_duration_sec: Optional[float] = None,
) -> CueSelectionResult:
    target_a_rel = max(float(pre_sec), float(analysis_sec - seam_sec - 2.0))
    target_b_rel = 2.0
    default_a_rel = float(choose_nearest_beat(beats_a, target_a_rel))
    default_b_rel = float(choose_first_beat_after(beats_b, target_b_rel))

    default_a_abs = float(a_analysis_start_sec + default_a_rel)
    default_b_abs = float(default_b_rel)

    if cue_a_override_sec is not None or cue_b_override_sec is not None:
        cue_a = float(cue_a_override_sec) if cue_a_override_sec is not None else default_a_abs
        cue_b = float(cue_b_override_sec) if cue_b_override_sec is not None else default_b_abs
        return CueSelectionResult(
            cue_a_sec=cue_a,
            cue_b_sec=cue_b,
            method="manual-override",
            debug={
                "manual_override": True,
                "default_auto_cues_sec": {"song_a": round(default_a_abs, 3), "song_b": round(default_b_abs, 3)},
            },
        )

    vocal_profile_a, vocal_debug_a = _extract_vocal_profile_demucs(
        y=y_a_analysis,
        sr=int(sr),
        window_start_sec=float(a_analysis_start_sec),
        track_label="song_a_analysis_window",
    )
    vocal_profile_b, vocal_debug_b = _extract_vocal_profile_demucs(
        y=y_b_analysis,
        sr=int(sr),
        window_start_sec=0.0,
        track_label="song_b_analysis_window",
    )
    vocal_debug = {
        "enabled": bool(_DEMUCS_ENABLED),
        "song_a": vocal_debug_a,
        "song_b": vocal_debug_b,
    }

    structure_result = _try_structure_aware_selection(
        song_a_path=song_a_path,
        song_b_path=song_b_path,
        song_a_duration_sec=song_a_duration_sec,
        song_b_duration_sec=song_b_duration_sec,
        pre_sec=float(pre_sec),
        seam_sec=float(seam_sec),
        post_sec=float(post_sec),
        vocal_profile_a=vocal_profile_a,
        vocal_profile_b=vocal_profile_b,
    )
    if structure_result is not None:
        structure_result.debug["manual_override"] = False
        structure_result.debug["default_local_auto_cues_sec"] = {
            "song_a": round(default_a_abs, 3),
            "song_b": round(default_b_abs, 3),
        }
        structure_result.debug["vocal_analysis"] = vocal_debug
        structure_result.debug["vocal_penalty_active"] = bool(vocal_profile_a is not None or vocal_profile_b is not None)
        return structure_result

    if beats_a.size < 4 or beats_b.size < 4:
        return CueSelectionResult(
            cue_a_sec=default_a_abs,
            cue_b_sec=default_b_abs,
            method="beat-fallback",
            debug={
                "reason": "insufficient_beats",
                "beat_counts": {"song_a": int(beats_a.size), "song_b": int(beats_b.size)},
                "vocal_analysis": vocal_debug,
            },
        )

    profiles_a = _compute_profiles(y_a_analysis, sr)
    profiles_b = _compute_profiles(y_b_analysis, sr)

    min_a = max(0.5, float(seam_sec + 0.5), float(pre_sec + (0.20 * seam_sec)))
    max_a = max(min_a + 0.1, float(analysis_sec - max(0.75, 0.25 * seam_sec)))
    min_b = max(0.75, float(0.12 * seam_sec))
    max_b = max(min_b + 0.1, float(analysis_sec - max((seam_sec + 0.75), (0.25 * post_sec))))

    raw_a = _build_candidates(beats_a, min_a, max_a, prefer_tail=True, limit=24)
    raw_b = _build_candidates(beats_b, min_b, max_b, prefer_tail=False, limit=24)
    if not raw_a or not raw_b:
        return CueSelectionResult(
            cue_a_sec=default_a_abs,
            cue_b_sec=default_b_abs,
            method="candidate-fallback",
            debug={
                "reason": "empty_candidate_set",
                "candidate_counts": {"song_a": len(raw_a), "song_b": len(raw_b)},
                "candidate_windows_sec": {
                    "song_a": [round(min_a, 3), round(max_a, 3)],
                    "song_b": [round(min_b, 3), round(max_b, 3)],
                },
                "vocal_analysis": vocal_debug,
            },
        )

    cands_a = [
        _make_candidate(
            t,
            idx,
            profiles_a,
            incoming=False,
            seam_sec=float(seam_sec),
            vocal_profile=vocal_profile_a,
            vocal_time_sec=float(a_analysis_start_sec + t),
        )
        for (t, idx) in raw_a
    ]
    cands_b = [
        _make_candidate(
            t,
            idx,
            profiles_b,
            incoming=True,
            seam_sec=float(seam_sec),
            vocal_profile=vocal_profile_b,
            vocal_time_sec=float(t),
        )
        for (t, idx) in raw_b
    ]

    scored_pairs: List[Dict[str, object]] = []
    best: Optional[Dict[str, object]] = None
    target_b = max(2.0, min(8.0, float(analysis_sec * 0.25)))

    for cand_a in cands_a:
        for cand_b in cands_b:
            total, comps = _score_pair(cand_a, cand_b, target_a=target_a_rel, target_b=target_b)
            item = {
                "score": float(total),
                "song_a_rel_sec": float(cand_a.time_sec),
                "song_b_rel_sec": float(cand_b.time_sec),
                "song_a_vocal_ratio": float(cand_a.vocal_ratio),
                "song_b_vocal_ratio": float(cand_b.vocal_ratio),
                "song_a_vocal_onset": float(cand_a.vocal_onset),
                "song_b_vocal_onset": float(cand_b.vocal_onset),
                "song_a_vocal_phrase": float(cand_a.vocal_phrase_score),
                "song_b_vocal_phrase": float(cand_b.vocal_phrase_score),
                "song_a_drum_anchor": float(cand_a.drum_anchor),
                "song_b_drum_anchor": float(cand_b.drum_anchor),
                "song_a_bass_energy": float(cand_a.bass_energy),
                "song_b_bass_energy": float(cand_b.bass_energy),
                "song_a_bass_stability": float(cand_a.bass_stability),
                "song_b_bass_stability": float(cand_b.bass_stability),
                "song_a_density": float(cand_a.instrumental_density),
                "song_b_density": float(cand_b.instrumental_density),
                "song_a_density_score": float(cand_a.density_score),
                "song_b_density_score": float(cand_b.density_score),
                "song_a_period_vocal_phrase": float(cand_a.period_vocal_phrase_score),
                "song_b_period_vocal_phrase": float(cand_b.period_vocal_phrase_score),
                "song_a_period_drum_anchor": float(cand_a.period_drum_anchor),
                "song_b_period_drum_anchor": float(cand_b.period_drum_anchor),
                "song_a_period_bass_energy": float(cand_a.period_bass_energy),
                "song_b_period_bass_energy": float(cand_b.period_bass_energy),
                "song_a_period_bass_stability": float(cand_a.period_bass_stability),
                "song_b_period_bass_stability": float(cand_b.period_bass_stability),
                "song_a_period_density": float(cand_a.period_density_score),
                "song_b_period_density": float(cand_b.period_density_score),
                "song_a_period_coverage": float(cand_a.period_coverage),
                "song_b_period_coverage": float(cand_b.period_coverage),
                "components": comps,
            }
            scored_pairs.append(item)
            if best is None or float(total) > float(best["score"]):
                best = item

    if best is None:
        return CueSelectionResult(
            cue_a_sec=default_a_abs,
            cue_b_sec=default_b_abs,
            method="score-fallback",
            debug={"reason": "no_scored_pairs", "vocal_analysis": vocal_debug},
        )

    scored_pairs = sorted(scored_pairs, key=lambda x: float(x["score"]), reverse=True)
    top_pairs = [
        {
            "score": round(float(item["score"]), 4),
            "song_a_rel_sec": round(float(item["song_a_rel_sec"]), 3),
            "song_b_rel_sec": round(float(item["song_b_rel_sec"]), 3),
            "song_a_vocal_ratio": round(float(item["song_a_vocal_ratio"]), 4),
            "song_b_vocal_ratio": round(float(item["song_b_vocal_ratio"]), 4),
            "song_a_vocal_phrase": round(float(item["song_a_vocal_phrase"]), 4),
            "song_b_vocal_phrase": round(float(item["song_b_vocal_phrase"]), 4),
            "song_a_drum_anchor": round(float(item["song_a_drum_anchor"]), 4),
            "song_b_drum_anchor": round(float(item["song_b_drum_anchor"]), 4),
            "song_a_bass_stability": round(float(item["song_a_bass_stability"]), 4),
            "song_b_bass_stability": round(float(item["song_b_bass_stability"]), 4),
            "song_a_density_score": round(float(item["song_a_density_score"]), 4),
            "song_b_density_score": round(float(item["song_b_density_score"]), 4),
            "song_a_period_vocal_phrase": round(float(item["song_a_period_vocal_phrase"]), 4),
            "song_b_period_vocal_phrase": round(float(item["song_b_period_vocal_phrase"]), 4),
            "song_a_period_drum_anchor": round(float(item["song_a_period_drum_anchor"]), 4),
            "song_b_period_drum_anchor": round(float(item["song_b_period_drum_anchor"]), 4),
            "song_a_period_bass_stability": round(float(item["song_a_period_bass_stability"]), 4),
            "song_b_period_bass_stability": round(float(item["song_b_period_bass_stability"]), 4),
            "song_a_period_density": round(float(item["song_a_period_density"]), 4),
            "song_b_period_density": round(float(item["song_b_period_density"]), 4),
            "song_a_period_coverage": round(float(item["song_a_period_coverage"]), 4),
            "song_b_period_coverage": round(float(item["song_b_period_coverage"]), 4),
            "components": {k: round(float(v), 4) for k, v in item["components"].items()},
        }
        for item in scored_pairs[:3]
    ]

    cue_a_abs = float(a_analysis_start_sec + float(best["song_a_rel_sec"]))
    cue_b_abs = float(best["song_b_rel_sec"])
    return CueSelectionResult(
        cue_a_sec=cue_a_abs,
        cue_b_sec=cue_b_abs,
        method="scored-auto",
        debug={
            "manual_override": False,
            "beat_counts": {"song_a": int(beats_a.size), "song_b": int(beats_b.size)},
            "candidate_counts": {"song_a": len(cands_a), "song_b": len(cands_b)},
            "candidate_windows_sec": {
                "song_a": [round(min_a, 3), round(max_a, 3)],
                "song_b": [round(min_b, 3), round(max_b, 3)],
            },
            "transition_period_sec": round(float(seam_sec), 3),
            "selected_rel_sec": {
                "song_a": round(float(best["song_a_rel_sec"]), 3),
                "song_b": round(float(best["song_b_rel_sec"]), 3),
            },
            "selected_mixability": {
                "song_a_ratio": round(float(best["song_a_vocal_ratio"]), 4),
                "song_b_ratio": round(float(best["song_b_vocal_ratio"]), 4),
                "song_a_vocal_onset": round(float(best["song_a_vocal_onset"]), 4),
                "song_b_vocal_onset": round(float(best["song_b_vocal_onset"]), 4),
                "song_a_vocal_phrase": round(float(best["song_a_vocal_phrase"]), 4),
                "song_b_vocal_phrase": round(float(best["song_b_vocal_phrase"]), 4),
                "song_a_drum_anchor": round(float(best["song_a_drum_anchor"]), 4),
                "song_b_drum_anchor": round(float(best["song_b_drum_anchor"]), 4),
                "song_a_bass_energy": round(float(best["song_a_bass_energy"]), 4),
                "song_b_bass_energy": round(float(best["song_b_bass_energy"]), 4),
                "song_a_bass_stability": round(float(best["song_a_bass_stability"]), 4),
                "song_b_bass_stability": round(float(best["song_b_bass_stability"]), 4),
                "song_a_density": round(float(best["song_a_density"]), 4),
                "song_b_density": round(float(best["song_b_density"]), 4),
                "song_a_density_score": round(float(best["song_a_density_score"]), 4),
                "song_b_density_score": round(float(best["song_b_density_score"]), 4),
                "song_a_period_vocal_phrase": round(float(best["song_a_period_vocal_phrase"]), 4),
                "song_b_period_vocal_phrase": round(float(best["song_b_period_vocal_phrase"]), 4),
                "song_a_period_drum_anchor": round(float(best["song_a_period_drum_anchor"]), 4),
                "song_b_period_drum_anchor": round(float(best["song_b_period_drum_anchor"]), 4),
                "song_a_period_bass_energy": round(float(best["song_a_period_bass_energy"]), 4),
                "song_b_period_bass_energy": round(float(best["song_b_period_bass_energy"]), 4),
                "song_a_period_bass_stability": round(float(best["song_a_period_bass_stability"]), 4),
                "song_b_period_bass_stability": round(float(best["song_b_period_bass_stability"]), 4),
                "song_a_period_density": round(float(best["song_a_period_density"]), 4),
                "song_b_period_density": round(float(best["song_b_period_density"]), 4),
                "song_a_period_coverage": round(float(best["song_a_period_coverage"]), 4),
                "song_b_period_coverage": round(float(best["song_b_period_coverage"]), 4),
            },
            "default_auto_cues_sec": {"song_a": round(default_a_abs, 3), "song_b": round(default_b_abs, 3)},
            "vocal_analysis": vocal_debug,
            "vocal_penalty_active": bool(vocal_profile_a is not None or vocal_profile_b is not None),
            "top_pairs": top_pairs,
            "period_scoring": {
                "enabled": True,
                "window_def": {"song_a": "[cue-seam, cue]", "song_b": "[cue, cue+seam]"},
                "overlap_simulation": "weighted vocal/bass clash precheck",
            },
        },
    )
