import argparse
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa  # type: ignore[reportMissingImports]
import numpy as np

from .audio_utils import (
    apply_edge_fades,
    clamp,
    crossfade_equal_length,
    decode_segment,
    ensure_length,
    estimate_bpm_and_beats,
    ffprobe_duration_sec,
    normalize_peak,
    resample_if_needed,
    safe_time_stretch,
    write_wav,
)
from .cuepoint_selector import select_mix_cuepoints

LOGGER = logging.getLogger(__name__)

DEFAULT_TARGET_SR = 32000
ACESTEP_INPUT_SR = 48000
STITCH_PREVIEW_SIDE_SEC = 10.0

PLUGIN_PRESETS: Dict[str, str] = {
    "Smooth Blend": "smooth seamless DJ transition, balanced energy, clean, no vocals",
    "EDM Build-up": "energetic EDM build-up transition with rising tension, clean, no vocals",
    "Percussive Bridge": "percussive bridge transition with rhythmic drums and clear groove, no vocals",
    "Ambient Wash": "ambient wash transition, spacious and atmospheric, soft energy curve, no vocals",
}

_ACESTEP_RUNTIME: Optional[Dict[str, Any]] = None
_DEMUCS_RUNTIME: Optional[Dict[str, Any]] = None
_DEMUCS_TRANSITION_ENABLED = os.getenv("AI_DJ_ENABLE_DEMUCS_TRANSITION", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_DEMUCS_MODEL_NAME = os.getenv("AI_DJ_DEMUCS_MODEL", "htdemucs").strip() or "htdemucs"
_DEMUCS_DEVICE_PREF = os.getenv("AI_DJ_DEMUCS_DEVICE", "cuda").strip().lower()
_DEMUCS_SEGMENT_SEC = 7.0
_REF_AUDIO_MODE = (os.getenv("AI_DJ_REFERENCE_AUDIO_MODE", "accompaniment-only") or "accompaniment-only").strip().lower()


@dataclass
class _DemucsStemBundle:
    vocals: np.ndarray
    drums: np.ndarray
    bass: np.ndarray
    other: np.ndarray
    accompaniment: np.ndarray
    sr: int
    method: str


@dataclass
class TransitionRequest:
    song_a_path: str
    song_b_path: str
    plugin_id: str = "Smooth Blend"
    instruction_text: str = ""
    pre_context_sec: float = 6.0
    repaint_width_sec: float = 4.0
    post_context_sec: float = 6.0
    analysis_sec: float = 45.0
    bpm_target: Optional[float] = None
    cue_a_sec: Optional[float] = None
    cue_b_sec: Optional[float] = None
    transition_base_mode: str = "B-base-fixed"
    transition_bars: int = 8
    creativity_strength: float = 7.0
    inference_steps: int = 8
    seed: int = 42
    output_dir: str = "outputs"
    output_stem: Optional[str] = None
    target_sr: int = DEFAULT_TARGET_SR
    keep_debug_files: bool = False

    # ACE-Step runtime config
    acestep_model_config: str = os.getenv("AI_DJ_ACESTEP_MODEL_CONFIG", "acestep-v15-turbo").strip()
    acestep_device: str = os.getenv("AI_DJ_ACESTEP_DEVICE", "auto").strip()
    acestep_project_root: str = os.getenv("AI_DJ_ACESTEP_PROJECT_ROOT", "").strip()
    acestep_prefer_source: Optional[str] = os.getenv("AI_DJ_ACESTEP_PREFER_SOURCE", "").strip() or None
    acestep_use_flash_attn: bool = False
    acestep_compile_model: bool = False
    acestep_offload_to_cpu: bool = False
    acestep_offload_dit_to_cpu: bool = False
    acestep_use_mlx_dit: bool = True

    def to_log_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TransitionResult:
    transition_path: str
    stitched_path: str
    rough_stitched_path: str
    hard_splice_path: str
    backend_used: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


def _slug(text: str) -> str:
    s = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text.strip())
    s = "_".join(part for part in s.split("_") if part)
    return s[:80] or "item"


def _deterministic_stem(request: TransitionRequest) -> str:
    if request.output_stem:
        return _slug(request.output_stem)

    payload = {
        "a": os.path.basename(request.song_a_path),
        "b": os.path.basename(request.song_b_path),
        "plugin": request.plugin_id,
        "instruction_text": request.instruction_text,
        "pre_context_sec": request.pre_context_sec,
        "repaint_width_sec": request.repaint_width_sec,
        "post_context_sec": request.post_context_sec,
        "analysis_sec": request.analysis_sec,
        "bpm_target": request.bpm_target,
        "cue_a_sec": request.cue_a_sec,
        "cue_b_sec": request.cue_b_sec,
        "transition_base_mode": request.transition_base_mode,
        "transition_bars": request.transition_bars,
        "creativity_strength": request.creativity_strength,
        "inference_steps": request.inference_steps,
        "seed": request.seed,
        "target_sr": request.target_sr,
        "acestep_model_config": request.acestep_model_config,
        "demucs_transition_enabled": _DEMUCS_TRANSITION_ENABLED,
        "demucs_model": _DEMUCS_MODEL_NAME,
        "reference_audio_mode": _REF_AUDIO_MODE,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:10]
    return f"transition_{_slug(Path(request.song_a_path).stem)}_to_{_slug(Path(request.song_b_path).stem)}_{digest}"


def _resolve_output_paths(request: TransitionRequest) -> Tuple[str, str, str, str, str]:
    os.makedirs(request.output_dir, exist_ok=True)
    stem = _deterministic_stem(request)
    transition_path = os.path.join(request.output_dir, f"{stem}_transition.wav")
    stitched_path = os.path.join(request.output_dir, f"{stem}_stitched.wav")
    rough_stitched_path = os.path.join(request.output_dir, f"{stem}_rough_stitched.wav")
    hard_splice_path = os.path.join(request.output_dir, f"{stem}_hard_splice.wav")
    rough_src_path = os.path.join(request.output_dir, f"{stem}_rough_src.wav")
    return transition_path, stitched_path, rough_stitched_path, hard_splice_path, rough_src_path


def _resolve_acestep_project_root(request: TransitionRequest) -> str:
    if request.acestep_project_root:
        os.makedirs(request.acestep_project_root, exist_ok=True)
        return request.acestep_project_root

    hf_data = "/data"
    if os.path.isdir(hf_data) and os.access(hf_data, os.W_OK):
        root = os.path.join(hf_data, "acestep_runtime")
        os.makedirs(root, exist_ok=True)
        return root

    root = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".acestep_runtime")
    os.makedirs(root, exist_ok=True)
    return root


def _build_caption(plugin_id: str, instruction_text: str) -> str:
    base = PLUGIN_PRESETS.get(plugin_id, PLUGIN_PRESETS["Smooth Blend"])
    extra = (instruction_text or "").strip()
    if not extra:
        return base
    return f"{base}. Additional instruction: {extra}"


def _resolve_half_double_tempo(bpm_ref: float, bpm_candidate: float) -> float:
    candidates = [0.5 * bpm_candidate, bpm_candidate, 2.0 * bpm_candidate]
    valid = [v for v in candidates if 40.0 <= float(v) <= 240.0]
    if not valid:
        return float(bpm_candidate)
    return float(min(valid, key=lambda x: abs(np.log2(max(1e-6, bpm_ref) / max(1e-6, x)))))


def _normalized_onset_envelope(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    if y.size <= 0:
        return np.zeros((1,), dtype=np.float32)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length).astype(np.float32)
    if onset.size == 0:
        return np.zeros((1,), dtype=np.float32)
    onset = onset - float(np.mean(onset))
    maximum = float(np.max(np.abs(onset)))
    if maximum > 1e-9:
        onset = onset / maximum
    return onset.astype(np.float32)


def _corr_similarity(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n <= 3:
        return 0.0
    a2 = a[:n].astype(np.float32)
    b2 = b[:n].astype(np.float32)
    denom = float(np.linalg.norm(a2) * np.linalg.norm(b2))
    if denom <= 1e-9:
        return 0.0
    raw = float(np.dot(a2, b2) / denom)
    return clamp((raw + 1.0) * 0.5, 0.0, 1.0)


def _rms(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))


def _resolve_demucs_device(torch_mod: Any) -> str:
    pref = (_DEMUCS_DEVICE_PREF or "").strip().lower()
    if pref == "cpu":
        return "cpu"
    if pref in {"cuda", "gpu"}:
        return "cuda" if bool(torch_mod.cuda.is_available()) else "cpu"
    return "cuda" if bool(torch_mod.cuda.is_available()) else "cpu"


def _load_demucs_runtime() -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    global _DEMUCS_RUNTIME
    if not _DEMUCS_TRANSITION_ENABLED:
        return None, {"enabled": False, "status": "disabled", "reason": "AI_DJ_ENABLE_DEMUCS_TRANSITION=0"}
    if _DEMUCS_RUNTIME is not None:
        return _DEMUCS_RUNTIME, {
            "enabled": True,
            "status": "ready",
            "model": _DEMUCS_RUNTIME.get("model_name"),
            "device": _DEMUCS_RUNTIME.get("device"),
        }

    try:
        import torch  # type: ignore[reportMissingImports]
        from demucs.pretrained import get_model  # type: ignore[reportMissingImports]

        model = get_model(_DEMUCS_MODEL_NAME)
        model.eval()
        device = _resolve_demucs_device(torch)
        model.to(device)
        _DEMUCS_RUNTIME = {
            "model": model,
            "torch": torch,
            "device": device,
            "model_name": _DEMUCS_MODEL_NAME,
        }
        return _DEMUCS_RUNTIME, {
            "enabled": True,
            "status": "ready",
            "model": _DEMUCS_MODEL_NAME,
            "device": device,
        }
    except Exception as exc:
        LOGGER.warning("Demucs transition runtime unavailable (%s). Falling back to non-stem transition path.", exc)
        return None, {
            "enabled": True,
            "status": "unavailable",
            "model": _DEMUCS_MODEL_NAME,
            "reason": str(exc),
        }


def _resample_to(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if int(orig_sr) == int(target_sr):
        return y.astype(np.float32)
    if y.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return librosa.resample(y.astype(np.float32), orig_sr=int(orig_sr), target_sr=int(target_sr)).astype(np.float32)


def _extract_demucs_stems(y: np.ndarray, sr: int, track_label: str) -> Tuple[Optional[_DemucsStemBundle], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "enabled": bool(_DEMUCS_TRANSITION_ENABLED),
        "track": track_label,
        "model": _DEMUCS_MODEL_NAME,
    }
    if y.size < int(max(1, sr) * 2.0):
        info["status"] = "skipped-short-audio"
        return None, info

    runtime, runtime_debug = _load_demucs_runtime()
    info.update(runtime_debug)
    if runtime is None:
        return None, info

    try:
        from demucs.apply import apply_model  # type: ignore[reportMissingImports]

        torch_mod = runtime["torch"]
        model = runtime["model"]
        device = str(runtime.get("device", "cpu"))

        mono = np.asarray(y, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            info["status"] = "empty"
            return None, info
        peak = float(np.max(np.abs(mono)))
        if peak > 1e-9:
            mono = mono / peak

        demucs_sr = int(getattr(model, "samplerate", 44100))
        work = _resample_to(mono, int(sr), demucs_sr)
        if work.size < int(max(1, demucs_sr) * 2.0):
            info["status"] = "skipped-short-audio"
            return None, info

        stereo = np.stack([work, work], axis=0)
        mix = torch_mod.from_numpy(stereo).unsqueeze(0).to(device)
        audio_sec = float(work.size / max(1, demucs_sr))
        use_split = audio_sec > (_DEMUCS_SEGMENT_SEC + 0.05)
        segment_sec = float(_DEMUCS_SEGMENT_SEC) if use_split else None

        try:
            with torch_mod.no_grad():
                estimates = apply_model(
                    model,
                    mix,
                    shifts=1,
                    split=use_split,
                    overlap=0.25,
                    progress=False,
                    device=device,
                    segment=segment_sec,
                )
        except Exception as exc:
            if device == "cuda":
                model.to("cpu")
                runtime["device"] = "cpu"
                device = "cpu"
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

        est = estimates.detach().cpu()
        est = est[0] if est.ndim == 4 else est
        if est.ndim != 3:
            raise RuntimeError(f"Unexpected demucs output shape: {tuple(est.shape)}")

        source_names = [str(s) for s in getattr(model, "sources", [])]
        if not source_names:
            raise RuntimeError("Demucs returned no source names.")
        if est.shape[0] != len(source_names):
            if est.shape[1] == len(source_names):
                est = est.permute(1, 0, 2)
            else:
                raise RuntimeError(f"Demucs source mismatch: shape {tuple(est.shape)}, sources {source_names}")

        def _stem(name: str) -> np.ndarray:
            if name in source_names:
                stem = est[source_names.index(name)].mean(dim=0).numpy().astype(np.float32)
                return _resample_to(stem, demucs_sr, int(sr))
            return np.zeros((mono.size,), dtype=np.float32)

        vocals = _stem("vocals")
        drums = _stem("drums")
        bass = _stem("bass")
        other = _stem("other")
        non_vocal_idxs = [i for i, s in enumerate(source_names) if s != "vocals"]
        if non_vocal_idxs:
            acc = est[non_vocal_idxs].sum(dim=0).mean(dim=0).numpy().astype(np.float32)
            accompaniment = _resample_to(acc, demucs_sr, int(sr))
        else:
            accompaniment = np.zeros((mono.size,), dtype=np.float32)

        target_n = int(mono.size)
        vocals = ensure_length(vocals, target_n)
        drums = ensure_length(drums, target_n)
        bass = ensure_length(bass, target_n)
        other = ensure_length(other, target_n)
        accompaniment = ensure_length(accompaniment, target_n)

        info.update(
            {
                "status": "ready",
                "method": "demucs-transition-stems",
                "split_mode": "chunked" if use_split else "full-window",
                "duration_sec": round(float(target_n / max(1, sr)), 3),
                "has_drums": bool("drums" in source_names),
                "has_bass": bool("bass" in source_names),
                "has_other": bool("other" in source_names),
                "device": runtime.get("device", device),
            }
        )
        return _DemucsStemBundle(
            vocals=vocals.astype(np.float32),
            drums=drums.astype(np.float32),
            bass=bass.astype(np.float32),
            other=other.astype(np.float32),
            accompaniment=accompaniment.astype(np.float32),
            sr=int(sr),
            method="demucs-transition-stems",
        ), info
    except Exception as exc:
        LOGGER.warning("Demucs stem extraction failed for %s (%s).", track_label, exc)
        info["status"] = "error"
        info["reason"] = str(exc)
        return None, info


def _slice_stem_bundle(bundle: Optional[_DemucsStemBundle], start_n: int, length_n: int) -> Optional[_DemucsStemBundle]:
    if bundle is None:
        return None
    s = int(max(0, start_n))
    n = int(max(0, length_n))
    e = s + n
    return _DemucsStemBundle(
        vocals=ensure_length(bundle.vocals[s:e], n),
        drums=ensure_length(bundle.drums[s:e], n),
        bass=ensure_length(bundle.bass[s:e], n),
        other=ensure_length(bundle.other[s:e], n),
        accompaniment=ensure_length(bundle.accompaniment[s:e], n),
        sr=int(bundle.sr),
        method=bundle.method,
    )


def _seconds_to_beats(seconds: float, bpm: float) -> float:
    return float(seconds) * (float(bpm) / 60.0)


def _beats_to_seconds(beats: float, bpm: float) -> float:
    return float(beats) * (60.0 / max(1e-6, float(bpm)))


def _quantize_seconds_to_beats(
    raw_sec: float,
    bpm: float,
    min_sec: float,
    max_sec: float,
    beat_step: int,
    min_beats: int,
) -> Tuple[float, int, float]:
    raw_sec = float(clamp(raw_sec, min_sec, max_sec))
    if bpm <= 1e-6:
        return raw_sec, int(round(_seconds_to_beats(raw_sec, 120.0))), _seconds_to_beats(raw_sec, 120.0)

    raw_beats = _seconds_to_beats(raw_sec, bpm)
    step = max(1, int(beat_step))
    min_beats_i = max(1, int(min_beats))
    max_allowed_beats = _seconds_to_beats(max_sec, bpm)
    max_beats_i = int(max(min_beats_i, np.floor(max_allowed_beats / step) * step))
    quant_beats = int(round(raw_beats / step) * step)
    quant_beats = int(clamp(float(quant_beats), float(min_beats_i), float(max_beats_i)))
    quant_sec = float(clamp(_beats_to_seconds(quant_beats, bpm), min_sec, max_sec))
    return quant_sec, quant_beats, raw_beats


def _phrase_lock_transition_shape(pre_sec: float, seam_sec: float, post_sec: float, bpm: float) -> Dict[str, Any]:
    pre_locked_sec, pre_beats, pre_raw_beats = _quantize_seconds_to_beats(
        raw_sec=pre_sec,
        bpm=bpm,
        min_sec=1.0,
        max_sec=20.0,
        beat_step=4,
        min_beats=2,
    )

    seam_raw_beats = _seconds_to_beats(seam_sec, bpm)
    seam_step = 8 if seam_raw_beats >= 8.0 else 4
    seam_locked_sec, seam_beats, _ = _quantize_seconds_to_beats(
        raw_sec=seam_sec,
        bpm=bpm,
        min_sec=1.0,
        max_sec=40.0,
        beat_step=seam_step,
        min_beats=2,
    )

    post_locked_sec, post_beats, post_raw_beats = _quantize_seconds_to_beats(
        raw_sec=post_sec,
        bpm=bpm,
        min_sec=1.0,
        max_sec=20.0,
        beat_step=4,
        min_beats=2,
    )

    return {
        "pre_sec": pre_locked_sec,
        "seam_sec": seam_locked_sec,
        "post_sec": post_locked_sec,
        "debug": {
            "bpm_ref": round(float(bpm), 3),
            "pre": {
                "raw_sec": round(float(pre_sec), 3),
                "locked_sec": round(float(pre_locked_sec), 3),
                "raw_beats": round(float(pre_raw_beats), 3),
                "locked_beats": int(pre_beats),
                "beat_step": 4,
            },
            "seam": {
                "raw_sec": round(float(seam_sec), 3),
                "locked_sec": round(float(seam_locked_sec), 3),
                "raw_beats": round(float(seam_raw_beats), 3),
                "locked_beats": int(seam_beats),
                "beat_step": int(seam_step),
            },
            "post": {
                "raw_sec": round(float(post_sec), 3),
                "locked_sec": round(float(post_locked_sec), 3),
                "raw_beats": round(float(post_raw_beats), 3),
                "locked_beats": int(post_beats),
                "beat_step": 4,
            },
        },
    }


def _stft_band_split(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(y.size)
    if n <= 0:
        z = np.zeros((0,), dtype=np.float32)
        return z, z, z

    n_fft = 2048 if n >= 2048 else 1024
    hop = max(128, n_fft // 4)
    y2 = ensure_length(y.astype(np.float32), max(n, n_fft))

    D = librosa.stft(y2, n_fft=n_fft, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_mask = (freqs <= 180.0).astype(np.float32)[:, None]
    mid_mask = ((freqs > 180.0) & (freqs <= 2500.0)).astype(np.float32)[:, None]
    high_mask = (freqs > 2500.0).astype(np.float32)[:, None]

    low = librosa.istft(D * low_mask, hop_length=hop, length=y2.size).astype(np.float32)
    mid = librosa.istft(D * mid_mask, hop_length=hop, length=y2.size).astype(np.float32)
    high = librosa.istft(D * high_mask, hop_length=hop, length=y2.size).astype(np.float32)
    return low[:n], mid[:n], high[:n]


def _dj_style_seam_mix(a_tail: np.ndarray, b_head: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = min(int(a_tail.size), int(b_head.size))
    if n <= 0:
        return np.zeros((0,), dtype=np.float32), {"method": "empty-input-fallback"}

    a = a_tail[:n].astype(np.float32)
    b = b_head[:n].astype(np.float32)
    try:
        a_low, a_mid, a_high = _stft_band_split(a, sr=sr)
        b_low, b_mid, b_high = _stft_band_split(b, sr=sr)
    except Exception as exc:
        LOGGER.warning("Band-split seam mixing failed (%s); using equal crossfade.", exc)
        return crossfade_equal_length(a, b), {"method": "crossfade-fallback", "error": str(exc)}

    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    high_in = x
    mid_in = np.power(x, 1.15).astype(np.float32)
    # Delay low-end handoff so kick/bass do not collide early.
    low_in = np.clip((x - 0.58) / 0.30, 0.0, 1.0).astype(np.float32)

    seam = (
        (a_high * (1.0 - high_in))
        + (b_high * high_in)
        + (a_mid * (1.0 - mid_in))
        + (b_mid * mid_in)
        + (a_low * (1.0 - low_in))
        + (b_low * low_in)
    ).astype(np.float32)

    return seam, {
        "method": "dj-eq-bass-swap",
        "low_handoff": {"start_ratio": 0.58, "end_ratio": 0.88},
        "bands_hz": {"low_max": 180, "mid_max": 2500},
    }


def _build_theme_reference_audio(
    a_pre: np.ndarray,
    a_tail: np.ndarray,
    b_head: np.ndarray,
    b_post: np.ndarray,
    sr: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    a_ctx = np.concatenate([a_pre, a_tail]).astype(np.float32)
    b_ctx = np.concatenate([b_head, b_post]).astype(np.float32)

    a_take_n = min(a_ctx.size, int(round(12.0 * sr)))
    b_take_n = min(b_ctx.size, int(round(12.0 * sr)))
    if a_take_n <= 0 or b_take_n <= 0:
        return np.zeros((0,), dtype=np.float32), {"enabled": False, "reason": "insufficient_context"}

    a_seg = a_ctx[-a_take_n:]
    b_seg = b_ctx[:b_take_n]
    overlap_n = min(int(round(0.45 * sr)), a_seg.size // 4, b_seg.size // 4)
    if overlap_n > 0:
        seam = crossfade_equal_length(a_seg[-overlap_n:], b_seg[:overlap_n])
        ref = np.concatenate([a_seg[:-overlap_n], seam, b_seg[overlap_n:]]).astype(np.float32)
    else:
        ref = np.concatenate([a_seg, b_seg]).astype(np.float32)

    ref = normalize_peak(apply_edge_fades(ref, sr=sr, fade_ms=20.0), peak=0.98)
    return ref, {
        "enabled": True,
        "method": "a-tail-b-head-theme-ref",
        "duration_sec": round(float(ref.size / max(1, sr)), 3),
        "segments_sec": {
            "song_a": round(float(a_seg.size / max(1, sr)), 3),
            "song_b": round(float(b_seg.size / max(1, sr)), 3),
            "overlap": round(float(overlap_n / max(1, sr)), 3),
        },
    }


def _left_pad_to_length(y: np.ndarray, target_n: int) -> np.ndarray:
    target_n = int(max(0, target_n))
    if y.size >= target_n:
        return y[-target_n:].astype(np.float32)
    return np.pad(y.astype(np.float32), (target_n - y.size, 0), mode="constant")


def _crossfade_join(a: np.ndarray, b: np.ndarray, fade_n: int) -> np.ndarray:
    if a.size <= 0:
        return b.astype(np.float32)
    if b.size <= 0:
        return a.astype(np.float32)
    n = int(max(0, fade_n))
    n = min(n, int(a.size), int(b.size))
    if n <= 0:
        return np.concatenate([a, b]).astype(np.float32)
    seam = crossfade_equal_length(a[-n:], b[:n])
    return np.concatenate([a[:-n], seam, b[n:]]).astype(np.float32)


def _build_period_reference_audio(period: np.ndarray, sr: int, source_mode: str = "full-period-a") -> Tuple[np.ndarray, Dict[str, Any]]:
    if period.size <= 0:
        return np.zeros((0,), dtype=np.float32), {"enabled": False, "reason": "empty-reference-period"}
    ref = normalize_peak(apply_edge_fades(period.astype(np.float32), sr=sr, fade_ms=20.0), peak=0.98)
    return ref, {
        "enabled": True,
        "method": "opposite-transition-period-reference",
        "source_mode": str(source_mode),
        "duration_sec": round(float(ref.size / max(1, sr)), 3),
    }


def _apply_transition_low_duck(
    y: np.ndarray,
    sr: int,
    duck_floor: float = 0.14,
    fade_out_end: float = 0.42,
    fade_in_start: float = 0.72,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = int(y.size)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32), {"enabled": False, "reason": "empty-audio"}

    try:
        low, mid, high = _stft_band_split(y.astype(np.float32), sr=sr)
    except Exception as exc:
        LOGGER.warning("Low-duck split failed (%s); skip ducking.", exc)
        return y.astype(np.float32), {"enabled": False, "reason": "split-failed", "error": str(exc)}

    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out_end = float(clamp(fade_out_end, 0.1, 0.9))
    in_start = float(clamp(max(out_end + 0.05, fade_in_start), 0.15, 0.95))
    floor = float(clamp(duck_floor, 0.03, 0.5))

    low_gain = np.full((n,), floor, dtype=np.float32)
    entry_mask = x <= out_end
    if np.any(entry_mask):
        low_gain[entry_mask] = (1.0 - ((x[entry_mask] / max(1e-6, out_end)) * (1.0 - floor))).astype(np.float32)
    exit_mask = x >= in_start
    if np.any(exit_mask):
        ramp = (x[exit_mask] - in_start) / max(1e-6, (1.0 - in_start))
        low_gain[exit_mask] = (floor + (ramp * (1.0 - floor))).astype(np.float32)

    y_out = (low * low_gain) + mid + high
    y_out = y_out.astype(np.float32)
    return y_out, {
        "enabled": True,
        "method": "low-duck-center",
        "duck_floor": round(float(floor), 4),
        "fade_out_end_ratio": round(float(out_end), 4),
        "fade_in_start_ratio": round(float(in_start), 4),
    }


def _build_one_bassline_stem_period(
    period_a: np.ndarray,
    period_b: np.ndarray,
    stems_a: Optional[_DemucsStemBundle],
    stems_b: Optional[_DemucsStemBundle],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    if stems_a is None or stems_b is None:
        return None, {"enabled": False, "reason": "missing-stems"}
    n = min(
        int(period_a.size),
        int(period_b.size),
        int(stems_a.vocals.size),
        int(stems_b.vocals.size),
        int(stems_a.bass.size),
        int(stems_b.bass.size),
    )
    if n <= 0:
        return None, {"enabled": False, "reason": "empty-period"}

    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    bass_in = np.clip((x - 0.60) / 0.28, 0.0, 1.0).astype(np.float32)
    # Keep lows lighter in the center, then restore toward each edge.
    center_bass_shape = (0.35 + (0.65 * np.abs((2.0 * x) - 1.0))).astype(np.float32)

    bass_mix = ((stems_a.bass[:n] * (1.0 - bass_in)) + (stems_b.bass[:n] * bass_in)).astype(np.float32)
    bass_mix = (bass_mix * center_bass_shape).astype(np.float32)

    acc_a = (stems_a.accompaniment[:n] - stems_a.bass[:n]).astype(np.float32)
    acc_b = (stems_b.accompaniment[:n] - stems_b.bass[:n]).astype(np.float32)
    inst_mix = ((acc_a * (1.0 - x)) + (acc_b * x)).astype(np.float32)

    vocal_side = np.where(x < 0.5, stems_a.vocals[:n], stems_b.vocals[:n]).astype(np.float32)
    vocal_shape = np.where(
        x < 0.5,
        np.clip(1.0 - ((x / 0.5) * 0.75), 0.25, 1.0),
        np.clip(((x - 0.5) / 0.5) * 0.75 + 0.25, 0.25, 1.0),
    ).astype(np.float32)
    vocals_mix = (vocal_side * vocal_shape * 0.26).astype(np.float32)

    stem_mix = (inst_mix + bass_mix + vocals_mix).astype(np.float32)
    return stem_mix, {
        "enabled": True,
        "method": "demucs-one-bassline-rule",
        "bass_handoff": {"start_ratio": 0.60, "end_ratio": 0.88},
        "center_bass_floor": 0.35,
        "vocal_sidechain_gain": 0.26,
    }


def _build_src_transition_period(
    period_a: np.ndarray,
    period_b: np.ndarray,
    sr: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    return _build_src_transition_period_with_stems(period_a, period_b, sr=sr, stems_a=None, stems_b=None)


def _build_src_transition_period_with_stems(
    period_a: np.ndarray,
    period_b: np.ndarray,
    sr: int,
    stems_a: Optional[_DemucsStemBundle] = None,
    stems_b: Optional[_DemucsStemBundle] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    directional, directional_debug = _dj_style_seam_mix(period_a, period_b, sr=sr)
    n = int(min(period_a.size, period_b.size))
    if n > 0:
        x = np.linspace(0.0, 1.0, n, dtype=np.float32)
        guide = ((period_a[:n] * (1.0 - x)) + (period_b[:n] * x)).astype(np.float32)
        src_period = ((0.70 * directional[:n]) + (0.30 * guide)).astype(np.float32)
    else:
        src_period = directional.astype(np.float32)

    demucs_mix, demucs_mix_debug = _build_one_bassline_stem_period(
        period_a=period_a,
        period_b=period_b,
        stems_a=stems_a,
        stems_b=stems_b,
    )
    if demucs_mix is not None and demucs_mix.size > 0:
        src_period = ((0.54 * src_period[: demucs_mix.size]) + (0.46 * demucs_mix)).astype(np.float32)
        if src_period.size < n:
            src_period = ensure_length(src_period, n)

    use_acc_ref = _REF_AUDIO_MODE in {"accompaniment-only", "accompaniment", "inst-only", "instrumental-only"}
    if use_acc_ref and stems_a is not None and stems_a.accompaniment.size > 0:
        reference_period = ensure_length(stems_a.accompaniment.astype(np.float32), int(period_a.size))
        ref_mode = "accompaniment-only"
    else:
        reference_period = period_a.astype(np.float32)
        ref_mode = "full-period-a"
    dominant = "song_b"

    src_period, low_duck_debug = _apply_transition_low_duck(src_period, sr=sr)
    src_period = normalize_peak(src_period, peak=0.99)
    return src_period, reference_period, {
        "method": "bar-period-layered-repaint-src-fixed-b-base",
        "base_mode": "B-base-fixed",
        "dominant_period": dominant,
        "demucs_one_bassline": demucs_mix_debug,
        "reference_mode": ref_mode,
        "guide_mix": {
            "enabled": True,
            "weight_directional": 0.70,
            "weight_time_direction_guide": 0.30,
            "behavior": "more-song-a-detail-at-entry-more-song-b-at-exit",
        },
        "directional_mix": directional_debug,
        "transition_low_profile": low_duck_debug,
    }


def _crossfade_join_frequency_aware(a: np.ndarray, b: np.ndarray, fade_n: int, sr: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    if a.size <= 0:
        return b.astype(np.float32), {"method": "prepend-empty"}
    if b.size <= 0:
        return a.astype(np.float32), {"method": "append-empty"}

    n = int(max(0, fade_n))
    n = min(n, int(a.size), int(b.size))
    if n <= 0:
        return np.concatenate([a, b]).astype(np.float32), {"method": "no-fade"}

    seg_a = a[-n:].astype(np.float32)
    seg_b = b[:n].astype(np.float32)
    seam, seam_debug = _dj_style_seam_mix(seg_a, seg_b, sr=sr)
    out = np.concatenate([a[:-n], seam, b[n:]]).astype(np.float32)
    return out, {"method": "frequency-aware-join", "fade_samples": int(n), "seam": seam_debug}


def _post_repaint_stem_correction(
    transition: np.ndarray,
    sr: int,
    anchor_a: Optional[_DemucsStemBundle] = None,
    anchor_b: Optional[_DemucsStemBundle] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    y = transition.astype(np.float32)
    if y.size <= 0:
        return np.zeros((0,), dtype=np.float32), {"enabled": False, "reason": "empty-transition"}

    stems, demucs_debug = _extract_demucs_stems(y, int(sr), track_label="post-repaint-transition")
    if stems is None:
        return y, {"enabled": False, "reason": "demucs-unavailable", "demucs": demucs_debug}

    n = int(min(stems.vocals.size, stems.drums.size, stems.bass.size, stems.other.size, y.size))
    if n <= 0:
        return y, {"enabled": False, "reason": "empty-stems", "demucs": demucs_debug}

    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    center = np.clip(np.minimum(x, 1.0 - x) / 0.18, 0.0, 1.0).astype(np.float32)

    bass_cur = max(1e-5, _rms(stems.bass[:n]))
    bass_ref_a = _rms(anchor_a.bass) if anchor_a is not None else bass_cur
    bass_ref_b = _rms(anchor_b.bass) if anchor_b is not None else bass_cur
    bass_gain_a = float(clamp(bass_ref_a / bass_cur, 0.65, 1.15))
    bass_gain_b = float(clamp(bass_ref_b / bass_cur, 0.65, 1.15))
    bass_linear = ((1.0 - x) * bass_gain_a) + (x * bass_gain_b)
    bass_center_shape = (0.72 + (0.28 * np.abs((2.0 * x) - 1.0))).astype(np.float32)
    bass_gain = (bass_linear * bass_center_shape).astype(np.float32)

    vocal_cur = max(1e-5, _rms(stems.vocals[:n]))
    vocal_ref_a = _rms(anchor_a.vocals) if anchor_a is not None else vocal_cur
    vocal_ref_b = _rms(anchor_b.vocals) if anchor_b is not None else vocal_cur
    vocal_gain_a = float(clamp(vocal_ref_a / vocal_cur, 0.42, 1.0))
    vocal_gain_b = float(clamp(vocal_ref_b / vocal_cur, 0.42, 1.0))
    vocal_linear = ((1.0 - x) * vocal_gain_a) + (x * vocal_gain_b)
    vocal_boundary_shape = (0.72 + (0.28 * center)).astype(np.float32)
    vocal_gain = (vocal_linear * vocal_boundary_shape).astype(np.float32)

    drum_gain = (1.05 - (0.08 * center)).astype(np.float32)
    other_gain = 1.0

    corrected = (
        (stems.vocals[:n] * vocal_gain)
        + (stems.drums[:n] * drum_gain)
        + (stems.bass[:n] * bass_gain)
        + (stems.other[:n] * other_gain)
    ).astype(np.float32)
    corrected = ensure_length(corrected, int(y.size))
    return corrected, {
        "enabled": True,
        "method": "demucs-post-repaint-boundary-rebalance",
        "demucs": demucs_debug,
        "gains": {
            "bass_start": round(float(bass_gain_a), 4),
            "bass_end": round(float(bass_gain_b), 4),
            "vocal_start": round(float(vocal_gain_a), 4),
            "vocal_end": round(float(vocal_gain_b), 4),
            "drum_edge_boost": 1.05,
        },
        "anchor_rms": {
            "bass_a": round(float(bass_ref_a), 6),
            "bass_b": round(float(bass_ref_b), 6),
            "vocal_a": round(float(vocal_ref_a), 6),
            "vocal_b": round(float(vocal_ref_b), 6),
        },
    }


def _assemble_substitute_mix(
    song_a_prefix: np.ndarray,
    transition: np.ndarray,
    song_b_suffix: np.ndarray,
    boundary_fade_n: int = 0,
    sr: int = DEFAULT_TARGET_SR,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    a = song_a_prefix.astype(np.float32) if song_a_prefix.size > 0 else np.zeros((0,), dtype=np.float32)
    t = transition.astype(np.float32) if transition.size > 0 else np.zeros((0,), dtype=np.float32)
    b = song_b_suffix.astype(np.float32) if song_b_suffix.size > 0 else np.zeros((0,), dtype=np.float32)
    joined, entry_debug = _crossfade_join_frequency_aware(a, t, boundary_fade_n, sr=sr)
    joined, exit_debug = _crossfade_join_frequency_aware(joined, b, boundary_fade_n, sr=sr)
    return joined.astype(np.float32), {
        "method": "dual-frequency-aware-boundary-joins",
        "entry": entry_debug,
        "exit": exit_debug,
    }


def _align_b_window_to_a_tail(
    a_tail: np.ndarray,
    y_b_stretched: np.ndarray,
    nominal_start_n: int,
    seam_n: int,
    post_n: int,
    sr: int,
    bpm_ref: float,
    a_tail_drums: Optional[np.ndarray] = None,
    y_b_stretched_drums: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    total_n = seam_n + post_n
    if y_b_stretched.size < total_n:
        return ensure_length(y_b_stretched, total_n), 0, {
            "method": "short-buffer-fallback",
            "candidate_count": 0,
        }

    beat_sec = 60.0 / max(1e-6, float(bpm_ref))
    search_sec = clamp(0.75 * beat_sec, 0.2, 1.2)
    search_n = int(round(search_sec * sr))

    nominal_start_n = int(clamp(float(nominal_start_n), 0.0, float(max(0, y_b_stretched.size - total_n))))
    lo = max(0, nominal_start_n - search_n)
    hi = min(y_b_stretched.size - total_n, nominal_start_n + search_n)

    _, beat_times_stretched = estimate_bpm_and_beats(y_b_stretched, sr)
    candidates: List[int] = []
    for bt in beat_times_stretched:
        idx = int(round(float(bt) * sr))
        if lo <= idx <= hi:
            candidates.append(idx)
    candidates.append(nominal_start_n)
    candidates = sorted(set(candidates))
    if not candidates:
        candidates = [nominal_start_n]

    use_drum_alignment = (
        isinstance(a_tail_drums, np.ndarray)
        and isinstance(y_b_stretched_drums, np.ndarray)
        and int(a_tail_drums.size) >= int(seam_n)
        and int(y_b_stretched_drums.size) >= int(y_b_stretched.size)
    )

    onset_a_mix = _normalized_onset_envelope(a_tail, sr)
    onset_a_drum = _normalized_onset_envelope(a_tail_drums[:seam_n], sr) if use_drum_alignment else onset_a_mix
    rms_a = _rms(a_tail)
    drum_rms_a = _rms(a_tail_drums[:seam_n]) if use_drum_alignment else 0.0
    best_idx = candidates[0]
    best_score = -1.0
    best_components = {"onset_mix": 0.0, "onset_drum": 0.0, "energy": 0.0, "drum_energy": 0.0, "distance": 0.0}

    distance_scale = max(1.0, 0.65 * search_n)
    for idx in candidates:
        seg = ensure_length(y_b_stretched[idx : idx + total_n], total_n)
        b_head = seg[:seam_n]

        onset_b_mix = _normalized_onset_envelope(b_head, sr)
        onset_score_mix = _corr_similarity(onset_a_mix, onset_b_mix)
        onset_score_drum = onset_score_mix
        drum_energy_score = 0.5
        onset_score = onset_score_mix
        if use_drum_alignment:
            seg_drums = ensure_length(y_b_stretched_drums[idx : idx + total_n], total_n)
            b_head_drums = seg_drums[:seam_n]
            onset_b_drum = _normalized_onset_envelope(b_head_drums, sr)
            onset_score_drum = _corr_similarity(onset_a_drum, onset_b_drum)
            onset_score = (0.78 * onset_score_drum) + (0.22 * onset_score_mix)
            drum_rms_b = _rms(b_head_drums)
            drum_gap = abs(drum_rms_a - drum_rms_b) / max(1e-4, drum_rms_a)
            drum_energy_score = clamp(1.0 - drum_gap, 0.0, 1.0)

        rms_b = _rms(b_head)
        energy_gap = abs(rms_a - rms_b) / max(1e-4, rms_a)
        energy_score = clamp(1.0 - energy_gap, 0.0, 1.0)

        dist = abs(idx - nominal_start_n)
        distance_score = float(np.exp(-dist / distance_scale))

        if use_drum_alignment:
            score = (0.62 * onset_score) + (0.18 * energy_score) + (0.10 * drum_energy_score) + (0.10 * distance_score)
        else:
            score = (0.56 * onset_score) + (0.26 * energy_score) + (0.18 * distance_score)
        if score > best_score:
            best_score = float(score)
            best_idx = int(idx)
            best_components = {
                "onset_mix": float(onset_score_mix),
                "onset_drum": float(onset_score_drum),
                "energy": float(energy_score),
                "drum_energy": float(drum_energy_score),
                "distance": float(distance_score),
            }

    aligned = ensure_length(y_b_stretched[best_idx : best_idx + total_n], total_n)
    return aligned, best_idx, {
        "method": "drum-led-beat-phase-transient-align" if use_drum_alignment else "beat-phase-transient-align",
        "used_drum_stems": bool(use_drum_alignment),
        "candidate_count": len(candidates),
        "search_sec": round(float(search_sec), 4),
        "search_samples": int(search_n),
        "nominal_start_sample": int(nominal_start_n),
        "best_start_sample": int(best_idx),
        "best_score": round(float(best_score), 6),
        "score_components": {k: round(float(v), 6) for k, v in best_components.items()},
    }


def _prepare_rough_transition(request: TransitionRequest) -> Dict[str, Any]:
    pre_sec_raw = clamp(request.pre_context_sec, 1.0, 20.0)
    post_sec_raw = clamp(request.post_context_sec, 1.0, 20.0)
    analysis_sec = clamp(request.analysis_sec, 10.0, 120.0)

    target_sr = int(request.target_sr)

    dur_a = ffprobe_duration_sec(request.song_a_path)
    dur_b = ffprobe_duration_sec(request.song_b_path)

    a_analysis_start = max(0.0, float(dur_a) - analysis_sec) if dur_a is not None else 0.0

    y_a_an, sr_a = decode_segment(request.song_a_path, a_analysis_start, analysis_sec, sr=target_sr, max_decode_sec=analysis_sec)
    y_b_an, sr_b = decode_segment(request.song_b_path, 0.0, analysis_sec, sr=target_sr, max_decode_sec=analysis_sec)
    bpm_a, beats_a = estimate_bpm_and_beats(y_a_an, sr_a)
    bpm_b, beats_b = estimate_bpm_and_beats(y_b_an, sr_b)

    if request.bpm_target is not None and 40.0 <= float(request.bpm_target) <= 220.0:
        bpm_a = float(request.bpm_target)

    bpm_a = float(bpm_a) if bpm_a is not None else 120.0
    bpm_b_detected = float(bpm_b) if bpm_b is not None else 120.0
    bpm_b_for_alignment = _resolve_half_double_tempo(bpm_a, bpm_b_detected)
    bars_requested = int(request.transition_bars)
    valid_bars = {4, 8, 16}
    transition_bars = bars_requested if bars_requested in valid_bars else 8
    seam_sec_raw = float(_beats_to_seconds(float(transition_bars * 4), bpm_a))
    seam_sec_raw = float(clamp(seam_sec_raw, 1.0, 40.0))
    seam_sec_ui_raw = seam_sec_raw
    base_mode = "B-base-fixed"

    phrase_lock = _phrase_lock_transition_shape(
        pre_sec=pre_sec_raw,
        seam_sec=seam_sec_raw,
        post_sec=post_sec_raw,
        bpm=bpm_a,
    )
    pre_sec = float(phrase_lock["pre_sec"])
    seam_sec = float(phrase_lock["seam_sec"])
    post_sec = float(phrase_lock["post_sec"])

    cue_selection = select_mix_cuepoints(
        y_a_analysis=y_a_an,
        y_b_analysis=y_b_an,
        sr=target_sr,
        analysis_sec=analysis_sec,
        pre_sec=pre_sec,
        seam_sec=seam_sec,
        post_sec=post_sec,
        a_analysis_start_sec=a_analysis_start,
        beats_a=beats_a,
        beats_b=beats_b,
        cue_a_override_sec=request.cue_a_sec,
        cue_b_override_sec=request.cue_b_sec,
        song_a_path=request.song_a_path,
        song_b_path=request.song_b_path,
        song_a_duration_sec=dur_a,
        song_b_duration_sec=dur_b,
    )
    cue_a = float(cue_selection.cue_a_sec)
    cue_b = float(cue_selection.cue_b_sec)

    stretch_rate_raw = bpm_a / max(1e-6, bpm_b_for_alignment)
    # Keep stronger musical coherence while avoiding very audible stretch artifacts.
    stretch_rate = clamp(stretch_rate_raw, 0.7, 1.35)

    pre_n = int(round(pre_sec * target_sr))
    seam_n = int(round(seam_sec * target_sr))
    post_n = int(round(post_sec * target_sr))

    # Song A transition period: bars before cue A.
    a_period_start = max(0.0, cue_a - seam_sec)
    period_a, _ = decode_segment(
        request.song_a_path,
        a_period_start,
        seam_sec,
        sr=target_sr,
        max_decode_sec=seam_sec + 2.0,
    )
    period_a = ensure_length(period_a, seam_n)
    period_a_stems, period_a_stem_debug = _extract_demucs_stems(period_a, target_sr, track_label="song-a-transition-period")

    # Repaint pre-context leading into the transition period.
    a_pre_start = max(0.0, a_period_start - pre_sec)
    a_pre, _ = decode_segment(
        request.song_a_path,
        a_pre_start,
        pre_sec,
        sr=target_sr,
        max_decode_sec=pre_sec + 2.0,
    )
    a_pre = _left_pad_to_length(a_pre, pre_n)

    cue_b_selected = cue_b
    stitch_preview_side_sec = float(STITCH_PREVIEW_SIDE_SEC)
    boundary_fade_beats = 2.0
    boundary_fade_sec = clamp(_beats_to_seconds(boundary_fade_beats, bpm_a), 0.08, 1.2)
    boundary_fade_n = int(round(boundary_fade_sec * target_sr))
    stitch_decode_side_sec = stitch_preview_side_sec + boundary_fade_sec
    cue_a_for_stitch = float(max(0.0, cue_a - seam_sec))
    if dur_a is not None:
        cue_a_for_stitch = clamp(cue_a_for_stitch, 0.0, float(dur_a))
    song_a_preview_start = max(0.0, cue_a_for_stitch - stitch_decode_side_sec)
    song_a_preview_dur = max(0.0, cue_a_for_stitch - song_a_preview_start)
    song_a_prefix, _ = decode_segment(
        request.song_a_path,
        song_a_preview_start,
        song_a_preview_dur,
        sr=target_sr,
        max_decode_sec=max(20.0, song_a_preview_dur + 2.0),
    )

    # Song B window: decode with pre-roll so we can phase-align on stretched beat grid.
    align_preroll_sec = clamp(0.75 * (60.0 / max(1e-6, bpm_a)), 0.2, 1.2)
    decode_start_b = max(0.0, cue_b_selected - (align_preroll_sec * stretch_rate))
    if dur_b is not None:
        decode_start_b = clamp(decode_start_b, 0.0, float(dur_b))
    desired_b_out_sec = seam_sec + max(post_sec, stitch_decode_side_sec) + (2.0 * align_preroll_sec)
    if dur_b is not None:
        # Decode only enough of Song B for alignment + transition + preview tail.
        remaining_sec = max(0.0, float(dur_b) - decode_start_b)
        raw_b_in_sec = clamp(min(remaining_sec, desired_b_out_sec * stretch_rate), 1.0, 360.0)
    else:
        raw_b_in_sec = clamp(desired_b_out_sec * stretch_rate, 1.0, 360.0)
    y_b_raw, _ = decode_segment(
        request.song_b_path,
        decode_start_b,
        raw_b_in_sec,
        sr=target_sr,
        max_decode_sec=raw_b_in_sec + 2.0,
    )
    y_b_stretched = safe_time_stretch(y_b_raw, rate=stretch_rate)
    y_b_stretched_stems, y_b_stem_debug = _extract_demucs_stems(
        y_b_stretched,
        target_sr,
        track_label="song-b-stretched-window",
    )
    nominal_b_start_n = int(round(align_preroll_sec * target_sr))
    y_b, aligned_b_start_n, b_alignment_debug = _align_b_window_to_a_tail(
        a_tail=period_a,
        y_b_stretched=y_b_stretched,
        nominal_start_n=nominal_b_start_n,
        seam_n=seam_n,
        post_n=post_n,
        sr=target_sr,
        bpm_ref=bpm_a,
        a_tail_drums=period_a_stems.drums if period_a_stems is not None else None,
        y_b_stretched_drums=y_b_stretched_stems.drums if y_b_stretched_stems is not None else None,
    )
    cue_b = float(decode_start_b + ((aligned_b_start_n / float(target_sr)) * stretch_rate))
    period_b = y_b[:seam_n]
    period_b_stems = _slice_stem_bundle(y_b_stretched_stems, aligned_b_start_n, seam_n)
    b_post = y_b[seam_n : seam_n + post_n]
    stitch_decode_n = int(round(stitch_decode_side_sec * target_sr))
    b_suffix_substitute = y_b_stretched[(aligned_b_start_n + seam_n) : (aligned_b_start_n + seam_n + stitch_decode_n)].astype(
        np.float32
    )
    if b_suffix_substitute.size == 0:
        b_suffix_substitute = np.zeros((0,), dtype=np.float32)

    rough_seam, reference_period, rough_mix_debug = _build_src_transition_period_with_stems(
        period_a=period_a,
        period_b=period_b,
        sr=target_sr,
        stems_a=period_a_stems,
        stems_b=period_b_stems,
    )
    rough_stitched = np.concatenate([a_pre, rough_seam, b_post]).astype(np.float32)
    reference_audio_clip, reference_audio_debug = _build_period_reference_audio(
        reference_period,
        sr=target_sr,
        source_mode=str(rough_mix_debug.get("reference_mode", "full-period-a")),
    )
    return {
        "target_sr": target_sr,
        "dur_a": dur_a,
        "dur_b": dur_b,
        "analysis_start_a_sec": a_analysis_start,
        "bpm_a": bpm_a,
        "bpm_b": bpm_b_detected,
        "bpm_b_for_alignment": bpm_b_for_alignment,
        "cue_a_sec": cue_a,
        "cue_b_sec": cue_b,
        "cue_b_selected_sec": cue_b_selected,
        "cue_selector_method": cue_selection.method,
        "cue_selector_debug": cue_selection.debug,
        "stretch_rate": stretch_rate,
        "stretch_rate_raw": stretch_rate_raw,
        "transition_base_mode": base_mode,
        "transition_bars": int(transition_bars),
        "b_alignment_debug": b_alignment_debug,
        "phrase_lock_debug": phrase_lock["debug"],
        "rough_mix_debug": rough_mix_debug,
        "reference_audio_debug": reference_audio_debug,
        "demucs_transition_debug": {
            "enabled": bool(_DEMUCS_TRANSITION_ENABLED),
            "period_a": period_a_stem_debug,
            "b_window_stretched": y_b_stem_debug,
            "period_b_from_aligned_window": {
                "status": "ready" if period_b_stems is not None else "unavailable",
                "source": "slice(song-b-stretched-window, aligned_start, seam_n)",
                "aligned_start_sample": int(aligned_b_start_n),
                "seam_n": int(seam_n),
            },
        },
        "pre_sec": pre_sec,
        "seam_sec": seam_sec,
        "post_sec": post_sec,
        "pre_sec_raw": pre_sec_raw,
        "seam_sec_raw": seam_sec_raw,
        "seam_sec_ui_raw": seam_sec_ui_raw,
        "post_sec_raw": post_sec_raw,
        "pre_n": pre_n,
        "seam_n": seam_n,
        "post_n": post_n,
        "rough_seam": rough_seam,
        "rough_stitched": rough_stitched,
        "song_a_prefix": song_a_prefix,
        "song_b_suffix_substitute": b_suffix_substitute,
        "reference_audio_clip": reference_audio_clip,
        "period_a_stem_bundle": period_a_stems,
        "period_b_stem_bundle": period_b_stems,
        "boundary_fade_n": int(boundary_fade_n),
        "boundary_fade_sec": float(boundary_fade_sec),
        "stitch_preview_side_sec": float(stitch_preview_side_sec),
        "stitch_decode_side_sec": float(stitch_decode_side_sec),
        "stitching_debug": {
            "mode": "replace-seam-no-insert",
            "transition_base_mode": base_mode,
            "transition_bars": int(transition_bars),
            "song_a_prefix_sec": round(float(song_a_prefix.size / max(1, target_sr)), 3),
            "transition_sec": round(float(seam_sec), 3),
            "song_b_suffix_sec": round(float(b_suffix_substitute.size / max(1, target_sr)), 3),
            "decode_start_b_sec": round(float(decode_start_b), 3),
            "cue_a_cut_sec": round(float(cue_a_for_stitch), 3),
            "cue_b_continuation_sec": round(float(cue_b + seam_sec), 3),
            "replaced_window_sec": round(float(seam_sec), 3),
            "boundary_fade_sec": round(float(boundary_fade_sec), 3),
            "stitch_preview_side_sec": round(float(stitch_preview_side_sec), 3),
            "stitch_decode_side_sec": round(float(stitch_decode_side_sec), 3),
        },
    }


def _extract_success_and_audios(result: Any) -> Tuple[bool, list, Optional[str]]:
    if isinstance(result, dict):
        success = bool(result.get("success", False))
        audios = result.get("audios", [])
        error = result.get("error") or result.get("status_message")
        return success, audios, error
    success = bool(getattr(result, "success", False))
    audios = getattr(result, "audios", [])
    error = getattr(result, "error", None) or getattr(result, "status_message", None)
    return success, audios, error


def _load_acestep_runtime(request: TransitionRequest) -> Dict[str, Any]:
    global _ACESTEP_RUNTIME

    project_root = _resolve_acestep_project_root(request)
    runtime_key = (project_root, request.acestep_model_config, request.acestep_device)

    if _ACESTEP_RUNTIME is not None and _ACESTEP_RUNTIME.get("key") == runtime_key:
        return _ACESTEP_RUNTIME

    try:
        from acestep.handler import AceStepHandler
        from acestep.inference import GenerationConfig, GenerationParams, generate_music
    except Exception as exc:
        raise RuntimeError(
            "ACE-Step is not installed or import failed. "
            "Install with: pip install git+https://github.com/ACE-Step/ACE-Step-1.5.git"
        ) from exc

    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=project_root,
        config_path=request.acestep_model_config,
        device=request.acestep_device,
        use_flash_attention=request.acestep_use_flash_attn,
        compile_model=request.acestep_compile_model,
        offload_to_cpu=request.acestep_offload_to_cpu,
        offload_dit_to_cpu=request.acestep_offload_dit_to_cpu,
        quantization=None,
        prefer_source=request.acestep_prefer_source,
        use_mlx_dit=request.acestep_use_mlx_dit,
    )
    if not ok:
        raise RuntimeError(f"ACE-Step initialize_service failed: {status}")

    _ACESTEP_RUNTIME = {
        "key": runtime_key,
        "project_root": project_root,
        "handler": handler,
        "GenerationParams": GenerationParams,
        "GenerationConfig": GenerationConfig,
        "generate_music": generate_music,
    }
    return _ACESTEP_RUNTIME


def _run_acestep_repaint(
    request: TransitionRequest,
    rough: Dict[str, Any],
    rough_src_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    runtime = _load_acestep_runtime(request)
    handler = runtime["handler"]
    GenerationParams = runtime["GenerationParams"]
    GenerationConfig = runtime["GenerationConfig"]
    generate_music = runtime["generate_music"]

    caption = _build_caption(request.plugin_id, request.instruction_text)

    rough_stitched = rough["rough_stitched"]
    rough_for_model = resample_if_needed(rough_stitched, rough["target_sr"], ACESTEP_INPUT_SR)
    write_wav(rough_src_path, rough_for_model, ACESTEP_INPUT_SR)
    reference_audio_path: Optional[str] = None
    reference_audio_clip = rough.get("reference_audio_clip")
    if isinstance(reference_audio_clip, np.ndarray) and reference_audio_clip.size > 0:
        reference_audio_path = (
            rough_src_path.replace("_rough_src.wav", "_theme_ref.wav")
            if rough_src_path.endswith("_rough_src.wav")
            else f"{rough_src_path}.theme_ref.wav"
        )
        reference_for_model = resample_if_needed(reference_audio_clip, rough["target_sr"], ACESTEP_INPUT_SR)
        write_wav(reference_audio_path, reference_for_model, ACESTEP_INPUT_SR)

    repaint_start = float(rough["pre_sec"])
    repaint_end = float(rough["pre_sec"] + rough["seam_sec"])
    total_duration = float(rough["pre_sec"] + rough["seam_sec"] + rough["post_sec"])
    bpm_hint = int(round(rough["bpm_a"])) if 30 <= rough["bpm_a"] <= 300 else None

    params = GenerationParams(
        task_type="repaint",
        src_audio=rough_src_path,
        reference_audio=reference_audio_path,
        repainting_start=repaint_start,
        repainting_end=repaint_end,
        caption=caption,
        lyrics="[Instrumental]",
        instrumental=True,
        bpm=bpm_hint,
        duration=total_duration,
        inference_steps=int(max(1, request.inference_steps)),
        guidance_scale=float(request.creativity_strength),
        seed=int(request.seed),
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_language=False,
    )
    config = GenerationConfig(
        batch_size=1,
        use_random_seed=False,
        seeds=[int(request.seed)],
        audio_format="wav",
    )

    result = generate_music(
        dit_handler=handler,
        llm_handler=None,
        params=params,
        config=config,
        save_dir=None,
        progress=None,
    )
    success, audios, error = _extract_success_and_audios(result)
    if not success or not audios:
        raise RuntimeError(error or "ACE-Step repaint returned no audio.")

    audio_item = audios[0]
    audio_tensor = audio_item.get("tensor")
    if audio_tensor is None:
        raise RuntimeError("ACE-Step repaint output missing audio tensor.")

    try:
        import torch
        if isinstance(audio_tensor, torch.Tensor):
            y = audio_tensor.detach().float().cpu().numpy()
        else:
            y = np.asarray(audio_tensor, dtype=np.float32)
    except Exception:
        y = np.asarray(audio_tensor, dtype=np.float32)

    if y.ndim == 2:
        y = np.mean(y, axis=0)
    elif y.ndim > 2:
        y = y.reshape(-1)
    y = y.astype(np.float32)

    model_sr = int(audio_item.get("sample_rate", ACESTEP_INPUT_SR))
    y = resample_if_needed(y, model_sr, rough["target_sr"])

    total_n = rough["pre_n"] + rough["seam_n"] + rough["post_n"]
    y = ensure_length(y, total_n)
    stitched = y[:total_n]
    seam_start = rough["pre_n"]
    seam_end = seam_start + rough["seam_n"]
    transition = stitched[seam_start:seam_end]
    return transition, stitched


def generate_transition_artifacts(request: TransitionRequest) -> TransitionResult:
    if not os.path.isfile(request.song_a_path):
        raise FileNotFoundError(f"Song A not found: {request.song_a_path}")
    if not os.path.isfile(request.song_b_path):
        raise FileNotFoundError(f"Song B not found: {request.song_b_path}")

    transition_path, stitched_path, rough_stitched_path, hard_splice_path, rough_src_path = _resolve_output_paths(request)

    LOGGER.info("Transition request args: %s", json.dumps(request.to_log_dict(), sort_keys=True))
    rough = _prepare_rough_transition(request)
    rough_stitched_audio = normalize_peak(
        apply_edge_fades(rough["rough_stitched"].astype(np.float32), rough["target_sr"], fade_ms=25.0),
        peak=0.98,
    )
    write_wav(rough_stitched_path, rough_stitched_audio, rough["target_sr"])
    hard_splice_audio = np.concatenate([rough["song_a_prefix"], rough["song_b_suffix_substitute"]]).astype(np.float32)
    hard_splice_audio = normalize_peak(hard_splice_audio, peak=0.98)
    write_wav(hard_splice_path, hard_splice_audio, rough["target_sr"])

    transition_audio = rough["rough_seam"]
    repaint_context_audio = rough["rough_stitched"]
    try:
        transition_audio, repaint_context_audio = _run_acestep_repaint(request, rough, rough_src_path)
    except Exception as exc:
        raise RuntimeError(f"ACE-Step repaint failed. Please verify ACE-Step runtime and model setup. {exc}") from exc

    backend_used = "acestep-repaint"

    transition_audio, post_repaint_stem_debug = _post_repaint_stem_correction(
        transition_audio.astype(np.float32),
        sr=int(rough["target_sr"]),
        anchor_a=rough.get("period_a_stem_bundle"),
        anchor_b=rough.get("period_b_stem_bundle"),
    )

    transition_audio, transition_low_profile_debug = _apply_transition_low_duck(
        transition_audio.astype(np.float32),
        sr=int(rough["target_sr"]),
    )

    stitched_audio, boundary_mix_debug = _assemble_substitute_mix(
        song_a_prefix=rough["song_a_prefix"],
        transition=transition_audio,
        song_b_suffix=rough["song_b_suffix_substitute"],
        boundary_fade_n=int(rough.get("boundary_fade_n", 0)),
        sr=int(rough["target_sr"]),
    )

    transition_audio = normalize_peak(apply_edge_fades(transition_audio, rough["target_sr"], fade_ms=25.0), peak=0.98)
    stitched_audio = normalize_peak(apply_edge_fades(stitched_audio, rough["target_sr"], fade_ms=25.0), peak=0.98)

    write_wav(transition_path, transition_audio, rough["target_sr"])
    write_wav(stitched_path, stitched_audio, rough["target_sr"])

    theme_ref_path = (
        rough_src_path.replace("_rough_src.wav", "_theme_ref.wav")
        if rough_src_path.endswith("_rough_src.wav")
        else f"{rough_src_path}.theme_ref.wav"
    )
    if not request.keep_debug_files:
        for tmp_path in (rough_src_path, theme_ref_path):
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    details = {
        "backend_used": backend_used,
        "generation_args": request.to_log_dict(),
        "bpm": {
            "song_a": round(float(rough["bpm_a"]), 3),
            "song_b": round(float(rough["bpm_b"]), 3),
            "song_b_for_alignment": round(float(rough["bpm_b_for_alignment"]), 3),
            "stretch_rate": round(float(rough["stretch_rate"]), 5),
            "stretch_rate_raw": round(float(rough["stretch_rate_raw"]), 5),
            "bpm_target_override": request.bpm_target,
        },
        "cue_points_sec": {
            "song_a": round(float(rough["cue_a_sec"]), 3),
            "song_b": round(float(rough["cue_b_sec"]), 3),
            "song_b_selected": round(float(rough["cue_b_selected_sec"]), 3),
            "selector_method": rough.get("cue_selector_method"),
        },
        "cue_selector": rough.get("cue_selector_debug"),
        "bpm_phase_alignment": rough.get("b_alignment_debug"),
        "phrase_lock": rough.get("phrase_lock_debug"),
        "rough_mix": rough.get("rough_mix_debug"),
        "reference_audio": rough.get("reference_audio_debug"),
        "demucs_transition": rough.get("demucs_transition_debug"),
        "stitching": rough.get("stitching_debug"),
        "boundary_mix": boundary_mix_debug,
        "post_repaint_stem_correction": post_repaint_stem_debug,
        "transition_low_profile": transition_low_profile_debug,
        "transition_strategy": {
            "name": "bar-defined-dual-base-repaint",
            "base_mode": rough.get("transition_base_mode"),
            "transition_bars": rough.get("transition_bars"),
            "boundary_fade_sec": round(float(rough.get("boundary_fade_sec", 0.0)), 3),
        },
        "clip_shape_sec": {
            "pre_context_sec_raw": round(float(rough["pre_sec_raw"]), 3),
            "pre_context_sec": round(float(rough["pre_sec"]), 3),
            "repaint_width_sec_ui_raw": round(float(rough.get("seam_sec_ui_raw", rough["seam_sec_raw"])), 3),
            "repaint_width_sec_raw": round(float(rough["seam_sec_raw"]), 3),
            "repaint_width_sec": round(float(rough["seam_sec"]), 3),
            "post_context_sec_raw": round(float(rough["post_sec_raw"]), 3),
            "post_context_sec": round(float(rough["post_sec"]), 3),
            "analysis_sec": round(float(request.analysis_sec), 3),
        },
        "durations_sec": {
            "song_a_total": rough["dur_a"],
            "song_b_total": rough["dur_b"],
            "analysis_start_a_sec": round(float(rough["analysis_start_a_sec"]), 3),
            "repaint_context_preview": round(float(repaint_context_audio.size / max(1, rough["target_sr"])), 3),
            "stitched_output": round(float(stitched_audio.size / max(1, rough["target_sr"])), 3),
        },
        "outputs": {
            "transition_path": transition_path,
            "stitched_path": stitched_path,
            "rough_stitched_path": rough_stitched_path,
            "hard_splice_path": hard_splice_path,
        },
    }
    LOGGER.info("Transition result details: %s", json.dumps(details, sort_keys=True))

    return TransitionResult(
        transition_path=transition_path,
        stitched_path=stitched_path,
        rough_stitched_path=rough_stitched_path,
        hard_splice_path=hard_splice_path,
        backend_used=backend_used,
        details=details,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic DJ transition generation (Phase A/B).")
    parser.add_argument("--song-a", required=True, help="Path to Song A audio file.")
    parser.add_argument("--song-b", required=True, help="Path to Song B audio file.")
    parser.add_argument("--plugin", default="Smooth Blend", choices=list(PLUGIN_PRESETS.keys()), help="Transition style plugin preset.")
    parser.add_argument("--instruction", default="", help="Extra text instruction for generation.")
    parser.add_argument("--pre-sec", type=float, default=6.0, help="Seconds before seam from Song A.")
    parser.add_argument("--repaint-sec", type=float, default=4.0, help="Repaint seam width in seconds.")
    parser.add_argument("--post-sec", type=float, default=6.0, help="Seconds after seam from Song B.")
    parser.add_argument("--analysis-sec", type=float, default=45.0, help="Analysis window in seconds.")
    parser.add_argument("--bpm-target", type=float, default=None, help="Optional BPM override target for Song A.")
    parser.add_argument("--cue-a-sec", type=float, default=None, help="Optional Song A cue override.")
    parser.add_argument("--cue-b-sec", type=float, default=None, help="Optional Song B cue override.")
    parser.add_argument(
        "--transition-bars",
        type=int,
        default=8,
        choices=[4, 8, 16],
        help="Transition period length in bars around cue points.",
    )
    parser.add_argument("--creativity", type=float, default=7.0, help="ACE-Step guidance strength.")
    parser.add_argument("--inference-steps", type=int, default=8, help="ACE-Step inference steps.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for output artifacts.")
    parser.add_argument("--output-stem", default=None, help="Optional fixed output stem.")
    parser.add_argument("--target-sr", type=int, default=DEFAULT_TARGET_SR, help="Output sample rate.")
    parser.add_argument("--keep-debug-files", action="store_true", help="Keep temporary rough source audio files.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args()

    req = TransitionRequest(
        song_a_path=args.song_a,
        song_b_path=args.song_b,
        plugin_id=args.plugin,
        instruction_text=args.instruction,
        pre_context_sec=args.pre_sec,
        repaint_width_sec=args.repaint_sec,
        post_context_sec=args.post_sec,
        analysis_sec=args.analysis_sec,
        bpm_target=args.bpm_target,
        cue_a_sec=args.cue_a_sec,
        cue_b_sec=args.cue_b_sec,
        transition_bars=args.transition_bars,
        creativity_strength=args.creativity,
        inference_steps=args.inference_steps,
        seed=args.seed,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        target_sr=args.target_sr,
        keep_debug_files=args.keep_debug_files,
    )

    result = generate_transition_artifacts(req)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
