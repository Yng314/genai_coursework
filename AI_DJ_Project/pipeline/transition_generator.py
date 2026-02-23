import argparse
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .audio_utils import (
    apply_edge_fades,
    choose_first_beat_after,
    choose_nearest_beat,
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

LOGGER = logging.getLogger(__name__)

DEFAULT_TARGET_SR = 32000
ACESTEP_INPUT_SR = 48000

PLUGIN_PRESETS: Dict[str, str] = {
    "Smooth Blend": "smooth seamless DJ transition, balanced energy, clean, no vocals",
    "EDM Build-up": "energetic EDM build-up transition with rising tension, clean, no vocals",
    "Percussive Bridge": "percussive bridge transition with rhythmic drums and clear groove, no vocals",
    "Ambient Wash": "ambient wash transition, spacious and atmospheric, soft energy curve, no vocals",
}

_ACESTEP_RUNTIME: Optional[Dict[str, Any]] = None


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
    creativity_strength: float = 7.0
    inference_steps: int = 8
    seed: int = 42
    output_dir: str = "outputs"
    output_stem: Optional[str] = None
    target_sr: int = DEFAULT_TARGET_SR
    force_fallback_crossfade: bool = False
    keep_debug_files: bool = False

    # ACE-Step runtime config
    acestep_model_config: str = os.getenv("AI_DJ_ACESTEP_MODEL_CONFIG", "acestep-v15-base").strip()
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
    backend_used: str
    used_fallback: bool
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
        "creativity_strength": request.creativity_strength,
        "inference_steps": request.inference_steps,
        "seed": request.seed,
        "target_sr": request.target_sr,
        "acestep_model_config": request.acestep_model_config,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:10]
    return f"transition_{_slug(Path(request.song_a_path).stem)}_to_{_slug(Path(request.song_b_path).stem)}_{digest}"


def _resolve_output_paths(request: TransitionRequest) -> Tuple[str, str, str]:
    os.makedirs(request.output_dir, exist_ok=True)
    stem = _deterministic_stem(request)
    transition_path = os.path.join(request.output_dir, f"{stem}_transition.wav")
    stitched_path = os.path.join(request.output_dir, f"{stem}_stitched.wav")
    rough_src_path = os.path.join(request.output_dir, f"{stem}_rough_src.wav")
    return transition_path, stitched_path, rough_src_path


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


def _prepare_rough_transition(request: TransitionRequest) -> Dict[str, Any]:
    pre_sec = clamp(request.pre_context_sec, 1.0, 20.0)
    seam_sec = clamp(request.repaint_width_sec, 1.0, 12.0)
    post_sec = clamp(request.post_context_sec, 1.0, 20.0)
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
    bpm_b = float(bpm_b) if bpm_b is not None else 120.0

    cue_a = float(request.cue_a_sec) if request.cue_a_sec is not None else float(
        a_analysis_start + choose_nearest_beat(beats_a, max(pre_sec, analysis_sec - seam_sec - 2.0))
    )
    cue_b = float(request.cue_b_sec) if request.cue_b_sec is not None else float(
        choose_first_beat_after(beats_b, 2.0)
    )

    stretch_rate = clamp(bpm_a / max(1e-6, bpm_b), 0.5, 2.0)

    pre_n = int(round(pre_sec * target_sr))
    seam_n = int(round(seam_sec * target_sr))
    post_n = int(round(post_sec * target_sr))

    # Song A window: pre-context + seam-tail
    a_start = max(0.0, cue_a - pre_sec)
    y_a_window, _ = decode_segment(
        request.song_a_path,
        a_start,
        pre_sec + seam_sec,
        sr=target_sr,
        max_decode_sec=pre_sec + seam_sec + 2.0,
    )
    y_a_window = ensure_length(y_a_window, pre_n + seam_n)
    a_pre = y_a_window[:pre_n]
    a_tail = y_a_window[pre_n : pre_n + seam_n]

    # Song B window: seam-head + post-context (after stretch)
    desired_b_out_sec = seam_sec + post_sec
    raw_b_in_sec = clamp(desired_b_out_sec * stretch_rate, 1.0, 120.0)
    y_b_raw, _ = decode_segment(
        request.song_b_path,
        cue_b,
        raw_b_in_sec,
        sr=target_sr,
        max_decode_sec=raw_b_in_sec + 2.0,
    )
    y_b = safe_time_stretch(y_b_raw, rate=stretch_rate)
    y_b = ensure_length(y_b, seam_n + post_n)
    b_head = y_b[:seam_n]
    b_post = y_b[seam_n : seam_n + post_n]

    rough_seam = crossfade_equal_length(a_tail, b_head)
    rough_stitched = np.concatenate([a_pre, rough_seam, b_post]).astype(np.float32)

    return {
        "target_sr": target_sr,
        "dur_a": dur_a,
        "dur_b": dur_b,
        "analysis_start_a_sec": a_analysis_start,
        "bpm_a": bpm_a,
        "bpm_b": bpm_b,
        "cue_a_sec": cue_a,
        "cue_b_sec": cue_b,
        "stretch_rate": stretch_rate,
        "pre_sec": pre_sec,
        "seam_sec": seam_sec,
        "post_sec": post_sec,
        "pre_n": pre_n,
        "seam_n": seam_n,
        "post_n": post_n,
        "rough_seam": rough_seam,
        "rough_stitched": rough_stitched,
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

    repaint_start = float(rough["pre_sec"])
    repaint_end = float(rough["pre_sec"] + rough["seam_sec"])
    total_duration = float(rough["pre_sec"] + rough["seam_sec"] + rough["post_sec"])
    bpm_hint = int(round(rough["bpm_a"])) if 30 <= rough["bpm_a"] <= 300 else None

    params = GenerationParams(
        task_type="repaint",
        src_audio=rough_src_path,
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

    transition_path, stitched_path, rough_src_path = _resolve_output_paths(request)

    LOGGER.info("Transition request args: %s", json.dumps(request.to_log_dict(), sort_keys=True))
    rough = _prepare_rough_transition(request)

    transition_audio = rough["rough_seam"]
    stitched_audio = rough["rough_stitched"]
    backend_used = "crossfade-fallback"
    used_fallback = True
    fallback_reason = None

    if not request.force_fallback_crossfade:
        try:
            transition_audio, stitched_audio = _run_acestep_repaint(request, rough, rough_src_path)
            backend_used = "acestep-repaint"
            used_fallback = False
        except Exception as exc:
            fallback_reason = str(exc)
            LOGGER.warning("ACE-Step repaint failed, falling back to deterministic crossfade: %s", exc)
    else:
        fallback_reason = "force_fallback_crossfade=true"

    transition_audio = normalize_peak(apply_edge_fades(transition_audio, rough["target_sr"], fade_ms=25.0), peak=0.98)
    stitched_audio = normalize_peak(apply_edge_fades(stitched_audio, rough["target_sr"], fade_ms=25.0), peak=0.98)

    write_wav(transition_path, transition_audio, rough["target_sr"])
    write_wav(stitched_path, stitched_audio, rough["target_sr"])

    if (not request.keep_debug_files) and os.path.exists(rough_src_path):
        try:
            os.remove(rough_src_path)
        except Exception:
            pass

    details = {
        "backend_used": backend_used,
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        "generation_args": request.to_log_dict(),
        "bpm": {
            "song_a": round(float(rough["bpm_a"]), 3),
            "song_b": round(float(rough["bpm_b"]), 3),
            "stretch_rate": round(float(rough["stretch_rate"]), 5),
            "bpm_target_override": request.bpm_target,
        },
        "cue_points_sec": {
            "song_a": round(float(rough["cue_a_sec"]), 3),
            "song_b": round(float(rough["cue_b_sec"]), 3),
        },
        "clip_shape_sec": {
            "pre_context_sec": round(float(rough["pre_sec"]), 3),
            "repaint_width_sec": round(float(rough["seam_sec"]), 3),
            "post_context_sec": round(float(rough["post_sec"]), 3),
            "analysis_sec": round(float(request.analysis_sec), 3),
        },
        "durations_sec": {
            "song_a_total": rough["dur_a"],
            "song_b_total": rough["dur_b"],
            "analysis_start_a_sec": round(float(rough["analysis_start_a_sec"]), 3),
        },
        "outputs": {
            "transition_path": transition_path,
            "stitched_path": stitched_path,
        },
    }
    LOGGER.info("Transition result details: %s", json.dumps(details, sort_keys=True))

    return TransitionResult(
        transition_path=transition_path,
        stitched_path=stitched_path,
        backend_used=backend_used,
        used_fallback=used_fallback,
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
    parser.add_argument("--creativity", type=float, default=7.0, help="ACE-Step guidance strength.")
    parser.add_argument("--inference-steps", type=int, default=8, help="ACE-Step inference steps.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for output artifacts.")
    parser.add_argument("--output-stem", default=None, help="Optional fixed output stem.")
    parser.add_argument("--target-sr", type=int, default=DEFAULT_TARGET_SR, help="Output sample rate.")
    parser.add_argument("--fallback-only", action="store_true", help="Skip ACE-Step and force deterministic crossfade fallback.")
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
        creativity_strength=args.creativity,
        inference_steps=args.inference_steps,
        seed=args.seed,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        target_sr=args.target_sr,
        force_fallback_crossfade=args.fallback_only,
        keep_debug_files=args.keep_debug_files,
    )

    result = generate_transition_artifacts(req)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

