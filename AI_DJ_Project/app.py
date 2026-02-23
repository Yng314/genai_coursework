import logging
from typing import Optional

import gradio as gr

from pipeline.transition_generator import (
    PLUGIN_PRESETS,
    TransitionRequest,
    generate_transition_artifacts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def _to_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _run_transition(
    song_a,
    song_b,
    plugin_id,
    instruction_text,
    pre_context_sec,
    repaint_width_sec,
    post_context_sec,
    analysis_sec,
    bpm_target,
    creativity_strength,
    inference_steps,
    seed,
    cue_a_sec,
    cue_b_sec,
    output_dir,
    force_fallback,
):
    if not song_a or not song_b:
        raise gr.Error("Please upload both Song A and Song B.")

    request = TransitionRequest(
        song_a_path=song_a,
        song_b_path=song_b,
        plugin_id=plugin_id,
        instruction_text=instruction_text or "",
        pre_context_sec=float(pre_context_sec),
        repaint_width_sec=float(repaint_width_sec),
        post_context_sec=float(post_context_sec),
        analysis_sec=float(analysis_sec),
        bpm_target=_to_optional_float(bpm_target),
        cue_a_sec=_to_optional_float(cue_a_sec),
        cue_b_sec=_to_optional_float(cue_b_sec),
        creativity_strength=float(creativity_strength),
        inference_steps=int(inference_steps),
        seed=int(seed),
        output_dir=(output_dir or "outputs").strip(),
        force_fallback_crossfade=bool(force_fallback),
    )

    try:
        result = generate_transition_artifacts(request)
    except Exception as exc:
        raise gr.Error(str(exc))

    status = (
        f"Backend used: {result.backend_used} | "
        f"Fallback: {result.used_fallback}"
    )
    return result.transition_path, result.stitched_path, result.details, status


def build_ui() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown(
            """
# AI DJ Transition Generator (Phase A/B)

This app follows the coursework refinement plan through **Phase B**:
- deterministic transition API (two songs in -> transition + stitched artifacts out)
- ACE-Step repaint seam generation
- automatic fallback to deterministic crossfade when ACE-Step fails
            """.strip()
        )

        with gr.Row():
            with gr.Column():
                song_a = gr.Audio(
                    label="Song A (mix out)",
                    type="filepath",
                    sources=["upload"],
                )
                song_b = gr.Audio(
                    label="Song B (mix in)",
                    type="filepath",
                    sources=["upload"],
                )

                plugin_id = gr.Dropdown(
                    label="Transition style plugin",
                    choices=list(PLUGIN_PRESETS.keys()),
                    value="Smooth Blend",
                )
                instruction_text = gr.Textbox(
                    label="Text instruction",
                    placeholder="e.g., smooth, rising energy, no vocals",
                    lines=2,
                )

            with gr.Column():
                pre_context_sec = gr.Slider(
                    minimum=1,
                    maximum=12,
                    value=6,
                    step=0.5,
                    label="Seconds before seam (Song A context)",
                )
                repaint_width_sec = gr.Slider(
                    minimum=1,
                    maximum=12,
                    value=4,
                    step=0.5,
                    label="Repaint seam width (seconds)",
                )
                post_context_sec = gr.Slider(
                    minimum=1,
                    maximum=12,
                    value=6,
                    step=0.5,
                    label="Seconds after seam (Song B context)",
                )
                analysis_sec = gr.Slider(
                    minimum=10,
                    maximum=90,
                    value=45,
                    step=5,
                    label="Analysis window (seconds)",
                )

                bpm_target = gr.Number(label="Optional BPM target override", value=None)
                creativity_strength = gr.Slider(
                    minimum=1.0,
                    maximum=12.0,
                    value=7.0,
                    step=0.5,
                    label="Creativity strength (guidance)",
                )
                inference_steps = gr.Slider(
                    minimum=1,
                    maximum=64,
                    value=8,
                    step=1,
                    label="ACE-Step inference steps",
                )
                seed = gr.Number(label="Seed", value=42, precision=0)

                cue_a_sec = gr.Number(label="Optional cue A override (sec)", value=None)
                cue_b_sec = gr.Number(label="Optional cue B override (sec)", value=None)

                output_dir = gr.Textbox(label="Output directory", value="outputs")
                force_fallback = gr.Checkbox(
                    label="Force deterministic crossfade fallback (debug)",
                    value=False,
                )

        run_btn = gr.Button("Generate transition artifacts", variant="primary")

        with gr.Row():
            transition_audio = gr.Audio(
                label="Generated transition clip",
                type="filepath",
            )
            stitched_audio = gr.Audio(
                label="Final stitched clip",
                type="filepath",
            )

        details = gr.JSON(label="Run details")
        status = gr.Textbox(label="Status", interactive=False)

        run_btn.click(
            fn=_run_transition,
            inputs=[
                song_a,
                song_b,
                plugin_id,
                instruction_text,
                pre_context_sec,
                repaint_width_sec,
                post_context_sec,
                analysis_sec,
                bpm_target,
                creativity_strength,
                inference_steps,
                seed,
                cue_a_sec,
                cue_b_sec,
                output_dir,
                force_fallback,
            ],
            outputs=[transition_audio, stitched_audio, details, status],
        )

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch()
