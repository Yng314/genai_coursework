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
    if isinstance(value, str) and not value.strip():
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
    transition_bars,
    pre_context_sec,
    post_context_sec,
    analysis_sec,
    bpm_target,
    creativity_strength,
    inference_steps,
    seed,
    cue_a_sec,
    cue_b_sec,
    output_dir,
):
    if not song_a or not song_b:
        raise gr.Error("Please upload both Song A and Song B.")

    request = TransitionRequest(
        song_a_path=song_a,
        song_b_path=song_b,
        plugin_id=plugin_id,
        instruction_text=instruction_text or "",
        transition_base_mode="B-base-fixed",
        transition_bars=int(transition_bars),
        pre_context_sec=float(pre_context_sec),
        repaint_width_sec=4.0,
        post_context_sec=float(post_context_sec),
        analysis_sec=float(analysis_sec),
        bpm_target=_to_optional_float(bpm_target),
        cue_a_sec=_to_optional_float(cue_a_sec),
        cue_b_sec=_to_optional_float(cue_b_sec),
        creativity_strength=float(creativity_strength),
        inference_steps=int(inference_steps),
        seed=int(seed),
        output_dir=(output_dir or "outputs").strip(),
    )

    try:
        result = generate_transition_artifacts(request)
    except Exception as exc:
        raise gr.Error(str(exc))

    return (
        result.transition_path,
        result.rough_stitched_path,
        result.hard_splice_path,
        result.stitched_path,
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown(
            """
# AI DJ Transition Generator (Phase A/B)

This app follows the coursework refinement plan through **Phase B**:
- deterministic transition API (two songs in -> transition + stitched artifacts out)
- ACE-Step repaint seam generation with bar-defined transition periods
            """.strip()
        )

        with gr.Row():
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

        with gr.Row():
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
            transition_bars = gr.Dropdown(
                label="Transition period length (bars)",
                choices=[4, 8, 16],
                value=8,
                info="Controls transition duration. Pipeline uses fixed B-base strategy with A as reference.",
            )

        with gr.Accordion("Advanced controls", open=False):
            with gr.Row():
                pre_context_sec = gr.Slider(
                    minimum=1,
                    maximum=12,
                    value=6,
                    step=0.5,
                    label="Seconds before seam (Song A context)",
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

            with gr.Row():
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

            with gr.Row():
                cue_a_sec = gr.Textbox(
                    label="Optional cue A override (sec)",
                    value="",
                    placeholder="Leave blank for auto cue selection",
                )
                cue_b_sec = gr.Textbox(
                    label="Optional cue B override (sec)",
                    value="",
                    placeholder="Leave blank for auto cue selection",
                )
                output_dir = gr.Textbox(label="Output directory", value="outputs")

        run_btn = gr.Button("Generate transition artifacts", variant="primary")

        with gr.Row():
            transition_audio = gr.Audio(
                label="Generated transition clip",
                type="filepath",
            )
            rough_stitched_audio = gr.Audio(
                label="No-repaint rough stitch (baseline)",
                type="filepath",
            )
            hard_splice_audio = gr.Audio(
                label="Hard splice baseline (no transition)",
                type="filepath",
            )
            stitched_audio = gr.Audio(
                label="Final stitched clip",
                type="filepath",
            )

        run_btn.click(
            fn=_run_transition,
            inputs=[
                song_a,
                song_b,
                plugin_id,
                instruction_text,
                transition_bars,
                pre_context_sec,
                post_context_sec,
                analysis_sec,
                bpm_target,
                creativity_strength,
                inference_steps,
                seed,
                cue_a_sec,
                cue_b_sec,
                output_dir,
            ],
            outputs=[transition_audio, rough_stitched_audio, hard_splice_audio, stitched_audio],
        )

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch()
