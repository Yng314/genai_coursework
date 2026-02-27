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

APP_CSS = """
.adv-item label,
.adv-item .gr-block-label,
.adv-item .gr-block-title {
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
"""

APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    radius_size="lg",
).set(
    block_radius="*radius_xl",
    input_radius="*radius_xl",
    button_large_radius="*radius_xl",
    button_medium_radius="*radius_xl",
    button_small_radius="*radius_xl",
)


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
        result.hard_splice_path,
        result.rough_stitched_path,
        result.stitched_path,
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(theme=APP_THEME, css=APP_CSS) as demo:
        gr.Markdown(
            """
<div style="text-align:center;">
  <h1>AI DJ Transition Generator</h1>
  <p>Upload two songs and generate a transition between them.</p>
</div>
            """.strip()
        )
        with gr.Row():
            gr.Markdown(
                """
### How to use
1. Upload **Song A** (current track) and **Song B** (next track).
2. Choose a **Transition style plugin**.
3. Optionally add **Text instruction** (e.g., smooth, rising energy, no vocals).
4. Select **Transition period length (bars)**.
5. Click **Generate transition artifacts**.
                """.strip(),
                container=False,
                elem_classes=["plain-info"],
            )
            gr.Markdown(
                """
### Outputs
- **Generated transition clip**: AI-generated repaint transition segment.
- **Hard splice baseline (no transition)**: direct cut baseline.
- **No-repaint rough stitch (baseline)**: stitched baseline without repaint.
- **Final stitched clip**: final result with transition inserted.
                """.strip(),
                container=False,
                elem_classes=["plain-info"],
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
                    min_width=320,
                    elem_classes=["adv-item"],
                )
                post_context_sec = gr.Slider(
                    minimum=1,
                    maximum=12,
                    value=6,
                    step=0.5,
                    label="Seconds after seam (Song B context)",
                    min_width=320,
                    elem_classes=["adv-item"],
                )

            with gr.Row():
                analysis_sec = gr.Slider(
                    minimum=10,
                    maximum=90,
                    value=45,
                    step=5,
                    label="Analysis window (seconds)",
                    min_width=320,
                    elem_classes=["adv-item"],
                )
                bpm_target = gr.Number(
                    label="Optional BPM target override",
                    value=None,
                    min_width=320,
                    elem_classes=["adv-item"],
                )

            with gr.Row():
                creativity_strength = gr.Slider(
                    minimum=1.0,
                    maximum=12.0,
                    value=7.0,
                    step=0.5,
                    label="Creativity strength (guidance)",
                    min_width=320,
                    elem_classes=["adv-item"],
                )
                inference_steps = gr.Slider(
                    minimum=1,
                    maximum=64,
                    value=8,
                    step=1,
                    label="ACE-Step inference steps",
                    min_width=320,
                    elem_classes=["adv-item"],
                )

            with gr.Row():
                seed = gr.Number(
                    label="Seed",
                    value=42,
                    precision=0,
                    min_width=320,
                    elem_classes=["adv-item"],
                )
                cue_a_sec = gr.Textbox(
                    label="Optional cue A override (sec)",
                    value="",
                    placeholder="Leave blank for auto cue selection",
                    min_width=320,
                    elem_classes=["adv-item"],
                )

            with gr.Row():
                cue_b_sec = gr.Textbox(
                    label="Optional cue B override (sec)",
                    value="",
                    placeholder="Leave blank for auto cue selection",
                    min_width=320,
                    elem_classes=["adv-item"],
                )
                output_dir = gr.Textbox(
                    label="Output directory",
                    value="outputs",
                    min_width=320,
                    elem_classes=["adv-item"],
                )

        run_btn = gr.Button("Generate transition artifacts", variant="primary")

        with gr.Row():
            transition_audio = gr.Audio(
                label="Generated transition clip",
                type="filepath",
            )
            hard_splice_audio = gr.Audio(
                label="Hard splice baseline (no transition)",
                type="filepath",
            )
            rough_stitched_audio = gr.Audio(
                label="No-repaint rough stitch (baseline)",
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
            outputs=[transition_audio, hard_splice_audio, rough_stitched_audio, stitched_audio],
        )

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch()
