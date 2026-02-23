# Coursework Detailed Plan (Refinement Track)

## Implementation Status (Repo)

- [x] Phase A baseline refactor
- [x] Phase B ACE-Step integration (`repaint`) + fallback
- [ ] Phase C plugin config system (`plugins.yaml`)
- [ ] Phase D additional interactive controls polish
- [ ] Phase E evaluation report artifacts
- [ ] Phase F final packaging

## 1. Project Positioning

**Track**: Refinement Demo (single base model + capability switching)

**One-sentence pitch**:
Build a DJ-transition generator by refining one base model (`ACE-Step/Ace-Step1.5`) into multiple switchable transition capabilities (plugins), controlled via structured UI parameters and text conditioning.

**Why this fits coursework**:
- Keep one base generative model as the core.
- Add plugin-like capability switching (different LoRAs / presets / task modes).
- Provide an interactive Gradio product experience instead of a raw script.

---

## 2. Final Demo Scope (What we will deliver)

1. Input:
- Song A (or song list + selected pair)
- Song B
- Transition style plugin (dropdown)
- Text instruction (e.g., "smooth, rising energy, no vocals")
- Sliders: duration, bpm target, key bias, creativity strength, seed

2. Core generation:
- Build an initial rough junction: tail of A + head of B
- Use `ACE-Step` (`repaint`) to regenerate only the seam region
- Optional style reference audio for better style consistency

3. Output:
- Generated transition clip
- Final stitched audio (A + generated transition + B)
- Metadata panel (all parameters + model/plugin used)

4. Switchable capabilities (plugin concept):
- `Smooth Blend`
- `EDM Build-up`
- `Percussive Bridge`
- `Ambient Wash`

Each plugin maps to different prompt templates + parameter defaults (+ optional LoRA).

---

## 3. Target Architecture

## 3.1 Pipeline
1. Preprocess: load audio, loudness normalize, optional BPM alignment
2. Build seam source: `A_tail + B_head`
3. Conditioning builder:
- UI sliders + user text
- plugin template -> final caption
4. ACE-Step inference:
- `task_type = repaint`
- controlled repaint window around seam
5. Postprocess:
- short boundary crossfades
- peak limiting / normalization
6. Export + preview in Gradio

## 3.2 Plugin Layer (Refinement core)
Use a single config table to switch capabilities:

- `plugin_id`
- prompt template
- repaint window policy
- generation defaults
- optional lora repo/path

This gives “single base model, multiple capabilities” without changing core pipeline code.

---

## 4. File Plan

1. `app.py`
- Gradio UI + orchestration
- model initialization and caching

2. `requirements.txt`
- runtime dependencies for Space/Colab

3. `plugins.yaml` (new)
- plugin definitions and defaults

4. `pipeline/transition_generator.py` (new)
- preprocessing, conditioning, repaint call, postprocessing

5. `pipeline/audio_utils.py` (new)
- slicing, normalization, seam construction, export

6. `docs/EXPERIMENTS.md` (new)
- baseline vs refined comparison records

7. `README.md` (update)
- run instructions + model notes + known limitations

---

## 5. Work Breakdown (Execution Steps)

## Phase A: Baseline refactor (Day 1-2)
1. Separate existing script logic into reusable functions.
2. Create deterministic transition API:
- input: two files + parameters
- output: one transition file + one final stitched file
3. Add logging for every generation argument.

**Exit criteria**:
- Can run one command/function to generate a transition artifact reproducibly.

## Phase B: ACE-Step integration (Day 3-4)
1. Add ACE-Step inference wrapper.
2. Implement `repaint` flow around seam window.
3. Add fallback mode: if inference fails, revert to classic crossfade.

**Exit criteria**:
- At least one successful AI-generated transition clip from A/B pair.

## Phase C: Refinement plugin system (Day 5-6)
1. Implement `plugins.yaml` loader.
2. Build 4 plugins (Smooth/Build-up/Percussive/Ambient).
3. Support plugin switching in one click.
4. Add optional LoRA loading hook per plugin.

**Exit criteria**:
- Same input pair + different plugins produce clearly different transitions.

## Phase D: Interactive controls (Day 7)
1. Add UI sliders and textbox:
- duration, creativity, bpm target, seed, repaint width
2. Convert UI state -> structured generation config.
3. Render parameter summary with output.

**Exit criteria**:
- Parameter changes are reflected in output and recorded.

## Phase E: Evaluation & comparison (Day 8-9)
1. Prepare baseline outputs (current non-GenAI crossfade).
2. Prepare refined outputs (plugin + repaint).
3. Compare with a fixed set of song pairs and prompts.
4. Document findings in `docs/EXPERIMENTS.md`.

**Exit criteria**:
- At least 3 pairs x 2 methods comparison available.

## Phase F: Packaging (Day 10)
1. Clean `app.py`, `requirements.txt`, startup flow.
2. Add model load-time notes and error messages.
3. Final smoke test on clean environment.

**Exit criteria**:
- App starts and produces output with one click for demo scenario.

---

## 6. LoRA Strategy

## Option 1 (Fast path, recommended first)
- Use base ACE-Step + prompt/plugin engineering only.
- Deliver working refinement demo quickly.

## Option 2 (Enhanced path)
- Add one or more existing ACE-Step-compatible LoRAs (if suitable).
- Map each plugin to a specific LoRA.

## Option 3 (Full custom, if time allows)
- Train custom LoRA for transition style domain.

### Minimal custom LoRA plan
1. Collect dataset of short transition-style clips (cleanly tagged).
2. For each sample: `audio + prompt text (+ optional lyrics/metadata)`.
3. Convert dataset with ACE-Step provided conversion script.
4. Train with conservative LoRA config first (`r`/`alpha` moderate).
5. Validate by A/B blind listening on fixed test pairs.

**Important**: Do Option 3 only after Option 1 is stable.

---

## 7. Evaluation Protocol

Use fixed test set:
- 3 to 5 song pairs
- 2 prompt intents per pair
- fixed seeds for reproducibility

Metrics:
1. Seam smoothness (subjective 1-5)
2. Style match to prompt (1-5)
3. Energy trajectory consistency (1-5)
4. Failure rate (generation errors / artifacts)

Output table columns:
- pair_id, method, plugin, prompt, seed, score_smooth, score_style, score_energy, notes

---

## 8. Risks and Fallbacks

1. Model inference too slow
- Fallback: shorter duration, fewer steps, smaller batch, CPU-safe fallback crossfade.

2. Unstable quality
- Fallback: generate K candidates and auto-pick by heuristic (loudness/clip-free + optional audio-text score).

3. LoRA path not ready in time
- Fallback: keep plugin switching via prompt templates + parameter presets (still valid refinement).

4. Memory issues
- Fallback: single-request mode, model cache reuse, lower sample length.

---

## 9. Definition of Done (Final Checklist)

1. One base model (`ACE-Step`) powers all transition generation.
2. At least 3 switchable plugins with clearly different behavior.
3. Text + slider controls affect generation outcomes.
4. End-to-end interactive demo works reliably.
5. Baseline vs refined comparisons documented.
6. Reproducible run path documented.

---

## 10. Suggested Immediate Next Actions (Today)

1. Create `app.py` skeleton with UI controls and fake outputs.
2. Implement seam extraction utility (`A_tail + B_head`).
3. Implement one ACE-Step repaint call path.
4. Add first plugin (`Smooth Blend`) and validate end-to-end.
5. Save first successful demo sample for coursework evidence.

