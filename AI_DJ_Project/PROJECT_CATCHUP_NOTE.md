# AI DJ Project Catch-Up Note

Last updated: 2026-02-19

## 1) Project Goal (Current Direction)

Build a **domain-specific AI DJ transition demo** for coursework Option 1 (Refinement):

- user uploads Song A and Song B
- system auto-detects cue points + BPM
- Song B is time-stretched to Song A BPM
- a generative model creates transition audio from text ("transition vibe")
- output is a **short transition clip only** (not full-song mix)

This scope is intentionally optimized for Hugging Face Spaces reliability.

---

## 2) Coursework Fit (Why this is Option 1)

This is a refinement of existing pipelines/models:

- existing generative pipeline (currently MusicGen, planned ACE-Step)
- wrapped in domain-specific DJ UX (cue/BPM/mix controls)
- not raw prompting only; structured controls for practical use

---

## 3) Current Implemented Pipeline (Already in `app.py`)

Current app file: `AI_DJ_Project/app.py`

### 3.1 Input + UI

- Upload `Song A` and `Song B`
- Set:
  - transition vibe text
  - transition type (`riser`, `drum fill`, `sweep`, `brake`, `scratch`, `impact`)
  - mode (`Overlay` or `Insert`)
  - pre/mix/post seconds
  - transition length + gain
  - optional BPM and cue overrides

### 3.2 Audio analysis and cueing

1. Probe duration with `ffprobe` (if available)
2. Decode only needed segments (ffmpeg first, librosa fallback)
3. Estimate BPM + beat times with `librosa.beat.beat_track`
4. Auto-cue strategy:
   - Song A: choose beat near end analysis window
   - Song B: choose first beat after ~2 seconds
5. Optional manual override for BPM and cue points

### 3.3 Tempo matching

- Compute stretch rate = `bpm_A / bpm_B` (clamped)
- Time-stretch Song B segment via `librosa.effects.time_stretch`

### 3.4 AI transition generation

- `@spaces.GPU` function `_generate_ai_transition(...)`
- Uses `facebook/musicgen-small`
- Prompt is domain-steered for DJ transition behavior
- Returns short generated transition audio

### 3.5 Assembly

- **Overlay mode**: crossfade A/B + overlay AI transition
- **Insert mode**: A -> AI transition -> B (with short anti-click fades)
- Edge fades + peak normalization before output

### 3.6 Output

- Output audio clip (NumPy audio to Gradio)
- JSON details:
  - BPM estimates
  - cue points
  - stretch rate
  - analysis settings

---

## 4) Full End-to-End Pipeline (Conceptual)

Upload A/B  
-> decode limited windows  
-> BPM + beat analysis  
-> auto-cue points  
-> stretch B to A BPM  
-> generate transition (GenAI)  
-> overlay/insert assembly  
-> normalize/fades  
-> return short transition clip + diagnostics

---

## 5) Planned Upgrade: ACE-Step + Custom LoRA

### 5.1 What ACE-Step is

ACE-Step 1.5 is a **full music-generation foundation model stack** (text-to-audio/music with editing/control workflows), not just a tiny SFX model.

Planned usage in this project:

- keep deterministic DJ logic (cue/BPM/stretch/assemble)
- swap transition generation backend from MusicGen to ACE-Step
- load custom LoRA adapter(s) to enforce DJ transition style

### 5.2 Integration strategy (recommended)

1. Keep current `app.py` flow unchanged for analysis/mixing
2. Introduce backend abstraction:
   - `MusicGenBackend` (fallback)
   - `AceStepBackend` (main target)
3. Add LoRA controls:
   - adapter selection
   - adapter scale
4. Continue returning short transition clips only

---

## 6) Genre-Specific LoRA Idea (Pop / Electronic / House / Dubstep / Techno)

## Is this a good idea?

**Yes, as a staged plan.**

It is a strong product and coursework idea because:

- user-selected genre can map to distinct transition style
- demonstrates clear domain-specific refinement
- supports explainable UX: "You picked House -> House-style transition LoRA"

### Important caveats

- Training one LoRA per genre increases data and compute requirements a lot
- Early quality may vary by genre and dataset size
- More adapters mean more evaluation and QA burden

### Practical rollout (recommended)

Phase 1 (safe):
- base model + one "general DJ transition" LoRA

Phase 2 (coursework-strong):
- 2-3 genre LoRAs (e.g., Pop / House / Dubstep)

Phase 3 (optional extension):
- larger genre library + auto-genre suggestion from uploaded songs

---

## 7) Proposed Genre LoRA Routing Logic

User selects uploaded-song genre (or manually selects transition style profile):

- Pop -> `lora_pop_transition`
- Electronic -> `lora_electronic_transition`
- House -> `lora_house_transition`
- Dubstep -> `lora_dubstep_transition`
- Techno -> `lora_techno_transition`
- Auto/Unknown -> `lora_general_transition`

Then:

1. load chosen LoRA
2. set LoRA scale
3. run ACE-Step generation for short transition duration
4. mix with A/B boundary clip

---

## 8) Data and Training Notes for LoRA

- Use only licensed/royalty-free/self-owned audio for dataset and demos
- Dataset should emphasize transition-like content (risers, fills, drops, sweeps, impacts)
- Include metadata/captions describing genre + transition intent
- Keep track of:
  - adapter name
  - dataset source and license
  - training config and epoch checkpoints

---

## 9) Current Risks / Constraints

- ACE-Step stack is heavier than MusicGen and needs careful deployment tuning
- Cold starts and memory behavior can be challenging on Spaces
- Auto-cueing is heuristic; may fail on hard tracks (manual override should remain)
- Time-stretch can introduce artifacts (expected in DJ contexts)

---

## 10) Fallback and Reliability Plan

- Keep MusicGen backend as fallback while integrating ACE-Step
- If ACE-Step init fails:
  - fail over to MusicGen backend
  - still return valid transition clip
- Preserve deterministic DSP path as model-agnostic baseline

---

## 11) "If I lost track" Quick Resume Checklist

1. Open `app.py` and confirm current backend is still working end-to-end
2. Verify demo still does:
   - cue detect
   - BPM match
   - transition generation
   - clip output
3. Re-read this note section 5/6/7
4. Continue with next implementation milestone:
   - backend abstraction
   - ACE-Step backend skeleton
   - single LoRA integration
   - then genre LoRA expansion

---

## 12) Next Concrete Milestones

M1: Refactor transition generation into backend interface  
M2: Implement `AceStepBackend` with base model inference  
M3: Add LoRA load/select/scale UI + runtime controls  
M4: Train first "general DJ transition" LoRA  
M5: Train 2-3 genre LoRAs and add genre routing  
M6: Compare outputs (base vs LoRA, genre A vs genre B) for coursework evidence

