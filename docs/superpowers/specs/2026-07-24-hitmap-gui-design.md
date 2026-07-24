# HitMap GUI Redesign — Design Spec

**Date:** 2026-07-24  
**Status:** Draft — awaiting user review  
**Product name (UI):** HitMap  
**Approach:** Stage-based shell with left control rail + waveform stage (Layout B)

## Problem

The current PySide6 UI is a flat stack of default Qt widgets. It works, but does not feel modern, branded, or easy to scan. Primary actions compete with secondary controls.

## Goals

- Look and feel like a professional music product named **HitMap**
- One clear composition: brand + left rail controls + dominant waveform stage
- Easy path: Load WAV → pick plugin → Convert → review → Save MIDI
- Progressive disclosure: tuck Device, preview source, Reset Position behind Advanced
- Keep all existing pipeline/controller behavior; this is a UI shell redesign

## Non-goals

- Changing detection/separation/MIDI algorithms
- Web UI or Electron rewrite
- Multi-window / multi-page wizard
- Rebranding the git repo / package name (UI brand is HitMap; repo may stay AudioToMidi)
- Full custom-painted waveform engine (keep pyqtgraph; restyle chrome around it)

## Visual direction

**Pro-audio hybrid (Layout B):**

| Token | Value | Role |
|-------|-------|------|
| Brand | HitMap (serif display) | Hero wordmark in left rail top |
| Tagline | Drum stem → MIDI | Small subtitle under brand |
| Surface | `#1a1d24` app chrome | Mid graphite, not pure black |
| Stage | `#0f1218` → `#151a22` | Waveform panel background |
| Accent | `#14b8a6` / `#5eead4` | Primary Convert, focus, sensitivity fill |
| Secondary CTA | `#0ea5e9` | Save MIDI (distinct from Convert) |
| Text | `#e2e8f0` / `#94a3b8` / `#64748b` | Primary / secondary / muted |
| Danger/clear | muted slate, not red-screaming | Clear All in Advanced |
| Fonts | Display: Georgia / “Palatino Linotype” fallback; UI: “Segoe UI” | Expressive brand, readable controls |

Avoid: purple gradients, cream+terracotta, broadsheet rules, neon glow stacks, pill-spam.

Onset marker colors stay voice-coded (existing waveform legend); restyle legend chips to match theme.

## Layout (approved)

```
┌──────────────┬────────────────────────────────────────┐
│ HitMap       │                                        │
│ tagline      │         WAVEFORM STAGE                 │
│              │         (stretch / hero)               │
│ SOURCE       │                                        │
│ [file/drop]  │                                        │
│ [plugin]     │                                        │
│ [Convert]    │                                        │
│              │                                        │
│ REVIEW       │                                        │
│ sensitivity  │                                        │
│ BPM          │                                        │
│ Play / Stop  │                                        │
│ [Save MIDI]  │                                        │
│              │                                        │
│ ▸ Advanced   │                                        │
│  (collapsed) │                                        │
└──────────────┴────────────────────────────────────────┘
```

**Rail width:** ~260–300px fixed; stage takes remaining width.  
**Window:** default ~1100×720; minimum ~900×560.

### Empty state (no WAV / no analysis)

- Stage shows centered quiet prompt: “Load a drum stem to begin” + short helper line
- Convert disabled until WAV selected
- Review controls disabled until analysis exists
- File area invites Browse (drag-and-drop if feasible with low risk; Browse remains primary)

### Ready state (after Convert)

- Status under brand: `N hits · ~BPM`
- Waveform + onset markers populated
- Review controls enabled
- Save MIDI becomes the clear export CTA

## Progressive disclosure

**Always visible:** file, plugin, Convert, waveform, sensitivity, BPM, Play/Stop (when preview supported), Save MIDI, Clear accessible but not competing (in Advanced or as quiet text button under Advanced).

**Advanced (collapsed by default):**

- Device (Auto / CPU / CUDA)
- Preview source (MIDI / Original / Both)
- Reset Position
- Clear All
- Plugin confidence warning (if low) — may also surface as a slim rail banner when relevant

## Interaction & motion

Keep motion subtle (2–3 intentional cues):

1. Convert progress: rail progress bar + status text; stage may show a soft indeterminate shimmer or dim overlay while busy
2. Analysis complete: brief accent flash on stage border / status line
3. Hover: primary buttons lift slightly (stylesheet `:hover`), Advanced expands with height animation if easy in Qt (`QPropertyAnimation` optional; instant expand OK if animation is flaky)

No gratuitous particle effects.

## Component map

| Current widget | New placement |
|----------------|---------------|
| path + Browse | Rail Source |
| plugin combo | Rail Source |
| Convert | Rail primary accent button |
| Save MIDI | Rail secondary CTA (sky) |
| Play / Stop | Rail Review |
| source combo | Advanced |
| Reset Position | Advanced |
| Clear All | Advanced |
| Device | Advanced |
| progress + elapsed | Under Convert or slim bar above stage |
| status label | Under brand / above Advanced |
| sensitivity | Rail Review |
| BPM spin | Rail Review |
| WaveformView | Stage (restyled bg, legend chips) |

## Technical approach

Stay on **PySide6**. Structure:

```
app/ui/
  theme.py          # QSS string + color/font constants
  main_window.py    # HitMap shell: rail + stage layout, wiring unchanged
  waveform_view.py  # Theme-aware colors; empty-state placeholder
  widgets.py        # Optional: SectionLabel, collapsible Advanced panel
```

- Apply global stylesheet in `main.py` or `MainWindow.__init__`
- Prefer QSS + layout refactor over rewriting controller
- Preserve all `PipelineController` signals/slots
- Window title: `HitMap`
- Tests: existing controller tests unchanged; add a lightweight UI smoke test that MainWindow constructs under offscreen Qt (optional)

## Error / busy states

- Busy: disable Convert / Browse / plugin / Advanced toggles that start work; keep Stop available during preview
- Failures: status text in rail (existing messages), optionally tint status muted amber for warnings
- Export success: status “Saved MIDI to …”

## Acceptance criteria

1. App launches as **HitMap** with left rail + waveform stage
2. Brand “HitMap” is the dominant first-viewport wordmark
3. Device / preview source / reset / clear live under collapsed Advanced
4. Convert and Save MIDI are visually distinct primary/secondary CTAs
5. Empty stage shows a clear prompt before load/convert
6. Full existing workflow still works (convert, sensitivity, BPM, preview, save)
7. Existing pytest suite still passes

## Implementation order

1. `theme.py` + apply QSS
2. Rebuild `main_window.py` layout (rail/stage) without behavior changes
3. Collapsible Advanced panel; move secondary controls
4. Empty-state stage + status under brand
5. Restyle `waveform_view` chrome/legend to theme
6. Polish busy/hover states; manual pass
7. Update README GUI screenshots/description lightly if needed

## Decisions log

- Visual direction: pro-audio hybrid (graphite + teal)
- Depth: full UX pass (C)
- Disclosure: A (core always on; device/preview extras tucked)
- Brand: HitMap
- Layout: B (side rail + stage) — approved in companion 2026-07-24
