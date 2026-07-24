# HitMap GUI Implementation Plan

> **For agentic workers:** Execute task-by-task. Steps use checkbox syntax.

**Goal:** Rebuild the PySide6 shell as HitMap — left control rail + waveform stage with theme and Advanced disclosure.

**Spec:** `docs/superpowers/specs/2026-07-24-hitmap-gui-design.md`

**Status:** Implemented. Full suite passing.

## Tasks

- [x] Task 1: `app/ui/theme.py` — colors + global QSS
- [x] Task 2: Rebuild `main_window.py` rail/stage layout + Advanced panel
- [x] Task 3: Restyle `waveform_view.py` + empty-state placeholder
- [x] Task 4: Theme applied in MainWindow; title HitMap
- [x] Task 5: Smoke test + full pytest
