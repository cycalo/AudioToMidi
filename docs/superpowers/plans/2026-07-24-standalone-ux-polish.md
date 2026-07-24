# Standalone UX Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Polish the standalone exe UX: visible model download progress, no console flash, ~1280px default width, Help → debug log dialog, integer BPM.

**Architecture:** Extend `ensure_checkpoint` with a progress callback wired through `separate` → Convert worker → existing UI progress signal. Add an in-memory ring log buffer + dialog. Suppress Windows console for frozen builds. Integer BPM via `normalize_bpm` + `QSpinBox`.

**Tech Stack:** Python, PySide6, urllib, logging, PyInstaller windowed build.

## Global Constraints

- Download still happens on first Convert (not launch prefetch).
- Cleanup policy unchanged (temps deleted; model cache kept).
- Debug log is option A: menu dialog + Save; no always-on file.
- Default window ≈ 1280px wide.
- BPM whole numbers only (40–240).

---

### Task 1: Integer BPM

**Files:** `pipeline/tempo.py`, `tests/test_tempo.py`, `app/ui/main_window.py`

- [x] Update `normalize_bpm` to round to nearest int then clamp; add test for fractional rounding
- [x] Replace `QDoubleSpinBox` with `QSpinBox`; integer status strings
- [x] Run `pytest tests/test_tempo.py tests/test_main_window.py -q`

### Task 2: Wider default window

**Files:** `app/ui/main_window.py`, `tests/test_main_window.py`

- [x] Target first-show width ≈ 1280; update smoke test
- [x] Run window test

### Task 3: Checkpoint download progress

**Files:** `pipeline/separation.py`, `app/controller.py`, `tests/test_separation.py`

- [x] Add `on_progress` to `ensure_checkpoint`; thread through `separate` / `_separate_demucs`
- [x] Convert worker reports download % via existing progress signal
- [x] Unit test with mocked download

### Task 4: Debug log buffer + dialog

**Files:** `app/log_buffer.py`, `app/ui/debug_log_dialog.py`, `app/ui/main_window.py`, `app/main.py`, `tests/test_log_buffer.py`

- [x] Ring buffer (~2000 lines) + logging handler
- [x] Help → View debug log… with Save / Clear / Close
- [x] Route startup cleanup + progress into buffer

### Task 5: Console flash suppression

**Files:** `app/main.py`, tests if any

- [x] `multiprocessing.freeze_support()` + Windows no-console child defaults for frozen builds

### Task 6: README note

**Files:** `README.md`

- [x] Clarify first-run download UI + cleanup/model cache behavior
