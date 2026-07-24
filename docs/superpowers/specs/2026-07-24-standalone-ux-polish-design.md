# Standalone UX Polish — Design Spec

**Date:** 2026-07-24  
**Status:** Approved for planning  
**Approach:** Targeted polish (Approach 1) — keep Convert-time model download; surface it in the UI; suppress console flash; add debug log dialog; wider default window; integer BPM. No cleanup policy change.

## Problem

Standalone exe users hit several rough edges:

1. **First Convert is very slow (~200s)** then subsequent runs are fast (~10s), with no visible explanation. The drumsep checkpoint (~167 MB) downloads on first use, but progress only goes to stderr (discarded in windowed builds).
2. **A PowerShell/console window flashes** when conversion finishes. The app does not launch a shell; this is almost certainly a child-process / windowed-PyInstaller + Torch/Demucs artifact.
3. **Default window is narrow** (~848px content width); waveform review needs more horizontal space.
4. **No way to inspect or save diagnostic logs** from the GUI.
5. **BPM allows fractional values**; users want whole numbers only.
6. **Cleanup behavior is unclear** to users (answered in product terms below; no code change required).

## Goals

- Show download progress in the existing Convert status/progress UI when the checkpoint is missing or corrupt.
- Eliminate visible console windows during Convert in the frozen Windows build.
- Default first-show window width ≈ 1280px (capped to screen).
- Help → View debug log… dialog with live in-memory log + Save log…
- BPM control and detection use integers only (40–240).
- Document and preserve existing temp-stem cleanup + model-cache retention.

## Non-goals

- Prefetching the model on app launch.
- Always-on log files or a docked diagnostics panel.
- Changing when stems are deleted or deleting the cached model.
- Quantization, variable tempo maps, or BPM half/double auto-correction.

## Architecture

```
Convert worker
  → ensure_checkpoint(on_progress=…)  # optional; UI via existing progress signal
  → separate / detect / estimate_bpm
App log buffer (ring) ← logging handlers + key pipeline/controller messages
Help → Debug Log dialog ← live view of buffer; Save writes .txt
Startup / quit cleanup unchanged (audiotomidi_* temps; model kept)
Windows frozen entry → multiprocessing freeze + no-console child flags
```

---

### 1. First-run model download UI

When Convert starts and the drumsep checkpoint is missing or fails SHA-256 verification:

1. Status text: `Downloading drum separation model (~167 MB)…`
2. Progress bar shows download percent when `Content-Length` is available; otherwise indeterminate (or percent omitted and bar still animates via staged updates if the toolkit requires a mode switch).
3. After successful download + verify, status returns to normal separation messages (`Separating stems on GPU…`, etc.).
4. If the model is already cached and valid, skip the download step entirely — no extra status flash.

**Implementation notes:**

- Extend `pipeline.separation.ensure_checkpoint` with an optional progress callback, e.g. `on_progress(downloaded: int, total: int | None) -> None`.
- The Convert job in `PipelineController.run_analysis` passes a callback that maps bytes → `report(message, percent)` on the existing worker progress signal (reuse the same UI path as today).
- Keep stderr prints for CLI/dev runs; GUI path must not rely on them.
- Download still happens at first Convert (not at launch).

---

### 2. PowerShell / console flash suppression

Goal: no visible console when Convert finishes (or during Convert) in the standalone exe.

Plan:

- Keep `console=False` in `build.spec`.
- On Windows frozen startup (`app/main.py`), call `multiprocessing.freeze_support()` and apply process-creation defaults so Torch/Demucs/multiprocessing children do not allocate a visible console (e.g. `CREATE_NO_WINDOW` / equivalent subprocess or `STARTUPINFO` defaults where we control spawn; document any library-level knobs used).
- Retain `ensure_stdio()` so libraries that write to stdout/stderr do not crash or force a new console.
- No “reveal in Explorer” or shell open on Convert/export complete (none exists today — this is suppression only).

Success criterion: converting a WAV in the built exe does not flash cmd/PowerShell/Windows Terminal.

---

### 3. Debug log dialog (option A)

- Menu: **Help → View debug log…** on the main window.
- Dialog contents:
  - Read-only, mono, auto-scrolling text view bound to an in-memory ring buffer (~2000 lines).
  - **Save log…** → `QFileDialog` → write `.txt`.
  - **Clear** → empty the buffer (and the view).
  - **Close**.
- Capture: app/controller/pipeline progress messages, errors, startup cleanup lines, model download lines, and standard `logging` output routed into the buffer.
- Quiet by default: no persistent file unless the user saves; Convert UX unchanged when the dialog is closed.
- Opening the dialog while Convert runs continues to append live lines.

---

### 4. Default window width

- On first `showEvent` fit, target width ≈ **1280px** (adjust `_STAGE_MIN_WIDTH` / fit math so total ≈ 1280 including rail + margins).
- Still clamp to `availableGeometry()`.
- Height behavior unchanged (fit rail content).
- Minimum size may rise slightly so a manually shrunk window stays usable, but must not exceed screen size.

---

### 5. Integer BPM only

- Replace `QDoubleSpinBox` with `QSpinBox`: range 40–240, step 1, no decimals.
- `estimate_bpm` / `normalize_bpm`: round to nearest integer and clamp; return type may remain `float` for MIDI writer compatibility but values are whole numbers (e.g. `120.0`).
- Status/session meta strings use integer formatting (e.g. `120 BPM`, not `120.0`).
- Export continues to pass spinbox value through `normalize_bpm` → `write_midi`.
- Update tempo unit tests for integer rounding behavior.

---

### 6. Cleanup policy (documented, unchanged)

| When | What |
|------|------|
| App quit | Delete current session `%TEMP%\audiotomidi_*` stems via `controller.cleanup` |
| Startup | Remove orphaned `audiotomidi_*` dirs left by crashed sessions |
| Clear All / new Convert | Discard previous session stems before creating a new temp dir |
| Model checkpoint | **Kept** under `models/drumsep/` (next to exe when frozen) for fast subsequent runs |

No code change required for cleanup in this work item beyond mentioning it in README if helpful.

---

## Error handling

- Download failure / SHA mismatch: Convert fails with a clear error string on the status label (existing `analysisFailed` path); partial `.th.part` cleaned as today.
- Log Save failure: show error in the dialog or status; do not crash.
- Console-suppression hooks must never prevent app launch if unavailable on a given Windows build.

## Testing

1. **Unit — `ensure_checkpoint` progress:** Mock URL response with known length; assert callback receives increasing `downloaded` and final dest exists (or use a tiny fixture file + monkeypatched URL).
2. **Unit — `normalize_bpm` / `estimate_bpm`:** Fractional inputs round to int; clamps unchanged; NaN → 120.
3. **Unit — log buffer:** Append past capacity; assert ring truncation; Save writes exact current contents.
4. **UI — BPM:** Spinbox is integer-only; analysis finished seeds an int.
5. **UI — window fit:** First show width ≈ 1280 on a large screen (or mocked available geometry).
6. **Manual — frozen exe:** First Convert shows download status/%; second Convert skips it; no console flash on finish.
7. **Existing:** Startup cleanup tests and tempo/MIDI tests still pass.

## Implementation order

1. Integer BPM (`tempo.py`, spinbox, tests, status strings).
2. Wider default window.
3. Checkpoint progress callback + Convert worker wiring.
4. In-memory log buffer + Help dialog + Save/Clear.
5. Windows console-suppression for frozen builds.
6. README timing/cleanup note tweak if needed.

## Out of scope / follow-ups

- Prefetch model on launch (Approach 2).
- Docked always-visible log panel.
- Auto-writing logs next to the exe.
- Investigating half/double BPM correction heuristics.
