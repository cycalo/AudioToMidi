# BPM Sync for MIDI Export — Design Spec

**Date:** 2026-07-23  
**Status:** Draft — awaiting user review  
**Problem:** Exported MIDI plays too slow/fast in a DAW when the project BPM differs from the hardcoded 120 BPM stamped into the file.

## Background

Events are stored as absolute seconds. `pretty_midi` converts those seconds to MIDI ticks using the file’s initial tempo. Many DAWs then schedule those ticks at the **project** tempo, ignoring or overriding the file tempo for playback speed.

If the file was written at 120 BPM but the session is 84 BPM:

- 1 second of audio → 2 beats of ticks at write time (120 BPM)
- Those 2 beats take ~1.43 s at 84 BPM → MIDI plays slow vs the drum track

Writing the MIDI with the correct BPM (e.g. 84) produces tick positions that match wall-clock time when the DAW is at that same BPM.

## Goal

Auto-detect BPM from the loaded drum audio, show an editable BPM control, and stamp that value into the MIDI on export. Do **not** quantize note times.

## Non-goals

- Quantization / snap-to-grid
- Variable tempo maps
- Time-signature meta events
- Time-stretching audio
- Changing onset detection timing

## Architecture

```
WAV → separate → detect onsets → events (seconds)
                ↘ estimate_bpm(wav) → AnalysisState.detected_bpm
UI BPM spinbox ← seeded from detected_bpm (user may override)
Save MIDI → write_midi(events, tempo=ui_bpm)
```

`write_midi(..., tempo=)` already exists. The gap is detection + UI wiring: the GUI currently always calls `export_midi` with the default 120.

## Components

### 1. `pipeline/tempo.py` (new)

Single responsibility: estimate a constant BPM from audio.

```python
def estimate_bpm(audio_path: PathLike, *, sr: Optional[int] = None) -> float:
    """Return estimated tempo in BPM (clamped to a sensible range)."""
```

**Implementation notes:**

- Use `librosa.beat.beat_track` on the source WAV (already a drum stem — good signal for tempo).
- Prefer the scalar/global tempo return; coerce array-like returns to a single float (`float(np.atleast_1d(tempo)[0])`).
- Round to 1 decimal place for display (e.g. `84.0`).
- Clamp to `[40.0, 240.0]`. If detection fails or returns NaN/empty, fall back to `120.0`.
- **No half/double-tempo correction in v1.** Librosa may return ~2× or ~½ the musical tempo; the editable BPM field is the escape hatch. Add heuristics later only if real tracks systematically fail.
- Keep the function pure and unit-testable (no Qt).

Helper:

```python
def normalize_bpm(bpm: float, *, default: float = 120.0) -> float:
    """Clamp and sanitize a user- or detector-provided BPM."""
```

### 2. `AnalysisState` (`app/controller.py`)

Add:

```python
detected_bpm: float = 120.0
```

Set during `run_analysis` after separation (or in parallel with onset detection on the original WAV). Sensitivity re-runs keep the same `detected_bpm` (tempo of the audio does not change when onset thresholds change).

### 3. Controller export

```python
def export_midi(self, output_path: str, *, tempo: float) -> None:
    write_midi(events, output_path, tempo=normalize_bpm(tempo))
```

Require an explicit `tempo` from the UI (no silent 120 default in the GUI path).

**CLI:** Out of scope for this change. Existing `--tempo` flags keep working; auto-detect there can be a follow-up.

### 4. UI (`app/ui/main_window.py`)

Add a BPM row near the sensitivity controls (visible after analysis):

- Label: `BPM`
- `QDoubleSpinBox`: range 40–240, step 0.1, decimals 1
- Enabled when `AnalysisState` exists and UI is not busy
- Seeded from `state.detected_bpm` when analysis finishes
- On Clear / new Convert: reset to disabled (or 120 until new detection completes)
- Tooltip: explain that this must match the DAW project tempo so MIDI plays in time; auto-detected from the audio, editable

On Save:

```python
self.controller.export_midi(path, tempo=self.bpm_spin.value())
```

Status text after convert may mention the detected BPM briefly, e.g.  
`Detected N events · BPM ~84.0 (editable). Review, then Save MIDI.`

### 5. Docs / comments

Update `pipeline/midi_writer.py` docstring: tempo is **not** cosmetic. It determines tick encoding; if it does not match the DAW project tempo, playback speed will be wrong even though note times were absolute seconds in memory.

## Data flow

| Step | What happens |
|------|----------------|
| Convert | Separate stems; detect onsets; `estimate_bpm(wav)` → `AnalysisState.detected_bpm` |
| Analysis finished | UI sets spinbox to `detected_bpm` |
| Sensitivity change | Re-detect onsets only; BPM spinbox value left as-is (user override preserved) |
| User edits BPM | Spinbox value used on next export only; does not re-run detection |
| Save MIDI | `write_midi(..., tempo=spinbox)` |
| Clear All | Clear state; disable BPM control |

## Error handling

| Case | Behavior |
|------|----------|
| `beat_track` fails / empty | Fall back to 120.0; still allow user edit |
| BPM outside 40–240 | Clamp via `normalize_bpm` |
| Export with no state | Existing guard (Save disabled) |
| Invalid spinbox (Qt prevents) | Spinbox range enforces bounds |

Detection errors must not fail the whole Convert job: wrap `estimate_bpm` so a failure logs/falls back rather than aborting onset detection.

## Testing

1. **Unit — `estimate_bpm` / `normalize_bpm`**
   - Synthetic click track at a known BPM (e.g. 120) → estimate within ±2 BPM, **or** exactly 2×/½× that tempo (allowed without auto-correction).
   - `normalize_bpm(30)` → 40; `normalize_bpm(300)` → 240; NaN → 120.

2. **Unit — MIDI tempo stamp**
   - `write_midi` with `tempo=84` → `pretty_midi` reports ~84 BPM initial tempo; note start times still match absolute seconds when read back via pretty_midi.

3. **Regression — tick speed vs DAW mismatch (documented test)**
   - Same events written at 120 vs 84: tick positions differ by the tempo ratio; seconds round-trip via pretty_midi remains correct for each file’s own tempo map.
   - Optional: assert microseconds-per-quarter ≈ `60e6 / bpm`.

4. **Controller / UI (lightweight)**
   - `AnalysisState` carries `detected_bpm`.
   - `export_midi` passes through the provided tempo (mock `write_midi` if existing controller tests do that).

## Acceptance criteria

- After Convert, UI shows an auto-detected BPM the user can edit.
- Saving MIDI with BPM set to the track tempo (e.g. 84) produces a file whose tempo meta matches that value.
- Importing that MIDI into a DAW session at the same BPM plays at the same speed as the source drum audio (wall-clock aligned).
- Note onset times are unchanged by BPM (no quantization).
- Convert still succeeds if tempo detection fails (fallback 120 + editable field).

## Implementation sketch (ordered)

1. Add `pipeline/tempo.py` + tests.
2. Wire `estimate_bpm` into `run_analysis` / `AnalysisState`.
3. Add BPM spinbox to main window; seed on analysis finished; pass on save.
4. Fix `midi_writer` docstring; extend existing MIDI tests for non-120 tempo.
5. Manual check: export at 84, import into DAW at 84, confirm speed matches.

## Future (explicitly deferred)

- Optional quantization strength / grid resolution
- Tempo map for drifting performances
- Time signature meta
- “Detect again” button separate from Convert
