# Test fixtures

Drop **real full-kit drum stems** (`.wav`) here to exercise source separation.
Audio files in this folder are gitignored (they can be large or copyrighted), so
they stay local to your machine. When at least one `.wav` is present, the
real-fixture tests in [../test_separation.py](../test_separation.py) run
automatically; otherwise they skip.

## How to test separation

```bash
# DSP fallback (fast, no download, 5 stems)
python pipeline/separation.py tests/fixtures/your_kit.wav -o tests/fixtures/your_kit_stems --backend dsp

# Demucs drumsep (best quality, 4 stems, one-time ~167 MB download)
python pipeline/separation.py tests/fixtures/your_kit.wav -o tests/fixtures/your_kit_stems --backend demucs
```

Then listen to each stem and confirm it is identifiable by ear (the true
Phase 2 acceptance bar).

## Backend stem sets

- `demucs`: kick, snare, toms, cymbals (hi-hat is inside cymbals).
- `dsp`: kick, snare, toms, hihat, cymbals.

## Separation quality notes (log per stem / genre)

Fill this in as real stems are tested, since quality varies by genre and by drum
voice (kick separates cleanest; toms/cymbals are the weakest link).

| Stem file | Genre / kit | Backend | Notes (SNR, bleed, artifacts) |
|---|---|---|---|
| _example.wav_ | _rock_ | _demucs_ | _pending_ |

## Phase 3 transcription notes (stems -> GM MIDI)

Generate a MIDI from a separated-stems directory and play it over the original:

```bash
python pipeline/merge.py tests/fixtures/drums_1_stems -o tests/fixtures/drums_1.mid
```

Observations on the four demucs fixtures (default settings, bleed suppression off):

| Fixture | Events | kick(36) | snare(38) | toms(45/47/50) | cymbals(49) | tom k |
|---|---|---|---|---|---|---|
| drums_1 | 175 | 43 | 39 | 37/0/25 | 31 | 2 |
| drums_2 | 339 | 92 | 101 | 63/0/36 | 47 | 2 |
| drums_3 | 214 | 52 | 50 | 30/0/15 | 67 | 2 |
| drums_4 | 221 | 61 | 58 | 17/0/19 | 66 | 2 |

- All four resolved toms to k=2 (low 45 + high 50), so mid tom (47) is unused; the
  adaptive selector preferred 2 clusters over 3 in every case.
- Tom onset counts run high, consistent with cymbal/snare bleed into the toms stem
  (Section 7 known limitation). Enable `--bleed-suppression` or raise
  `--velocity-floor` to trim it once validated by ear.
