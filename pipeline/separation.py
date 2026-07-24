"""Source separation: split a full-kit drum WAV into per-voice stems.

Two backends:

- ``demucs``: the ``inagoy/drumsep`` Hybrid Demucs checkpoint, run via
  ``demucs-infer``. Produces 4 ML stems (kick, snare, toms, cymbals), then a
  hybrid DSP pass extracts ``hihat`` from the original mix (6–12 kHz band) so
  hi-hat onsets are not lumped into crash cymbals. The checkpoint is downloaded
  on first use and cached locally.
- ``dsp``: an in-house, ML-free fallback (HPSS + per-instrument frequency
  masking + a kick transient gate). No download, CPU-only. Produces 5 stems
  (kick, snare, toms, hihat, cymbals).

Each run writes a ``manifest.json`` describing the backend, device, sample rate,
and the stem set produced, so later stages don't have to guess which stems exist.

TODO: Phase 3 — consume these stems for per-stem onset detection and tom pitch
clustering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import librosa
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.paths import repo_root  # noqa: E402

# --- drumsep checkpoint (inagoy/drumsep, Hybrid Demucs) --------------------
DRUMSEP_SIGNATURE = "49469ca8"
DRUMSEP_FILENAME = f"{DRUMSEP_SIGNATURE}.th"
DRUMSEP_URL = os.environ.get(
    "AUDIOTOMIDI_DRUMSEP_URL",
    "https://huggingface.co/NeoPy/UVR/resolve/main/drumsep/model.th",
)
DRUMSEP_SHA256 = "aefaa8543c9b9c75e22f5f32b53ab86dfe416457849af1383ff1aef83401423f"


def _models_dir() -> Path:
    """Writable checkpoint cache: next to the .exe when frozen, else repo models/."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "models" / "drumsep"
    return repo_root() / "models" / "drumsep"


MODELS_DIR = _models_dir()

# The checkpoint's internal source names are Spanish; translate to our English ones.
_SOURCE_TRANSLATION = {
    "bombo": "kick",
    "redoblante": "snare",
    "platillos": "cymbals",
    "toms": "toms",
}

DEMUCS_STEMS = ("kick", "snare", "toms", "cymbals")
# Effective stem set after the hybrid hi-hat DSP post-pass on demucs runs.
DEMUCS_OUTPUT_STEMS = ("kick", "snare", "toms", "hihat", "cymbals")
DSP_STEMS = ("kick", "snare", "toms", "hihat", "cymbals")

DEFAULT_BACKEND = "demucs"

# DSP STFT parameters.
_N_FFT = 2048
_HOP_LENGTH = 512


# --------------------------------------------------------------------------
# Device / checkpoint helpers
# --------------------------------------------------------------------------
def resolve_device(device: str = "auto") -> str:
    """Resolve ``"auto"`` to ``"cuda"`` when available, else ``"cpu"``."""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def device_options() -> list[tuple[str, str]]:
    """Return ``(value, label)`` pairs for GUI/CLI device selection.

    Always includes ``auto`` and ``cpu``. Adds ``cuda`` when a CUDA GPU is
    available, with the detected device name in the label when possible.
    """
    options: list[tuple[str, str]] = [
        ("auto", "Auto (use GPU if available)"),
        ("cpu", "CPU"),
    ]
    try:
        import torch

        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = "CUDA"
            options.append(("cuda", f"GPU — {gpu_name}"))
    except Exception:
        pass
    return options


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_checkpoint(
    *,
    verify: bool = True,
    on_progress: Optional[Callable[[int, Optional[int]], None]] = None,
) -> Path:
    """Download and cache the drumsep checkpoint, returning its local path.

    Downloads only on first use (into ``models/drumsep/49469ca8.th``) and verifies
    the SHA-256 unless ``verify=False``. The download URL can be overridden with
    the ``AUDIOTOMIDI_DRUMSEP_URL`` environment variable.

    ``on_progress(downloaded_bytes, total_bytes_or_None)`` is invoked during
    download when provided (skipped entirely when the checkpoint is already
    cached and valid).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / DRUMSEP_FILENAME

    if dest.exists():
        if not verify or _sha256(dest) == DRUMSEP_SHA256:
            return dest
        # Corrupt/partial cache: remove and re-download.
        dest.unlink()

    tmp = dest.with_suffix(".th.part")
    print(
        f"Downloading drumsep checkpoint (~167 MB) from {DRUMSEP_URL} ...",
        file=sys.stderr,
    )
    request = urllib.request.Request(DRUMSEP_URL, headers={"User-Agent": "AudioToMidi/0.2"})
    with urllib.request.urlopen(request) as response, open(tmp, "wb") as out:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header else None
        downloaded = 0
        next_report = 10 * (1 << 20)
        if on_progress is not None:
            on_progress(0, total)
        for chunk in iter(lambda: response.read(1 << 20), b""):
            out.write(chunk)
            downloaded += len(chunk)
            if on_progress is not None:
                on_progress(downloaded, total)
            if downloaded >= next_report:
                pct = f" ({downloaded * 100 // total}%)" if total else ""
                print(f"  {downloaded >> 20} MB{pct}", file=sys.stderr)
                next_report += 10 * (1 << 20)

    if verify:
        actual = _sha256(tmp)
        if actual != DRUMSEP_SHA256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                "drumsep checkpoint SHA-256 mismatch: "
                f"expected {DRUMSEP_SHA256}, got {actual}. "
                "Set AUDIOTOMIDI_DRUMSEP_URL to a trusted mirror."
            )
    tmp.replace(dest)
    return dest


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------
def _read_audio_tensor(path: str):
    """Read audio as a torch tensor of shape (channels, samples) plus its sample rate."""
    import torch

    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (samples, channels)
    wav = torch.from_numpy(data.T).contiguous()  # (channels, samples)
    return wav, int(sr)


def _write_stem(out_dir: Path, name: str, samples: np.ndarray, sr: int) -> Path:
    """Write a stem WAV. ``samples`` is (channels, n) or (n,) float32."""
    path = out_dir / f"{name}.wav"
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 2:
        array = array.T  # soundfile wants (n, channels)
    sf.write(str(path), array, sr)
    return path


# --------------------------------------------------------------------------
# Demucs backend
# --------------------------------------------------------------------------
def _separate_demucs(
    input_path: str,
    out_dir: Path,
    *,
    device: str,
    shifts: int,
    overlap: float,
    on_checkpoint_progress: Optional[Callable[[int, Optional[int]], None]] = None,
) -> Dict[str, Path]:
    import torch
    from demucs_infer.apply import apply_model
    from demucs_infer.audio import convert_audio
    from demucs_infer.pretrained import get_model

    ensure_checkpoint(on_progress=on_checkpoint_progress)
    model = get_model(DRUMSEP_SIGNATURE, repo=MODELS_DIR)
    model.to(device)
    model.eval()

    wav, sr = _read_audio_tensor(input_path)
    wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)

    # Per-track normalization (matches demucs' own inference path).
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)

    with torch.no_grad():
        estimates = apply_model(
            model,
            wav[None],
            device=device,
            shifts=shifts,
            split=True,
            overlap=overlap,
            progress=True,
        )[0]
    estimates = estimates * (ref.std() + 1e-8) + ref.mean()

    stems: Dict[str, Path] = {}
    for source_name, audio in zip(model.sources, estimates):
        english = _SOURCE_TRANSLATION.get(source_name.lower(), source_name.lower())
        stems[english] = _write_stem(
            out_dir, english, audio.cpu().numpy(), model.samplerate
        )
    return stems


def _band_mask(freqs: np.ndarray, low: float, high: float, roll: float = 20.0) -> np.ndarray:
    """Soft band mask over ``freqs`` with raised-cosine roll-offs at the edges."""
    mask = np.zeros_like(freqs, dtype=np.float64)
    inside = (freqs >= low) & (freqs <= high)
    mask[inside] = 1.0

    lo_edge = (freqs >= low - roll) & (freqs < low)
    mask[lo_edge] = 0.5 * (1.0 + np.cos(np.pi * (low - freqs[lo_edge]) / roll))
    hi_edge = (freqs > high) & (freqs <= high + roll)
    mask[hi_edge] = 0.5 * (1.0 + np.cos(np.pi * (freqs[hi_edge] - high) / roll))
    return mask


def _envelope_follow(x: np.ndarray, attack: int, release: int) -> np.ndarray:
    """One-pole attack/release envelope follower over a 0..1 signal."""
    out = np.zeros_like(x)
    prev = 0.0
    a_coeff = 1.0 / max(1, attack)
    r_coeff = 1.0 / max(1, release)
    for i, value in enumerate(x):
        coeff = a_coeff if value > prev else r_coeff
        prev = prev + coeff * (value - prev)
        out[i] = prev
    return out


def _hihat_band_mask(freqs: np.ndarray) -> np.ndarray:
    """Hi-hat band (6–12 kHz), shared by DSP and hybrid demucs post-pass."""
    return _band_mask(freqs, 6000, 12000)


def _extract_hihat_stem(y: np.ndarray, sr: int) -> np.ndarray:
    """Isolate hi-hat energy from a mono mix via STFT band masking."""
    n = len(y)
    stft = librosa.stft(y, n_fft=_N_FFT, hop_length=_HOP_LENGTH)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_N_FFT)
    spec = _hihat_band_mask(freqs)[:, None] * stft
    return librosa.istft(spec, hop_length=_HOP_LENGTH, length=n).astype(np.float32)


def _write_hihat_stem(input_path: str, out_dir: Path) -> Path:
    """Extract and write ``hihat.wav`` from the original full-kit WAV."""
    y, sr = librosa.load(input_path, sr=None, mono=True)
    return _write_stem(out_dir, "hihat", _extract_hihat_stem(y, sr), sr)


# --------------------------------------------------------------------------
# DSP backend (cukas/drumsep-style, ML-free)
# --------------------------------------------------------------------------
def _separate_dsp(input_path: str, out_dir: Path) -> Dict[str, Path]:
    y, sr = librosa.load(input_path, sr=None, mono=True)
    n = len(y)

    stft = librosa.stft(y, n_fft=_N_FFT, hop_length=_HOP_LENGTH)
    harmonic, percussive = librosa.decompose.hpss(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_N_FFT)

    kick_band = _band_mask(freqs, 20, 100)
    snare_band = np.maximum(_band_mask(freqs, 150, 300), _band_mask(freqs, 2000, 4000))
    snare_low = _band_mask(freqs, 150, 300)
    hihat_band = _hihat_band_mask(freqs)
    # Cymbals cover a broad high range but exclude the hi-hat sub-band.
    cymbals_band = _band_mask(freqs, 3000, 16000) * (1.0 - hihat_band)
    # Toms sit in the low-mids but avoid the kick and snare fundamentals.
    toms_band = _band_mask(freqs, 80, 400) * (1.0 - kick_band) * (1.0 - snare_low)

    def _istft(mask: np.ndarray, source: np.ndarray) -> np.ndarray:
        spec = mask[:, None] * source
        return librosa.istft(spec, hop_length=_HOP_LENGTH, length=n).astype(np.float32)

    kick_spec = kick_band[:, None] * percussive
    energy = np.sqrt((np.abs(kick_spec) ** 2).sum(axis=0))
    if energy.max() > 0:
        energy = energy / energy.max()
    gate = _envelope_follow(energy, attack=3, release=8)
    kick = librosa.istft(kick_spec * gate[None, :], hop_length=_HOP_LENGTH, length=n).astype(
        np.float32
    )

    stems_audio = {
        "kick": kick,
        "snare": _istft(snare_band, percussive),
        "toms": _istft(toms_band, percussive),
        "hihat": _extract_hihat_stem(y, sr),
        "cymbals": _istft(cymbals_band, stft),
    }

    stems: Dict[str, Path] = {}
    for name in DSP_STEMS:
        stems[name] = _write_stem(out_dir, name, stems_audio[name], sr)
    return stems


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def separate(
    input_path: str,
    out_dir: str,
    *,
    backend: Optional[str] = None,
    device: str = "auto",
    shifts: int = 0,
    overlap: float = 0.25,
    on_checkpoint_progress: Optional[Callable[[int, Optional[int]], None]] = None,
) -> dict:
    """Separate a full-kit drum WAV into per-voice stems.

    Args:
        input_path: Path to the input WAV.
        out_dir: Directory to write stem WAVs and ``manifest.json`` into.
        backend: ``"demucs"`` (default) or ``"dsp"``. Falls back to the
            ``AUDIOTOMIDI_SEPARATION_BACKEND`` env var, then ``"demucs"``.
        device: ``"auto"``, ``"cpu"``, or ``"cuda"`` (demucs backend only).
        shifts: Demucs shift-trick averaging count (0 = fastest).
        overlap: Demucs segment overlap fraction.
        on_checkpoint_progress: Optional callback for drumsep model download
            progress ``(downloaded_bytes, total_bytes_or_None)``.

    Returns:
        A manifest dict: ``{"backend", "device", "sample_rate", "source",
        "stems": {name: path}}``.
    """
    backend = backend or os.environ.get("AUDIOTOMIDI_SEPARATION_BACKEND", DEFAULT_BACKEND)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if backend == "demucs":
        resolved_device = resolve_device(device)
        stems = _separate_demucs(
            input_path,
            out_path,
            device=resolved_device,
            shifts=shifts,
            overlap=overlap,
            on_checkpoint_progress=on_checkpoint_progress,
        )
        stems["hihat"] = _write_hihat_stem(input_path, out_path)
    elif backend == "dsp":
        resolved_device = "cpu"
        stems = _separate_dsp(input_path, out_path)
    else:
        raise ValueError(f"Unknown backend: {backend!r} (expected 'demucs' or 'dsp')")

    # All stems share the sample rate of the written files.
    sample_rate = int(sf.info(str(next(iter(stems.values())))).samplerate)

    manifest = {
        "backend": backend,
        "device": resolved_device,
        "sample_rate": sample_rate,
        "source": Path(input_path).name,
        "stems": {name: path.name for name, path in stems.items()},
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Return absolute stem paths for immediate programmatic use.
    manifest_result = dict(manifest)
    manifest_result["stems"] = {name: str(path) for name, path in stems.items()}
    return manifest_result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a full-kit drum WAV into per-voice stems (Phase 2)."
    )
    parser.add_argument("input", type=Path, help="Input full-kit drum WAV.")
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input_stem>_stems next to the input).",
    )
    parser.add_argument(
        "--backend",
        choices=("demucs", "dsp"),
        default=None,
        help="Separation backend (default: demucs, or AUDIOTOMIDI_SEPARATION_BACKEND).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device for the demucs backend: auto | cpu | cuda.",
    )
    parser.add_argument(
        "--shifts",
        type=int,
        default=0,
        help="Demucs shift-trick averaging count (0 = fastest).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Demucs segment overlap fraction.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if not args.input.exists():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2

    out_dir = args.out_dir or args.input.with_name(f"{args.input.stem}_stems")
    manifest = separate(
        str(args.input),
        str(out_dir),
        backend=args.backend,
        device=args.device,
        shifts=args.shifts,
        overlap=args.overlap,
    )

    print(
        f"Separated '{args.input.name}' with backend={manifest['backend']} "
        f"(device={manifest['device']}, sr={manifest['sample_rate']}) -> {out_dir}"
    )
    for name, path in manifest["stems"].items():
        print(f"  {name}: {Path(path).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
