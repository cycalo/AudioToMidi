# Getting Started

Short guides for running AudioToMidi from the console and packaging a Windows `.exe`.

---

## 1. Run from the console

**Needs:** Windows, Python 3.11+ (3.13+ recommended), ~2 GB free disk.

```bash
git clone https://github.com/cycalo/AudioToMidi.git
cd AudioToMidi

python -m venv .venv
```

**Activate the venv** (pick the shell you use):

| Shell | Activate |
| ----- | -------- |
| **Git Bash / MINGW** | `source .venv/Scripts/activate` |
| **PowerShell** | `.\.venv\Scripts\Activate.ps1` |
| **cmd.exe** | `.venv\Scripts\activate.bat` |

Then:

```bash
pip install -r requirements.txt
python app/main.py
```

Or without activating, call the venv Python directly:

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe app/main.py
```

That opens the GUI. First Convert may download the Demucs checkpoint (~167 MB). Optional GPU (CUDA) speeds separation; CPU works but is slower.

**Tip:** Keep the venv activated (`(.venv)` in the prompt) whenever you run the app from source.

---

## 2. Build the Windows exe

**Needs:** Same setup as above (venv + `requirements.txt`, which includes PyInstaller).

From the repo root, with the venv activated:

```bash
pyinstaller build.spec
```

Or without activating:

```bash
.venv/Scripts/pyinstaller.exe build.spec
```

**Output:** `dist/AudioToMidi/AudioToMidi.exe`

This is a **one-folder** build. Ship or run the whole `dist/AudioToMidi/` folder — do not move only the `.exe`.

Rebuild after pulling so `mappings/` and `Preview Kit/` are bundled. First run of the packaged app still downloads the Demucs checkpoint on demand (cached under `models/drumsep/` next to the exe).
