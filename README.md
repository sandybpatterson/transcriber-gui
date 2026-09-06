# TranscriberGUI

A Windows desktop app that converts video files into text transcripts using
a local, free AI speech-to-text model
([faster-whisper](https://github.com/SYSTRAN/faster-whisper), a
CTranslate2-based reimplementation of OpenAI's Whisper). Runs entirely
offline — no cloud API costs, no privacy concerns.

Drop in a video, pick a model size, and get a searchable, editable,
timestamped transcript you can export as plain text, subtitles, or a
formatted Word document. An optional speaker-diarization pass (via
[pyannote.audio](https://github.com/pyannote/pyannote-audio)) labels who
said what.

## Screenshot

*(coming soon)*

## Features

- Drag-and-drop or browse to select a video (`.mp4`, `.mov`, `.mkv`, `.avi`)
- Audio extraction via `ffmpeg` (fetched automatically on first run — no
  need to have it on your `PATH`)
- Transcription via `faster-whisper`, with a choice of model size
  (`tiny` / `base` / `small` / `medium` / `large-v3-turbo`) and language
  auto-detection
- Transcription runs on a background thread, so the UI never freezes, and
  shows live progress
- Scrollable, searchable, editable transcript panel with timestamps
- Export to `.txt` (plain), `.srt` (subtitles), or `.docx` (formatted
  document)
- Optional speaker diarization (checkbox) to label speakers in the output

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/sandybpatterson/transcriber-gui.git
cd transcriber-gui
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python main.py
```

`ffmpeg` does not need to be installed separately — it is fetched
automatically the first time the app extracts audio from a video.

Speaker diarization uses a gated Hugging Face model
(`pyannote/speaker-diarization-3.1`). To use it, accept the model's terms on
Hugging Face and set the `HF_TOKEN` environment variable to an access token
before checking the diarization box.

## Building a standalone .exe

```bash
pip install pyinstaller
pyinstaller build.spec
```

The finished executable lands in `dist/TranscriberGUI/`.

## Project layout

```
transcriber-gui/
├── main.py                 # entry point, launches QApplication
├── ui/
│   ├── main_window.py       # file picker, model/language dropdowns, start button, progress bar
│   └── results_view.py      # timestamped transcript panel, search, export buttons
├── core/
│   ├── engine.py            # ffmpeg extraction + faster-whisper, run in a QThread worker
│   └── diarization.py       # optional pyannote speaker-labeling pass
├── exporters.py             # writes .txt, .srt, .docx from transcript data
├── requirements.txt
└── build.spec               # PyInstaller config
```

## License

MIT — see [LICENSE](LICENSE).
