# TranscriberGUI — Handoff
### For Claude: read this first before doing anything else.

---

## What This Is

A Windows desktop app that converts video files into text transcripts using
a local, free, offline speech-to-text model (`faster-whisper`, a
CTranslate2-based reimplementation of OpenAI's Whisper). No cloud API, no
per-use cost, no privacy concerns — everything runs on the user's own
machine.

**Owner's own name for this may be Sandy B. Patterson / Trenton** — the same
person behind the other repos in this account (`sandybpatterson.github.io`,
`soda-bot`, `original-bug-book`, `beyond-ice-and-steam`). This project was
scaffolded standalone; **check with the owner before assuming it should stay
that way** — the original brief flagged it might later integrate with or
replace functionality in Soda Bot (a separate, unrelated text-to-speech
podcast-publishing tool in the `soda-bot` repo — the reverse direction of
what this app does). As of this writing, Soda Bot has no transcription
feature of its own to integrate with, so standalone is the right call for
now, but that's worth re-checking if this project picks back up later.

---

## Git Workflow

**Push directly to `main`.** No PR process has been used on this repo so
far — every commit to date went straight to `main`. Keep doing that unless
the owner asks for a PR.

---

## Status: Working, Tested, Not Yet Packaged

### What's built
Full pipeline, all wired together:
- `main.py` — entry point
- `ui/main_window.py` — drag-drop zone, browse button, model/language
  dropdowns, diarization checkbox, Start/Cancel buttons, progress bar,
  status label, results table, three export buttons
- `ui/results_view.py` — searchable, editable, timestamped transcript table
- `core/engine.py` — `extract_audio()` (ffmpeg via `imageio-ffmpeg`, auto-
  fetched, no PATH dependency), `transcribe_audio()` (faster-whisper),
  `TranscriptionWorker` (QThread-safe pipeline runner with cooperative
  cancellation)
- `core/diarization.py` — optional pyannote.audio speaker-labeling pass,
  gated behind an `HF_TOKEN` env var
- `exporters.py` — `.txt` / `.srt` / `.docx` output

### What's actually been tested (not just written)
This matters because a scaffold that only passed a syntax check is a
different thing from one that's been run. As of 2026-09-06, verified live,
offscreen, in a throwaway venv:
- Window and results-table construction
- All three exporters producing correct output from real `Segment` data
- `extract_audio()` against a real synthetic video (espeak-ng speech muxed
  into an ffmpeg test pattern) — produces a correct mono 16kHz WAV
- The Cancel button, end-to-end, against the two realistic hang shapes this
  app can actually produce: a stall between transcript segments (cooperative
  cancel), and a genuinely blocking call with no reachable checkpoint (a
  hung socket read, modeled with a real blocking `sleep()` since it behaves
  identically from the GIL's perspective) — force-stop path confirmed
  working
- The race between cancelling a stuck run and immediately starting a new
  one — confirmed the new run's results aren't clobbered by the old one's
  eventually-abandoned worker

**Not tested**: actual Whisper model inference and actual pyannote
diarization against real audio. The sandbox this was built in blocks
`huggingface.co` at the network policy level, so no model weights could
ever be downloaded there. The code against both libraries' documented APIs
is straightforward, but "the API is used correctly" and "verified against a
real model's actual output" are different claims — the first is what's true
here, not the second. If picking this up again, running a real transcription
against a real short video is the highest-value next check.

### What's not done
- **No `.exe` exists yet.** Nobody has run PyInstaller. Also: this can only
  be built to a real Windows `.exe` *on a Windows machine* — PyInstaller
  doesn't cross-compile, and this project has so far only ever been
  developed in a Linux sandbox. `build.spec` is written and untested.
- No CI, no automated test suite (the tests described above were one-off
  manual verification scripts in a scratch directory, not checked into the
  repo — nothing to run automatically, nothing wired up).
- No app icon, no code signing.

---

## Facebook Videos (added 2026-10-17)

URL input, batch URLs, and the "use existing captions" option all work for
Facebook as well as YouTube — `yt-dlp` already had the extractors; the
changes were around it:
- `url_fallback_name()` in `core/engine.py` — the old "everything after the
  last `/`" fallback turned `facebook.com/watch/?v=ID` into `?v=ID`; it now
  reads the `v` query parameter (fixes YouTube `watch?v=` names too).
- **Sign-in cookies** — a "Sign in for private videos" dropdown in the main
  window (`cookies_source` / `cookies_file` in QSettings), applied to every
  yt-dlp call via `ytdlp_auth_opts()`. Firefox only, deliberately: Chrome
  127+ and Edge use app-bound cookie encryption on Windows that yt-dlp
  can't read, so offering them would be a setting that silently fails.
  Chrome/Edge users export a `cookies.txt` instead.
- `explain_download_error()` appends a plain-English "you probably need to
  sign in" hint to yt-dlp's cryptic Facebook login errors.
- Caption language matching adds `xx_.*` patterns, since Facebook keys
  caption tracks by locale (`en_US`) where YouTube uses `en`.
- The channel watcher is still YouTube-only. Not touched.

**Not tested against real Facebook.** The sandbox's proxy blocks
facebook.com outright, so verification was offline only: helper functions,
yt-dlp extractor matching for each URL shape, `.srt` caption parsing, and
the main window building headless. First real-world check: one public
Facebook video, then one friends-only video with Firefox sign-in.

## A Real Bug Found by Actually Running It, and How It Was Fixed

Worth understanding before touching `core/engine.py` or
`ui/main_window.py` again, since the reasoning shapes both files:

**The bug:** the very first live end-to-end test hung for 98 minutes.
`faster-whisper`'s model download (from Hugging Face Hub, on first run) has
no reachable cancellation checkpoint from application code, and the UI had
no Cancel button at all — an indeterminate progress bar and a disabled
Start button, forever, if the network stalls. This is a realistic failure
mode for real users too: a flaky connection, or literally no internet the
first time someone opens an "offline" transcription tool.

**The fix, in two parts:**
1. `HF_HUB_DOWNLOAD_TIMEOUT` is now set (defaulting to 30s, `os.environ.
   setdefault` so it respects a value the user's already set) before both
   the transcription and diarization model downloads. This is the fix for
   the actual root cause — a dead connection now fails with a catchable
   exception in a bounded window instead of hanging at the source.
2. A Cancel button, for the general case (the user wants to abort a
   legitimately-running-but-slow transcription, not just a hung download).
   Cooperative: the worker checks a cancel flag between pipeline steps
   (after extraction, after each transcript segment, after diarization).
   For a call stuck with no reachable checkpoint, there's a force-stop
   fallback via `QThread.terminate()`.

**Two things learned the hard way while building the force-stop path, both
confirmed by direct measurement, not assumed — worth knowing before
"fixing" this code to look more conventional:**
- `QThread.wait()`, even called with an explicit bounded timeout (e.g.
  `wait(1000)`), can simply never return if the target thread has no
  cancellation point. An *unbounded* wait after `terminate()` is a common
  pattern elsewhere; here it would have frozen the GUI thread right along
  with the worker.
- `QObject.disconnect()` carries the exact same risk, against the exact
  same kind of thread. This is why the abandoned-worker safety net is an
  identity guard (`guarded()` in `_start_transcription`, checking "is the
  emitting worker still the current one?") rather than disconnecting the
  old worker's signals — disconnecting was the first approach tried, and it
  reintroduced the hang it was meant to prevent.

**Known residual limitation, left as a code comment, not silently
accepted:** if a worker is truly never yielding the GIL at all (a tight
pure-Python or non-releasing-native loop with zero I/O), even the UI
update in the force-stop path can be delayed behind it. This is an inherent
CPython single-GIL constraint, not fixable from the application side short
of moving transcription into a separate process (worth considering if this
project grows, not worth doing now). It doesn't come up in practice here:
`extract_audio()` is subprocess/I/O-bound and `faster-whisper`'s native
inference releases the GIL during computation the same way blocking I/O
does — both behave like the sleep-based test case above, which is
confirmed working.

---

## What To Do Next

1. **Get a real model downloaded and transcribing on a machine with normal
   internet access** — the biggest untested gap. Confirm actual Whisper
   output looks right on real speech, not just mocked segments.
2. **Build the `.exe` on an actual Windows machine** — `pip install
   pyinstaller && pyinstaller build.spec`, confirm it launches, confirm
   ffmpeg auto-fetch works from a clean machine with no prior Python
   environment.
3. Only after both of those: consider a UI polish pass (the layout works
   but has never been refined against real usage — e.g., the empty results
   table takes up a lot of vertical space before any transcript exists).
4. Revisit the Soda Bot integration question with the owner once this has
   seen real use — see the note at the top.

## Self-update from GitHub (added November 2026)

The app runs straight from this git clone (`C:\src	ranscriber-gui`, the
desktop shortcut uses its `venv`). **Check for updates** in the main window
(`core/updater.py`) runs `git fetch`, lists new commits, and on Yes does a
fast-forward `git pull`, re-runs `pip install -r requirements.txt` only if
that file changed, then relaunches Scribe and closes the old window.
- Greyed out while a transcription, a batch or a channel check is running
  (polled once a second); re-checked right before pulling and before
  restarting -- running work is never killed by an update.
- Refuses to update if this folder has uncommitted code changes.
- Workflow: every change is committed and pushed to
  `sandybpatterson/transcriber-gui`, so other copies can pick it up.
- The window title shows the running commit, e.g. `Scribe (fcb2553)`.

---

## Batch improvements (added November 2026)

The multi-URL batch flow ("Transcribe multiple URLs...") was reworked
around the real scenario of pasting several YouTube videos (or a playlist)
and walking away:
- A pasted YouTube playlist link expands into its videos
  (`is_playlist_url` / `expand_playlist_url` in `core/engine.py` -- only a
  `youtube.com/playlist` page expands; a watch URL carrying `list=` stays
  one video). Duplicate URLs are dropped.
- Titles (and lengths) look themselves up shortly after pasting -- the
  "Look up titles" button is now a manual re-check, and Start works even
  before the lookup runs (it starts automatically once titles are in).
- The queue table has a Length column; the batch saves .txt/.srt/.docx as
  picked in the setup dialog (was .txt only).
- Re-running a batch skips videos already saved in the output folder, so
  a cancelled batch resumes instead of redoing finished work.
- A summary dialog at the end lists Done / Skipped / Failed per video,
  with "Open output folder" and "Retry remaining" (failed + never-ran
  only). The status line also names the video currently running.
- Verified headless against real YouTube metadata (playlist expansion,
  title+duration lookup, dedupe) in a throwaway venv; the GUI itself was
  not launched there -- first real check should be a small batch on the
  Windows PC.

*Last updated: September 2026*
