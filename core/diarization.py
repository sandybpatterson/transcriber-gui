"""Optional speaker-diarization pass using pyannote.audio."""
from __future__ import annotations

import logging
import os
import threading
from typing import Callable, Optional

from PySide6.QtCore import QObject, Signal

from core.engine import Segment

logger = logging.getLogger(__name__)

TOKEN_CHECK_TIMEOUT_SECONDS = 30

# All three are gated on Hugging Face and all three get pulled in by
# actually running a diarization pass, not just constructing the pipeline
# -- confirmed the hard way, one at a time, against real audio:
# segmentation-3.0 loads during pipeline construction, and
# speaker-diarization-community-1 (a clustering sub-step, "plda/xvec_
# transform.npz") only gets fetched once the pipeline actually processes
# a file. There may be more not yet hit by real usage; this list only
# reflects what's been found so far.
REQUIRED_REPOS = (
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "pyannote/speaker-diarization-community-1",
)


def verification_key(token: str) -> str:
    """Ties a "verified" cache entry to both the token AND the current
    set of required repos -- so adding a newly-discovered gated
    dependency (this has happened more than once) automatically
    invalidates a prior verification instead of leaving a stale
    "verified" state for a check that covered fewer repos than are now
    needed."""
    return token + "::" + ",".join(REQUIRED_REPOS)


def check_whoami(token: str) -> None:
    """Verifies `token` itself is a real, valid Hugging Face token.

    Split out from repo-access checks because HF's gated-repo error text
    ("...must be authenticated... please log in") reads the same whether
    the token itself is wrong or the token is fine but just hasn't been
    granted access to a specific repo -- checked separately so a failure
    message can actually tell those two cases apart.
    """
    from huggingface_hub import HfApi

    logger.info("Checking Hugging Face token (whoami)")
    try:
        HfApi().whoami(token=token)
    except Exception as exc:  # noqa: BLE001 - surface Hugging Face's own reason
        logger.warning("Token failed whoami check: %s", exc)
        raise RuntimeError(
            f"That token itself isn't valid: {exc}\n\n"
            "Double-check it was copied in full with no missing characters, "
            "or create a new one at https://huggingface.co/settings/tokens."
        ) from exc
    logger.info("Token passed whoami check")


def check_repo_access(repo_id: str, token: str) -> None:
    """Verifies `token` can actually access the gated repo `repo_id`.

    Downloads just the repo's small config.yaml rather than the full
    pipeline (which would also pull in real model weights) -- enough to
    trigger the same gate check a full pipeline load would hit, much
    faster.
    """
    from huggingface_hub import hf_hub_download

    logger.info("Checking access to %s", repo_id)
    try:
        hf_hub_download(repo_id, "config.yaml", token=token)
    except Exception as exc:  # noqa: BLE001 - surface whatever Hugging Face actually said
        logger.exception("Failed downloading config.yaml from %s", repo_id)
        raise RuntimeError(
            f"Can't access {repo_id} yet: {exc}\n\n"
            f"Visit https://huggingface.co/{repo_id} while logged in and "
            "request/accept access, then test again."
        ) from exc
    logger.info("Access to %s confirmed", repo_id)


def check_hf_token(token: str) -> None:
    """Verifies `token` can actually access every gated model this needs."""
    check_whoami(token)
    for repo_id in REQUIRED_REPOS:
        check_repo_access(repo_id, token)
    logger.info("Token check passed for all required repos")


class BackgroundCheckWorker(QObject):
    """Runs an arbitrary zero-argument callable on a background thread so
    the UI doesn't block on whatever network calls it makes, with a hard
    external deadline the callable itself can't be trusted to respect.

    Used for the whoami check, each individual repo-access check, and the
    combined all-repos check -- same shape, same failure mode (a stuck
    network call with no way to know from the outside), so the same
    protection applies uniformly rather than being bolted on per call site.
    """

    succeeded = Signal()
    failed = Signal(str)

    def __init__(self, fn: Callable[[], None], timeout: int = TOKEN_CHECK_TIMEOUT_SECONDS):
        super().__init__()
        self._fn = fn
        self._timeout = timeout

    def run(self) -> None:
        # Confirmed directly, twice: huggingface_hub's whoami() passes no
        # timeout at all to its underlying HTTP request (not governed by
        # HF_HUB_DOWNLOAD_TIMEOUT, which only covers the download path),
        # and socket.setdefaulttimeout() doesn't help either -- requests/
        # urllib3 explicitly sets timeout=None on the socket when none is
        # given, overriding the global default. A hard external deadline
        # via a background thread is the only thing that actually bounds
        # this reliably, regardless of what the callable does internally.
        #
        # Uses a plain daemon thread rather than
        # concurrent.futures.ThreadPoolExecutor -- also confirmed
        # directly: the executor registers a global atexit hook that
        # joins every thread it ever created regardless of an individual
        # shutdown(wait=False) call, so a still-stuck call would prevent
        # the whole app process from ever fully exiting, even after the
        # window is closed. A daemon thread carries no such join-on-exit
        # behavior -- it's simply abandoned.
        result: dict = {}

        def target() -> None:
            try:
                self._fn()
                result["ok"] = True
            except Exception as exc:  # noqa: BLE001 - message already formatted by the callable
                result["error"] = str(exc)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=self._timeout)

        if thread.is_alive():
            logger.warning("Background check timed out after %ds", self._timeout)
            self.failed.emit(
                f"Timed out after {self._timeout}s. Hugging Face may be slow "
                "or unreachable right now -- try again in a moment."
            )
        elif "error" in result:
            self.failed.emit(result["error"])
        else:
            self.succeeded.emit()


def _load_wav_as_tensor(path: str) -> dict:
    """Reads a 16-bit PCM WAV file into pyannote's documented in-memory
    audio format: {"waveform": (channel, time) tensor, "sample_rate": int}.

    Stdlib `wave` only understands PCM, which is exactly what
    extract_audio()/download_audio() always produce, so no general-purpose
    decoding is needed here.
    """
    import wave

    import numpy as np
    import torch

    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise RuntimeError(f"Expected 16-bit PCM WAV audio, got {sample_width * 8}-bit")

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).T
    else:
        samples = samples.reshape(1, -1)

    return {"waveform": torch.from_numpy(samples), "sample_rate": sample_rate}


def diarize_audio(
    audio_path: str,
    hf_token: Optional[str] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> list[tuple[float, float, str]]:
    """Run pyannote's speaker-diarization pipeline, returning (start, end, speaker) turns."""
    # Same reasoning as transcribe_audio's model download: bound the Hub
    # fetch so a dead connection fails fast instead of hanging with no
    # reachable cancellation point.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

    from pyannote.audio import Pipeline

    token = hf_token or os.environ.get("HF_TOKEN")
    logger.info("Loading pyannote diarization pipeline")
    # This step alone can take a minute or more the first time (loading
    # torch plus several neural network models with no natural progress
    # percentage to report) -- confirmed live: a user watching a static
    # "Identifying speakers..." message with no further detail couldn't
    # tell it apart from a hang. At least name what's actually happening.
    if on_progress:
        on_progress("Identifying speakers (loading model -- first run can take a minute)...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token,
    )
    logger.info("Running diarization on %s", audio_path)
    if on_progress:
        on_progress("Identifying speakers (analyzing audio)...")
    # Not passing audio_path directly: pyannote.audio now decodes audio via
    # torchcodec exclusively (its own io.py: "relies on torchcodec for
    # reading", no other backend), which itself needs real FFmpeg *shared
    # library* DLLs on the system -- the standalone ffmpeg.exe this app
    # bundles via imageio-ffmpeg doesn't provide those, so torchcodec's
    # native loader fails outright. Confirmed live. Loading the WAV
    # ourselves (already a clean 16kHz mono PCM file from extract_audio())
    # and passing pyannote's documented in-memory alternative -- a
    # {"waveform", "sample_rate"} dict -- sidesteps torchcodec entirely.
    output = pipeline(_load_wav_as_tensor(audio_path))
    # This version's pipeline returns a DiarizeOutput wrapper (also
    # carrying exclusive_speaker_diarization and speaker_embeddings) --
    # .speaker_diarization is the actual Annotation with .itertracks().
    diarization = output.speaker_diarization

    return [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]


def assign_speakers(
    segments: list[Segment],
    speaker_segments: list[tuple[float, float, str]],
) -> list[Segment]:
    """Label each transcript segment with whichever speaker turn overlaps it most."""
    for segment in segments:
        best_speaker = None
        best_overlap = 0.0
        for start, end, speaker in speaker_segments:
            overlap = min(segment.end, end) - max(segment.start, start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        segment.speaker = best_speaker
    return segments
