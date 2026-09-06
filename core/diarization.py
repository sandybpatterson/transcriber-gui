"""Optional speaker-diarization pass using pyannote.audio."""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from PySide6.QtCore import QObject, Signal

from core.engine import Segment

logger = logging.getLogger(__name__)

TOKEN_CHECK_TIMEOUT_SECONDS = 30

# Both models are gated on Hugging Face (confirmed directly: downloading
# even their small config.yaml fails with GatedRepoError when unauthorized)
# -- segmentation-3.0 is a dependency pyannote's pipeline loads internally,
# so a token needs access to both, not just the one named in diarize_audio.
REQUIRED_REPOS = ("pyannote/speaker-diarization-3.1", "pyannote/segmentation-3.0")


def check_hf_token(token: str) -> None:
    """Verifies `token` can actually access both gated models this needs.

    Downloads just each repo's small config.yaml rather than the full
    pipeline (which would also pull in real model weights) -- enough to
    trigger the same gate check a full pipeline load would hit, much
    faster. Raises with a message naming the specific blocking repo and a
    link to it on success partial failure; returns normally if both are
    reachable.
    """
    from huggingface_hub import HfApi, hf_hub_download

    logger.info("Checking Hugging Face token (whoami)")
    # HF's gated-repo error text ("...must be authenticated... please log
    # in") reads the same whether the token itself is wrong or the token
    # is fine but just hasn't been granted access to that specific repo --
    # checked separately here so the message can actually tell those apart
    # instead of leaving it ambiguous.
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

    for repo_id in REQUIRED_REPOS:
        logger.info("Checking access to %s", repo_id)
        try:
            hf_hub_download(repo_id, "config.yaml", token=token)
        except Exception as exc:  # noqa: BLE001 - surface whatever Hugging Face actually said
            logger.warning("No access to %s: %s", repo_id, exc)
            raise RuntimeError(
                f"Can't access {repo_id} yet: {exc}\n\n"
                f"Visit https://huggingface.co/{repo_id} while logged in and "
                "request/accept access, then test again."
            ) from exc
        logger.info("Access to %s confirmed", repo_id)
    logger.info("Token check passed for all required repos")


class HfTokenTestWorker(QObject):
    """Runs check_hf_token() on a background thread so the UI doesn't
    block on the network calls it makes."""

    succeeded = Signal()
    failed = Signal(str)

    def __init__(self, token: str):
        super().__init__()
        self.token = token

    def run(self) -> None:
        # check_hf_token() ultimately calls huggingface_hub's whoami(),
        # which was found -- by direct testing -- to pass no timeout at
        # all to its underlying HTTP request, and isn't governed by
        # HF_HUB_DOWNLOAD_TIMEOUT either (that only covers the actual
        # download path). socket.setdefaulttimeout() doesn't help here
        # either: also confirmed directly, requests/urllib3 explicitly
        # sets timeout=None on the socket when none is given, which
        # overrides the global default. A hard external deadline via a
        # background thread is the only thing that actually bounds this.
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
                check_hf_token(self.token)
                result["ok"] = True
            except Exception as exc:  # noqa: BLE001 - message already formatted by check_hf_token
                result["error"] = str(exc)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=TOKEN_CHECK_TIMEOUT_SECONDS)

        if thread.is_alive():
            logger.warning("Token check timed out after %ds", TOKEN_CHECK_TIMEOUT_SECONDS)
            self.failed.emit(
                f"Timed out after {TOKEN_CHECK_TIMEOUT_SECONDS}s checking your token. "
                "Hugging Face may be slow or unreachable right now -- try again in a moment."
            )
        elif "error" in result:
            self.failed.emit(result["error"])
        else:
            self.succeeded.emit()


def diarize_audio(audio_path: str, hf_token: Optional[str] = None) -> list[tuple[float, float, str]]:
    """Run pyannote's speaker-diarization pipeline, returning (start, end, speaker) turns."""
    # Same reasoning as transcribe_audio's model download: bound the Hub
    # fetch so a dead connection fails fast instead of hanging with no
    # reachable cancellation point.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

    from pyannote.audio import Pipeline

    token = hf_token or os.environ.get("HF_TOKEN")
    logger.info("Loading pyannote diarization pipeline")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )
    logger.info("Running diarization on %s", audio_path)
    diarization = pipeline(audio_path)

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
