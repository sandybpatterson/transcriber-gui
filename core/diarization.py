"""Optional speaker-diarization pass using pyannote.audio."""
from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import QObject, Signal

from core.engine import Segment

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

    # HF's gated-repo error text ("...must be authenticated... please log
    # in") reads the same whether the token itself is wrong or the token
    # is fine but just hasn't been granted access to that specific repo --
    # checked separately here so the message can actually tell those apart
    # instead of leaving it ambiguous.
    try:
        HfApi().whoami(token=token)
    except Exception as exc:  # noqa: BLE001 - surface Hugging Face's own reason
        raise RuntimeError(
            f"That token itself isn't valid: {exc}\n\n"
            "Double-check it was copied in full with no missing characters, "
            "or create a new one at https://huggingface.co/settings/tokens."
        ) from exc

    for repo_id in REQUIRED_REPOS:
        try:
            hf_hub_download(repo_id, "config.yaml", token=token)
        except Exception as exc:  # noqa: BLE001 - surface whatever Hugging Face actually said
            raise RuntimeError(
                f"Can't access {repo_id} yet: {exc}\n\n"
                f"Visit https://huggingface.co/{repo_id} while logged in and "
                "request/accept access, then test again."
            ) from exc


class HfTokenTestWorker(QObject):
    """Runs check_hf_token() on a background thread so the UI doesn't
    block on the network calls it makes."""

    succeeded = Signal()
    failed = Signal(str)

    def __init__(self, token: str):
        super().__init__()
        self.token = token

    def run(self) -> None:
        try:
            check_hf_token(self.token)
            self.succeeded.emit()
        except Exception as exc:  # noqa: BLE001 - message already formatted by check_hf_token
            self.failed.emit(str(exc))


def diarize_audio(audio_path: str, hf_token: Optional[str] = None) -> list[tuple[float, float, str]]:
    """Run pyannote's speaker-diarization pipeline, returning (start, end, speaker) turns."""
    # Same reasoning as transcribe_audio's model download: bound the Hub
    # fetch so a dead connection fails fast instead of hanging with no
    # reachable cancellation point.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

    from pyannote.audio import Pipeline

    token = hf_token or os.environ.get("HF_TOKEN")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )
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
