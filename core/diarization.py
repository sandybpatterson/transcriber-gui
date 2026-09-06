"""Optional speaker-diarization pass using pyannote.audio."""
from __future__ import annotations

import os
from typing import Optional

from core.engine import Segment


def diarize_audio(audio_path: str, hf_token: Optional[str] = None) -> list[tuple[float, float, str]]:
    """Run pyannote's speaker-diarization pipeline, returning (start, end, speaker) turns."""
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
