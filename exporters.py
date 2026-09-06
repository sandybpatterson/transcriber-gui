"""Write transcript segments out as .txt, .srt, or .docx."""
from __future__ import annotations

from pathlib import Path

from core.engine import Segment


def _format_srt_timestamp(seconds: float) -> str:
    millis = round(seconds * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def export_txt(segments: list[Segment], output_path: str) -> None:
    lines = [
        f"[{segment.speaker}] {segment.text}" if segment.speaker else segment.text
        for segment in segments
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def export_srt(segments: list[Segment], output_path: str) -> None:
    lines = []
    for index, segment in enumerate(segments, start=1):
        prefix = f"[{segment.speaker}] " if segment.speaker else ""
        lines.append(str(index))
        lines.append(f"{_format_srt_timestamp(segment.start)} --> {_format_srt_timestamp(segment.end)}")
        lines.append(f"{prefix}{segment.text}")
        lines.append("")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def export_docx(segments: list[Segment], output_path: str) -> None:
    from docx import Document

    document = Document()
    document.add_heading(Path(output_path).stem, level=1)

    for segment in segments:
        timestamp = _format_srt_timestamp(segment.start)[:-4]
        speaker = f" {segment.speaker}:" if segment.speaker else ""
        paragraph = document.add_paragraph()
        run = paragraph.add_run(f"[{timestamp}]{speaker} ")
        run.bold = True
        paragraph.add_run(segment.text)

    document.save(output_path)
