from pathlib import Path

from datactrl.discovery.classify import classify_file
from datactrl.discovery.walk import iter_files
from datactrl.types import Modality

FIXTURE = Path(__file__).parent / "fixtures" / "messy"


def test_classify_extensions():
    assert classify_file(FIXTURE / "nested" / "labs.csv") == Modality.STRUCTURED
    assert classify_file(FIXTURE / "clinic_note.txt") == Modality.NOTE
    assert classify_file(FIXTURE / "nested" / "demographics.json") == Modality.STRUCTURED
    assert classify_file(FIXTURE / "nested" / "prose_note.json") == Modality.NOTE
    assert classify_file(FIXTURE / "junk" / "readme.bin") is None


def test_classify_skips_non_dicom_binaries(tmp_path: Path):
    svs = tmp_path / "slide.svs"
    svs.write_bytes(b"not-dicom")
    png = tmp_path / "preview.png"
    png.write_bytes(b"\x89PNG")
    assert classify_file(svs) is None
    assert classify_file(png) is None


def test_walk_skips_hidden(tmp_path: Path):
    visible = tmp_path / "a.txt"
    visible.write_text("hi")
    hidden_dir = tmp_path / ".cache"
    hidden_dir.mkdir()
    (hidden_dir / "secret.txt").write_text("nope")
    files = iter_files(tmp_path)
    assert files == [visible.resolve()]
