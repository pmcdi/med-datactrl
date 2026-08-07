from pathlib import Path

from datactrl.discovery.classify import classify_file
from datactrl.discovery.walk import iter_files
from datactrl.types import Modality

FIXTURE = Path(__file__).parent / "fixtures" / "messy"


def test_classify_extensions():
    assert classify_file(FIXTURE / "nested" / "labs.csv") == Modality.STRUCTURED
    assert classify_file(FIXTURE / "clinic_note.txt") == Modality.NOTE
    assert (
        classify_file(FIXTURE / "nested" / "demographics.json") == Modality.STRUCTURED
    )
    assert classify_file(FIXTURE / "nested" / "prose_note.json") == Modality.NOTE
    assert classify_file(FIXTURE / "junk" / "readme.bin") is None
    assert classify_file(Path("scan.nii")) == Modality.NIFTI
    assert classify_file(Path("scan.nii.gz")) == Modality.NIFTI
    assert classify_file(Path("Scan.NII.GZ")) == Modality.NIFTI
    assert classify_file(Path("archive.tar.gz")) is None


def test_walk_skips_hidden(tmp_path: Path):
    visible = tmp_path / "a.txt"
    visible.write_text("hi")
    hidden_dir = tmp_path / ".cache"
    hidden_dir.mkdir()
    (hidden_dir / "secret.txt").write_text("nope")
    files = iter_files(tmp_path)
    assert files == [visible.resolve()]
