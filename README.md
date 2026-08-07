# med-datactrl

Local medical data access harness: open a filesystem root, auto-discover DICOM / structured data / clinical notes / NIfTI, and fetch bounded slices for humans or LLM tools.

See [DESIGN.md](DESIGN.md) for scope.

## Quick start

```python
from datactrl import Case

case = Case.open("/path/to/case")
report = case.discover()
print(report.to_dict())  # case_id, resources, counts, unknown

# Progressive disclosure: describe → fetch(view=...)
for ref in report.resources:
    print(case.describe(ref.id)["views"])

# Structured rows
for table in report.structured:
    print(case.fetch(table.id, view="rows", limit=10))

# Notes
for note in report.notes:
    print(case.fetch(note.id, view="text", max_chars=2000))

# DICOM metadata / preview (any slice_index)
for series in report.dicom:
    print(case.describe(series.id)["links"])  # structural SEG↔CT etc.
print(case.fetch(report.dicom[0].id, view="metadata"))
print(case.fetch(report.dicom[0].id, view="preview", slice_index=0))
print(case.dicom_groups("CT,SEG"))

# NIfTI volumes (always discovered by .nii / .nii.gz)
for volume in report.nifti:
    print(case.fetch(volume.id, view="metadata"))
    print(case.fetch(volume.id, view="preview", slice_index=0))
```

### Optional NIfTI scan/mask patterns

NIfTI discovery does not require a folder layout. When your dataset uses consistent path templates, pass `NiftiCrawlOptions` to enrich matches and create mask→scan links via [med-imagetools `niftiindex`](https://github.com/bhklab/med-imagetools/pull/438):

```python
from datactrl import Case, NiftiCrawlOptions

case = Case.open(
    "/path/to/case",
    nifti=NiftiCrawlOptions(
        scan_name_pattern="images/{PatientID}_{Modality}.nii.gz",
        mask_name_pattern="masks/{PatientID}_{Modality}_{ROI}.nii.gz",
        # Writes index/cache under this directory (default: <root>/.datactrl/nifti)
        output_dir="/tmp/nifti-index",
        force=True,
    ),
)
for volume in case.report.nifti:
    print(volume.path, volume.meta.get("file_type"), case.describe(volume.id)["links"])
```

Unmatched NIfTI files remain first-class resources without crawl metadata. Crawl artifacts (`index.csv`, `crawl_cache.json`) are written under `output_dir/<dataset_name>/`.

> **Dependency note:** NIfTI crawl support currently pins `med-imagetools` to commit [`ccaa96f`](https://github.com/bhklab/med-imagetools/commit/ccaa96f5622e0da19f41c37b0cfa5651a9bace9f) on branch `JoshuaSiraj/nifti-index` ([PR #438](https://github.com/bhklab/med-imagetools/pull/438)). Switch back to a PyPI release once that lands.

## Links

Resources can be connected in two ways:

| Source | Origin | Persisted? |
|--------|--------|------------|
| `structural` | Adapters (e.g. DICOM SEG→CT references, NIfTI mask→scan from optional patterns) | Yes — derived from files |
| `asserted` | User or agent via `assert_link` | No — session-only overlay on the open `Case` |

Structural links appear automatically after discovery (see `case.describe(series_id)["links"]` for DICOM). Asserted links let you record cross-resource relationships the harness did not infer — for example, that a note mentions a lab table, or that two findings support the same hypothesis.

```python
case = Case.open("/path/to/case")
note = case.report.notes[0]
table = case.report.structured[0]

# Record a relationship between two known resources
link = case.assert_link(note.id, table.id, relation="mentions")
# → {"from_id": "...", "to_id": "...", "relation": "mentions", "source": "asserted", "meta": {}}

# Asserted links are merged into describe() for either endpoint
case.describe(note.id)["links"]   # includes the link above
case.links(note.id)               # structural + asserted for one resource
case.links()                      # case-wide links (DICOM/NIfTI structural + all asserted)
```

Notes:

- Both `from_id` and `to_id` must already exist in the discovery index; unknown ids raise `ResourceNotFound`.
- `relation` is a free-form label (`mentions`, `supports`, `same_patient`, …).
- Optional `meta={...}` attaches extra key/value data to the link.
- Asserted links are **not written to disk** — they live only for the lifetime of the `Case` object (or tool session). Re-open the case and they are gone.

Via tool dispatch (same semantics, JSON in/out):

```python
from datactrl.tools import dispatch

dispatch("assert_link", {
    "root": "/path/to/case",
    "from_id": "note:abc123",
    "to_id": "structured:def456",
    "relation": "mentions",
})
dispatch("links", {"root": "/path/to/case", "resource_id": "note:abc123"})
```

## Tool dispatch

```python
from datactrl.tools import TOOL_DEFINITIONS, dispatch

print(dispatch("discover", {"root": "/path/to/case"}))
# Then: describe / fetch / links / assert_link / dicom_tree / dicom_groups
```

## Install

```bash
pixi install
# or
pip install -e .
```

## Tests

```bash
pixi run pytest
# or
pytest
```
