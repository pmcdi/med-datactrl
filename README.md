# med-datactrl

Local medical data access harness: open a filesystem root, auto-discover DICOM / structured data / clinical notes, and fetch bounded slices for humans or LLM tools.

See [DESIGN.md](DESIGN.md) for scope.

## Quick start

```python
from datactrl import Case

case = Case.open("/path/to/case")
report = case.discover()
print(report.to_inventory())  # compact ids/labels/counts for LLM tools
print(report.to_dict())       # full resource list

# Progressive disclosure: describe → fetch(view=...)
for ref in report.resources:
    print(case.describe(ref.id)["views"])

# Structured rows
for table in report.structured:
    print(case.fetch(table.id, view="rows", limit=10))

# Notes (meta: note_date + confidence/flags, note_type, text_quality)
for note in report.notes:
    print(case.fetch(note.id, view="text", max_chars=2000))
    print(case.fetch(note.id, view="sections"))                     # headers + char offsets
    print(case.fetch(note.id, view="span", start=0, end=200, context_chars=100))
print(case.notes_search("portal vein thrombosis", limit=8))         # BM25 hits with offsets
# PDF notes: view="page", page_index=N returns one page (pdf_base64 + text + quality)

# DICOM metadata / preview (any slice_index)
for series in report.dicom:
    print(case.describe(series.id)["links"])  # structural SEG↔CT etc.
print(case.fetch(report.dicom[0].id, view="metadata"))
print(case.fetch(report.dicom[0].id, view="preview", slice_index=0))
print(case.dicom_select("liver CT"))  # ranked series + mm-based slice, no pixels
print(case.dicom_select("which structures are contoured"))  # RTSTRUCT + ROI names
print(case.dicom_groups("CT,SEG"))
```

## Links

Resources can be connected in two ways:

| Source | Origin | Persisted? |
|--------|--------|------------|
| `structural` | Adapters (e.g. DICOM SEG→CT references from imagetools) | Yes — derived from files |
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
case.links()                      # case-wide links (DICOM structural + all asserted)
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
# compact inventory; then: describe / fetch / notes_search / links / assert_link / dicom_tree / dicom_groups / dicom_select
# Gemini-style declarations: gemini_function_declarations()
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
