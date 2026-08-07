# Design

## North star

Build a **local medical data access harness**: open a filesystem root, auto-discover DICOM / structured data / clinical notes / NIfTI, and expose a small Discover → Describe → Fetch API that humans and LLMs can use.

Success means an agent can **inventory a case and pull only what it needs**, not that we build an EHR, PACS, or clinical decision system.

## Narrow

**In scope**

- Local directory as the unit of work (any layout; auto-discover)
- Four modalities: DICOM (via med-imagetools), structured tables/records, clinical notes, NIfTI
- DICOM cross-series references (SEG/RTSTRUCT/etc. ↔ parent CT/MR/PT) via imagetools Interlacer
- NIfTI volumes by extension; optional path templates for scan/mask linking via imagetools nifti crawl
- Progressive disclosure: list → describe/schema → bounded fetch
- Stable resource IDs, clear errors, size/row/char limits by default
- Python API first; tool schemas that mirror that API

**Out of scope (unless we explicitly expand later)**

- Agent loops, memory, planning, compaction
- FHIR clients, remote PACS, hospital auth/SSO
- Diagnosis, summarization products, or clinical safety claims
- Embedding pipelines, training, or foundation-model hosting
- Pathology/WSI, genomics, or other modalities before the core set is solid
- Required manifests or mandatory folder layouts (optional NIfTI patterns are opt-in only)
- Dumping whole series, tables, or charts into context to be helpful

## Reliable

- Discovery is cheap and conservative: classify with clear rules; unknown stays unknown
- Fetch is bounded: defaults for rows, chars, and image previews; truncation is explicit
- One code path for library callers and tools — tools never bypass the Case API
- Fail loudly with typed errors instead of silent empty success
- Prefer correct small returns over impressive large ones

## Decision test

1. Does this help **find or fetch** medical data from a local case?
2. Can it ship with a **hard limit** and a clear failure mode?
3. Does it work without assuming a folder layout?

**One-line reminder:** We are a navigable environment for medical files — not an agent, not an EHR, not a model.
