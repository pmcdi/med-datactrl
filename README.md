# med-datactrl

## API

### `Controller`

| Method | Description |
|--------|-------------|
| `from_patient_directory(directory)` | Discover modalities from file paths and return a configured controller. |
| `get_dicom_patient_info()` | Return a dict mapping patient IDs to lists of DICOM series tree nodes. |
| `get_dicom_series_metadata(series_id)` | Return the full DICOM metadata dict for a series. |
| `get_dicom_image(series_id)` | Return a basic LLM-ready PNG payload for one DICOM series. |
| `get_dicom_overlay_image(series_id)` | Return a basic overlay-marked PNG payload for one DICOM series. |
| `list_dicom_atoms()` | Return sorted names of all registered DICOM atom getters. |
| `call_dicom_atom(name, series_id)` | Invoke a DICOM atom getter and return its value. |
| `get_pathology_info()` | Return pathology slide summaries. |
| `get_slide_metadata(slide_id)` | Return OpenSlide properties for a slide. |
| `get_pathology_image(slide_id)` | Return a basic LLM-ready PNG payload for one pathology slide. |
| `get_pathology_overlay_image(slide_id)` | Return a basic overlay-marked PNG payload for one pathology slide. |
| `list_slide_atoms()` | Return pathology slide atom getters. |
| `call_slide_atom(name, slide_id)` | Invoke a pathology slide atom getter. |
| `list_clinical_notes()` | Return clinical note summaries. |
| `get_clinical_note(note_id)` | Return the full text or JSON content of a clinical note. |
