# med-datactrl

## API

### `Controller`

| Method | Description |
|--------|-------------|
| `from_dicom_directory(directory)` | Crawl a DICOM directory and return a configured controller. |
| `get_patient_info()` | Return a dict mapping patient IDs to lists of series tree nodes. |
| `get_series_metadata(series_id)` | Return the full metadata dict for a series. |
| `list_atoms()` | Return sorted names of all registered atom getters. |
| `call_atom(name, series_id)` | Invoke an atom getter and return its value. |

### Atoms

Atoms are dynamically registered based on the tags found during crawling. Naming follows two patterns:

| Pattern | Example | Description |
|---------|---------|-------------|
| `get_{Tag}` | `get_Modality` | Base DICOM tags shared across modalities. |
| `get_{modality}_{Tag}` | `get_ct_SliceThickness` | Modality-specific tags or computed fields. |

Modality-scoped atoms raise `ValueError` if called on a series with a different modality.

