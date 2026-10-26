"""Default fetch bounds and hard ceilings."""

DEFAULT_TABLE_LIMIT = 20
MAX_TABLE_LIMIT = 1_000

DEFAULT_NOTE_CHARS = 4_000
MAX_NOTE_CHARS = 200_000
DEFAULT_NOTE_PREVIEW_CHARS = 500
MAX_SPAN_CHARS = 8_000
MAX_SPAN_CONTEXT_CHARS = 2_000
DEFAULT_NOTE_SEARCH_HITS = 8
MAX_NOTE_SEARCH_HITS = 50
MAX_NOTE_SECTIONS = 200
# Characters read per note at discovery for date/type/quality metadata.
NOTE_META_SNIFF_CHARS = 8_000

DEFAULT_JSON_SNIFF_BYTES = 64_000

# DICOM preview: max edge length after optional downscale (pixels).
MAX_DICOM_PREVIEW_EDGE = 1024

# DICOM volume: reject above these instead of cropping the grid.
# 512×512×128 voxels. 64 MB covers that grid as int16 before gzip.
MAX_VOLUME_VOXELS = 33_554_432
MAX_VOLUME_BYTES = 64_000_000
