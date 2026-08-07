"""Typed harness errors."""


class DataCtrlError(Exception):
    """Base error for the harness."""


class ResourceNotFound(DataCtrlError):
    """Requested resource ID is not in the open case."""


class LimitExceeded(DataCtrlError):
    """Caller asked for more than an allowed bound."""


class ModalityUnavailable(DataCtrlError):
    """Requested modality has no resources in this case."""


class UnknownView(DataCtrlError):
    """Requested fetch view is not supported for this resource."""


class InvalidFetchOptions(DataCtrlError):
    """Fetch options failed validation."""
