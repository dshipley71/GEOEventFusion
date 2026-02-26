"""GEOEventFusion I/O package.

File read/write operations only — no business logic in this layer.
"""

from geoeventfusion.io.persistence import load_json, save_json

__all__ = [
    "save_json",
    "load_json",
]
