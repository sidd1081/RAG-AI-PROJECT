import json
from datetime import datetime, timezone
from pathlib import Path

from app.config.settings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    INDEX_SCHEMA_VERSION,
)

MANIFEST_FILENAME = "manifest.json"


def manifest_path(directory: Path) -> Path:
    return directory / MANIFEST_FILENAME


def write_manifest(directory: Path):
    manifest = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(manifest_path(directory), "w") as f:
        json.dump(manifest, f, indent=2)


def manifest_is_compatible(directory: Path) -> bool:
    """
    Refuses to treat an index as loadable if it was built with a different
    embedding model, dimension, or schema version than what's currently
    configured. Silently loading an incompatible index would produce
    corrupted similarity scores (or a hard crash on dimension mismatch) -
    failing loudly here and forcing a rebuild is the safe behavior.
    """
    path = manifest_path(directory)
    if not path.exists():
        return False

    try:
        with open(path, "r") as f:
            manifest = json.load(f)
    except Exception:
        return False

    return (
        manifest.get("schema_version") == INDEX_SCHEMA_VERSION
        and manifest.get("embedding_model_name") == EMBEDDING_MODEL_NAME
        and manifest.get("embedding_dimension") == EMBEDDING_DIMENSION
    )
