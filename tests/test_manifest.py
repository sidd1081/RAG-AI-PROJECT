import json
from app.rag.manifest import write_manifest, manifest_is_compatible


def test_freshly_written_manifest_is_compatible(tmp_path):
    write_manifest(tmp_path)
    assert manifest_is_compatible(tmp_path) is True


def test_missing_manifest_is_not_compatible(tmp_path):
    assert manifest_is_compatible(tmp_path) is False


def test_manifest_with_wrong_schema_version_is_not_compatible(tmp_path):
    write_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"

    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = -1
    manifest_path.write_text(json.dumps(manifest))

    assert manifest_is_compatible(tmp_path) is False


def test_manifest_with_wrong_embedding_model_is_not_compatible(tmp_path):
    write_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"

    manifest = json.loads(manifest_path.read_text())
    manifest["embedding_model_name"] = "some/other-model"
    manifest_path.write_text(json.dumps(manifest))

    assert manifest_is_compatible(tmp_path) is False
