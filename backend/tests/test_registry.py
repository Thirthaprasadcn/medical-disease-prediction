from backend.models.registry.gate import validate_entry
from backend.models.registry.registry import ensure_demo_entry, get_active_entry, list_entries


def test_gate_requires_fields():
    errors = validate_entry({}, require_files=False)
    assert errors


def test_demo_bootstrap():
    ensure_demo_entry()
    entries = list_entries()
    assert any(e["model_id"] == "pneumonia_cnn" for e in entries)
    active = get_active_entry()
    assert active is not None
