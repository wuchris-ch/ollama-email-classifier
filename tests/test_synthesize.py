"""The committed dataset must be exactly reproducible from synthesize.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

spec = importlib.util.spec_from_file_location("synthesize", DATA / "synthesize.py")
synthesize_mod = importlib.util.module_from_spec(spec)
sys.modules["synthesize"] = synthesize_mod
spec.loader.exec_module(synthesize_mod)


def test_synthesis_is_deterministic():
    a = synthesize_mod.synthesize(n_per_class=5, seed=42)
    b = synthesize_mod.synthesize(n_per_class=5, seed=42)
    assert [(e.subject, e.body, e.label) for e in a] == [(e.subject, e.body, e.label) for e in b]


def test_committed_jsonl_matches_regeneration(tmp_path):
    splits = {"train": (100, 1), "dev": (25, 2), "test": (34, 3)}
    for name, (n_per_class, seed) in splits.items():
        emails = synthesize_mod.synthesize(n_per_class=n_per_class, seed=seed)
        regenerated = tmp_path / f"{name}.jsonl"
        synthesize_mod.write_jsonl(emails, regenerated)
        assert regenerated.read_text() == (DATA / f"{name}.jsonl").read_text(), (
            f"data/{name}.jsonl does not match synthesize.py output — "
            "regenerate with `python data/synthesize.py` and commit"
        )


def test_splits_are_balanced():
    for name, expected_per_class in (("train", 100), ("dev", 25), ("test", 34)):
        rows = [json.loads(line) for line in (DATA / f"{name}.jsonl").read_text().splitlines() if line.strip()]
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        assert all(c == expected_per_class for c in counts.values()), counts
