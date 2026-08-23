"""Guards the real-data provenance manifest (tracks/dataset_manifest.json).

Dependency-free: validates the *committed* manifest's shape so a stale or malformed
manifest fails the suite, without importing the track loaders (which need mne/wfdb).
Regenerate the manifest with tools/build_dataset_manifest.py; integrity-check a local
cache with `--verify`.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "..", "tracks", "dataset_manifest.json")

TRACKS = {"sleep_edf", "ecg_cinc2017", "ctg_ctu_uhb", "emg_ninapro", "har", "bci_eegmmidb"}
REQUIRED = ["name", "dataset", "dataset_version", "license", "url", "citation",
            "split_unit", "smoke_test_records", "cache"]


def _manifest():
    with open(MANIFEST) as f:
        return json.load(f)


def test_manifest_covers_all_six_tracks():
    m = _manifest()
    assert m.get("schema", "").startswith("bsp-dataset-manifest")
    assert set(m["datasets"]) == TRACKS


def test_each_dataset_has_provenance_fields():
    for tid, d in _manifest()["datasets"].items():
        for field in REQUIRED:
            assert d.get(field) not in (None, "", []), f"{tid}: missing/empty {field}"
        assert isinstance(d["smoke_test_records"], list) and d["smoke_test_records"]


def test_cached_datasets_have_a_fingerprint_and_valid_checksums():
    for tid, d in _manifest()["datasets"].items():
        c = d["cache"]
        assert c["status"] in ("cached", "not-downloaded")
        if c["status"] == "cached":
            assert c["n_files"] > 0 and c["total_bytes"] > 0
            assert isinstance(c["aggregate_sha256"], str) and len(c["aggregate_sha256"]) == 64
            for rec in c["smoke_record_files"]:
                assert len(rec["sha256"]) == 64
                assert rec["record"] in d["smoke_test_records"]
