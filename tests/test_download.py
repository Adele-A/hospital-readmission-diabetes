"""Light tests for download helpers that do not touch the network."""

import hashlib

from src.data.download import sha256_of


def test_sha256_of_matches_hashlib(tmp_path):
    f = tmp_path / "blob.bin"
    payload = b"readmission-portfolio\n" * 100
    f.write_bytes(payload)
    assert sha256_of(f) == hashlib.sha256(payload).hexdigest()
