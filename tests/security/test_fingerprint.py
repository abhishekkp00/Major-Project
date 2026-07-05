from src.security import (
    collect_identifiers,
    build_canonical_string,
    compute_fingerprint_hash,
    get_fingerprint_hash,
)


class TestFingerprint:
    def test_fingerprint_hash_is_64_hex_chars(self):
        h = get_fingerprint_hash()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_fingerprint_is_reproducible(self):
        h1 = get_fingerprint_hash()
        h2 = get_fingerprint_hash()
        assert h1 == h2

    def test_fingerprint_changes_with_different_identifiers(self):
        ids_a = {"machine_id": "aaa", "cpu_model": "Intel X", "disk_uuid": "uuid-1"}
        ids_b = {"machine_id": "bbb", "cpu_model": "Intel X", "disk_uuid": "uuid-1"}
        h_a = compute_fingerprint_hash(build_canonical_string(ids_a))
        h_b = compute_fingerprint_hash(build_canonical_string(ids_b))
        assert h_a != h_b

    def test_canonical_string_is_sorted(self):
        ids = {"z_key": "zzz", "a_key": "aaa", "m_key": "mmm"}
        canon = build_canonical_string(ids)
        assert canon.index("a_key") < canon.index("m_key") < canon.index("z_key")

    def test_unavailable_sources_produce_consistent_hash(self):
        ids = {"machine_id": "UNAVAILABLE", "cpu_model": "UNAVAILABLE", "disk_uuid": "UNAVAILABLE"}
        h1 = compute_fingerprint_hash(build_canonical_string(ids))
        h2 = compute_fingerprint_hash(build_canonical_string(ids))
        assert h1 == h2

    def test_raw_identifiers_not_in_hash_output(self):
        ids = collect_identifiers()
        fp_hash = compute_fingerprint_hash(build_canonical_string(ids))
        for v in ids.values():
            if v != "UNAVAILABLE":
                assert v not in fp_hash
