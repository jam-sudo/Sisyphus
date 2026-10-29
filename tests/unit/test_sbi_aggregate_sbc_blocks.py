"""Check that block aggregation preserves counts and refuses missing cases."""

import json

import pytest

from scripts.sbi_aggregate_sbc_blocks import aggregate_blocks


def test_aggregate_blocks_counts_and_missing_case(tmp_path):
    paths = []
    for start in (0, 2):
        block = {
            "posterior": "test.pt", "validation_set": "test.json", "sampler": "corrected",
            "n_calibration": 4, "n_posterior": 500, "levels": [50, 80, 90, 95],
            "seed": 0, "ks_threshold": 0.01, "coverage_tol": 0.10,
            "case_start": start, "case_end": start + 2,
            "drugs": [{
                "name": "test", "n_available": 4, "n_valid": 2,
                "ranks": [[100, 200, 300], [400, 300, 200]],
                "coverage_counts": {str(level): [1, 1, 1]
                                    for level in (50, 80, 90, 95)},
            }],
        }
        path = tmp_path / f"{start}.json"
        path.write_text(json.dumps(block))
        paths.append(path)
    result = aggregate_blocks(paths, "multi_drug", 1)
    assert result["drugs"][0]["n_valid"] == 4
    assert result["drugs"][0]["coverage"]["50"] == [0.5, 0.5, 0.5]
    with pytest.raises(ValueError, match="Incomplete"):
        aggregate_blocks(paths[:1], "multi_drug", 1)
