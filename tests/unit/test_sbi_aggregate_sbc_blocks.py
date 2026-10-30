"""Check that block aggregation preserves counts and refuses missing cases."""

import json

import pytest

from scripts.sbi_aggregate_sbc_blocks import _holm_rejection_count, aggregate_blocks, main


def test_holm_rejections_use_one_flow_family():
    assert _holm_rejection_count([0.004, 0.006, 0.2], 0.01) == 0
    assert _holm_rejection_count([0.0001, 0.006, 0.2], 0.01) == 1


def test_aggregate_blocks_counts_and_missing_case(tmp_path, monkeypatch):
    paths = []
    for start in (0, 2):
        block = {
            "posterior": "test.pt", "validation_set": "test.json", "sampler": "corrected",
            "n_calibration": 4, "n_posterior": 500, "levels": [50, 80, 90, 95],
            "seed": 0, "ks_threshold": 0.01, "coverage_tol": 0.10,
            "populations": [{"name": "adult", "bw": 70.0, "age": 35.0}],
            "case_start": start, "case_end": start + 2,
            "drugs": [{
                "name": "test", "n_available": 4, "n_valid": 2,
                "ranks": [[100, 200, 300], [400, 300, 200]],
                "coverage_counts": {str(level): [1, 1, 1]
                                    for level in (50, 80, 90, 95)},
                "rejection_fallback_cases": int(start == 0),
                "forced_exact_rejection_cases": int(start == 0),
            }],
        }
        path = tmp_path / f"{start}.json"
        path.write_text(json.dumps(block))
        paths.append(path)
    result = aggregate_blocks(paths, "multi_drug", 1)
    assert result["validation_set"] == "data/sbi/validation_drug_set.json"
    assert result["populations"][0]["bw"] == 70.0
    assert result["drugs"][0]["n_valid"] == 4
    assert result["drugs"][0]["coverage"]["50"] == [0.5, 0.5, 0.5]
    assert result["forced_exact_rejection_cases"] == 1
    assert result["sampling_error_blocks"] == 0
    out = tmp_path.parent / "aggregated.json"
    monkeypatch.setattr("sys.argv", [
        "sbi_aggregate_sbc_blocks.py", "--flow", "multi_drug", "--blocks", str(tmp_path),
        "--expected-pairs", "1", "--out", str(out),
    ])
    main()
    written = json.loads(out.read_text())
    assert written["calibrated_null_global_pass_probability"] > 0
    assert written["holm_global_gate"]["predeclared_at_utc"] == "2026-10-29T15:40:44Z"
    assert (written["holm_global_gate"]["calibrated_null_pass_probability"]
            >= written["calibrated_null_global_pass_probability"])
    assert written["size_calibrated_global_gate"]["ks_test_count"] == 3
    assert written["size_calibrated_global_gate"]["coverage_test_count"] == 12
    assert written["size_calibrated_global_gate"]["predeclared_at_utc"] == (
        "2026-10-29T19:14:47Z"
    )
    with pytest.raises(ValueError, match="Incomplete"):
        aggregate_blocks(paths[:1], "multi_drug", 1)


def test_aggregate_blocks_reports_sampling_errors(tmp_path):
    block = {
        "posterior": "test.pt", "validation_set": "test.json", "sampler": "corrected",
        "n_calibration": 2, "n_posterior": 500, "levels": [50, 80, 90, 95],
        "seed": 0, "ks_threshold": 0.01, "coverage_tol": 0.10,
        "case_start": 0, "case_end": 2,
        "drugs": [{
            "name": "test", "n_available": 2,
            "sampling_error": "Exact rejection yielded 7/500 samples",
        }],
    }
    path = tmp_path / "0.json"
    path.write_text(json.dumps(block))
    result = aggregate_blocks([path], "multi_drug", 1)
    assert result["sampling_error_blocks"] == 1
    assert result["drugs"][0]["pass"] is False
